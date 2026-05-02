"""FireRedASR2 (Xiaohongshu) backend — Mandarin SOTA.

FireRedASR2-AED / FireRedASR2-LLM achieve SOTA Mandarin CER (avg 2.89% on
4 public benchmarks) and outperform Doubao-ASR, Qwen3-ASR-1.7B, Fun-ASR,
and Fun-ASR-Nano-2512 on the FireRedTeam evaluations. Supports Mandarin,
20+ Chinese dialects/accents, English, and code-switching.

Real-time characteristics — read this before picking it for live dialogue
------------------------------------------------------------------------

FireRedASR2 is a **non-causal Attention-based Encoder-Decoder**. It is *not*
a streaming model. WhisperLiveKit drives it through the LocalAgreement
policy, which means each ``OnlineASRProcessor`` cycle re-runs full
inference on the growing audio buffer (up to ``buffer_trimming_sec``
seconds — 15 s by default).

In practice that gives **quasi-realtime** transcription suitable for live
captioning / meeting transcription where ~1-2 s latency is acceptable. For
sub-second dialogue UI (robot interaction, voice control), use Voxtral
Realtime (~480 ms) or one of the SimulStreaming variants (Qwen3-SimulKV,
Qwen3-MLX-Simul, SimulStreaming) instead.

Implementation notes
--------------------

* The upstream Python API in ``fireredasr2s.fireredasr2`` expects **batches
  of file paths**, not numpy arrays:

      model.transcribe([uttid], [wav_path]) -> [{"text": ..., "timestamp": ...}]

  WhisperLiveKit's :class:`OnlineASRProcessor` passes numpy buffers, so this
  wrapper writes each chunk to a per-call temp WAV (16 kHz mono PCM) and
  feeds the path. On Linux the temp directory is ``/dev/shm`` (tmpfs in
  RAM) when available, eliminating disk I/O from the hot path; macOS and
  Windows fall back to the system temp dir.

* Word-level timestamps are available with the AED variant when
  ``return_timestamp=True`` is set on ``FireRedAsr2Config``; the LLM variant
  does not produce per-word timing.

* No streaming AlignAtt support — FireRed v2 is non-causal AED. The
  backend is therefore wired through the LocalAgreement policy only.

* This module never imports ``fireredasr2s`` at import time. All model code
  is loaded lazily inside :meth:`FireRedASR2.load_model`, so the file is
  safe to import in cassette-replay sandboxes that have no GPU and no HF
  access.

References
----------
* https://github.com/FireRedTeam/FireRedASR2S
* https://huggingface.co/FireRedTeam/FireRedASR2-AED
* https://huggingface.co/FireRedTeam/FireRedASR2-LLM
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


# Short convenience names → HuggingFace / local model identifiers.
# Users can also pass a full HF id or a local directory via ``--model-dir``.
FIRERED_MODEL_MAPPING = {
    "firered2-aed": "FireRedTeam/FireRedASR2-AED",
    "firered2-llm": "FireRedTeam/FireRedASR2-LLM",
    "firered-aed":  "FireRedTeam/FireRedASR2-AED",
    "firered-llm": "FireRedTeam/FireRedASR2-LLM",
    "firered":      "FireRedTeam/FireRedASR2-AED",  # default to AED (smaller, faster, has word ts)
    "firered2":     "FireRedTeam/FireRedASR2-AED",
}

# CER-aware sentence boundary marks; used by ``segments_end_ts``.
_PUNCTUATION_ENDS = set(".!?。！？；;")


def _resolve_variant_and_path(model_size: Optional[str], model_dir: Optional[str]) -> tuple:
    """Return (variant, model_path) where variant is 'aed' or 'llm'.

    AED is the default — smaller, faster, supports word timestamps.
    """
    if model_dir:
        path = model_dir
        variant = "llm" if "LLM" in os.path.basename(path).upper() else "aed"
        return variant, path

    key = (model_size or "firered").lower()
    path = FIRERED_MODEL_MAPPING.get(key, model_size or FIRERED_MODEL_MAPPING["firered"])
    variant = "llm" if "LLM" in path.upper() else "aed"
    return variant, path


def _resolve_tmp_root() -> str:
    """Return the fastest writable temp directory for buffered audio.

    Linux ``/dev/shm`` is a tmpfs mount backed by RAM, so writing the
    per-call WAV there avoids disk I/O entirely (~1-2 ms vs ~5-10 ms on
    a typical SSD). Falls back to the OS default temp dir on macOS,
    Windows, or any system where ``/dev/shm`` is missing or read-only.
    """
    if os.name == "posix":
        candidate = "/dev/shm"
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return tempfile.gettempdir()


class FireRedASR2(ASRBase):
    """LocalAgreement-compatible wrapper around FireRedASR2."""

    sep = ""           # CJK tokens carry no separators between them
    SAMPLING_RATE = 16000

    def __init__(
        self,
        lan: str = "auto",
        model_size: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        logfile=sys.stderr,
        **kwargs,
    ):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self._call_counter = 0  # for unique uttids per session
        # Per-call temp WAVs land here (/dev/shm on Linux). Each call uses
        # tempfile.mkstemp inside this dir so there is no persistent file
        # to leak — the directory itself is shared (e.g. /dev/shm) and
        # not owned by us.
        self._tmp_root = _resolve_tmp_root()
        self._variant, self._model_path = _resolve_variant_and_path(model_size, model_dir)
        self.model = self.load_model(model_size, cache_dir, model_dir)
        logger.info("FireRedASR2 temp dir: %s", self._tmp_root)

    # ── Model loading (lazy import to keep this file safe on Web sandboxes) ──

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        try:
            import torch
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        except ImportError as exc:
            raise ImportError(
                "FireRedASR2 backend requires the 'fireredasr2s' package. "
                "Install it from https://github.com/FireRedTeam/FireRedASR2S "
                "(no PyPI release at time of writing)."
            ) from exc

        use_gpu = bool(torch.cuda.is_available())
        if self._variant == "aed":
            cfg = FireRedAsr2Config(
                use_gpu=use_gpu,
                use_half=False,
                beam_size=3,
                nbest=1,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=True,
            )
        else:  # llm
            cfg = FireRedAsr2Config(
                use_gpu=use_gpu,
                decode_min_len=0,
                repetition_penalty=3.0,
                llm_length_penalty=1.0,
                temperature=1.0,
            )

        logger.info(
            "Loading FireRedASR2 (%s) from %s (gpu=%s)",
            self._variant, self._model_path, use_gpu,
        )
        return FireRedAsr2.from_pretrained(self._variant, self._model_path, cfg)

    # ── Inference ──

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        """Write the audio buffer to a temp WAV and run a single-utterance batch.

        The WAV lives in ``self._tmp_root`` (``/dev/shm`` on Linux when
        available) and is unlinked immediately after inference returns,
        so disk usage is bounded by one in-flight call.
        """
        self._call_counter += 1
        uttid = f"buf_{self._call_counter:06d}"

        # mkstemp gives a unique path inside the chosen tmpfs/temp dir
        # without us needing to maintain our own collision-avoiding scheme.
        fd, wav_path = tempfile.mkstemp(
            prefix="firered_", suffix=".wav", dir=self._tmp_root,
        )
        os.close(fd)

        # FireRedASR2 requires 16 kHz 16-bit mono PCM. Audio entering this
        # method is float32 [-1, 1] from the audio pipeline.
        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767).astype(np.int16)
        sf.write(wav_path, audio_i16, self.SAMPLING_RATE, subtype="PCM_16")

        try:
            results = self.model.transcribe([uttid], [wav_path])
        except Exception:
            logger.exception("FireRedASR2 transcribe failed")
            results = [{"uttid": uttid, "text": "", "timestamp": []}]
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        if not results:
            return {"text": "", "timestamp": [], "_audio_duration": len(audio) / self.SAMPLING_RATE}

        result = dict(results[0])
        result["_audio_duration"] = len(audio) / self.SAMPLING_RATE
        return result

    # ── Result extraction ──

    def ts_words(self, result) -> List[ASRToken]:
        text = (result.get("text") or "").strip()
        if not text:
            return []

        timestamps = result.get("timestamp") or []
        # FireRedASR2-AED produces (word, start_s, end_s) tuples. The LLM
        # variant returns no timestamps; fall back to even spacing.
        if timestamps:
            tokens: List[ASRToken] = []
            for i, item in enumerate(timestamps):
                if isinstance(item, dict):
                    word = item.get("text") or item.get("word") or ""
                    start = float(item.get("start_ms", 0)) / 1000.0 \
                        if "start_ms" in item else float(item.get("start", 0.0))
                    end = float(item.get("end_ms", 0)) / 1000.0 \
                        if "end_ms" in item else float(item.get("end", 0.0))
                else:
                    word, start, end = item
                if not word:
                    continue
                # Match faster-whisper convention: tokens after the first carry
                # a leading space so ''.join works in Segment.from_tokens.
                token_text = word if i == 0 else " " + word
                tokens.append(ASRToken(
                    start=float(start), end=float(end), text=token_text,
                    detected_language=self.original_language or "zh",
                ))
            return tokens

        # No timestamps — distribute words uniformly across the audio span.
        # CJK output has no whitespace so we tokenise per character and emit
        # tokens without a leading space; ASCII output is whitespace-tokenised
        # and follows the faster-whisper convention of " <word>".
        is_ascii = text.isascii()
        units = text.split() if is_ascii else list(text)
        duration = result.get("_audio_duration", 0.0) or 0.0
        if not units or duration <= 0:
            return []
        step = duration / len(units)
        return [
            ASRToken(
                start=round(i * step, 3),
                end=round((i + 1) * step, 3),
                text=u if i == 0 else (" " + u if is_ascii else u),
                detected_language=self.original_language or "zh",
            )
            for i, u in enumerate(units)
        ]

    def segments_end_ts(self, result) -> List[float]:
        timestamps = result.get("timestamp") or []
        if not timestamps:
            duration = result.get("_audio_duration", 0.0) or 0.0
            return [duration] if duration > 0 else []

        ends: List[float] = []
        last_end = 0.0
        for item in timestamps:
            if isinstance(item, dict):
                word = item.get("text") or item.get("word") or ""
                end = float(item.get("end_ms", 0)) / 1000.0 \
                    if "end_ms" in item else float(item.get("end", 0.0))
            else:
                word, _start, end = item
            last_end = float(end)
            if word and word.rstrip()[-1:] in _PUNCTUATION_ENDS:
                ends.append(last_end)
        if not ends or ends[-1] != last_end:
            ends.append(last_end)
        return ends

    def use_vad(self) -> bool:
        # FireRedASR2 has its own integrated VAD (FireRedVAD) when used as the
        # full ``FireRedAsr2System``, but the per-utterance ``FireRedAsr2``
        # entry point used here does not gate on VAD — leave it to
        # WhisperLiveKit's pipeline.
        return False
