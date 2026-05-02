"""SenseVoice (Alibaba FunAudioLLM) backend — multilingual unified model.

SenseVoice is a non-autoregressive end-to-end speech understanding model
covering Mandarin, Cantonese, English, Japanese, and Korean in one model,
with built-in speech emotion recognition and audio event detection. Useful
for HRI / robotics where a single model needs to accept any of the above
without language switching.

Implementation notes
--------------------

* Loaded through ``funasr.AutoModel`` with ``model="iic/SenseVoiceSmall"``.
  ``AutoModel.generate(input=...)`` accepts a numpy float32 array directly
  (or a file path), returning ``[{"text": "<|zh|><|HAPPY|>...transcription"}]``.

* Output text is prefixed with metadata tags ``<|lang|><|emotion|>
  <|event|><|itn|>`` that must be stripped before joining with downstream
  WhisperLiveKit tokens. We expose the parsed metadata via
  :class:`ASRToken.detected_language` and store emotion / event on the
  result object so callers can read them via ``ts_words`` if interested.

* SenseVoice-Small is non-autoregressive and emits no per-word timestamps
  by default. The 2024-11 update added CTC-alignment-based timestamps but
  not all checkpoints expose them. When timestamps are missing we fall
  back to even-spaced word boundaries — same strategy as FireRedASR2-LLM
  and Qwen3-ASR.

* No SimulStreaming policy support (non-causal). Wired through
  LocalAgreement only.

References
----------
* https://github.com/FunAudioLLM/SenseVoice
* https://huggingface.co/FunAudioLLM/SenseVoiceSmall
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


# Default checkpoint. ``model_dir`` or a HF id passed via ``--model-size``
# can override this in TranscriptionEngine init.
DEFAULT_SENSEVOICE_MODEL = "iic/SenseVoiceSmall"

# SenseVoice metadata tag names → values it emits. Used to decide which
# tag class a captured ``<|...|>`` belongs to.
_LANG_TAGS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
_EVENT_TAGS = {"BGM", "Speech", "Applause", "Laughter", "Cry", "Sneeze",
               "Breath", "Cough"}
_EMO_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED",
             "SURPRISED", "EMO_UNKNOWN"}
_ITN_TAGS = {"woitn", "withitn"}

_TAG_RE = re.compile(r"<\|([^|>]+)\|>")
_PUNCTUATION_ENDS = set(".!?。！？；;")


def _parse_metadata(raw: str) -> Dict[str, Any]:
    """Strip SenseVoice ``<|...|>`` tags and return them alongside clean text.

    Returns a dict with: text (str without tags), language (str|None),
    emotion (str|None), audio_event (str|None), itn (str|None).
    """
    language: Optional[str] = None
    emotion: Optional[str] = None
    event: Optional[str] = None
    itn: Optional[str] = None

    def _capture(match: re.Match) -> str:
        nonlocal language, emotion, event, itn
        tag = match.group(1)
        if tag in _LANG_TAGS:
            language = tag
        elif tag in _EMO_TAGS:
            emotion = tag
        elif tag in _EVENT_TAGS:
            event = tag
        elif tag in _ITN_TAGS:
            itn = tag
        return ""

    cleaned = _TAG_RE.sub(_capture, raw or "").strip()
    return {
        "text": cleaned,
        "language": language,
        "emotion": emotion,
        "audio_event": event,
        "itn": itn,
    }


class SenseVoiceASR(ASRBase):
    """LocalAgreement-compatible wrapper around SenseVoice-Small."""

    sep = ""           # mostly used with CJK output
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
        self.model = self.load_model(model_size, cache_dir, model_dir)

    # ── Model loading (lazy import) ──

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError(
                "SenseVoice backend requires the 'funasr' package. "
                "Install it with: pip install funasr"
            ) from exc

        # Resolve which checkpoint to load.
        if model_dir:
            target = model_dir
        elif model_size and ("/" in model_size or model_size.startswith(".")):
            target = model_size
        else:
            target = DEFAULT_SENSEVOICE_MODEL

        # We disable the bundled VAD (vad_model=None) because WhisperLiveKit
        # already runs Silero VAD in the audio pipeline.
        logger.info("Loading SenseVoice from %s ...", target)
        return AutoModel(
            model=target,
            trust_remote_code=True,
            vad_model=None,
            disable_update=True,
        )

    # ── Inference ──

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        # SenseVoice-Small accepts a numpy float32 1-D array directly.
        audio_in = np.asarray(audio, dtype=np.float32)
        # Per-call language: if the caller pinned a language, pass it through;
        # otherwise let SenseVoice auto-detect.
        language = self.original_language if self.original_language else "auto"

        try:
            results = self.model.generate(
                input=audio_in,
                cache={},
                language=language,
                use_itn=True,
                merge_vad=False,
            )
        except Exception:
            logger.exception("SenseVoice generate failed")
            results = [{"text": ""}]

        if not results:
            return {"text": "", "_audio_duration": len(audio) / self.SAMPLING_RATE}

        first = dict(results[0])
        meta = _parse_metadata(first.get("text", ""))
        merged = {**first, **meta}
        merged["_audio_duration"] = len(audio) / self.SAMPLING_RATE
        return merged

    # ── Result extraction ──

    def ts_words(self, result) -> List[ASRToken]:
        text = (result.get("text") or "").strip()
        if not text:
            return []
        detected = result.get("language") or self.original_language

        # If the upstream model produced char/word-level timestamps (rare for
        # SenseVoice-Small), honour them.
        timestamps = result.get("timestamp") or []
        if timestamps:
            return self._tokens_from_timestamps(timestamps, detected)

        # Otherwise, distribute uniformly over the audio span.
        units = list(text) if not text.isascii() else text.split()
        duration = result.get("_audio_duration", 0.0) or 0.0
        if not units or duration <= 0:
            return []
        step = duration / len(units)
        return [
            ASRToken(
                start=round(i * step, 3),
                end=round((i + 1) * step, 3),
                text=u if i == 0 else (" " + u if text.isascii() else u),
                detected_language=detected,
            )
            for i, u in enumerate(units)
        ]

    @staticmethod
    def _tokens_from_timestamps(items, detected_language) -> List[ASRToken]:
        tokens: List[ASRToken] = []
        for i, item in enumerate(items):
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
            tokens.append(ASRToken(
                start=float(start), end=float(end),
                text=word if i == 0 else (" " + word if word.isascii() else word),
                detected_language=detected_language,
            ))
        return tokens

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
        return False
