"""Record/replay cassettes for ASR backends — GPU-free pipeline testing.

The cassette mechanism captures the interaction between ``OnlineASRProcessor``
and an ASR backend (the three-tuple ``transcribe → ts_words → segments_end_ts``)
so that the surrounding pipeline (FFmpeg, VAD, LocalAgreement / SimulStreaming,
DiffTracker, output formatting) can be exercised end-to-end without loading any
model.

Tier 1 (Web sandbox)::

    from whisperlivekit.test_cassettes import CassetteASR

    asr = CassetteASR.load("tests/cassettes/qwen3_ja_short_001.json")
    # asr exposes transcribe/ts_words/segments_end_ts/use_vad/sep
    # plug it into TestHarness.replay() or OnlineASRProcessor directly

Tier 2 (GPU machine)::

    from whisperlivekit.test_cassettes import CassetteRecorder
    real_asr = ...   # any ASRBase or Qwen3ASR / VoxtralHFStreamingASR / ...
    rec = CassetteRecorder(real_asr, cassette_id="qwen3_ja_short_001")
    online = OnlineASRProcessor(rec)
    # ... run the pipeline normally ...
    rec.save("tests/cassettes/qwen3_ja_short_001.json")

Cassette JSON schema (v1)::

    {
      "schema_version": 1,
      "cassette_id": "qwen3_ja_short_001",
      "backend": "qwen3",
      "model_id": "Qwen/Qwen3-ASR-1.7B",
      "language": "ja",
      "sep": "",
      "use_vad": false,
      "sampling_rate": 16000,
      "recorded_at": "2026-05-02T12:00:00Z",
      "audio_sha256": "...",
      "calls": [
        {
          "audio_len_samples": 8000,
          "audio_sha256": "...",
          "init_prompt": "",
          "tokens": [
            {"start": 0.12, "end": 0.45, "text": "今日", "speaker": -1,
             "detected_language": "ja", "probability": null}
          ],
          "segment_ends": [0.5]
        },
        ...
      ]
    }

Streaming backends (Voxtral HF, SimulStreaming) that do not expose the simple
three-tuple are out of scope for v1; they need a separate cassette family.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)

CASSETTE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _audio_to_bytes(audio: Any) -> bytes:
    """Normalise an audio buffer to canonical bytes for hashing."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32, copy=False).tobytes()
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    raise TypeError(f"Unsupported audio type for hashing: {type(audio)!r}")


def audio_hash(audio: Any) -> str:
    """Return a stable hex sha256 over the audio buffer."""
    return hashlib.sha256(_audio_to_bytes(audio)).hexdigest()


# ---------------------------------------------------------------------------
# Token (de)serialisation
# ---------------------------------------------------------------------------


def token_to_dict(t: ASRToken) -> Dict[str, Any]:
    return {
        "start": t.start,
        "end": t.end,
        "text": t.text,
        "speaker": t.speaker,
        "detected_language": t.detected_language,
        "probability": t.probability,
    }


def token_from_dict(d: Dict[str, Any]) -> ASRToken:
    return ASRToken(
        start=d.get("start", 0.0),
        end=d.get("end", 0.0),
        text=d.get("text", ""),
        speaker=d.get("speaker", -1),
        detected_language=d.get("detected_language"),
        probability=d.get("probability"),
    )


# ---------------------------------------------------------------------------
# Cassette data model
# ---------------------------------------------------------------------------


@dataclass
class CassetteCall:
    audio_len_samples: int
    audio_sha256: str
    init_prompt: str
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    segment_ends: List[float] = field(default_factory=list)


@dataclass
class Cassette:
    cassette_id: str
    backend: str
    model_id: str = ""
    language: Optional[str] = None
    sep: str = " "
    use_vad: bool = False
    sampling_rate: int = 16000
    schema_version: int = CASSETTE_SCHEMA_VERSION
    recorded_at: str = ""
    audio_sha256: str = ""
    calls: List[CassetteCall] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        logger.info("Cassette saved: %s (%d calls)", p, len(self.calls))

    @classmethod
    def load(cls, path: str | Path) -> "Cassette":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ver = data.get("schema_version", 1)
        if ver != CASSETTE_SCHEMA_VERSION:
            raise ValueError(
                f"Cassette schema {ver} not supported "
                f"(expected {CASSETTE_SCHEMA_VERSION})"
            )
        calls = [CassetteCall(**c) for c in data.get("calls", [])]
        data["calls"] = calls
        return cls(**data)


# ---------------------------------------------------------------------------
# Recorder — wraps a real ASR and writes a cassette
# ---------------------------------------------------------------------------


class CassetteRecorder:
    """Wrap a real ASR backend and capture every transcribe call.

    Forwards everything to the wrapped ASR via __getattr__, but intercepts
    ``transcribe``, ``ts_words`` and ``segments_end_ts`` to record the
    inputs, derived tokens and segment ends.

    Save the cassette via ``save(path)`` after the pipeline run completes.
    """

    def __init__(
        self,
        asr: Any,
        cassette_id: str,
        backend: str = "",
        model_id: str = "",
        language: Optional[str] = None,
    ) -> None:
        self._asr = asr
        self._cassette = Cassette(
            cassette_id=cassette_id,
            backend=backend or getattr(asr, "backend_choice", asr.__class__.__name__),
            model_id=model_id,
            language=language if language is not None else getattr(asr, "original_language", None),
            sep=getattr(asr, "sep", " "),
            use_vad=bool(getattr(asr, "use_vad", lambda: False)()),
            sampling_rate=getattr(asr, "SAMPLING_RATE", 16000),
            recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )
        self._pending_call: Optional[CassetteCall] = None
        self._pending_result: Any = None

    # The OnlineASRProcessor calls transcribe → ts_words → segments_end_ts in
    # this exact order on a single result, so we stash the result between
    # calls and finalise the cassette entry on segments_end_ts.

    def transcribe(self, audio: Any, init_prompt: str = ""):
        result = self._asr.transcribe(audio, init_prompt=init_prompt)
        self._pending_call = CassetteCall(
            audio_len_samples=int(np.asarray(audio).shape[-1]) if hasattr(audio, "shape") else len(audio),
            audio_sha256=audio_hash(audio),
            init_prompt=init_prompt or "",
        )
        self._pending_result = result
        return result

    def ts_words(self, result) -> List[ASRToken]:
        tokens = self._asr.ts_words(result)
        if self._pending_call is not None:
            self._pending_call.tokens = [token_to_dict(t) for t in tokens]
        return tokens

    def segments_end_ts(self, result) -> List[float]:
        ends = self._asr.segments_end_ts(result)
        if self._pending_call is not None:
            self._pending_call.segment_ends = [float(e) for e in ends]
            self._cassette.calls.append(self._pending_call)
            self._pending_call = None
            self._pending_result = None
        return ends

    def use_vad(self) -> bool:
        return bool(self._asr.use_vad())

    @property
    def sep(self) -> str:
        return getattr(self._asr, "sep", " ")

    def __getattr__(self, name: str) -> Any:
        # Forward anything we don't override (original_language, tokenizer, ...).
        return getattr(self._asr, name)

    def save(self, path: str | Path) -> None:
        self._cassette.save(path)


# ---------------------------------------------------------------------------
# Replayer — pretends to be an ASR backend, returns recorded tokens
# ---------------------------------------------------------------------------


class CassetteMissError(LookupError):
    """Raised when the cassette has no entry matching the requested call."""


class _ReplayResult:
    """Sentinel object returned from CassetteASR.transcribe.

    Carries the matched call so ts_words/segments_end_ts can recover the
    recorded tokens without needing to re-hash the audio.
    """

    __slots__ = ("call",)

    def __init__(self, call: CassetteCall) -> None:
        self.call = call


class CassetteASR:
    """Drop-in replacement for an ASR backend, served from a cassette.

    Expected interface (matches ASRBase / Qwen3ASR / FasterWhisperASR):
      - sep
      - SAMPLING_RATE
      - original_language
      - transcribe(audio, init_prompt="")
      - ts_words(result)
      - segments_end_ts(result)
      - use_vad()
    """

    def __init__(self, cassette: Cassette, strict: bool = True) -> None:
        self._cassette = cassette
        self._strict = strict
        self.sep = cassette.sep
        self.SAMPLING_RATE = cassette.sampling_rate
        self.original_language = cassette.language
        self.backend_choice = cassette.backend
        self.tokenizer = None
        self._next_call_idx = 0
        self._index_by_hash: Dict[str, List[int]] = {}
        for i, call in enumerate(cassette.calls):
            self._index_by_hash.setdefault(call.audio_sha256, []).append(i)

    # ── Construction helpers ──

    @classmethod
    def load(cls, path: str | Path, strict: bool = True) -> "CassetteASR":
        return cls(Cassette.load(path), strict=strict)

    # ── ASR interface ──

    def transcribe(self, audio: Any, init_prompt: str = ""):
        h = audio_hash(audio)
        # Prefer hash match (deterministic) and fall back to call-order match
        # so that tests remain robust to small audio normalisation drift.
        idxs = self._index_by_hash.get(h, [])
        call: Optional[CassetteCall] = None
        for idx in idxs:
            if idx >= self._next_call_idx:
                call = self._cassette.calls[idx]
                self._next_call_idx = idx + 1
                break
        if call is None:
            if self._strict:
                raise CassetteMissError(
                    f"Cassette {self._cassette.cassette_id!r} has no matching "
                    f"entry for call #{self._next_call_idx} "
                    f"(audio_sha256={h[:12]}…, len={getattr(audio, 'shape', (len(audio),))})"
                )
            # Non-strict fallback: empty result so the pipeline can keep going.
            call = CassetteCall(
                audio_len_samples=int(np.asarray(audio).shape[-1]) if hasattr(audio, "shape") else len(audio),
                audio_sha256=h,
                init_prompt=init_prompt or "",
                tokens=[],
                segment_ends=[],
            )
        return _ReplayResult(call)

    def ts_words(self, result: _ReplayResult) -> List[ASRToken]:
        return [token_from_dict(d) for d in result.call.tokens]

    def segments_end_ts(self, result: _ReplayResult) -> List[float]:
        return list(result.call.segment_ends)

    def use_vad(self) -> bool:
        return self._cassette.use_vad

    # ── Diagnostics ──

    @property
    def cassette_id(self) -> str:
        return self._cassette.cassette_id

    @property
    def n_calls(self) -> int:
        return len(self._cassette.calls)

    @property
    def calls_consumed(self) -> int:
        return self._next_call_idx
