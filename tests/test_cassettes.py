"""Unit tests for the cassette record/replay mechanism.

These tests verify the cassette layer in isolation (no model, no pipeline)
so they run on a GPU-less Web sandbox. End-to-end TestHarness.replay()
coverage requires committed cassette fixtures and lives in
``test_pipeline.py`` once those exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from whisperlivekit.test_cassettes import (
    Cassette,
    CassetteASR,
    CassetteMissError,
    CassetteRecorder,
    audio_hash,
    token_from_dict,
    token_to_dict,
)
from whisperlivekit.timed_objects import ASRToken

# ---------------------------------------------------------------------------
# Fake ASR — minimal three-tuple backend for exercising the recorder
# ---------------------------------------------------------------------------


class _FakeASR:
    """Deterministic stand-in for an ASRBase implementation."""

    sep = ""
    SAMPLING_RATE = 16000
    backend_choice = "fake"
    original_language = "ja"

    def __init__(self):
        self.calls = 0
        # Each call returns these tokens in order:
        self._scripts = [
            [ASRToken(start=0.0, end=0.4, text="今日", detected_language="ja")],
            [
                ASRToken(start=0.4, end=0.8, text="は", detected_language="ja"),
                ASRToken(start=0.8, end=1.4, text="良い", detected_language="ja"),
            ],
            [ASRToken(start=1.4, end=2.0, text="天気", detected_language="ja")],
        ]

    def transcribe(self, audio, init_prompt=""):
        idx = min(self.calls, len(self._scripts) - 1)
        self.calls += 1
        return {"idx": idx, "audio_len": int(np.asarray(audio).shape[-1])}

    def ts_words(self, result):
        return list(self._scripts[result["idx"]])

    def segments_end_ts(self, result):
        toks = self._scripts[result["idx"]]
        return [toks[-1].end] if toks else []

    def use_vad(self):
        return False


# ---------------------------------------------------------------------------
# Hash + token round-trip
# ---------------------------------------------------------------------------


def test_audio_hash_stable_across_dtype_views():
    """Same float32 samples → same hash regardless of contiguity tricks."""
    a = np.linspace(-1, 1, 8000, dtype=np.float32)
    b = a.copy()
    assert audio_hash(a) == audio_hash(b)


def test_audio_hash_differs_for_different_audio():
    a = np.zeros(8000, dtype=np.float32)
    b = np.ones(8000, dtype=np.float32)
    assert audio_hash(a) != audio_hash(b)


def test_token_round_trip_preserves_fields():
    t = ASRToken(start=0.1, end=0.5, text="hello",
                 speaker=2, detected_language="en", probability=0.9)
    t2 = token_from_dict(token_to_dict(t))
    assert t2.start == t.start
    assert t2.end == t.end
    assert t2.text == t.text
    assert t2.speaker == t.speaker
    assert t2.detected_language == t.detected_language
    assert t2.probability == t.probability


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def test_recorder_captures_three_tuple_per_call(tmp_path: Path):
    asr = _FakeASR()
    rec = CassetteRecorder(asr, cassette_id="fake_001")

    audio_a = np.zeros(8000, dtype=np.float32)
    audio_b = np.ones(8000, dtype=np.float32)

    # Simulate the OnlineASRProcessor's exact call order.
    for audio in (audio_a, audio_b):
        res = rec.transcribe(audio, init_prompt="")
        toks = rec.ts_words(res)
        ends = rec.segments_end_ts(res)
        assert toks, "fake ASR should produce tokens"
        assert ends, "fake ASR should produce segment ends"

    out = tmp_path / "fake_001.json"
    rec.save(out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["backend"] == "fake"
    assert len(data["calls"]) == 2
    assert data["calls"][0]["audio_sha256"] == audio_hash(audio_a)
    assert data["calls"][1]["audio_sha256"] == audio_hash(audio_b)
    assert data["calls"][0]["tokens"][0]["text"] == "今日"


def test_recorder_attribute_passthrough():
    asr = _FakeASR()
    rec = CassetteRecorder(asr, cassette_id="fake_002")
    # __getattr__ delegates anything we don't override.
    assert rec.original_language == "ja"
    assert rec.SAMPLING_RATE == 16000


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------


def test_replay_roundtrip_returns_recorded_tokens(tmp_path: Path):
    asr = _FakeASR()
    rec = CassetteRecorder(asr, cassette_id="fake_rt")

    audio_a = np.zeros(8000, dtype=np.float32)
    audio_b = np.ones(8000, dtype=np.float32)

    for audio in (audio_a, audio_b):
        res = rec.transcribe(audio, init_prompt="")
        rec.ts_words(res)
        rec.segments_end_ts(res)

    out = tmp_path / "fake_rt.json"
    rec.save(out)

    replayer = CassetteASR.load(out)
    assert replayer.cassette_id == "fake_rt"
    assert replayer.n_calls == 2
    assert replayer.original_language == "ja"
    assert replayer.SAMPLING_RATE == 16000

    # Replay both calls and verify the tokens match what the recorder saw.
    res_a = replayer.transcribe(audio_a)
    toks_a = replayer.ts_words(res_a)
    assert [t.text for t in toks_a] == ["今日"]

    res_b = replayer.transcribe(audio_b)
    toks_b = replayer.ts_words(res_b)
    assert [t.text for t in toks_b] == ["は", "良い"]
    assert replayer.calls_consumed == 2


def test_replay_strict_miss_raises(tmp_path: Path):
    cassette = Cassette(cassette_id="empty", backend="fake")
    out = tmp_path / "empty.json"
    cassette.save(out)

    replayer = CassetteASR.load(out, strict=True)
    with pytest.raises(CassetteMissError):
        replayer.transcribe(np.zeros(1600, dtype=np.float32))


def test_replay_lenient_returns_empty_on_miss(tmp_path: Path):
    cassette = Cassette(cassette_id="empty", backend="fake")
    out = tmp_path / "empty.json"
    cassette.save(out)

    replayer = CassetteASR.load(out, strict=False)
    res = replayer.transcribe(np.zeros(1600, dtype=np.float32))
    assert replayer.ts_words(res) == []
    assert replayer.segments_end_ts(res) == []


def test_cassette_load_rejects_unknown_schema(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema_version": 999, "cassette_id": "x", "backend": "y"}))
    with pytest.raises(ValueError, match="schema"):
        Cassette.load(p)


def test_cassette_engine_backend_loads_via_core(tmp_path: Path):
    """The 'cassette' backend wires a CassetteASR into TranscriptionEngine."""
    asr = _FakeASR()
    rec = CassetteRecorder(asr, cassette_id="core_smoke")
    audio = np.zeros(8000, dtype=np.float32)
    res = rec.transcribe(audio)
    rec.ts_words(res)
    rec.segments_end_ts(res)
    out = tmp_path / "core_smoke.json"
    rec.save(out)

    from whisperlivekit.core import TranscriptionEngine
    TranscriptionEngine.reset()
    engine = TranscriptionEngine(
        backend="cassette",
        cassette_path=str(out),
        vac=False,
        diarization=False,
        pcm_input=True,
    )
    try:
        assert isinstance(engine.asr, CassetteASR)
        assert engine.asr.cassette_id == "core_smoke"
        assert engine.asr.n_calls == 1
    finally:
        TranscriptionEngine.reset()
