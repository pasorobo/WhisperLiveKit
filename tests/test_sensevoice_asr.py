"""Isolation tests for the SenseVoice backend wrapper.

These tests do not load the SenseVoice model — they exercise the
metadata-tag parser and ``ts_words`` / ``segments_end_ts`` against
handcrafted result dicts that mimic FunASR's ``AutoModel.generate`` output.

End-to-end coverage requires a GPU machine with HF / ModelScope access and
lives in a future cassette-replay test.
"""

from __future__ import annotations

import pytest

from whisperlivekit.sensevoice_asr import SenseVoiceASR, _parse_metadata

# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def test_parse_metadata_strips_all_tags():
    raw = "<|zh|><|HAPPY|><|Speech|><|woitn|>你好世界。"
    meta = _parse_metadata(raw)
    assert meta["text"] == "你好世界。"
    assert meta["language"] == "zh"
    assert meta["emotion"] == "HAPPY"
    assert meta["audio_event"] == "Speech"
    assert meta["itn"] == "woitn"


def test_parse_metadata_handles_no_tags():
    meta = _parse_metadata("Hello world")
    assert meta["text"] == "Hello world"
    assert meta["language"] is None
    assert meta["emotion"] is None
    assert meta["audio_event"] is None


def test_parse_metadata_handles_empty():
    meta = _parse_metadata("")
    assert meta["text"] == ""
    assert all(meta[k] is None for k in ("language", "emotion", "audio_event", "itn"))


def test_parse_metadata_japanese_with_laughter_event():
    raw = "<|ja|><|NEUTRAL|><|Laughter|><|withitn|>こんにちは"
    meta = _parse_metadata(raw)
    assert meta["text"] == "こんにちは"
    assert meta["language"] == "ja"
    assert meta["audio_event"] == "Laughter"


def test_parse_metadata_english_with_applause():
    raw = "<|en|><|HAPPY|><|Applause|><|withitn|>Thank you"
    meta = _parse_metadata(raw)
    assert meta["text"] == "Thank you"
    assert meta["language"] == "en"
    assert meta["emotion"] == "HAPPY"
    assert meta["audio_event"] == "Applause"


def test_parse_metadata_unknown_tag_kept_in_none():
    """Unknown tags are stripped but do not crash; nothing is captured."""
    meta = _parse_metadata("<|MYSTERY|>hello")
    assert meta["text"] == "hello"
    assert meta["language"] is None


# ---------------------------------------------------------------------------
# Token extraction — bypass __init__ so no model is loaded
# ---------------------------------------------------------------------------


def _make_wrapper(language=None):
    asr = SenseVoiceASR.__new__(SenseVoiceASR)
    asr.original_language = language
    asr.transcribe_kargs = {}
    return asr


def test_ts_words_uniform_cjk_spacing():
    asr = _make_wrapper()
    result = {"text": "你好世界", "language": "zh", "_audio_duration": 4.0}
    tokens = asr.ts_words(result)
    assert len(tokens) == 4
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[-1].end == pytest.approx(4.0)
    # CJK tokens carry no leading space — joining must reproduce the original.
    assert "".join(t.text for t in tokens) == "你好世界"
    assert all(t.detected_language == "zh" for t in tokens)


def test_ts_words_uniform_english_spacing():
    asr = _make_wrapper()
    result = {"text": "hello speech world", "language": "en", "_audio_duration": 3.0}
    tokens = asr.ts_words(result)
    assert len(tokens) == 3
    assert tokens[0].text == "hello"
    assert tokens[1].text == " speech"
    assert tokens[2].text == " world"


def test_ts_words_with_explicit_timestamps():
    asr = _make_wrapper()
    result = {
        "text": "你好",
        "language": "zh",
        "timestamp": [
            {"text": "你", "start_ms": 100, "end_ms": 400},
            {"text": "好", "start_ms": 400, "end_ms": 800},
        ],
        "_audio_duration": 1.0,
    }
    tokens = asr.ts_words(result)
    assert tokens[0].start == pytest.approx(0.1)
    assert tokens[1].end == pytest.approx(0.8)


def test_ts_words_returns_empty_for_no_text():
    asr = _make_wrapper()
    assert asr.ts_words({"text": "", "_audio_duration": 1.0}) == []


def test_ts_words_propagates_detected_language_from_metadata():
    """Even when caller's lan was 'auto', detected language overrides per-token."""
    asr = _make_wrapper(language=None)
    result = {"text": "안녕하세요", "language": "ko", "_audio_duration": 2.0}
    tokens = asr.ts_words(result)
    assert all(t.detected_language == "ko" for t in tokens)


def test_segments_end_ts_falls_back_to_audio_duration():
    asr = _make_wrapper()
    assert asr.segments_end_ts({"text": "...", "_audio_duration": 5.5}) == [5.5]


def test_use_vad_is_false():
    asr = _make_wrapper()
    assert asr.use_vad() is False
