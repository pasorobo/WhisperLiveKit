"""Isolation tests for the FireRedASR2 backend wrapper.

These tests do not load the FireRedASR2 model — they construct the wrapper
without calling __init__ and exercise ``ts_words`` / ``segments_end_ts``
against handcrafted result dicts that mimic the upstream output format
documented at https://github.com/FireRedTeam/FireRedASR2S.

End-to-end coverage requires a GPU machine with HF / ModelScope access and
lives in a future cassette-replay test.
"""

from __future__ import annotations

import pytest

from whisperlivekit.firered_asr import (
    FIRERED_MODEL_MAPPING,
    FireRedASR2,
    _resolve_variant_and_path,
)

# ---------------------------------------------------------------------------
# Variant / path resolution
# ---------------------------------------------------------------------------


def test_default_resolves_to_aed():
    variant, path = _resolve_variant_and_path(None, None)
    assert variant == "aed"
    assert path == FIRERED_MODEL_MAPPING["firered"]


def test_explicit_llm_keyword():
    variant, path = _resolve_variant_and_path("firered2-llm", None)
    assert variant == "llm"
    assert "LLM" in path


def test_local_model_dir_with_llm_in_name():
    variant, path = _resolve_variant_and_path(None, "/models/FireRedASR2-LLM")
    assert variant == "llm"
    assert path == "/models/FireRedASR2-LLM"


def test_local_model_dir_with_aed_in_name():
    variant, path = _resolve_variant_and_path(None, "/models/FireRedASR2-AED")
    assert variant == "aed"
    assert path == "/models/FireRedASR2-AED"


# ---------------------------------------------------------------------------
# Token extraction — bypass __init__ so no model is loaded
# ---------------------------------------------------------------------------


def _make_wrapper(language="zh"):
    """Construct a FireRedASR2 instance without loading the model."""
    asr = FireRedASR2.__new__(FireRedASR2)
    asr.original_language = language
    asr.transcribe_kargs = {}
    return asr


def test_ts_words_with_tuple_timestamps():
    asr = _make_wrapper()
    result = {
        "uttid": "buf_000001",
        "text": "你好世界",
        "timestamp": [
            ("你", 0.42, 0.66),
            ("好", 0.66, 1.10),
            ("世", 1.10, 1.34),
            ("界", 1.34, 2.04),
        ],
        "_audio_duration": 2.32,
    }
    tokens = asr.ts_words(result)
    assert len(tokens) == 4
    assert tokens[0].text == "你"             # first token has no leading space
    assert tokens[1].text == " 好"            # subsequent tokens prefixed with " "
    assert tokens[0].start == pytest.approx(0.42)
    assert tokens[3].end == pytest.approx(2.04)
    assert all(t.detected_language == "zh" for t in tokens)


def test_ts_words_with_dict_timestamps_in_ms():
    asr = _make_wrapper()
    result = {
        "text": "你好",
        "timestamp": [
            {"text": "你", "start_ms": 490, "end_ms": 690},
            {"text": "好", "start_ms": 690, "end_ms": 1090},
        ],
        "_audio_duration": 1.5,
    }
    tokens = asr.ts_words(result)
    assert len(tokens) == 2
    assert tokens[0].start == pytest.approx(0.49)
    assert tokens[1].end == pytest.approx(1.09)


def test_ts_words_falls_back_to_uniform_spacing():
    """LLM variant emits no timestamps — spacing should be uniform across audio."""
    asr = _make_wrapper(language="zh")
    result = {"text": "今天天气真好", "timestamp": [], "_audio_duration": 6.0}
    tokens = asr.ts_words(result)
    assert len(tokens) == 6
    assert tokens[0].start == pytest.approx(0.0)
    assert tokens[-1].end == pytest.approx(6.0)
    assert tokens[3].text == "气"  # 4th char ("气"), no leading space because text isn't ASCII


def test_ts_words_returns_empty_for_empty_text():
    asr = _make_wrapper()
    assert asr.ts_words({"text": "", "timestamp": [], "_audio_duration": 1.0}) == []
    assert asr.ts_words({"text": None, "timestamp": [], "_audio_duration": 1.0}) == []


def test_segments_end_ts_uses_punctuation_boundaries():
    asr = _make_wrapper()
    result = {
        "text": "你好。世界！",
        "timestamp": [
            ("你", 0.0, 0.4),
            ("好", 0.4, 0.7),
            ("。", 0.7, 0.8),
            ("世", 0.8, 1.1),
            ("界", 1.1, 1.4),
            ("！", 1.4, 1.5),
        ],
    }
    ends = asr.segments_end_ts(result)
    assert 0.8 in ends      # end of "。"
    assert 1.5 in ends      # end of "！"


def test_segments_end_ts_falls_back_to_audio_duration():
    asr = _make_wrapper()
    result = {"text": "no timestamps", "timestamp": [], "_audio_duration": 3.7}
    assert asr.segments_end_ts(result) == [3.7]


def test_use_vad_is_false():
    asr = _make_wrapper()
    assert asr.use_vad() is False
