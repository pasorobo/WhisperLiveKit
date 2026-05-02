"""Tests for named configuration presets."""

from __future__ import annotations

import sys

import pytest

from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.presets import get_preset, list_preset_names

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_preset_names_is_sorted_and_nonempty():
    names = list_preset_names()
    assert names, "PRESETS should not be empty"
    assert names == sorted(names)


def test_get_preset_returns_independent_copy():
    a = get_preset("ja-realtime")
    a["lan"] = "mutated"
    b = get_preset("ja-realtime")
    assert b["lan"] == "ja", "preset returned a shared dict"


def test_get_preset_unknown_lists_available():
    with pytest.raises(KeyError) as excinfo:
        get_preset("does-not-exist")
    msg = str(excinfo.value)
    assert "does-not-exist" in msg
    assert "ja-realtime" in msg


@pytest.mark.parametrize("name", list_preset_names())
def test_every_preset_uses_known_config_fields(name):
    """Catch typos: every key must map to a WhisperLiveKitConfig dataclass field."""
    from dataclasses import fields
    known = {f.name for f in fields(WhisperLiveKitConfig)}
    preset = get_preset(name)
    unknown = set(preset) - known
    assert not unknown, f"preset {name!r} references unknown config keys: {unknown}"


@pytest.mark.parametrize("name", list_preset_names())
def test_every_preset_loads_cleanly(name):
    """Every registered preset must produce a valid WhisperLiveKitConfig."""
    cfg = WhisperLiveKitConfig.from_preset(name)
    assert isinstance(cfg, WhisperLiveKitConfig)


# ---------------------------------------------------------------------------
# Spot-check key presets — guards against accidental edits
# ---------------------------------------------------------------------------


def test_ja_realtime_uses_voxtral():
    p = get_preset("ja-realtime")
    assert p["backend"] == "voxtral"
    assert p["lan"] == "ja"
    assert p["min_chunk_size"] == 0.5


def test_zh_accuracy_targets_chinese():
    p = get_preset("zh-accuracy")
    assert p["lan"] == "zh"


def test_apple_silicon_presets_use_mlx_backend():
    for name in ("apple-silicon-ja", "apple-silicon-zh"):
        p = get_preset(name)
        assert "mlx" in p["backend"], f"{name} should target an MLX backend"


# ---------------------------------------------------------------------------
# from_preset overrides
# ---------------------------------------------------------------------------


def test_from_preset_applies_values():
    cfg = WhisperLiveKitConfig.from_preset("ja-realtime")
    assert cfg.backend == "voxtral"
    assert cfg.lan == "ja"
    assert cfg.min_chunk_size == 0.5


def test_from_preset_overrides_win():
    cfg = WhisperLiveKitConfig.from_preset(
        "ja-realtime", lan="en", min_chunk_size=0.2,
    )
    assert cfg.lan == "en"
    assert cfg.min_chunk_size == 0.2
    # Values not in overrides come from the preset
    assert cfg.backend == "voxtral"


def test_from_preset_unknown_raises():
    with pytest.raises(KeyError):
        WhisperLiveKitConfig.from_preset("nonexistent")


# ---------------------------------------------------------------------------
# CLI integration via parse_args
# ---------------------------------------------------------------------------


def _run_parse_args(*argv):
    """Invoke the CLI parser with a synthetic argv."""
    from whisperlivekit.parse_args import parse_args
    saved = sys.argv
    sys.argv = ["wlk", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = saved


def test_cli_preset_applies_when_no_other_flags():
    cfg = _run_parse_args("--preset", "ja-realtime")
    assert cfg.backend == "voxtral"
    assert cfg.lan == "ja"
    assert cfg.min_chunk_size == 0.5


def test_cli_explicit_flag_overrides_preset():
    cfg = _run_parse_args("--preset", "ja-realtime", "--lan", "en")
    assert cfg.backend == "voxtral"   # from preset
    assert cfg.lan == "en"            # explicit flag wins


def test_cli_no_preset_keeps_argparse_defaults():
    cfg = _run_parse_args()
    assert cfg.backend == "auto"
    assert cfg.lan == "auto"
