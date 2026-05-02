"""Named configuration presets for common WhisperLiveKit deployments.

A preset is a small dict of ``WhisperLiveKitConfig`` field values mapped to
a short name. Presets capture the recommended backend / language / streaming
policy / buffer settings for typical use cases (per-language accuracy,
low-latency real-time, Apple Silicon native, multilingual robotics, …).

Usage
-----

Programmatic::

    from whisperlivekit import WhisperLiveKitConfig
    cfg = WhisperLiveKitConfig.from_preset("ja-realtime")
    # User overrides win:
    cfg = WhisperLiveKitConfig.from_preset("ja-realtime", min_chunk_size=0.3)

CLI::

    wlk --preset ja-realtime           # apply preset defaults
    wlk --preset ja-realtime --lan en  # preset + explicit override

Realtime tiers
--------------

The ``-realtime`` presets target true-streaming backends (Voxtral, Qwen3
SimulStreaming variants) with sub-second latency. The ``-accuracy`` and
``-broadcast`` presets target quasi-realtime backends (FireRedASR2, Qwen3
LocalAgreement) where each LocalAgreement cycle re-runs full inference;
expect ~1-2 s latency on GPU but better CER. ``hri-multilang`` uses
SenseVoice, which is non-streaming but covers zh/en/yue/ja/ko with one
model — pick it when single-model multilingual matters more than
sub-second response.
"""

from __future__ import annotations

from typing import Any, Dict, List

# All keys must be valid WhisperLiveKitConfig field names. Boolean flags that
# the CLI exposes negated (e.g. --no-vad → no_vad in argparse) are *not*
# supported as preset keys: keep boolean ASR settings at their config-level
# defaults and let the user toggle via CLI flags directly.
PRESETS: Dict[str, Dict[str, Any]] = {
    # ── Japanese ───────────────────────────────────────────────────────────
    # ja-accuracy: quasi-realtime (Qwen3 LocalAgreement, ~1-2s lag)
    "ja-accuracy": {
        "backend": "qwen3",
        "lan": "ja",
        "buffer_trimming": "segment",
        "buffer_trimming_sec": 15.0,
    },
    # ja-realtime: TRUE streaming (Voxtral Mini Realtime, ~480ms)
    "ja-realtime": {
        "backend": "voxtral",
        "lan": "ja",
        "min_chunk_size": 0.5,
    },
    # ja-broadcast: quasi-realtime, sentence-aware trim for long-form
    "ja-broadcast": {
        "backend": "qwen3",
        "lan": "ja",
        "buffer_trimming": "sentence",
        "buffer_trimming_sec": 20.0,
    },

    # ── Chinese (Mandarin) ────────────────────────────────────────────────
    # zh-accuracy: quasi-realtime — FireRedASR2 holds the public Mandarin
    # SOTA (avg CER 2.89% on 4 benches, outperforming Qwen3-ASR-1.7B /
    # Doubao-ASR / Fun-ASR). FireRed is non-causal AED so each
    # LocalAgreement cycle re-runs full inference (~1-2 s latency); use
    # zh-realtime instead when sub-second response matters more than CER.
    "zh-accuracy": {
        "backend": "firered",
        "lan": "zh",
        "buffer_trimming_sec": 15.0,
    },
    # zh-realtime: TRUE streaming (Qwen3 SimulStreaming with KV cache,
    # ~300-500 ms). Slightly worse CER than FireRed but sub-second.
    "zh-realtime": {
        "backend": "qwen3-simul-kv",
        "lan": "zh",
        "min_chunk_size": 0.5,
    },

    # ── Multilingual ──────────────────────────────────────────────────────
    "ja-zh-en": {
        "backend": "qwen3",
        "lan": "auto",
    },
    "hri-multilang": {
        # Robotics / human-robot interaction: multilingual + emotion + events
        # in a single non-streaming model. Quasi-realtime — SenseVoice
        # processes whole chunks per call, so expect ~1-2 s lag. If
        # sub-second response is required, switch to "ja-realtime"
        # (Voxtral) and lose the emotion/event side-channel.
        "backend": "sensevoice",
        "lan": "auto",
        "min_chunk_size": 0.5,
    },

    # ── Apple Silicon (MLX) ───────────────────────────────────────────────
    "apple-silicon-ja": {
        "backend": "qwen3-mlx-simul",
        "lan": "ja",
    },
    "apple-silicon-zh": {
        "backend": "qwen3-mlx-simul",
        "lan": "zh",
    },

    # ── English baseline ──────────────────────────────────────────────────
    "en-fast": {
        "backend": "faster-whisper",
        "lan": "en",
        "model_size": "large-v3-turbo",
    },
}


def list_preset_names() -> List[str]:
    """Return preset names in stable sorted order."""
    return sorted(PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    """Return a fresh dict copy of the named preset.

    Raises:
        KeyError: if the name is not registered. The error message lists all
            available names to make CLI typos easy to recover from.
    """
    if name not in PRESETS:
        available = ", ".join(list_preset_names())
        raise KeyError(f"Unknown preset {name!r}. Available: {available}")
    return dict(PRESETS[name])
