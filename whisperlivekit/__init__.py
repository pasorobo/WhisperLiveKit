from .audio_processor import AudioProcessor
from .config import WhisperLiveKitConfig
from .core import TranscriptionEngine
from .firered_asr import FireRedASR2
from .parse_args import parse_args
from .presets import PRESETS, get_preset, list_preset_names
from .sensevoice_asr import SenseVoiceASR
from .test_cassettes import (
    Cassette,
    CassetteASR,
    CassetteMissError,
    CassetteRecorder,
)
from .test_client import TranscriptionResult, transcribe_audio
from .test_harness import TestHarness, TestState
from .web.web_interface import get_inline_ui_html, get_web_interface_html

__all__ = [
    "WhisperLiveKitConfig",
    "TranscriptionEngine",
    "AudioProcessor",
    "parse_args",
    "transcribe_audio",
    "TranscriptionResult",
    "TestHarness",
    "TestState",
    "get_web_interface_html",
    "get_inline_ui_html",
    # ASR backends (lazy model loading — class import is safe without deps)
    "FireRedASR2",
    "SenseVoiceASR",
    # Cassette record/replay (Tier 1 GPU-free testing)
    "Cassette",
    "CassetteASR",
    "CassetteMissError",
    "CassetteRecorder",
    # Configuration presets
    "PRESETS",
    "get_preset",
    "list_preset_names",
]
