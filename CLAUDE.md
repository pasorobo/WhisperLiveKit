# CLAUDE.md -- WhisperLiveKit

## Build & Test

Install for development:

```sh
pip install -e ".[test]"
```

Test with real audio using `TestHarness` (requires models + audio files):

```python
import asyncio
from whisperlivekit import TestHarness

async def main():
    async with TestHarness(model_size="base", lan="en", diarization=True) as h:
        await h.feed("audio.wav", speed=1.0)     # feed at real-time
        await h.drain(2.0)                         # let ASR catch up
        h.print_state()                            # see current output

        await h.silence(7.0, speed=1.0)            # 7s silence
        await h.wait_for_silence()                 # verify detection

        result = await h.finish()
        print(f"WER: {result.wer('expected text'):.2%}")
        print(f"Speakers: {result.speakers}")
        print(f"Text at 3s: {result.text_at(3.0)}")

asyncio.run(main())
```

### Cassette replay (GPU-free testing)

For environments without a GPU or model weights (Claude Code Web, CI sandboxes,
quick iteration on glue logic), the **cassette mechanism** replays a previously
recorded ASR session. The full pipeline (FFmpeg, VAD, online policy,
DiffTracker, output formatting) runs for real; only the model inference is
served from a JSON file.

```python
# Tier 1 — Web sandbox / no GPU: replay an existing cassette
async with TestHarness.replay("tests/cassettes/qwen3_ja_short_001.json") as h:
    await h.feed("tests/fixtures/ja_short.wav", speed=0)
    result = await h.finish()
    print(result.text)
```

```python
# Tier 2 — GPU machine: record once, commit the cassette JSON
async with TestHarness.record(
    cassette_path="tests/cassettes/qwen3_ja_short_001.json",
    backend="qwen3", lan="ja",
) as h:
    await h.feed("tests/fixtures/ja_short.wav", speed=0)
    await h.finish()
# Cassette is auto-saved on clean __aexit__.
```

The CLI helper `scripts/record_cassettes.py` wraps this for batch recording.
The cassette is hash-keyed on the exact audio buffer passed to `transcribe()`,
so changes to buffer-trimming or chunking will produce a `CassetteMissError`
on Tier 1 — that is intentional. Re-record on the GPU machine when this fires.

The `cassette` backend is also exposed via `WhisperLiveKitConfig` (set
`backend="cassette"` and `cassette_path=...`) so cassettes plug into anything
that accepts a `TranscriptionEngine`. SimulStreaming and Voxtral HF use
non-three-tuple flows and need a future cassette family — v1 covers
LocalAgreement-style backends (Whisper, FasterWhisper, MLXWhisper, Qwen3,
SenseVoice, FireRedASR2).

#### Web-sandbox network restrictions

When developing on Claude Code Web (or any restricted-egress sandbox), expect:

- ✅ `github.com` and `raw.githubusercontent.com` are reachable — small audio
  fixtures hosted on GitHub raw URLs can be fetched.
- ❌ `huggingface.co`, `huggingface-inference.co`, OpenAI's Whisper CDN, and
  most public dataset CDNs (OpenSLR, archive.org, Wikimedia Commons) are
  blocked. This means **`faster-whisper`, `openai-whisper`, and HF datasets
  cannot download model weights or large datasets** in the sandbox.
- → Real-model cassette recording must happen on a machine with HF access
  (e.g. a GPU workstation). The Web sandbox can write/read cassettes,
  exercise the pipeline glue, and run `compute_cer` / `compute_wer` against
  the recorded outputs, but cannot itself produce a cassette from scratch.

## Configuration presets

Common deployment configurations are registered in `whisperlivekit/presets.py`
under short names. Apply one via `WhisperLiveKitConfig.from_preset(name, **overrides)`
or `wlk --preset <name>`. Explicit CLI flags override preset values.

| Preset | Purpose |
|---|---|
| `ja-accuracy` | Japanese, accuracy-first (Qwen3 + segment trim 15s) |
| `ja-realtime` | Japanese, low-latency (Voxtral 480ms + 0.5s chunks) |
| `ja-broadcast` | Japanese long-form / broadcast (sentence-level trim) |
| `zh-accuracy` | Chinese accuracy (Qwen3; FireRed becomes the default in Phase 1) |
| `zh-realtime` | Chinese low-latency (Qwen3 SimulStreaming-KV) |
| `ja-zh-en` | Multilingual auto-detect (Qwen3) |
| `hri-multilang` | Robotics / human-robot interaction (Qwen3 + small chunks) |
| `apple-silicon-{ja,zh}` | MLX on Apple Silicon |
| `en-fast` | English baseline (faster-whisper large-v3-turbo) |

## Architecture

WhisperLiveKit is a real-time speech transcription system using WebSockets.

- **TranscriptionEngine** (singleton) loads models once at startup and is shared across all sessions.
- **AudioProcessor** is created per WebSocket session. It runs an async producer-consumer pipeline: FFmpeg decodes audio, Silero VAD detects speech, the ASR backend transcribes, and results stream back to the client.
- Two streaming policies:
  - **LocalAgreement** (HypothesisBuffer) -- confirms tokens only when consecutive inferences agree.
  - **SimulStreaming** (AlignAtt attention-based) -- emits tokens as soon as alignment attention is confident.
- 8 ASR backends: WhisperASR, FasterWhisperASR, MLXWhisper, VoxtralMLX, VoxtralHF, Qwen3, FireRedASR2 (Mandarin SOTA), SenseVoice (multilingual zh/en/yue/ja/ko + emotion / event detection).
- **SessionASRProxy** wraps the shared ASR with a per-session language override, using a lock to safely swap `original_language` during `transcribe()`.
- **DiffTracker** implements a snapshot-then-diff protocol for bandwidth-efficient incremental WebSocket updates (opt-in via `?mode=diff`).

## Key Files

| File | Purpose |
|---|---|
| `config.py` | `WhisperLiveKitConfig` dataclass -- single source of truth for configuration |
| `core.py` | `TranscriptionEngine` singleton, `online_factory()`, diarization/translation factories |
| `audio_processor.py` | Per-session async pipeline (FFmpeg -> VAD -> ASR -> output) |
| `basic_server.py` | FastAPI server: WebSocket `/asr`, REST `/v1/audio/transcriptions`, CLI `wlk` |
| `timed_objects.py` | `ASRToken`, `Segment`, `FrontData` data structures |
| `diff_protocol.py` | `DiffTracker` -- snapshot-then-diff WebSocket protocol |
| `session_asr_proxy.py` | `SessionASRProxy` -- thread-safe per-session language wrapper |
| `parse_args.py` | CLI argument parser, returns `WhisperLiveKitConfig` |
| `test_client.py` | Headless WebSocket test client (`wlk-test`) |
| `test_harness.py` | In-process testing harness (`TestHarness`) for real E2E testing |
| `test_cassettes.py` | Record/replay layer (`CassetteRecorder` / `CassetteASR`) — GPU-free pipeline tests via JSON fixtures under `tests/cassettes/` |
| `firered_asr.py` | `FireRedASR2` — Mandarin SOTA (CER 2.89% avg-4); supports 20+ Chinese dialects, English, code-switching. Wraps the upstream batch+file-path API by writing each chunk to a temp WAV. |
| `sensevoice_asr.py` | `SenseVoiceASR` — non-autoregressive multilingual model (zh/en/yue/ja/ko) with emotion + audio-event side-channels. Strips SenseVoice metadata tags before emitting tokens. |
| `presets.py` | Named configuration presets (`ja-realtime`, `zh-accuracy`, `hri-multilang`, …) loaded via `WhisperLiveKitConfig.from_preset()` or `--preset`. |
| `local_agreement/online_asr.py` | `OnlineASRProcessor` for LocalAgreement policy |
| `simul_whisper/` | SimulStreaming policy implementation (AlignAtt) |

## Key Patterns

- **TranscriptionEngine** uses double-checked locking for thread-safe singleton initialization. Never create a second instance in production. Use `TranscriptionEngine.reset()` in tests only to switch backends.
- **WhisperLiveKitConfig** dataclass is the single source of truth. Use `from_namespace()` (from argparse) or `from_kwargs()` (programmatic). `parse_args()` returns a `WhisperLiveKitConfig`, not a raw Namespace.
- **online_factory()** in `core.py` routes to the correct online processor class based on backend and policy.
- **FrontData.to_dict()** is the canonical output format for WebSocket messages.
- **SessionASRProxy** uses `__getattr__` delegation -- it forwards everything except `transcribe()` to the wrapped ASR.
- The server exposes `self.args` as a `Namespace` on `TranscriptionEngine` for backward compatibility with `AudioProcessor`.

## Adding a New ASR Backend

1. Create `whisperlivekit/my_backend.py` with a class implementing:
   - `transcribe(audio, init_prompt="")` -- run inference on audio array
   - `ts_words(result)` -- extract timestamped words from result
   - `segments_end_ts(result)` -- extract segment end timestamps
   - `use_vad()` -- whether this backend needs external VAD
2. Set required attributes on the class: `sep`, `original_language`, `backend_choice`, `SAMPLING_RATE`, `confidence_validation`, `tokenizer`, `buffer_trimming`, `buffer_trimming_sec`.
3. Register in `core.py`:
   - Add an `elif` branch in `TranscriptionEngine._do_init()` to instantiate the backend.
   - Add a routing case in `online_factory()` to return the appropriate online processor.
4. Add the backend choice to CLI args in `parse_args.py`.
5. (optional) Register a preset in `presets.py` so users can opt in via
   `--preset <name>`.
6. (optional) Add an isolation test under `tests/` that constructs the
   wrapper via `__new__` (to skip model loading) and exercises
   `ts_words` / `segments_end_ts` against handcrafted result dicts —
   see `tests/test_firered_asr.py` and `tests/test_sensevoice_asr.py`.

### Reference implementations

| Backend | Style | What it demonstrates |
|---|---|---|
| `firered_asr.py` (`FireRedASR2`) | Batch + file-path upstream API | How to wrap a backend whose API takes paths, not numpy arrays — write each chunk to a temp WAV. Variant resolution (AED vs LLM) from `model_size`/`model_dir`. |
| `sensevoice_asr.py` (`SenseVoiceASR`) | Numpy-in, metadata-tagged text out | How to parse a backend that emits `<\|lang\|><\|emotion\|><\|event\|><\|itn\|>` prefix tags and route them into `ASRToken.detected_language` plus side-channel metadata. |
| `qwen3_asr.py` (`Qwen3ASR`) | Numpy-in, ForcedAligner timestamps | Recommended template for any AED model with native word-level timestamps — closest to what an `ASRBase` subclass should look like. |

All three are non-causal AED and route through LocalAgreement only;
SimulStreaming (AlignAtt) requires an alignment-heads JSON and a model
that supports causal attention masking — see `qwen3_simul.py` for that
pattern.

## Testing with TestHarness

`TestHarness` wraps AudioProcessor in-process for full pipeline testing without a server.

Key methods:
- `feed(path, speed=1.0)` -- feed audio at controlled speed (0 = instant)
- `silence(duration, speed=1.0)` -- inject silence (>5s triggers silence detection)
- `drain(seconds)` -- wait for ASR to catch up without feeding audio
- `finish(timeout)` -- signal end-of-audio, wait for pipeline to drain
- `state` -- current `TestState` with lines, buffers, speakers, timestamps
- `wait_for(predicate)` / `wait_for_text()` / `wait_for_silence()` / `wait_for_speakers(n)`
- `snapshot_at(audio_time)` -- historical state at a given audio position
- `on_update(callback)` -- register callback for each state update

`TestState` provides:
- `text`, `committed_text` -- full or committed-only transcription
- `speakers`, `n_speakers`, `has_silence` -- speaker/silence info
- `line_at(time_s)`, `speaker_at(time_s)`, `text_at(time_s)` -- query by timestamp
- `lines_between(start, end)`, `text_between(start, end)` -- query by time range
- `wer(reference)`, `wer_detailed(reference)` -- WER evaluation against ground truth
- `cer(reference)`, `cer_detailed(reference)` -- CER evaluation (CJK languages: ja/zh/ko)
- `speech_lines`, `silence_segments` -- filtered line lists

For CJK languages, prefer `cer()` over `wer()`. CER strips whitespace and CJK/ASCII
punctuation before scoring (see `whisperlivekit.metrics.normalize_cjk_text`), so
results are robust to formatting differences between reference and hypothesis.
Word-level WER for ja/zh requires a tokenizer (MeCab or jieba); that is left as
future work. For now, use `cer()` for ja/zh and the existing `wer()` for
whitespace-tokenized languages (en, fr, ...).

## OpenAI-Compatible REST API

The server exposes an OpenAI-compatible batch transcription endpoint:

```bash
# Transcribe a file (drop-in replacement for OpenAI)
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F response_format=verbose_json

# Works with the OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
result = client.audio.transcriptions.create(model="whisper-1", file=open("audio.mp3", "rb"))
print(result.text)
```

Supported `response_format` values: `json`, `verbose_json`, `text`, `srt`, `vtt`.
The `model` parameter is accepted but ignored (uses the server's configured backend).

## Do NOT

- Do not create a second `TranscriptionEngine` instance. It is a singleton; the constructor returns the existing instance after the first call.
- Do not modify `original_language` on the shared ASR directly. Use `SessionASRProxy` for per-session language overrides.
- Do not assume the frontend handles diff protocol messages. Diff mode is opt-in (`?mode=diff`) and ignored by default.
- Do not write mock-based unit tests. Use `TestHarness` with real audio for pipeline testing. The one exception is cassette-replay tests (`TestHarness.replay(...)`) which are still real-pipeline tests — only the model inference is replaced with a recorded JSON.
