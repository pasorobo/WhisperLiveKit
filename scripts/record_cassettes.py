"""Record ASR cassettes against a real model on a GPU machine.

Run this on a machine that can actually load the model (e.g. RTX 5090 / Apple
Silicon) so that the resulting JSON cassettes can be committed to the repo
and replayed by Tier 1 (Web sandbox) tests.

Examples
--------

Record a Qwen3-ASR session against a Japanese sample::

    python scripts/record_cassettes.py \\
        --backend qwen3 --lan ja \\
        --audio path/to/ja_short.wav \\
        --output tests/cassettes/qwen3_ja_short_001.json

Record a faster-whisper baseline against a Chinese sample::

    python scripts/record_cassettes.py \\
        --backend faster-whisper --model-size large-v3-turbo --lan zh \\
        --audio path/to/zh_short.wav \\
        --output tests/cassettes/fw_v3turbo_zh_short_001.json

Notes
-----

* The cassette captures the (audio_buffer, init_prompt) → (tokens, segment_ends)
  triples produced by the OnlineASRProcessor. Replay is hash-keyed on the
  exact audio bytes, so any change to buffer-trimming or chunking logic will
  cause a CassetteMissError on Tier 1 — that is intentional, re-record then.

* SimulStreaming / Voxtral HF use their own non-three-tuple flows and are
  not yet supported by this v1 cassette format.
"""

import argparse
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("record_cassettes")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True, help="Path to the input audio file.")
    p.add_argument("--output", required=True, help="Cassette JSON output path.")
    p.add_argument("--cassette-id", default=None,
                   help="Cassette identifier (defaults to output filename stem).")
    p.add_argument("--backend", default="faster-whisper",
                   help="ASR backend to load (must support transcribe/ts_words/segments_end_ts).")
    p.add_argument("--model-size", default="base", help="Model size or HF id.")
    p.add_argument("--lan", default="auto", help="Language code (e.g. ja, zh, en, auto).")
    p.add_argument("--speed", type=float, default=0.0,
                   help="Audio feed speed (0=instant, 1=real-time, 2=2x).")
    p.add_argument("--chunk-duration", type=float, default=1.0,
                   help="Chunk size in seconds for the harness feed.")
    p.add_argument("--vac", action="store_true",
                   help="Enable Voice Activity Controller (default off for determinism).")
    p.add_argument("--diarization", action="store_true",
                   help="Enable diarization during recording.")
    p.add_argument("--reference", default=None,
                   help="Optional ground-truth text — logged alongside the cassette.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def record(args: argparse.Namespace) -> None:
    from whisperlivekit.test_harness import TestHarness

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    output_path = Path(args.output).expanduser().resolve()
    cassette_id = args.cassette_id or output_path.stem

    logger.info("Recording cassette %r", cassette_id)
    logger.info("  audio    = %s", audio_path)
    logger.info("  backend  = %s (model=%s, lan=%s)", args.backend, args.model_size, args.lan)
    logger.info("  output   = %s", output_path)

    harness = TestHarness.record(
        cassette_path=str(output_path),
        cassette_id=cassette_id,
        backend=args.backend,
        model_size=args.model_size,
        lan=args.lan,
        vac=args.vac,
        diarization=args.diarization,
    )

    async with harness as h:
        await h.feed(str(audio_path), speed=args.speed, chunk_duration=args.chunk_duration)
        await h.drain(5.0)
        result = await h.finish(timeout=120)

    logger.info("Transcription: %r", result.text[:120])
    logger.info("Lines: %d  speakers: %d  silence: %s",
                len(result.lines), result.n_speakers, result.has_silence)
    if args.reference:
        try:
            wer = result.wer(args.reference)
            logger.info("WER vs reference: %.2f%%", wer * 100)
        except Exception:
            logger.exception("Could not compute WER")

    logger.info("Cassette saved to %s", output_path)


def main() -> None:
    args = parse_cli()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
