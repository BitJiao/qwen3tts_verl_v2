#!/usr/bin/env python3
"""Create tiny repo-local Qwen3-TTS smoke data.

The generated files are intentionally small and deterministic. They are useful
for checking paths, dataset parsing, and short training entrypoints; they are
not meaningful training data.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def write_wav(path: Path, *, seconds: float = 0.5, sample_rate: int = 24000, freq: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(seconds * sample_rate)
    amplitude = 0.15
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(n_samples):
            value = amplitude * math.sin(2.0 * math.pi * freq * idx / sample_rate)
            frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
        wav.writeframes(bytes(frames))


def audio_codes(num_frames: int = 12, num_groups: int = 16) -> list[list[int]]:
    return [[0 for _ in range(num_groups)] for _ in range(num_frames)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_relative_or_original(path: Path) -> str:
    """Keep committed smoke manifests portable across clone locations."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/smoke")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ref_wav = out_dir / "ref.wav"
    target_wav = out_dir / "target.wav"
    write_wav(ref_wav, freq=220.0)
    write_wav(target_wav, freq=330.0)
    ref_wav_entry = repo_relative_or_original(ref_wav)
    target_wav_entry = repo_relative_or_original(target_wav)

    sft_rows = [
        {
            "audio": target_wav_entry,
            "text": "hello world",
            "ref_audio": ref_wav_entry,
            "audio_codes": audio_codes(),
        },
        {
            "audio": target_wav_entry,
            "text": "a short smoke test",
            "ref_audio": ref_wav_entry,
            "audio_codes": audio_codes(),
        },
    ]
    grpo_rows = [
        {
            "text": "hello world",
            "ref_audio": ref_wav_entry,
            "ref_text": "hello",
            "language": "en",
            "target_duration": 0.5,
        },
        {
            "text": "a short smoke test",
            "ref_audio": ref_wav_entry,
            "ref_text": "hello",
            "language": "en",
            "target_duration": 0.5,
        },
    ]

    write_jsonl(out_dir / "train_with_codes.jsonl", sft_rows)
    write_jsonl(out_dir / "train_grpo.jsonl", grpo_rows)
    (out_dir / "seedtts_meta.lst").write_text(
        f"smoke_000|hello|{ref_wav_entry}|hello world|{target_wav_entry}\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# Qwen3-TTS Smoke Data\n\n"
        "Tiny deterministic files for repository smoke checks. These files are "
        "not meaningful training data.\n",
        encoding="utf-8",
    )

    print(f"Wrote smoke data to {out_dir}")


if __name__ == "__main__":
    main()
