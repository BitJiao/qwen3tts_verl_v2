"""Generate and score Qwen3-TTS outputs for SeedTTS-style evaluation sets."""

from __future__ import annotations

import atexit
import argparse
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from recipe.qwen3_tts import wer_sim_reward
from recipe.qwen3_tts.grpo_trainer import load_tts, torch_dtype


_STAGING_DIR: Path | None = None


@dataclass
class SeedTTSSample:
    index: int
    sample_id: str
    text: str
    ref_audio: str
    ref_text: str | None = None
    language: str = "Auto"
    gt_audio: str | None = None
    raw: dict[str, Any] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path() -> str:
    return str(_repo_root() / "models" / "Qwen3-TTS-12Hz-1.7B-Base")


def _sanitize_id(value: str, fallback: str) -> str:
    value = str(value or "").strip()
    if not value:
        value = fallback
    value = value.replace("\\", "/")
    stem = Path(value).with_suffix("").as_posix()
    parts = [part for part in stem.split("/") if part not in {"", ".", ".."}]
    stem = "/".join(parts)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", "/"} else "_" for ch in stem)
    return safe.strip("/") or fallback


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        else:
            return value
    return None


def _resolve_path(path: str | None, base_dir: Path | None) -> str | None:
    if not path:
        return None
    path_obj = Path(str(path)).expanduser()
    if path_obj.is_absolute() or base_dir is None:
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def _language_from_text_or_arg(text: str, language: str | None) -> str:
    if language and language != "auto_from_text":
        return language
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


def _staging_dir() -> Path:
    global _STAGING_DIR
    if _STAGING_DIR is None:
        _STAGING_DIR = Path(tempfile.mkdtemp(prefix="qwen3tts_seedtts_ref_"))
        atexit.register(shutil.rmtree, str(_STAGING_DIR), True)
    return _STAGING_DIR


def _stage_audio_dict(row: dict[str, Any], audio: dict[str, Any], index: int) -> str | None:
    path = audio.get("path")
    if path:
        return str(path)

    rel_path = str(row.get("ref_audio_path") or row.get("sample_id") or f"{index:06d}.wav")
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    if ".." in Path(rel_path).parts:
        rel_path = f"{index:06d}.wav"
    out_path = (_staging_dir() / rel_path).with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_bytes = audio.get("bytes")
    if audio_bytes:
        out_path.write_bytes(audio_bytes)
        return str(out_path)

    array = audio.get("array")
    sampling_rate = audio.get("sampling_rate") or audio.get("sr") or 24000
    if array is not None:
        sf.write(out_path, np.asarray(array, dtype=np.float32), int(sampling_rate))
        return str(out_path)
    return None


def _sample_from_json(row: dict[str, Any], index: int, base_dir: Path | None, language: str | None) -> SeedTTSSample:
    text = _first_present(row, ("text", "target_text", "prompt", "sentence", "utt_text", "transcript"))
    ref_audio = _first_present(
        row,
        ("ref_audio", "ref_audio_path", "prompt_audio", "reference_audio", "audio_prompt", "audio"),
    )
    if isinstance(ref_audio, dict):
        ref_audio = _stage_audio_dict(row, ref_audio, index)
    if not text:
        raise ValueError(f"row {index} has no target text field")
    if not ref_audio:
        raise ValueError(f"row {index} has no reference audio field")

    sample_id = _first_present(row, ("id", "uid", "sample_id", "filename", "audio_id", "utt_id"))
    sample_id = _sanitize_id(str(sample_id or ""), f"{index:06d}")
    row_language = row.get("language") or row.get("lang") or language
    return SeedTTSSample(
        index=index,
        sample_id=sample_id,
        text=str(text),
        ref_audio=_resolve_path(str(ref_audio), base_dir) or str(ref_audio),
        ref_text=_first_present(row, ("ref_text", "prompt_text", "reference_text")),
        language=_language_from_text_or_arg(str(text), str(row_language) if row_language else None),
        gt_audio=_resolve_path(str(_first_present(row, ("gt_audio", "ground_truth", "target_audio")) or ""), base_dir),
        raw=row,
    )


def _sample_from_meta_line(line: str, index: int, base_dir: Path | None, language: str | None) -> SeedTTSSample:
    parts = [part.strip() for part in line.rstrip("\n").split("|")]
    if len(parts) < 4:
        raise ValueError(f"meta line {index} needs at least 4 pipe-separated fields")
    filename, ref_text, ref_audio, text = parts[:4]
    gt_audio = parts[4] if len(parts) >= 5 else None
    sample_id = _sanitize_id(filename, f"{index:06d}")
    return SeedTTSSample(
        index=index,
        sample_id=sample_id,
        text=text,
        ref_audio=_resolve_path(ref_audio, base_dir) or ref_audio,
        ref_text=ref_text or None,
        language=_language_from_text_or_arg(text, language),
        gt_audio=_resolve_path(gt_audio, base_dir),
        raw={
            "filename": filename,
            "ref_text": ref_text,
            "ref_audio": ref_audio,
            "text": text,
            "gt_audio": gt_audio,
        },
    )


def load_samples(path: str, *, language: str | None, max_samples: int | None) -> list[SeedTTSSample]:
    input_path = Path(path).expanduser()
    if input_path.is_file():
        base_dir = input_path.parent
        samples: list[SeedTTSSample] = []
        with input_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    samples.append(_sample_from_json(json.loads(line), len(samples), base_dir, language))
                else:
                    samples.append(_sample_from_meta_line(line, len(samples), base_dir, language))
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ValueError(f"{path} is not a local file and datasets could not be imported: {exc}") from exc

    split = "train"
    if language in {"en", "zh"}:
        split = language
    dataset = load_dataset(path, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [_sample_from_json(dict(row), idx, None, language) for idx, row in enumerate(dataset)]


def _worker_samples(samples: list[SeedTTSSample], rank: int, world_size: int) -> list[SeedTTSSample]:
    return [sample for pos, sample in enumerate(samples) if pos % world_size == rank]


def _manifest_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"manifest.rank{rank}.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate_worker(
    rank: int,
    world_size: int,
    samples: list[SeedTTSSample],
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["visible_devices"][rank]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype(args_dict["dtype"])
    random.seed(args_dict["seed"] + rank)
    np.random.seed(args_dict["seed"] + rank)
    torch.manual_seed(args_dict["seed"] + rank)

    output_dir = Path(args_dict["output_dir"])
    audio_dir = output_dir / "wav_res"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tts = load_tts(args_dict["model_path"], dtype, args_dict["attn_implementation"], device)
    tts.model.eval()

    records: list[dict[str, Any]] = []
    subset = _worker_samples(samples, rank, world_size)
    iterator = tqdm(subset, desc=f"gpu{rank}", position=rank, disable=args_dict["disable_tqdm"])
    for sample in iterator:
        wav_path = audio_dir / f"{sample.sample_id}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = None
        sample_rate = 0
        duration = 0.0
        try:
            wavs, sample_rate = tts.generate_voice_clone(
                text=sample.text,
                language=sample.language,
                ref_audio=sample.ref_audio,
                ref_text=sample.ref_text,
                x_vector_only_mode=args_dict["x_vector_only_mode"],
                non_streaming_mode=args_dict["non_streaming_mode"],
                do_sample=args_dict["do_sample"],
                top_k=args_dict["top_k"],
                top_p=args_dict["top_p"],
                temperature=args_dict["temperature"],
                subtalker_dosample=args_dict["do_sample"],
                subtalker_top_k=args_dict["subtalker_top_k"],
                subtalker_top_p=args_dict["subtalker_top_p"],
                subtalker_temperature=args_dict["subtalker_temperature"],
                max_new_tokens=args_dict["max_new_tokens"],
            )
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            duration = len(wav) / float(sample_rate)
            sf.write(wav_path, wav, sample_rate)
        except Exception as exc:  # Keep the full eval moving and report failures in manifest.
            status = "error"
            error = repr(exc)

        elapsed = time.perf_counter() - started
        records.append(
            {
                "index": sample.index,
                "id": sample.sample_id,
                "text": sample.text,
                "ref_audio": sample.ref_audio,
                "ref_text": sample.ref_text,
                "language": sample.language,
                "gt_audio": sample.gt_audio,
                "wav_path": str(wav_path),
                "sample_rate": sample_rate,
                "duration": duration,
                "latency": elapsed,
                "rtf": elapsed / duration if duration > 0 else None,
                "status": status,
                "error": error,
            }
        )

    _write_jsonl(_manifest_path(output_dir, rank), records)
    ok = sum(1 for record in records if record["status"] == "ok")
    return {"rank": rank, "total": len(records), "ok": ok, "error": len(records) - ok}


def _visible_devices(devices: str) -> list[str]:
    if devices.strip().lower() == "auto":
        if not torch.cuda.is_available():
            return [""]
        return [str(idx) for idx in range(torch.cuda.device_count())]
    return [item.strip().removeprefix("cuda:") for item in devices.split(",") if item.strip()]


def merge_manifests(output_dir: Path, world_size: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = _manifest_path(output_dir, rank)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    records.sort(key=lambda item: item["index"])
    _write_jsonl(output_dir / "manifest.jsonl", records)
    write_wav_res_ref_text(records, output_dir)
    return records


def write_seedtts_meta(samples: list[SeedTTSSample], output_dir: Path) -> None:
    path = output_dir / "meta.lst"
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            gt_audio = sample.gt_audio or ""
            ref_text = sample.ref_text or ""
            f.write(f"{sample.sample_id}|{ref_text}|{sample.ref_audio}|{sample.text}|{gt_audio}\n")


def write_wav_res_ref_text(records: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        rows.append(f"{record['wav_path']}|{record['ref_audio']}|{record['text']}\n")

    for name in ("wav_res_ref_text", "wav_res_ref_text.txt"):
        (output_dir / name).write_text("".join(rows), encoding="utf-8")


def summarize(records: list[dict[str, Any]], output_dir: Path, elapsed: float) -> dict[str, Any]:
    ok_records = [record for record in records if record["status"] == "ok"]
    durations = [float(record["duration"]) for record in ok_records if record.get("duration")]
    latencies = [float(record["latency"]) for record in ok_records if record.get("latency")]
    rtfs = [float(record["rtf"]) for record in ok_records if record.get("rtf") is not None]
    summary = {
        "total": len(records),
        "ok": len(ok_records),
        "error": len(records) - len(ok_records),
        "wall_seconds": elapsed,
        "throughput_qps": len(ok_records) / elapsed if elapsed > 0 else 0.0,
        "audio_seconds": float(sum(durations)),
        "audio_seconds_per_wall_second": float(sum(durations) / elapsed) if elapsed > 0 else 0.0,
        "latency_mean": float(np.mean(latencies)) if latencies else None,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "rtf_mean": float(np.mean(rtfs)) if rtfs else None,
        "rtf_p95": float(np.percentile(rtfs, 95)) if rtfs else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def score_manifest(manifest_path: str, output_dir: str, *, asr_backend: str, asr_model_path: str | None) -> dict[str, Any]:
    os.environ["REWARD_ASR_BACKEND"] = asr_backend
    if asr_model_path:
        os.environ["ASR_MODEL_PATH"] = asr_model_path

    records: list[dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    scored: list[dict[str, Any]] = []
    for record in tqdm(records, desc="score"):
        if record.get("status") != "ok":
            scored.append({**record, "score": -1.0, "wer_score": None, "sim_score": None})
            continue
        wav, sr = sf.read(record["wav_path"], always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sample = {
            "text": record.get("text", ""),
            "ref_audio": record.get("ref_audio"),
            "language": record.get("language"),
        }
        hyp_text = wer_sim_reward._transcribe(wav, sr, language=sample["language"])
        wer_score = wer_sim_reward._wer_score(sample["text"], hyp_text) if hyp_text else 0.0
        sim_score = wer_sim_reward._cosine_score(
            wer_sim_reward._mfcc_embedding(wav, sr),
            wer_sim_reward._ref_embedding(sample["ref_audio"]),
        )
        scored.append({**record, "hyp_text": hyp_text, "wer_score": wer_score, "sim_score": sim_score})

    out_dir = Path(output_dir)
    _write_jsonl(out_dir / "scores.jsonl", scored)
    ok = [item for item in scored if item.get("wer_score") is not None]
    summary = {
        "evaluated": len(ok),
        "skipped": len(scored) - len(ok),
        "wer_score_mean": float(np.mean([item["wer_score"] for item in ok])) if ok else None,
        "sim_score_mean": float(np.mean([item["sim_score"] for item in ok])) if ok else None,
    }
    (out_dir / "score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="8-GPU Qwen3-TTS generation for SeedTTS-style eval sets.")
    parser.add_argument("--input", "--input_jsonl", "--meta", dest="input_path", required=False)
    parser.add_argument("--model_path", default=_default_model_path())
    parser.add_argument("--output_dir", default="results/qwen3_tts_seedtts")
    parser.add_argument("--devices", default="auto", help="Comma-separated CUDA ids, or auto.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--language", default="auto_from_text", help="en, zh, Auto, or auto_from_text.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non_streaming_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x_vector_only_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing non-empty output_dir before generation.")
    parser.add_argument("--score_only", action="store_true", help="Score an existing manifest instead of generating.")
    parser.add_argument("--manifest", default=None, help="Manifest path for --score_only. Defaults to output_dir/manifest.jsonl.")
    parser.add_argument("--asr_backend", default="none", choices=["none", "transformers", "faster_whisper"])
    parser.add_argument("--asr_model_path", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        manifest = args.manifest or str(output_dir / "manifest.jsonl")
        summary = score_manifest(
            manifest,
            str(output_dir),
            asr_backend=args.asr_backend,
            asr_model_path=args.asr_model_path,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.input_path:
        parser.error("--input is required unless --score_only is set")
    if not Path(args.model_path).is_dir():
        parser.error(f"--model_path does not exist: {args.model_path}")

    visible_devices = _visible_devices(args.devices)
    if not visible_devices:
        parser.error("--devices resolved to an empty list")

    samples = load_samples(args.input_path, language=args.language, max_samples=args.max_samples)
    if not samples:
        parser.error("no samples loaded")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            parser.error(f"--output_dir is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    write_seedtts_meta(samples, output_dir)

    args_dict = vars(args).copy()
    args_dict["visible_devices"] = visible_devices
    args_dict["output_dir"] = str(output_dir)
    world_size = len(visible_devices)
    started = time.perf_counter()
    ctx = get_context("spawn")
    with ctx.Pool(processes=world_size) as pool:
        worker_results = pool.starmap(
            _generate_worker,
            [(rank, world_size, samples, args_dict) for rank in range(world_size)],
        )
    records = merge_manifests(output_dir, world_size)
    summary = summarize(records, output_dir, time.perf_counter() - started)
    summary["workers"] = worker_results
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
