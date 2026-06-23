import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_MODEL_ID = "RMSnow/SpeechJudge-GRM"
_WARNED = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        print(f"[speechjudge_reward] {message}", file=sys.stderr, flush=True)
        _WARNED.add(key)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_model_path() -> str:
    return str(_repo_root() / "pretrained" / "SpeechJudge-GRM")


def _speechjudge_repo_path() -> str | None:
    value = os.environ.get("SPEECHJUDGE_REPO")
    if value:
        return value

    default = _repo_root() / "third_party" / "SpeechJudge"
    if default.exists():
        return str(default)
    return None


def _ensure_speechjudge_import_path() -> None:
    repo_path = _speechjudge_repo_path()
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    infer_path = str(Path(repo_path) / "infer") if repo_path else None
    if infer_path and infer_path not in sys.path:
        sys.path.insert(0, infer_path)


def _finite_wav(wav: np.ndarray | None) -> np.ndarray | None:
    if wav is None:
        return None
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0 or not np.isfinite(wav).all():
        return None
    return wav


def _validate_wav(sample: dict[str, Any], wav: np.ndarray | None, sample_rate: int) -> np.ndarray | None:
    wav = _finite_wav(wav)
    if wav is None:
        return None
    duration = len(wav) / float(sample_rate)
    min_duration = float(sample.get("min_duration", os.environ.get("SPEECHJUDGE_MIN_DURATION", "0.4")))
    max_duration = float(sample.get("max_duration", os.environ.get("SPEECHJUDGE_MAX_DURATION", "20.0")))
    if duration < min_duration or duration > max_duration:
        return None
    return wav


def _write_temp_wav(wav: np.ndarray, sample_rate: int, directory: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="speechjudge_", suffix=".wav", dir=directory, delete=False)
    handle.close()
    wav = np.clip(wav, -1.0, 1.0)
    sf.write(handle.name, wav, sample_rate)
    return handle.name


def _score_from_rating(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    # SpeechJudge outputs 1-10. Normalize for GRPO reward scale.
    return float(np.clip((score - 1.0) / 9.0, 0.0, 1.0))


def _extract_single_score(text: str) -> float | None:
    patterns = [
        r"(?:score|rating)\D{0,20}(\d+(?:\.\d+)?)",
        r"\b(\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\b(10(?:\.0+)?|[1-9](?:\.\d+)?)\b",
        r"^\s*(\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return _score_from_rating(matches[-1])
    return None


@lru_cache(maxsize=1)
def _load_model():
    _ensure_speechjudge_import_path()
    model_path = os.environ.get("SPEECHJUDGE_MODEL_PATH", _default_model_path())
    try:
        from infer.main_grm import load_model
    except Exception as exc:
        raise RuntimeError(
            "Cannot import SpeechJudge. Clone https://github.com/AmphionTeam/SpeechJudge "
            "to $SPEECHJUDGE_REPO or this repo's third_party/SpeechJudge, and install "
            "qwen-omni-utils plus the SpeechJudge transformer dependencies."
        ) from exc

    try:
        return load_model(model_path)
    except ImportError as exc:
        if "flash_attn" not in str(exc) and "FlashAttention2" not in str(exc):
            raise
        _warn_once("flash_attn", "flash_attn is unavailable; loading SpeechJudge-GRM with eager attention.")

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=os.environ.get("SPEECHJUDGE_DEVICE_MAP", "auto"),
        attn_implementation=os.environ.get("SPEECHJUDGE_ATTN_IMPLEMENTATION", "eager"),
    )
    return model, processor


def _compare_pair(target_text: str, wav_path_a: str, wav_path_b: str) -> tuple[float | None, float | None]:
    model, processor = _load_model()
    from infer.main_grm import compare_wavs

    rating, result = compare_wavs(processor, model, target_text, wav_path_a, wav_path_b)
    if rating:
        return _score_from_rating(rating.get("output_a")), _score_from_rating(rating.get("output_b"))

    _warn_once("parse_pair", f"Could not parse pairwise SpeechJudge output: {result!r}")
    return None, None


def _score_single(target_text: str, wav_path: str) -> float:
    _ensure_speechjudge_import_path()
    model, processor = _load_model()

    try:
        from utils import build_qwen_omni_inputs, build_rm_conversation
    except Exception as exc:
        raise RuntimeError("SpeechJudge infer/utils.py is not importable") from exc

    conversation = build_rm_conversation(wav_path, target_text)
    omni_inputs = build_qwen_omni_inputs(processor, conversation)
    omni_inputs = omni_inputs.to(model.device).to(model.dtype)
    prompt_length = omni_inputs["input_ids"].shape[1]
    text_ids = model.generate(
        **omni_inputs,
        use_audio_in_video=False,
        do_sample=True,
        return_audio=False,
        max_new_tokens=int(os.environ.get("SPEECHJUDGE_MAX_NEW_TOKENS", "64")),
    )
    text_ids = text_ids[:, prompt_length:]
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    score = _extract_single_score(text)
    if score is None:
        _warn_once("parse_single", f"Could not parse single SpeechJudge output: {text!r}")
        return -1.0
    return score


def _score_with_server(target_text: str, wav_paths: list[str]) -> list[float] | None:
    server_url = os.environ.get("SPEECHJUDGE_SERVER_URL")
    if not server_url:
        return None

    payload = json.dumps({"target_text": target_text, "wav_paths": wav_paths}).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/score",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        timeout = float(os.environ.get("SPEECHJUDGE_SERVER_TIMEOUT", "600"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        _warn_once("server_failed", f"SpeechJudge server request failed; assigning -1 to failed samples: {exc}")
        return [-1.0] * len(wav_paths)

    scores = result.get("scores")
    if not isinstance(scores, list) or len(scores) != len(wav_paths):
        _warn_once("server_bad_response", f"SpeechJudge server returned invalid response: {result!r}")
        return [-1.0] * len(wav_paths)
    return [float(score) for score in scores]


def compute_score(sample, wav, sample_rate, audio_codes) -> float:
    del audio_codes
    scores = compute_scores(sample=sample, wavs=[wav], sample_rate=sample_rate, audio_codes_list=[None])
    return scores[0]


def compute_scores(sample, wavs, sample_rate, audio_codes_list) -> list[float]:
    del audio_codes_list
    if not wavs:
        return []

    target_text = sample.get("text") or sample.get("target_text") or ""
    if not target_text:
        return [-1.0] * len(wavs)

    valid_wavs = [_validate_wav(sample, wav, sample_rate) for wav in wavs]
    scores: list[float] = [-1.0] * len(wavs)
    if not any(wav is not None for wav in valid_wavs):
        return scores

    tmp_dir = os.environ.get("SPEECHJUDGE_TMP_DIR")
    temp_context = tempfile.TemporaryDirectory(prefix="speechjudge_reward_") if not tmp_dir else None
    work_dir = tmp_dir or temp_context.name
    os.makedirs(work_dir, exist_ok=True)
    paths: list[str | None] = []

    try:
        for wav in valid_wavs:
            paths.append(_write_temp_wav(wav, sample_rate, work_dir) if wav is not None else None)

        valid_indices = [idx for idx, path in enumerate(paths) if path is not None]
        valid_paths = [paths[idx] for idx in valid_indices]
        server_scores = _score_with_server(target_text, valid_paths)
        if server_scores is not None:
            for idx, score in zip(valid_indices, server_scores):
                scores[idx] = score
            return scores

        mode = os.environ.get("SPEECHJUDGE_SCORING_MODE", "single").lower()
        if mode in {"pairwise", "pair"} and len(paths) > 1:
            baseline_idx = next((idx for idx, path in enumerate(paths) if path is not None), None)
            if baseline_idx is not None:
                scores[baseline_idx] = 0.5
                baseline_path = paths[baseline_idx]
                for idx, path in enumerate(paths):
                    if path is None or idx == baseline_idx:
                        continue
                    base_score, cand_score = _compare_pair(target_text, baseline_path, path)
                    if cand_score is not None:
                        scores[idx] = cand_score
                    if base_score is not None:
                        scores[baseline_idx] = max(scores[baseline_idx], base_score)
                return scores

        for idx, path in enumerate(paths):
            if path is None:
                continue
            try:
                scores[idx] = _score_single(target_text, path)
            except Exception as exc:
                _warn_once("score_failed", f"SpeechJudge scoring failed; returning -1 for failed samples: {exc}")
                scores[idx] = -1.0
        return scores
    finally:
        if temp_context is not None:
            temp_context.cleanup()
        else:
            for path in paths:
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
