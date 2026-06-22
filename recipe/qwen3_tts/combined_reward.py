import os
import sys
from typing import Any

import numpy as np

from recipe.qwen3_tts import speechjudge_reward, wer_sim_reward


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _is_enabled(weight: float) -> bool:
    return weight > 0.0


def _valid_wav(sample: dict[str, Any], wav: np.ndarray | None, sample_rate: int) -> np.ndarray | None:
    if wav is None:
        return None
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0 or not np.isfinite(wav).all():
        return None

    duration = len(wav) / float(sample_rate)
    min_duration = float(sample.get("min_duration", os.environ.get("REWARD_MIN_DURATION", "0.4")))
    max_duration = float(sample.get("max_duration", os.environ.get("REWARD_MAX_DURATION", "20.0")))
    if duration < min_duration or duration > max_duration:
        return None
    return wav


def _duration_score(sample: dict[str, Any], wav: np.ndarray, sample_rate: int) -> float:
    target = sample.get("target_duration")
    if target is None:
        return 0.0
    try:
        target = float(target)
    except (TypeError, ValueError):
        return 0.0
    if target <= 0:
        return 0.0

    duration = len(wav) / float(sample_rate)
    relative_error = abs(duration - target) / max(target, 1e-6)
    return float(np.clip(1.0 - relative_error, 0.0, 1.0))


def _wer_sim_components(
    sample: dict[str, Any],
    valid_wavs: list[np.ndarray | None],
    sample_rate: int,
    use_wer: bool,
    use_sim: bool,
) -> tuple[list[float], list[float]]:
    wer_scores = [0.0] * len(valid_wavs)
    sim_scores = [0.0] * len(valid_wavs)

    if not use_wer and not use_sim:
        return wer_scores, sim_scores

    ref_audio = sample.get("ref_audio")
    if use_sim and not ref_audio:
        sim_scores = [-1.0] * len(valid_wavs)

    target_text = sample.get("text") or sample.get("target_text") or ""
    language = sample.get("language", None)
    concrete_wavs = [wav for wav in valid_wavs if wav is not None]

    hyp_texts: list[str] = []
    if use_wer and concrete_wavs:
        hyp_texts = wer_sim_reward._transcribe_many(concrete_wavs, sample_rate, language=language)

    ref_emb = None
    if use_sim and ref_audio:
        ref_emb = wer_sim_reward._ref_embedding(ref_audio)

    valid_cursor = 0
    for idx, wav in enumerate(valid_wavs):
        if wav is None:
            wer_scores[idx] = -1.0
            sim_scores[idx] = -1.0
            continue

        if use_wer:
            hyp_text = hyp_texts[valid_cursor] if valid_cursor < len(hyp_texts) else ""
            wer_scores[idx] = wer_sim_reward._wer_score(target_text, hyp_text) if hyp_text else 0.0

        if use_sim and ref_emb is not None:
            gen_emb = wer_sim_reward._mfcc_embedding(wav, sample_rate)
            sim_scores[idx] = wer_sim_reward._cosine_score(gen_emb, ref_emb)

        valid_cursor += 1

    return wer_scores, sim_scores


def _maybe_log_components(
    totals: list[float],
    wer_scores: list[float],
    sim_scores: list[float],
    judge_scores: list[float],
    duration_scores: list[float],
    weights: dict[str, float],
) -> None:
    if os.environ.get("COMBINED_REWARD_LOG_COMPONENTS", "0").lower() not in {"1", "true", "yes"}:
        return

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    print(
        "[combined_reward] "
        f"total={mean(totals):.4f} "
        f"wer={mean(wer_scores):.4f} "
        f"sim={mean(sim_scores):.4f} "
        f"judge={mean(judge_scores):.4f} "
        f"duration={mean(duration_scores):.4f} "
        f"weights={weights}",
        file=sys.stderr,
        flush=True,
    )


def compute_score(sample, wav, sample_rate, audio_codes) -> float:
    scores = compute_scores(sample=sample, wavs=[wav], sample_rate=sample_rate, audio_codes_list=[audio_codes])
    return scores[0]


def compute_scores(sample, wavs, sample_rate, audio_codes_list) -> list[float]:
    del audio_codes_list
    if not wavs:
        return []

    weights = {
        "wer": _env_float("REWARD_WER_WEIGHT", "0.3"),
        "sim": _env_float("REWARD_SIM_WEIGHT", "0.2"),
        "judge": _env_float("REWARD_JUDGE_WEIGHT", "0.5"),
        "duration": _env_float("REWARD_DURATION_WEIGHT", "0.0"),
    }
    total_weight = sum(weight for weight in weights.values() if weight > 0.0)
    if total_weight <= 0:
        raise ValueError("At least one combined reward weight must be positive.")

    invalid_value = _env_float("REWARD_INVALID_VALUE", "-1.0")
    valid_wavs = [_valid_wav(sample, wav, sample_rate) for wav in wavs]

    wer_scores, sim_scores = _wer_sim_components(
        sample,
        valid_wavs,
        sample_rate,
        use_wer=_is_enabled(weights["wer"]),
        use_sim=_is_enabled(weights["sim"]),
    )

    if _is_enabled(weights["judge"]):
        judge_scores = speechjudge_reward.compute_scores(
            sample=sample,
            wavs=valid_wavs,
            sample_rate=sample_rate,
            audio_codes_list=[None] * len(wavs),
        )
    else:
        judge_scores = [0.0] * len(wavs)

    duration_scores = [
        _duration_score(sample, wav, sample_rate) if wav is not None and _is_enabled(weights["duration"]) else 0.0
        for wav in valid_wavs
    ]

    totals: list[float] = []
    for idx, wav in enumerate(valid_wavs):
        if wav is None:
            totals.append(invalid_value)
            continue

        total = (
            weights["wer"] * wer_scores[idx]
            + weights["sim"] * sim_scores[idx]
            + weights["judge"] * judge_scores[idx]
            + weights["duration"] * duration_scores[idx]
        ) / total_weight
        totals.append(float(total))

    _maybe_log_components(totals, wer_scores, sim_scores, judge_scores, duration_scores, weights)
    return totals
