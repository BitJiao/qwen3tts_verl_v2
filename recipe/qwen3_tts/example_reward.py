import numpy as np


def _audio_code_quality(audio_codes):
    codes = np.asarray(audio_codes.detach().cpu() if hasattr(audio_codes, "detach") else audio_codes)
    if codes.size == 0:
        return 0.0
    flat = codes.reshape(-1)
    if flat.size <= 1:
        return 0.0

    unique_ratio = len(np.unique(flat)) / max(float(flat.size), 1.0)
    repeat_ratio = float(np.mean(flat[1:] == flat[:-1]))
    values, counts = np.unique(flat, return_counts=True)
    del values
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    entropy_score = min(entropy / 6.0, 1.0)
    return float(np.clip(0.45 * unique_ratio + 0.45 * entropy_score + 0.10 * (1.0 - repeat_ratio), 0.0, 1.0))


def compute_score(sample, wav, sample_rate, audio_codes):
    if wav is None or len(wav) == 0 or not np.isfinite(wav).all():
        return -1.0

    duration = len(wav) / sample_rate
    min_duration = float(sample.get("min_duration", 0.4))
    max_duration = float(sample.get("max_duration", 20.0))
    if duration < min_duration or duration > max_duration:
        return -1.0

    # Replace this with MOS, ASR WER, speaker similarity, or task-specific reward.
    target_duration = float(sample.get("target_duration", min(max_duration, 6.0)))
    duration_score = 1.0 - min(abs(duration - target_duration) / max(target_duration, 1e-6), 1.0)
    mean_abs = float(np.abs(wav).mean())
    peak_abs = float(np.abs(wav).max())
    amplitude_score = min(mean_abs * 20.0, 1.0)
    clipping_penalty = 1.0 if peak_abs <= 0.98 else 0.0
    silence_penalty = 1.0 if mean_abs >= 1e-4 else 0.0
    code_quality = _audio_code_quality(audio_codes)
    return (
        0.45 * duration_score
        + 0.20 * amplitude_score
        + 0.25 * code_quality
        + 0.05 * clipping_penalty
        + 0.05 * silence_penalty
    )
