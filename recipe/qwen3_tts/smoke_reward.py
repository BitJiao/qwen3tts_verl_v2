import numpy as np


def compute_score(sample, wav, sample_rate, audio_codes):
    del sample, sample_rate
    if wav is None or len(wav) == 0 or not np.isfinite(wav).all():
        return -1.0
    codes = np.asarray(audio_codes.detach().cpu() if hasattr(audio_codes, "detach") else audio_codes)
    return float((int(codes[:, 0].sum()) % 997) / 997.0)
