import os
import re
import string
import sys
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf


_ASR = None
_ASR_ERROR_REPORTED = False


def _warn_once(message: str):
    global _ASR_ERROR_REPORTED
    if not _ASR_ERROR_REPORTED:
        print(f"[wer_sim_reward] {message}", file=sys.stderr, flush=True)
        _ASR_ERROR_REPORTED = True


def _load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def _prepare_wav(wav: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sample_rate != target_sr:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)
    return wav


def _mfcc_embedding(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = _prepare_wav(wav, sample_rate, target_sr=16000)
    if wav.size == 0:
        return np.zeros(40, dtype=np.float32)
    wav = wav / max(float(np.max(np.abs(wav))), 1e-6)
    mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=20)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)
    return feat.astype(np.float32)


@lru_cache(maxsize=512)
def _ref_embedding(ref_audio: str) -> np.ndarray:
    wav, sr = _load_audio(ref_audio, target_sr=16000)
    return _mfcc_embedding(wav, sr)


def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    # MFCC cosine is only a proxy. Clamp to [0, 1] so higher is always better.
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def _get_asr():
    global _ASR
    if _ASR is not None:
        return _ASR

    backend = os.environ.get("REWARD_ASR_BACKEND", "none").lower()
    model_path = os.environ.get("ASR_MODEL_PATH", "openai/whisper-small")

    if backend in {"", "none", "off", "false", "0"}:
        _ASR = False
        return _ASR

    try:
        if backend == "faster_whisper":
            from faster_whisper import WhisperModel

            _ASR = ("faster_whisper", WhisperModel(model_path, device=os.environ.get("ASR_DEVICE", "cuda")))
        elif backend == "transformers":
            import torch
            from transformers import pipeline

            device = int(os.environ.get("ASR_DEVICE_INDEX", "0")) if torch.cuda.is_available() else -1
            dtype = torch.float16 if device >= 0 else torch.float32
            _ASR = (
                "transformers",
                pipeline(
                    "automatic-speech-recognition",
                    model=model_path,
                    device=device,
                    torch_dtype=dtype,
                ),
            )
        else:
            raise ValueError(f"unknown REWARD_ASR_BACKEND={backend}")
    except Exception as exc:  # Reward should degrade instead of crashing training.
        _warn_once(f"ASR disabled because loading failed: {exc}")
        _ASR = False
    return _ASR


def _transcribe(wav: np.ndarray, sample_rate: int, language: str | None = None) -> str:
    asr = _get_asr()
    if not asr:
        return ""

    wav16 = _prepare_wav(wav, sample_rate, target_sr=16000)
    backend, model = asr
    try:
        if backend == "faster_whisper":
            segments, _ = model.transcribe(wav16, language=None if language in {None, "Auto"} else language)
            return "".join(segment.text for segment in segments)

        generate_kwargs = {}
        if language and language != "Auto":
            generate_kwargs["language"] = language
        result = model({"array": wav16, "sampling_rate": 16000}, generate_kwargs=generate_kwargs)
        return str(result.get("text", ""))
    except Exception as exc:
        _warn_once(f"ASR disabled because transcription failed: {exc}")
        return ""


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    table = str.maketrans("", "", string.punctuation + "，。！？；：、“”‘’（）【】《》…")
    return text.translate(table).strip()


def _tokenize_for_error_rate(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if re.search(r"[\u4e00-\u9fff]", text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _wer_score(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize_for_error_rate(reference)
    hyp_tokens = _tokenize_for_error_rate(hypothesis)
    if not ref_tokens:
        return 0.0
    error_rate = _edit_distance(ref_tokens, hyp_tokens) / max(float(len(ref_tokens)), 1.0)
    return float(np.clip(1.0 - error_rate, 0.0, 1.0))


def compute_score(sample, wav, sample_rate, audio_codes) -> float:
    del audio_codes
    if wav is None or len(wav) == 0 or not np.isfinite(wav).all():
        return -1.0

    duration = len(wav) / float(sample_rate)
    min_duration = float(sample.get("min_duration", 0.4))
    max_duration = float(sample.get("max_duration", 20.0))
    if duration < min_duration or duration > max_duration:
        return -1.0

    ref_audio = sample.get("ref_audio")
    if not ref_audio:
        return -1.0

    target_text = sample.get("text") or sample.get("target_text") or ""
    language = sample.get("language", None)

    hyp_text = _transcribe(wav, sample_rate, language=language)
    wer_component = _wer_score(target_text, hyp_text) if hyp_text else 0.0

    gen_emb = _mfcc_embedding(wav, sample_rate)
    ref_emb = _ref_embedding(ref_audio)
    sim_component = _cosine_score(gen_emb, ref_emb)

    wer_weight = float(os.environ.get("REWARD_WER_WEIGHT", "0.6"))
    sim_weight = float(os.environ.get("REWARD_SIM_WEIGHT", "0.4"))
    total = max(wer_weight + sim_weight, 1e-6)
    return float((wer_weight * wer_component + sim_weight * sim_component) / total)
