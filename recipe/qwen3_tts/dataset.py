# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for verl SFT from Qwen3-TTS finetuning/dataset.py.

import json
from pathlib import Path
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset
from transformers import AutoProcessor

from verl.utils.fs import copy_local_path_from_hdfs

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]


class Qwen3TTSSFTDataset(Dataset):
    """Qwen3-TTS SFT dataset.

    Expected fields per row:
    ``audio`` target wav path, ``text`` transcript, ``ref_audio`` reference wav
    path, and precomputed ``audio_codes`` from Qwen3-TTS ``prepare_data.py``.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
        model_config: Qwen3TTSConfig | None = None,
    ):
        del tokenizer
        self.config = config
        self.model_config: Qwen3TTSConfig = model_config or config.model_config
        self.lag_num = config.get("lag_num", -1)
        assert self.lag_num == -1, "Qwen3-TTS verl SFT currently supports lag_num=-1 only"

        processor_path = config.get("processor_path", None)
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                processor_path or config.model_path,
                fix_mistral_regex=True,
            )
        self.processor = processor

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.data_list = self._load_files(list(parquet_files))

        if max_samples > 0:
            self.data_list = self.data_list[:max_samples]

        print(f"Qwen3-TTS dataset len: {len(self.data_list)}")

    def _load_files(self, files: list[str]) -> list[dict[str, Any]]:
        data = []
        for data_file in files:
            local_file = copy_local_path_from_hdfs(data_file, verbose=True)
            suffix = Path(local_file).suffix.lower()
            if suffix == ".jsonl":
                with open(local_file, encoding="utf-8") as f:
                    data.extend(json.loads(line) for line in f if line.strip())
            elif suffix == ".json":
                with open(local_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                data.extend(loaded if isinstance(loaded, list) else loaded["data"])
            elif suffix == ".parquet":
                data.extend(pd.read_parquet(local_file).to_dict("records"))
            else:
                raise ValueError(f"Unsupported Qwen3-TTS data file type: {local_file}")
        return data

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, audio_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        items = audios if isinstance(audios, list) else [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for audio in items:
            if isinstance(audio, str):
                out.append(self._load_audio_to_np(audio))
            elif isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], np.ndarray):
                out.append((audio[0].astype(np.float32), int(audio[1])))
            elif isinstance(audio, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int):
        assert sr == 24000, "Qwen3-TTS speaker encoder currently expects 24kHz reference audio"
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text_ids = self._tokenize_text(self._build_assistant_text(item["text"]))
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        ref_audio = item["ref_audio"]
        ref_audio_list = ref_audio if isinstance(ref_audio, list) else [ref_audio]
        ref_wav, ref_sr = self._normalize_audio_inputs(ref_audio_list)[0]
        ref_mel = self.extract_mels(audio=ref_wav, sr=ref_sr)

        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        item_length = [sample["text_ids"].shape[1] + sample["audio_codes"].shape[0] for sample in batch]
        max_length = max(item_length) + 8
        batch_size = len(batch)

        input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long)
        codec_ids = torch.zeros((batch_size, max_length, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        codec_0_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        cfg = self.model_config
        for i, sample in enumerate(batch):
            text_ids = sample["text_ids"]
            audio_codec_0 = sample["audio_codes"][:, 0]
            audio_codecs = sample["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = cfg.tts_pad_token_id
            input_ids[i, 7, 0] = cfg.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = cfg.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = cfg.tts_pad_token_id
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    cfg.talker_config.codec_nothink_id,
                    cfg.talker_config.codec_think_bos_id,
                    cfg.talker_config.codec_think_eos_id,
                    0,
                    cfg.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = cfg.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = cfg.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = cfg.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = cfg.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = cfg.talker_config.codec_eos_token_id

            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False
            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        ref_mels = torch.cat([sample["ref_mel"] for sample in batch], dim=0)
        loss_mask = codec_0_labels.ne(-100)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "loss_mask": loss_mask,
        }
