import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModel

import recipe.qwen3_tts.register  # noqa: F401
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def parse_args():
    parser = argparse.ArgumentParser(description="Export verl FSDP Qwen3-TTS checkpoint to custom_voice format.")
    parser.add_argument("--checkpoint_dir", required=True, help="verl checkpoint dir, e.g. global_step_100")
    parser.add_argument("--base_model_dir", required=True, help="Original Qwen3-TTS Base checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Output custom_voice HF directory")
    parser.add_argument("--speaker_name", default="speaker_test", help="Speaker name to write into config.json")
    parser.add_argument("--speaker_id", type=int, default=3000, help="Codec embedding row used for this speaker")
    parser.add_argument("--ref_audio", default=None, help="24kHz reference wav for speaker embedding")
    parser.add_argument("--train_jsonl", default=None, help="Use first row's ref_audio when --ref_audio is omitted")
    parser.add_argument("--device", default=None, help="Device for speaker embedding extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code to HF loaders")
    return parser.parse_args()


def resolve_ref_audio(args) -> str:
    if args.ref_audio:
        return args.ref_audio

    if not args.train_jsonl:
        raise ValueError("Either --ref_audio or --train_jsonl is required for speaker embedding extraction.")

    with open(args.train_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ref_audio = item["ref_audio"]
            if isinstance(ref_audio, list):
                if not ref_audio:
                    raise ValueError("First train_jsonl row has an empty ref_audio list.")
                return ref_audio[0]
            return ref_audio

    raise ValueError(f"No valid rows found in {args.train_jsonl}")


def load_audio_to_np(audio_path: str) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32), int(sr)


@torch.inference_mode()
def extract_speaker_embedding(base_model_dir: str, ref_audio: str, device: str | None, trust_remote_code: bool):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    model = AutoModel.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    if getattr(model, "speaker_encoder", None) is None:
        raise ValueError("Speaker embedding extraction requires a Qwen3-TTS Base checkpoint with speaker_encoder.")

    audio, sr = load_audio_to_np(ref_audio)
    if sr != 24000:
        raise ValueError(f"Qwen3-TTS speaker encoder expects 24kHz ref_audio, got {sr}: {ref_audio}")

    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    speaker_embedding = model.speaker_encoder(mels.to(device).to(dtype))[0].detach().cpu()
    del model
    return speaker_embedding


def merge_fsdp_state_dict(checkpoint_dir: str, trust_remote_code: bool):
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_dir,
        target_dir=None,
        hf_model_config_path=os.path.join(checkpoint_dir, "huggingface"),
        trust_remote_code=trust_remote_code,
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    return merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)


def prepare_output_dir(base_model_dir: str, output_dir: str, overwrite: bool):
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    shutil.copytree(base_model_dir, output_dir)

    for pattern in [
        "model*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model.bin.index.json",
    ]:
        for path in glob.glob(str(output / pattern)):
            os.remove(path)


def patch_config(output_dir: str, speaker_name: str, speaker_id: int):
    config_path = Path(output_dir) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.setdefault("talker_config", {})
    talker_config["spk_id"] = {speaker_name: speaker_id}
    talker_config["spk_is_dialect"] = {speaker_name: False}

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def patch_state_dict(state_dict: dict[str, torch.Tensor], speaker_embedding: torch.Tensor, speaker_id: int):
    for key in list(state_dict.keys()):
        if key.startswith("speaker_encoder"):
            del state_dict[key]

    embedding_key = "talker.model.codec_embedding.weight"
    if embedding_key not in state_dict:
        raise KeyError(f"Missing {embedding_key} in merged checkpoint.")

    weight = state_dict[embedding_key]
    if speaker_id >= weight.shape[0]:
        raise ValueError(f"speaker_id={speaker_id} is out of range for {embedding_key} with shape {tuple(weight.shape)}")

    state_dict[embedding_key][speaker_id] = speaker_embedding.to(device=weight.device, dtype=weight.dtype)
    return state_dict


def main():
    args = parse_args()
    ref_audio = resolve_ref_audio(args)
    speaker_embedding = extract_speaker_embedding(
        args.base_model_dir,
        ref_audio=ref_audio,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    state_dict = merge_fsdp_state_dict(args.checkpoint_dir, trust_remote_code=args.trust_remote_code)
    state_dict = patch_state_dict(state_dict, speaker_embedding, speaker_id=args.speaker_id)

    prepare_output_dir(args.base_model_dir, args.output_dir, overwrite=args.overwrite)
    patch_config(args.output_dir, speaker_name=args.speaker_name, speaker_id=args.speaker_id)
    save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))

    print(f"Exported Qwen3-TTS custom_voice model to {args.output_dir}")
    print(f"Speaker: {args.speaker_name} -> {args.speaker_id}, ref_audio={ref_audio}")


if __name__ == "__main__":
    main()
