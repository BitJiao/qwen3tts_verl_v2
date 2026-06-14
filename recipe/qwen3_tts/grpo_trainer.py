import argparse
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from finetuning.dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import AutoConfig

import recipe.qwen3_tts.register  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight RL trainer for Qwen3-TTS Base voice-clone post-training.")
    parser.add_argument("--model_path", default="/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_tts_grpo")
    parser.add_argument("--reward_fn", default=None, help="Python reward function as module:function or /path/file.py:function")
    parser.add_argument("--algorithm", default="grpo", choices=["grpo", "ppo", "gspo"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rollout_devices",
        default=None,
        help="Comma-separated devices for parallel rollout generation, e.g. cuda:0,cuda:1 or auto.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sub_talker_loss_coef", type=float, default=0.3)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--policy_epochs", type=int, default=1)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--clip_ratio_low", type=float, default=None)
    parser.add_argument("--clip_ratio_high", type=float, default=None)
    parser.add_argument("--ratio_clip_min", type=float, default=-10.0)
    parser.add_argument("--ratio_clip_max", type=float, default=10.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--non_streaming_mode", action="store_true", default=True)
    parser.add_argument("--streaming_mode", dest="non_streaming_mode", action="store_false")
    parser.add_argument("--x_vector_only_mode", action="store_true", default=True)
    parser.add_argument("--icl_mode", dest="x_vector_only_mode", action="store_false")
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_local_model_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


def canonical_device(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(device)


def parse_rollout_devices(spec: str | None, default_device: torch.device) -> list[str]:
    if spec is None or spec.strip() == "":
        return [canonical_device(default_device)]

    spec = spec.strip()
    if spec.lower() == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        return ["cpu"]

    devices = []
    seen = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key = canonical_device(item)
        if key not in seen:
            devices.append(key)
            seen.add(key)
    if not devices:
        raise ValueError("--rollout_devices did not contain any valid devices")
    return devices


def move_tts_to_device(tts: Qwen3TTSModel, device: torch.device | str) -> Qwen3TTSModel:
    device = torch.device(device)
    tts.model.to(device)
    tts.device = device

    speech_tokenizer = getattr(tts.model, "speech_tokenizer", None)
    if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
        speech_tokenizer.model.to(device)
        speech_tokenizer.device = device

    return tts


def load_tts(
    local_model_dir: str,
    dtype: torch.dtype,
    attn_implementation: str,
    device: torch.device | str,
) -> Qwen3TTSModel:
    tts = Qwen3TTSModel.from_pretrained(
        local_model_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return move_tts_to_device(tts, device)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def import_reward_fn(spec: str | None) -> Callable[..., float]:
    if spec is None:
        return default_reward_fn

    if ":" not in spec:
        raise ValueError("--reward_fn must be module:function or /path/file.py:function")

    module_name, fn_name = spec.split(":", 1)
    if module_name.endswith(".py") or os.path.exists(module_name):
        module_path = Path(module_name).resolve()
        loaded = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if loaded is None or loaded.loader is None:
            raise ValueError(f"Cannot import reward module from {module_path}")
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Reward target is not callable: {spec}")
    return fn


def default_reward_fn(sample: dict[str, Any], wav: np.ndarray, sample_rate: int, **_: Any) -> float:
    if wav is None or len(wav) == 0 or not np.isfinite(wav).all():
        return -1.0
    duration = float(len(wav)) / float(sample_rate)
    max_duration = float(sample.get("max_duration", 30.0))
    if duration <= 0.2 or duration > max_duration:
        return -1.0
    return min(duration / max_duration, 1.0)


def call_reward(
    reward_fn: Callable[..., float],
    sample: dict[str, Any],
    wav: np.ndarray,
    sample_rate: int,
    codes: torch.Tensor,
) -> float:
    try:
        value = reward_fn(sample=sample, wav=wav, sample_rate=sample_rate, audio_codes=codes)
    except TypeError:
        value = reward_fn(sample, wav, sample_rate, codes)
    return float(value)


@torch.no_grad()
def generate_voice_clone_rollouts(
    tts: Qwen3TTSModel,
    sample: dict[str, Any],
    group_size: int,
    args,
) -> tuple[list[torch.Tensor], list[np.ndarray], int]:
    text = sample["text"]
    language = sample.get("language", "Auto")
    ref_audio = sample["ref_audio"]
    ref_text = sample.get("ref_text")

    texts = [text] * group_size
    languages = [language] * group_size
    ref_audios = [ref_audio] * group_size
    ref_texts = [ref_text] * group_size
    xvec_modes = [args.x_vector_only_mode] * group_size

    input_ids = tts._tokenize_texts([tts._build_assistant_text(t) for t in texts])
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=ref_audios,
        ref_text=ref_texts,
        x_vector_only_mode=xvec_modes,
    )
    voice_clone_prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)

    ref_ids = []
    for item in prompt_items:
        if item.ref_text is None or item.ref_text == "":
            ref_ids.append(None)
        else:
            ref_ids.append(tts._tokenize_texts([tts._build_ref_text(item.ref_text)])[0])

    codes, _ = tts.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt,
        languages=languages,
        non_streaming_mode=args.non_streaming_mode,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        subtalker_dosample=True,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        max_new_tokens=args.max_new_tokens,
    )

    decode_codes = []
    for i, generated in enumerate(codes):
        ref_code = voice_clone_prompt.get("ref_code", [None] * group_size)[i]
        if ref_code is None:
            decode_codes.append(generated)
        else:
            decode_codes.append(torch.cat([ref_code.to(generated.device), generated], dim=0))

    wavs_all, sample_rate = tts.model.speech_tokenizer.decode([{"audio_codes": c} for c in decode_codes])

    wavs = []
    for i, wav in enumerate(wavs_all):
        ref_code = voice_clone_prompt.get("ref_code", [None] * group_size)[i]
        if ref_code is None:
            wavs.append(wav)
            continue
        ref_len = int(ref_code.shape[0])
        total_len = int(decode_codes[i].shape[0])
        cut = int(ref_len / max(total_len, 1) * wav.shape[0])
        wavs.append(wav[cut:])

    return codes, wavs, sample_rate


def split_count(total: int, parts: int) -> list[tuple[int, int]]:
    parts = max(parts, 1)
    base, remainder = divmod(total, parts)
    offsets = []
    start = 0
    for idx in range(parts):
        count = base + (1 if idx < remainder else 0)
        if count <= 0:
            continue
        offsets.append((start, count))
        start += count
    return offsets


@torch.no_grad()
def generate_voice_clone_rollouts_parallel(
    rollout_models: list[Qwen3TTSModel],
    sample: dict[str, Any],
    group_size: int,
    args,
) -> tuple[list[torch.Tensor], list[np.ndarray], int]:
    if len(rollout_models) == 1 or group_size == 1:
        return generate_voice_clone_rollouts(rollout_models[0], sample, group_size, args)

    chunks = split_count(group_size, len(rollout_models))
    results: list[tuple[int, list[torch.Tensor], list[np.ndarray], int]] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(generate_voice_clone_rollouts, rollout_models[idx], sample, count, args): start
            for idx, (start, count) in enumerate(chunks)
        }
        for future in as_completed(futures):
            start = futures[future]
            codes, wavs, sample_rate = future.result()
            results.append((start, codes, wavs, sample_rate))

    results.sort(key=lambda item: item[0])
    sample_rates = {item[3] for item in results}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Rollout workers returned different sample rates: {sorted(sample_rates)}")

    codes_all: list[torch.Tensor] = []
    wavs_all: list[np.ndarray] = []
    for _, codes, wavs, _ in results:
        codes_all.extend(codes)
        wavs_all.extend(wavs)
    return codes_all, wavs_all, results[0][3]


@torch.no_grad()
def generate_prompt_batch_rollouts_parallel(
    rollout_models: list[Qwen3TTSModel],
    prompt_batch: list[dict[str, Any]],
    group_size: int,
    args,
) -> list[tuple[int, list[torch.Tensor], list[np.ndarray], int]]:
    if len(prompt_batch) == 1:
        codes, wavs, sample_rate = generate_voice_clone_rollouts_parallel(
            rollout_models,
            prompt_batch[0],
            group_size,
            args,
        )
        return [(0, codes, wavs, sample_rate)]

    results: list[tuple[int, list[torch.Tensor], list[np.ndarray], int]] = []
    workers = max(1, min(len(rollout_models), len(prompt_batch)))
    for wave_start in range(0, len(prompt_batch), workers):
        wave = prompt_batch[wave_start : wave_start + workers]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                executor.submit(
                    generate_voice_clone_rollouts,
                    rollout_models[local_idx],
                    sample,
                    group_size,
                    args,
                ): wave_start + local_idx
                for local_idx, sample in enumerate(wave)
            }
            for future in as_completed(futures):
                sample_idx = futures[future]
                codes, wavs, sample_rate = future.result()
                results.append((sample_idx, codes, wavs, sample_rate))

    results.sort(key=lambda item: item[0])
    return results


def build_training_batch(sample: dict[str, Any], codes: torch.Tensor, processor, config, device: torch.device):
    item = dict(sample)
    item.setdefault("audio", item["ref_audio"])
    item["audio_codes"] = codes.detach().cpu().tolist()
    dataset = TTSDataset([item], processor, config)
    batch = dataset.collate_fn([dataset[0]])
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def qwen3_tts_nll(model, batch: dict[str, torch.Tensor], sub_talker_loss_coef: float):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    speaker_embedding = model.speaker_encoder(ref_mels.to(device).to(dtype)).detach()
    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    loss = outputs.loss + sub_talker_loss_coef * sub_talker_loss
    return loss, outputs.loss.detach(), sub_talker_loss.detach()


def policy_loss_from_nll(
    nll: torch.Tensor,
    advantage: torch.Tensor,
    old_nll: torch.Tensor | None,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.algorithm == "grpo":
        return advantage * nll, torch.ones_like(nll.detach())

    if old_nll is None:
        raise ValueError(f"{args.algorithm} requires old_nll")

    log_ratio = (old_nll.detach() - nll).clamp(args.ratio_clip_min, args.ratio_clip_max)
    ratio = torch.exp(log_ratio)
    low = args.clip_ratio_low if args.clip_ratio_low is not None else args.clip_ratio
    high = args.clip_ratio_high if args.clip_ratio_high is not None else args.clip_ratio
    clipped_ratio = ratio.clamp(1.0 - low, 1.0 + high)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    return -surrogate, ratio.detach()


def group_advantages(rewards: list[float], eps: float) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std) < eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


def remove_model_files(output_dir: str):
    output = Path(output_dir)
    patterns = ["model*.safetensors", "model.safetensors.index.json", "pytorch_model*.bin", "pytorch_model.bin.index.json"]
    for pattern in patterns:
        for path in output.glob(pattern):
            path.unlink()


def save_checkpoint(tts: Qwen3TTSModel, base_model_dir: str, output_dir: str, overwrite: bool = True):
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        shutil.rmtree(output)
    shutil.copytree(base_model_dir, output_dir)
    remove_model_files(output_dir)
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in tts.model.state_dict().items()
    }
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    tts.processor.save_pretrained(output_dir)


def sync_rollout_models(source: Qwen3TTSModel, rollout_models: list[Qwen3TTSModel]):
    if len(rollout_models) <= 1:
        return
    source_state = source.model.state_dict()
    for rollout_model in rollout_models[1:]:
        rollout_model.model.load_state_dict(source_state, strict=True)


def main():
    args = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.group_size < 2:
        raise ValueError("--group_size must be >= 2 for grouped RL post-training")
    if args.policy_epochs < 1:
        raise ValueError("--policy_epochs must be >= 1")

    local_model_dir = resolve_local_model_dir(args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype if device.type != "cpu" else "fp32")
    rollout_devices = parse_rollout_devices(args.rollout_devices, device)
    main_device = canonical_device(device)
    print(
        json.dumps(
            {
                "requested_rollout_devices": rollout_devices,
                "train_device": main_device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tts = load_tts(local_model_dir, dtype, args.attn_implementation, device)
    if getattr(tts.model, "speaker_encoder", None) is None:
        raise ValueError("Qwen3-TTS GRPO training requires a Base checkpoint with speaker_encoder.")

    rollout_models = [tts]
    extra_rollout_devices = [rollout_device for rollout_device in rollout_devices if rollout_device != main_device]
    for rollout_device in extra_rollout_devices:
        print(json.dumps({"loading_rollout_device": rollout_device}, ensure_ascii=False), flush=True)
        rollout_model = load_tts(local_model_dir, dtype, args.attn_implementation, rollout_device)
        rollout_model.model.eval()
        rollout_models.append(rollout_model)
    print(
        json.dumps(
            {
                "rollout_devices": [canonical_device(next(model.model.parameters()).device) for model in rollout_models],
                "train_device": main_device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    config = AutoConfig.from_pretrained(local_model_dir)
    data = load_jsonl(args.train_jsonl)
    steps_per_epoch = math.ceil(len(data) / args.prompt_batch_size) if data else 0
    planned_steps = steps_per_epoch * args.num_epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    print(
        json.dumps(
            {
                "train_samples": len(data),
                "prompt_batch_size": args.prompt_batch_size,
                "group_size": args.group_size,
                "algorithm": args.algorithm,
                "policy_epochs": args.policy_epochs,
                "num_epochs": args.num_epochs,
                "max_steps": args.max_steps,
                "steps_per_epoch": steps_per_epoch,
                "planned_steps": planned_steps,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    reward_fn = import_reward_fn(args.reward_fn)
    optimizer = AdamW(tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_root = Path(args.output_dir)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        if args.shuffle:
            random.shuffle(data)

        for start in range(0, len(data), args.prompt_batch_size):
            step_start_time = time.perf_counter()
            prompt_batch = data[start : start + args.prompt_batch_size]
            optimizer.zero_grad(set_to_none=True)

            step_rewards = []
            step_losses = []
            step_codec_losses = []
            step_sub_losses = []
            step_advantages = []
            step_ratios = []
            zero_advantage_groups = 0

            for rollout_model in rollout_models:
                rollout_model.model.eval()
            rollout_start_time = time.perf_counter()
            rollout_results = generate_prompt_batch_rollouts_parallel(
                rollout_models,
                prompt_batch,
                args.group_size,
                args,
            )
            rollout_seconds = time.perf_counter() - rollout_start_time

            train_start_time = time.perf_counter()
            policy_items = []
            for sample_idx, codes_list, wavs, sample_rate in rollout_results:
                sample = prompt_batch[sample_idx]
                rewards = [
                    call_reward(reward_fn, sample=sample, wav=wav, sample_rate=sample_rate, codes=codes)
                    for codes, wav in zip(codes_list, wavs)
                ]
                advantages = group_advantages(rewards, args.advantage_eps).to(device)
                if torch.count_nonzero(advantages).item() == 0:
                    zero_advantage_groups += 1
                step_rewards.extend(rewards)
                step_advantages.extend(float(value) for value in advantages.detach().cpu())

                tts.model.train()
                for rollout_idx, codes in enumerate(codes_list):
                    train_batch = build_training_batch(sample, codes, tts.processor, config, device)
                    old_nll = None
                    if args.algorithm in {"ppo", "gspo"}:
                        with torch.no_grad():
                            old_nll, _, _ = qwen3_tts_nll(
                                tts.model,
                                train_batch,
                                sub_talker_loss_coef=args.sub_talker_loss_coef,
                            )
                    policy_items.append((train_batch, advantages[rollout_idx], old_nll))

            loss_scale = float(max(len(policy_items), 1) * args.policy_epochs)
            for _ in range(args.policy_epochs):
                for train_batch, advantage, old_nll in policy_items:
                    nll, codec_loss, sub_loss = qwen3_tts_nll(
                        tts.model,
                        train_batch,
                        sub_talker_loss_coef=args.sub_talker_loss_coef,
                    )
                    raw_policy_loss, ratio = policy_loss_from_nll(nll, advantage, old_nll, args)
                    loss = raw_policy_loss / loss_scale
                    loss.backward()
                    step_losses.append(float(loss.detach().cpu()))
                    step_codec_losses.append(float(codec_loss.cpu()))
                    step_sub_losses.append(float(sub_loss.cpu()))
                    step_ratios.append(float(ratio.cpu()))

            grad_norm = torch.nn.utils.clip_grad_norm_(tts.model.parameters(), args.max_grad_norm)
            optimizer.step()
            sync_rollout_models(tts, rollout_models)
            train_seconds = time.perf_counter() - train_start_time

            global_step += 1
            step_seconds = time.perf_counter() - step_start_time
            generated_rollouts = len(rollout_results) * args.group_size
            reward_mean = float(np.mean(step_rewards)) if step_rewards else math.nan
            reward_std = float(np.std(step_rewards)) if step_rewards else math.nan
            loss_mean = float(np.mean(step_losses)) if step_losses else math.nan
            advantage_abs_mean = float(np.mean(np.abs(step_advantages))) if step_advantages else math.nan
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "algorithm": args.algorithm,
                        "reward_mean": reward_mean,
                        "reward_std": reward_std,
                        "loss": loss_mean,
                        "codec_0_loss": float(np.mean(step_codec_losses)) if step_codec_losses else math.nan,
                        "sub_talker_loss": float(np.mean(step_sub_losses)) if step_sub_losses else math.nan,
                        "advantage_abs_mean": advantage_abs_mean,
                        "ratio_mean": float(np.mean(step_ratios)) if step_ratios else math.nan,
                        "ratio_std": float(np.std(step_ratios)) if step_ratios else math.nan,
                        "zero_advantage_groups": zero_advantage_groups,
                        "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "generated_rollouts": generated_rollouts,
                        "rollout_seconds": rollout_seconds,
                        "train_seconds": train_seconds,
                        "step_seconds": step_seconds,
                        "rollouts_per_second": generated_rollouts / max(step_seconds, 1e-6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(tts, local_model_dir, str(output_root / f"global_step_{global_step}"), overwrite=True)

            if args.max_steps > 0 and global_step >= args.max_steps:
                save_checkpoint(tts, local_model_dir, str(output_root / "final"), overwrite=True)
                return

        save_checkpoint(tts, local_model_dir, str(output_root / f"epoch_{epoch}"), overwrite=True)

    save_checkpoint(tts, local_model_dir, str(output_root / "final"), overwrite=True)


if __name__ == "__main__":
    main()
