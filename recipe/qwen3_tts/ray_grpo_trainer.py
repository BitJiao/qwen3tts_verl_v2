import json
import math
import os
import random
import shutil
import socket
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoConfig

from recipe.qwen3_tts.grpo_trainer import (
    build_training_batch,
    call_rewards,
    format_eta,
    generate_voice_clone_rollouts,
    group_advantages,
    import_reward_fn,
    load_jsonl,
    load_tts,
    parse_rollout_devices,
    policy_loss_from_nll,
    qwen3_tts_nll,
    resolve_local_model_dir,
    save_checkpoint,
    torch_dtype,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _split_count(total: int, parts: int) -> list[int]:
    parts = max(parts, 1)
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def _make_rollout_assignments(
    prompt_batch: list[dict[str, Any]],
    group_size: int,
    num_workers: int,
) -> list[list[dict[str, Any]]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    if not prompt_batch:
        return assignments

    if len(prompt_batch) >= num_workers:
        for sample_idx, sample in enumerate(prompt_batch):
            assignments[sample_idx % num_workers].append(
                {"sample_idx": sample_idx, "sample": sample, "count": group_size}
            )
        return assignments

    worker_cursor = 0
    worker_splits = _split_count(num_workers, len(prompt_batch))
    for sample_idx, (sample, workers_for_sample) in enumerate(zip(prompt_batch, worker_splits)):
        counts = _split_count(group_size, workers_for_sample)
        for local_worker_idx, count in enumerate(counts):
            if count <= 0:
                continue
            worker_idx = (worker_cursor + local_worker_idx) % num_workers
            assignments[worker_idx].append({"sample_idx": sample_idx, "sample": sample, "count": count})
        worker_cursor += workers_for_sample

    return assignments


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else math.nan


def _runtime_env() -> dict[str, dict[str, str]]:
    env_names = [
        "PYTHONPATH",
        "QWEN3_TTS_REPO",
        "SPEECHJUDGE_SERVER_URL",
        "SPEECHJUDGE_REPO",
        "SPEECHJUDGE_MODEL_PATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ]
    return {"env_vars": {name: os.environ[name] for name in env_names if name in os.environ}}


def _infer_num_workers(args, device: torch.device) -> int:
    if args.ray_num_workers is not None:
        if args.ray_num_workers < 1:
            raise ValueError("--ray_num_workers must be >= 1")
        return args.ray_num_workers

    if args.rollout_devices is None or str(args.rollout_devices).strip() == "":
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1

    rollout_devices = parse_rollout_devices(args.rollout_devices, device)
    return max(1, len(rollout_devices))


class Qwen3TTSRayWorker:
    def setup(
        self,
        args_dict: dict[str, Any],
        local_model_dir: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> dict[str, Any]:
        self.args = SimpleNamespace(**args_dict)
        self.local_model_dir = local_model_dir
        self.rank = rank
        self.world_size = world_size

        torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
        torch.set_num_interop_threads(max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))))

        seed = int(self.args.seed) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        if world_size > 1:
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=int(self.args.ray_pg_timeout_s)),
            )

        dtype = torch_dtype(self.args.dtype if self.device.type != "cpu" else "fp32")
        self.tts = load_tts(local_model_dir, dtype, self.args.attn_implementation, self.device)
        if getattr(self.tts.model, "speaker_encoder", None) is None:
            raise ValueError("Qwen3-TTS GRPO training requires a Base checkpoint with speaker_encoder.")

        self.config = AutoConfig.from_pretrained(local_model_dir)
        self.reward_fn = import_reward_fn(self.args.reward_fn)
        self.optimizer = AdamW(
            self.tts.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return {
            "rank": rank,
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "device": str(self.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def rollout(self, rollout_assignments: list[dict[str, Any]]) -> dict[str, Any]:
        rollout_start_time = time.perf_counter()
        self.tts.model.eval()

        records = []
        generated_rollouts = 0
        with torch.no_grad():
            for item in rollout_assignments:
                codes_list, wavs, sample_rate = generate_voice_clone_rollouts(
                    self.tts,
                    item["sample"],
                    int(item["count"]),
                    self.args,
                )
                generated_rollouts += len(codes_list)
                record = {
                    "sample_idx": int(item["sample_idx"]),
                    "sample": item["sample"],
                    "codes_list": [codes.detach().cpu() for codes in codes_list],
                }
                if self.args.ray_reward_on_worker:
                    record["rewards"] = call_rewards(
                        self.reward_fn,
                        sample=item["sample"],
                        wavs=wavs,
                        sample_rate=sample_rate,
                        codes_list=codes_list,
                    )
                else:
                    record["wavs"] = wavs
                    record["sample_rate"] = sample_rate
                records.append(record)

        return {
            "rank": self.rank,
            "records": records,
            "generated_rollouts": generated_rollouts,
            "rollout_seconds": time.perf_counter() - rollout_start_time,
        }

    def _sync_gradients(self):
        if self.world_size <= 1:
            return

        params = [param for param in self.tts.model.parameters() if param.requires_grad]
        flags = torch.tensor(
            [1 if param.grad is not None else 0 for param in params],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)

        for param, flag in zip(params, flags.tolist()):
            if flag == 0:
                param.grad = None
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    def train_step(self, train_records: list[dict[str, Any]], loss_scale: float) -> dict[str, Any]:
        step_start_time = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        step_losses: list[float] = []
        step_codec_losses: list[float] = []
        step_sub_losses: list[float] = []
        step_ratios: list[float] = []

        train_start_time = time.perf_counter()
        policy_items = []
        for record in train_records:
            sample = record["sample"]
            codes_list = record["codes_list"]
            advantages = torch.tensor(record["advantages"], dtype=torch.float32, device=self.device)
            self.tts.model.train()
            for rollout_idx, codes in enumerate(codes_list):
                train_batch = build_training_batch(sample, codes, self.tts.processor, self.config, self.device)
                old_nll = None
                if self.args.algorithm in {"ppo", "gspo"}:
                    with torch.no_grad():
                        old_nll, _, _ = qwen3_tts_nll(
                            self.tts.model,
                            train_batch,
                            sub_talker_loss_coef=self.args.sub_talker_loss_coef,
                        )
                policy_items.append((train_batch, advantages[rollout_idx], old_nll))

        for _ in range(self.args.policy_epochs):
            for train_batch, advantage, old_nll in policy_items:
                nll, codec_loss, sub_loss = qwen3_tts_nll(
                    self.tts.model,
                    train_batch,
                    sub_talker_loss_coef=self.args.sub_talker_loss_coef,
                )
                raw_policy_loss, ratio = policy_loss_from_nll(nll, advantage, old_nll, self.args)
                loss = raw_policy_loss / loss_scale
                loss.backward()
                step_losses.append(float(loss.detach().cpu()))
                step_codec_losses.append(float(codec_loss.cpu()))
                step_sub_losses.append(float(sub_loss.cpu()))
                step_ratios.append(float(ratio.cpu()))

        self._sync_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tts.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        train_seconds = time.perf_counter() - train_start_time

        return {
            "rank": self.rank,
            "local_records": len(train_records),
            "local_rollouts": sum(len(record["codes_list"]) for record in train_records),
            "losses": step_losses,
            "codec_losses": step_codec_losses,
            "sub_losses": step_sub_losses,
            "ratios": step_ratios,
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "train_seconds": train_seconds,
            "step_seconds": time.perf_counter() - step_start_time,
        }

    def save(self, output_dir: str):
        save_checkpoint(self.tts, self.local_model_dir, output_dir, overwrite=True)
        return output_dir

    def shutdown(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return True


def ray_main(args):
    import ray

    if args.group_size < 2:
        raise ValueError("--group_size must be >= 2 for grouped RL post-training")
    if args.policy_epochs < 1:
        raise ValueError("--policy_epochs must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_model_dir = resolve_local_model_dir(args.model_path)
    driver_device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    num_workers = _infer_num_workers(args, driver_device)

    if not ray.is_initialized():
        ray.init(address=args.ray_address, runtime_env=_runtime_env())

    cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    use_gpus = cluster_gpus > 0
    if torch.cuda.is_available() and cluster_gpus < num_workers:
        raise RuntimeError(
            f"Ray sees {cluster_gpus:g} GPU(s), but ray_num_workers={num_workers}. "
            "Check CUDA_VISIBLE_DEVICES or restart the Ray cluster."
        )

    master_addr = args.ray_master_addr
    if master_addr is None:
        master_addr = ray.util.get_node_ip_address()
    master_port = args.ray_master_port if args.ray_master_port > 0 else _find_free_port()

    print(
        json.dumps(
            {
                "ray_enabled": True,
                "ray_address": args.ray_address,
                "ray_num_workers": num_workers,
                "ray_worker_gpus": 1 if use_gpus else 0,
                "ray_cluster_gpus": cluster_gpus,
                "torch_dist_master": f"{master_addr}:{master_port}",
                "rollout_devices_hint": args.rollout_devices,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    worker_cls = ray.remote(
        num_gpus=1 if use_gpus else 0,
        num_cpus=args.ray_num_cpus_per_worker,
    )(Qwen3TTSRayWorker)
    workers = [worker_cls.remote() for _ in range(num_workers)]

    try:
        setup_refs = [
            worker.setup.remote(vars(args), local_model_dir, rank, num_workers, master_addr, master_port)
            for rank, worker in enumerate(workers)
        ]
        setup_info = ray.get(setup_refs)
        print(json.dumps({"ray_workers": setup_info}, ensure_ascii=False), flush=True)

        data = load_jsonl(args.train_jsonl)
        reward_fn = import_reward_fn(args.reward_fn)
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

        output_root = Path(args.output_dir)
        if output_root.exists() and args.overwrite:
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        global_step = 0
        train_begin_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            if args.shuffle:
                random.shuffle(data)

            for start in range(0, len(data), args.prompt_batch_size):
                step_start_time = time.perf_counter()
                prompt_batch = data[start : start + args.prompt_batch_size]
                rollout_assignments = _make_rollout_assignments(prompt_batch, args.group_size, num_workers)
                loss_scale = float(max(len(prompt_batch) * args.group_size, 1) * args.policy_epochs)

                rollout_refs = [
                    worker.rollout.remote(rollout_assignments[rank])
                    for rank, worker in enumerate(workers)
                ]
                worker_rollouts = ray.get(rollout_refs)

                records_by_sample: dict[int, list[tuple[int, dict[str, Any]]]] = {}
                generated_rollouts = 0
                rollout_seconds = 0.0
                for worker_rank, worker_result in enumerate(worker_rollouts):
                    generated_rollouts += int(worker_result["generated_rollouts"])
                    rollout_seconds = max(rollout_seconds, float(worker_result["rollout_seconds"]))
                    for record in worker_result["records"]:
                        records_by_sample.setdefault(int(record["sample_idx"]), []).append((worker_rank, record))

                train_records_by_worker: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
                step_rewards: list[float] = []
                step_advantages: list[float] = []
                zero_advantage_groups = 0
                for sample_idx in sorted(records_by_sample):
                    sample = prompt_batch[sample_idx]
                    flat_codes = []
                    flat_rewards = []
                    flat_wavs = []
                    sample_rate = None
                    slices = []
                    for worker_rank, record in records_by_sample[sample_idx]:
                        start_offset = len(flat_codes)
                        flat_codes.extend(record["codes_list"])
                        if args.ray_reward_on_worker:
                            flat_rewards.extend(record["rewards"])
                        else:
                            flat_wavs.extend(record["wavs"])
                            if sample_rate is None:
                                sample_rate = record["sample_rate"]
                            elif sample_rate != record["sample_rate"]:
                                raise RuntimeError(
                                    f"Ray workers returned different sample rates for sample {sample_idx}: "
                                    f"{sample_rate} vs {record['sample_rate']}"
                                )
                        end_offset = len(flat_codes)
                        slices.append((worker_rank, record, start_offset, end_offset))

                    if not args.ray_reward_on_worker:
                        flat_rewards = call_rewards(
                            reward_fn,
                            sample=sample,
                            wavs=flat_wavs,
                            sample_rate=sample_rate,
                            codes_list=flat_codes,
                        )

                    if len(flat_rewards) != len(flat_codes):
                        raise RuntimeError(
                            f"Reward count mismatch for sample {sample_idx}: "
                            f"{len(flat_rewards)} rewards for {len(flat_codes)} rollouts"
                        )

                    advantages = group_advantages(flat_rewards, args.advantage_eps)
                    if torch.count_nonzero(advantages).item() == 0:
                        zero_advantage_groups += 1
                    advantage_values = [float(value) for value in advantages.tolist()]
                    step_rewards.extend(flat_rewards)
                    step_advantages.extend(advantage_values)

                    for worker_rank, record, start_offset, end_offset in slices:
                        train_records_by_worker[worker_rank].append(
                            {
                                "sample": record["sample"],
                                "codes_list": record["codes_list"],
                                "advantages": advantage_values[start_offset:end_offset],
                            }
                        )

                step_refs = [
                    worker.train_step.remote(train_records_by_worker[rank], loss_scale)
                    for rank, worker in enumerate(workers)
                ]
                worker_stats = ray.get(step_refs)

                global_step += 1
                step_seconds = time.perf_counter() - step_start_time
                elapsed_seconds = time.perf_counter() - train_begin_time
                avg_step_seconds = elapsed_seconds / max(global_step, 1)
                remaining_steps = max(planned_steps - global_step, 0)
                eta_seconds = remaining_steps * avg_step_seconds
                finish_at = datetime.now() + timedelta(seconds=eta_seconds)

                step_losses = [value for stats in worker_stats for value in stats["losses"]]
                step_codec_losses = [value for stats in worker_stats for value in stats["codec_losses"]]
                step_sub_losses = [value for stats in worker_stats for value in stats["sub_losses"]]
                step_ratios = [value for stats in worker_stats for value in stats["ratios"]]
                train_seconds = max(float(stats["train_seconds"]) for stats in worker_stats)

                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "total_steps": planned_steps,
                            "remaining_steps": remaining_steps,
                            "progress_percent": 100.0 * global_step / max(planned_steps, 1),
                            "elapsed": format_eta(elapsed_seconds),
                            "eta": format_eta(eta_seconds),
                            "eta_seconds": eta_seconds,
                            "finish_at": finish_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "avg_step_seconds": avg_step_seconds,
                            "algorithm": args.algorithm,
                            "ray_num_workers": num_workers,
                            "reward_mean": _mean(step_rewards),
                            "reward_std": _std(step_rewards),
                            "loss": _mean(step_losses),
                            "codec_0_loss": _mean(step_codec_losses),
                            "sub_talker_loss": _mean(step_sub_losses),
                            "advantage_abs_mean": _mean([abs(value) for value in step_advantages]),
                            "ratio_mean": _mean(step_ratios),
                            "ratio_std": _std(step_ratios),
                            "zero_advantage_groups": zero_advantage_groups,
                            "grad_norm": _mean([float(stats["grad_norm"]) for stats in worker_stats]),
                            "generated_rollouts": generated_rollouts,
                            "local_rollouts": [int(stats["local_rollouts"]) for stats in worker_stats],
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
                    ray.get(workers[0].save.remote(str(output_root / f"global_step_{global_step}")))

                if args.max_steps > 0 and global_step >= args.max_steps:
                    ray.get(workers[0].save.remote(str(output_root / "final")))
                    return

            ray.get(workers[0].save.remote(str(output_root / f"epoch_{epoch}")))

        ray.get(workers[0].save.remote(str(output_root / "final")))
    finally:
        shutdown_refs = [worker.shutdown.remote() for worker in workers]
        try:
            ray.get(shutdown_refs, timeout=30)
        except Exception:
            pass
        ray.shutdown()
