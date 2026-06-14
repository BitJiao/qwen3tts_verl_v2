import logging
import os

import hydra
import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from recipe.qwen3_tts.dataset import Qwen3TTSSFTDataset
from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.device import auto_set_device, get_device_name

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def _noop_loss(*args, **kwargs):
    raise RuntimeError("Qwen3-TTS loss is computed inside FSDPEngineWithQwen3TTS.forward_step")


class Qwen3TTSSFTTrainer(SFTTrainer):
    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

        config = TrainingWorkerConfig(
            model_type="qwen3_tts",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=_noop_loss)
        self.engine = self.training_client.engine

    def _build_dataset(self):
        config = self.config
        self.train_dataset = Qwen3TTSSFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=self.model_config.tokenizer,
            config=config.data,
            processor=self.model_config.processor,
            max_samples=config.data.get("train_max_samples", -1),
            model_config=self.model_config.hf_config,
        )
        if config.data.val_files:
            self.val_dataset = Qwen3TTSSFTDataset(
                parquet_files=config.data.val_files,
                tokenizer=self.model_config.tokenizer,
                config=config.data,
                processor=self.model_config.processor,
                max_samples=config.data.get("val_max_samples", -1),
                model_config=self.model_config.hf_config,
            )
        else:
            self.val_dataset = None

    def _build_dataloader(self):
        config = self.config
        device_name = get_device_name()
        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        batch_seqlens = data["attention_mask"].sum(dim=-1).to(self.device_name)
        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor, input_tensor=batch_seqlens, group=dp_group
        )
        return output_tensor.tolist()


def run_qwen3_tts_sft(config):
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group()
    trainer = Qwen3TTSSFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="../../verl/trainer/config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    auto_set_device(config)
    run_qwen3_tts_sft(config)


if __name__ == "__main__":
    main()
