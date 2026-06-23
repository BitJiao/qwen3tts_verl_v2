#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${VERL_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base}
QWEN3_TTS_REPO=${QWEN3_TTS_REPO:-${VERL_ROOT}/third_party/Qwen3-TTS}
TRAIN_JSONL=${TRAIN_JSONL:-train_with_codes.jsonl}
VAL_JSONL=${VAL_JSONL:-}
PROJECT_NAME=${PROJECT_NAME:-qwen3-tts-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_tts_12hz_base}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
LR=${LR:-2e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${QWEN3_TTS_REPO}:${PYTHONPATH:-}"

torchrun --standalone --nnodes="${NNODES}" --nproc_per_node="${N_GPUS_PER_NODE}" \
  -m recipe.qwen3_tts.sft_trainer \
  model.path="${MODEL_PATH}" \
  model.external_lib=recipe.qwen3_tts.register \
  model.trust_remote_code=false \
  model.enable_gradient_checkpointing=true \
  model.use_remove_padding=false \
  engine.strategy=fsdp \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.use_dynamic_bsz=false \
  engine.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  engine.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  optim.lr="${LR}" \
  optim.weight_decay=0.01 \
  data.train_files="${TRAIN_JSONL}" \
  data.val_files="${VAL_JSONL}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.num_workers=2 \
  +data.model_path="${MODEL_PATH}" \
  +data.processor_path="${MODEL_PATH}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.logger="['console']" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  "$@"
