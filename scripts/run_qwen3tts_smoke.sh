#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/qwen3tts_common.sh"

VENV_DIR="$(qwen3tts_abs_path "${VENV_DIR:-.venv}" "${VERL_ROOT}")"
QWEN3_TTS_REPO="$(qwen3tts_abs_path "${QWEN3_TTS_REPO:-third_party/Qwen3-TTS}" "${VERL_ROOT}")"
MODEL_PATH="$(qwen3tts_abs_path "${MODEL_PATH:-models/Qwen3-TTS-12Hz-1.7B-Base}" "${VERL_ROOT}")"
SMOKE_DATA_DIR="$(qwen3tts_abs_path "${SMOKE_DATA_DIR:-data/smoke}" "${VERL_ROOT}")"
RUN_TRAINING="${RUN_TRAINING:-0}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
SMOKE_DEVICE="${SMOKE_DEVICE:-cuda:0}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON=python3
fi

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${QWEN3_TTS_REPO}:${PYTHONPATH:-}"

"${PYTHON}" scripts/create_qwen3tts_smoke_data.py --output_dir "${SMOKE_DATA_DIR}"
"${PYTHON}" scripts/check_qwen3_tts_env.py

case "${RUN_TRAINING}" in
  1|true|True|TRUE|yes|Yes|YES)
    ;;
  *)
    echo "Smoke environment/data checks passed. Set RUN_TRAINING=1 to run 1-GPU SFT and GRPO smoke."
    exit 0
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "Place Qwen3-TTS weights under models/ or rerun setup with DOWNLOAD_MODEL=1." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${SMOKE_DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
  gpu_idx="${BASH_REMATCH[1]}"
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_idx}" | tr -d ' ')"
  if [[ "${free_mb}" =~ ^[0-9]+$ && "${free_mb}" -lt "${MIN_FREE_GPU_MB}" ]]; then
    echo "GPU ${gpu_idx} only has ${free_mb} MiB free; need at least ${MIN_FREE_GPU_MB} MiB for smoke training." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${SMOKE_DEVICE#cuda:}" \
VENV_DIR="${VENV_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL="${SMOKE_DATA_DIR}/train_with_codes.jsonl" \
N_GPUS_PER_NODE=1 \
TRAIN_BATCH_SIZE=1 \
MICRO_BATCH_SIZE_PER_GPU=1 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=-1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh

VENV_DIR="${VENV_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL="${SMOKE_DATA_DIR}/train_grpo.jsonl" \
OUTPUT_DIR=checkpoints/qwen3_tts_grpo_smoke \
USE_RAY=0 \
DEVICE="${SMOKE_DEVICE}" \
ROLLOUT_DEVICES="${SMOKE_DEVICE}" \
GROUP_SIZE=2 \
PROMPT_BATCH_SIZE=1 \
MAX_STEPS=1 \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
REWARD_ASR_BACKEND=none \
ATTN_IMPLEMENTATION=eager \
MAX_NEW_TOKENS=32 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
