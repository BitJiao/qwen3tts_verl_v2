#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
QWEN3_TTS_REPO="${QWEN3_TTS_REPO:-/opt/data/private/jsj/Qwen3-TTS-main}"

cd "${VERL_ROOT}"
mkdir -p logs "${QWEN3_TTS_REPO}/checkpoints"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ALGORITHM="${ALGORITHM:-grpo}"
LOG="${LOG:-logs/${ALGORITHM}_all_g8_eager_${TS}.log}"

export QWEN3_TTS_REPO
export ALGORITHM
export MODEL_PATH="${MODEL_PATH:-/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base}"
export TRAIN_JSONL="${TRAIN_JSONL:-${QWEN3_TTS_REPO}/data/minds14_qwen3tts_all/all_grpo.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${QWEN3_TTS_REPO}/checkpoints/qwen3_tts_${ALGORITHM}_all_g8_eager_${TS}}"
export DEVICE="${DEVICE:-cuda:0}"
export ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-4}"
export MAX_STEPS="${MAX_STEPS:-10}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export REWARD_FN="${REWARD_FN:-recipe.qwen3_tts.wer_sim_reward:compute_score}"
export REWARD_ASR_BACKEND="${REWARD_ASR_BACKEND:-transformers}"
export ASR_MODEL_PATH="${ASR_MODEL_PATH:-/opt/data/private/jsj/models/openai-whisper-small}"
export ASR_DEVICE_INDEX="${ASR_DEVICE_INDEX:-0}"

echo "LOG=${LOG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ALGORITHM=${ALGORITHM}"
echo "REWARD_FN=${REWARD_FN}"
echo "REWARD_ASR_BACKEND=${REWARD_ASR_BACKEND}"
echo "ASR_MODEL_PATH=${ASR_MODEL_PATH}"

bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh "$@" 2>&1 | tee "${LOG}"
