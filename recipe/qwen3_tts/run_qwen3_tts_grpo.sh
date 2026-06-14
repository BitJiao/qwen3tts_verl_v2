#!/usr/bin/env bash
set -euo pipefail

QWEN3_TTS_REPO=${QWEN3_TTS_REPO:-/opt/data/private/jsj/Qwen3-TTS-main}
MODEL_PATH=${MODEL_PATH:-/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base}
TRAIN_JSONL=${TRAIN_JSONL:-${QWEN3_TTS_REPO}/data/minds14_qwen3tts/zh-CN_grpo.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/qwen3_tts_grpo}
REWARD_FN=${REWARD_FN:-recipe.qwen3_tts.wer_sim_reward:compute_score}
PYTHON=${PYTHON:-${QWEN3_TTS_REPO}/.venv/bin/python}
ALGORITHM=${ALGORITHM:-grpo}
DEVICE=${DEVICE:-cuda:0}
ROLLOUT_DEVICES=${ROLLOUT_DEVICES:-}
DTYPE=${DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
GROUP_SIZE=${GROUP_SIZE:-4}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-20}
POLICY_EPOCHS=${POLICY_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-1.0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTHONPATH="$(pwd):${QWEN3_TTS_REPO}:${PYTHONPATH:-}"

ARGS=(
  --model_path "${MODEL_PATH}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --algorithm "${ALGORITHM}"
  --dtype "${DTYPE}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --group_size "${GROUP_SIZE}"
  --prompt_batch_size "${PROMPT_BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --lr "${LR}"
  --save_freq "${SAVE_FREQ}"
  --policy_epochs "${POLICY_EPOCHS}"
  --clip_ratio "${CLIP_RATIO}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_k "${TOP_K}"
  --top_p "${TOP_P}"
)

if [[ -n "${CLIP_RATIO_LOW}" ]]; then
  ARGS+=(--clip_ratio_low "${CLIP_RATIO_LOW}")
fi

if [[ -n "${CLIP_RATIO_HIGH}" ]]; then
  ARGS+=(--clip_ratio_high "${CLIP_RATIO_HIGH}")
fi

if [[ -n "${ROLLOUT_DEVICES}" ]]; then
  ARGS+=(--rollout_devices "${ROLLOUT_DEVICES}")
fi

if [[ -n "${REWARD_FN}" ]]; then
  ARGS+=(--reward_fn "${REWARD_FN}")
fi

"${PYTHON}" -m recipe.qwen3_tts.grpo_trainer "${ARGS[@]}" "$@"
