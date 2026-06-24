#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${VERL_ROOT}/scripts/qwen3tts_common.sh"

QWEN3_TTS_REPO=${QWEN3_TTS_REPO:-${VERL_ROOT}/third_party/Qwen3-TTS}
MODEL_PATH=${MODEL_PATH:-${VERL_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base}
INPUT_JSONL=${INPUT_JSONL:-${SEEDTTS_META:-}}
OUTPUT_DIR=${OUTPUT_DIR:-${VERL_ROOT}/results/qwen3_tts_seedtts}
DEVICES=${DEVICES:-auto}
OVERWRITE=${OVERWRITE:-0}
VENV_DIR=${VENV_DIR:-${VERL_ROOT}/.venv}
QWEN3_TTS_REPO="$(qwen3tts_abs_path "${QWEN3_TTS_REPO}" "${VERL_ROOT}")"
MODEL_PATH="$(qwen3tts_abs_path "${MODEL_PATH}" "${VERL_ROOT}")"
if [[ -n "${INPUT_JSONL}" ]]; then
  INPUT_JSONL="$(qwen3tts_abs_path "${INPUT_JSONL}" "${VERL_ROOT}")"
fi
OUTPUT_DIR="$(qwen3tts_abs_path "${OUTPUT_DIR}" "${VERL_ROOT}")"
VENV_DIR="$(qwen3tts_abs_path "${VENV_DIR}" "${VERL_ROOT}")"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON="${VENV_DIR}/bin/python"
  else
    PYTHON=python3
  fi
fi

if [[ -z "${INPUT_JSONL}" ]]; then
  echo "Set INPUT_JSONL=data/seedtts/meta.lst or SEEDTTS_META=data/seedtts/meta.lst" >&2
  exit 1
fi

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${QWEN3_TTS_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

ARGS=(
  --input "${INPUT_JSONL}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --devices "${DEVICES}"
)

case "${OVERWRITE}" in
  1|true|True|TRUE|yes|Yes|YES)
    ARGS+=(--overwrite)
    ;;
esac

"${PYTHON}" -m recipe.qwen3_tts.seedtts_eval \
  "${ARGS[@]}" \
  "$@"
