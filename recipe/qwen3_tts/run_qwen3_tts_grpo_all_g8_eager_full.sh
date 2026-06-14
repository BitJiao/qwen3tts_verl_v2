#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-grpo}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"

bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_all_g8_eager.sh" "$@"
