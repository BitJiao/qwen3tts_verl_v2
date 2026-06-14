#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-ppo}"
export POLICY_EPOCHS="${POLICY_EPOCHS:-1}"
export CLIP_RATIO="${CLIP_RATIO:-0.2}"

bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_all_g8_eager.sh" "$@"
