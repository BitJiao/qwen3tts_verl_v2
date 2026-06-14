#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ALGORITHM="${ALGORITHM:-gspo}"
export POLICY_EPOCHS="${POLICY_EPOCHS:-1}"
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.0003}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.0004}"

bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_all_g8_eager.sh" "$@"
