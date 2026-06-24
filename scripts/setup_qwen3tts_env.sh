#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

QWEN3_TTS_REPO="${QWEN3_TTS_REPO:-${VERL_REPO}/third_party/Qwen3-TTS}"
QWEN3_TTS_GIT="${QWEN3_TTS_GIT:-https://github.com/QwenLM/Qwen3-TTS.git}"
QWEN3_TTS_REF="${QWEN3_TTS_REF:-main}"
QWEN3_TTS_UPDATE="${QWEN3_TTS_UPDATE:-0}"
QWEN3_TTS_SOURCE_DIR="${QWEN3_TTS_SOURCE_DIR:-${VERL_REPO}/models/sources/Qwen3-TTS}"
QWEN3_TTS_SOURCE_ARCHIVE="${QWEN3_TTS_SOURCE_ARCHIVE:-${VERL_REPO}/models/sources/Qwen3-TTS-source.tar.gz}"
VENV_DIR="${VENV_DIR:-${VERL_REPO}/.venv}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  else
    PYTHON_BIN=python3
  fi
fi
MODEL_PATH="${MODEL_PATH:-${VERL_REPO}/models/Qwen3-TTS-12Hz-1.7B-Base}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.3.1 torchaudio==2.3.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
RUNTIME_REQUIREMENTS="${RUNTIME_REQUIREMENTS:-${VERL_REPO}/requirements-qwen3tts-verl.txt}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is required. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg and ffprobe must be on PATH. Install them with apt before running audio workflows." >&2
  exit 1
fi

mkdir -p "$(dirname "${QWEN3_TTS_REPO}")"

is_qwen3_tts_tree() {
  local path="$1"
  [[ -f "${path}/pyproject.toml" && -d "${path}/qwen_tts" ]]
}

copy_qwen3_tts_tree() {
  local src="$1"
  local dst="$2"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
}

extract_qwen3_tts_archive() {
  local archive="$1"
  local dst="$2"
  local tmpdir
  tmpdir="$(mktemp -d)"
  tar -xzf "${archive}" -C "${tmpdir}"

  local candidate=""
  if is_qwen3_tts_tree "${tmpdir}"; then
    candidate="${tmpdir}"
  else
    while IFS= read -r dir; do
      if is_qwen3_tts_tree "${dir}"; then
        candidate="${dir}"
        break
      fi
    done < <(find "${tmpdir}" -mindepth 1 -maxdepth 2 -type d | sort)
  fi

  if [[ -z "${candidate}" ]]; then
    rm -rf "${tmpdir}"
    echo "${archive} does not contain a Qwen3-TTS source tree." >&2
    exit 1
  fi

  copy_qwen3_tts_tree "${candidate}" "${dst}"
  rm -rf "${tmpdir}"
}

if [[ -d "${QWEN3_TTS_REPO}/.git" ]]; then
  if [[ "${QWEN3_TTS_UPDATE}" == "1" ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "git is required when QWEN3_TTS_UPDATE=1" >&2
      exit 1
    fi
    if ! git -C "${QWEN3_TTS_REPO}" diff --quiet || ! git -C "${QWEN3_TTS_REPO}" diff --cached --quiet; then
      echo "${QWEN3_TTS_REPO} has local changes. Refusing to update; rerun without QWEN3_TTS_UPDATE=1." >&2
      exit 1
    fi
    echo "Updating Qwen3-TTS source: ${QWEN3_TTS_REPO}"
    git -C "${QWEN3_TTS_REPO}" fetch --depth 1 origin "${QWEN3_TTS_REF}"
    git -C "${QWEN3_TTS_REPO}" checkout FETCH_HEAD
  else
    echo "Using existing Qwen3-TTS git checkout: ${QWEN3_TTS_REPO}"
  fi
elif is_qwen3_tts_tree "${QWEN3_TTS_REPO}"; then
  echo "Using existing Qwen3-TTS source tree: ${QWEN3_TTS_REPO}"
elif [[ ! -e "${QWEN3_TTS_REPO}" ]]; then
  if is_qwen3_tts_tree "${QWEN3_TTS_SOURCE_DIR}"; then
    echo "Copying bundled Qwen3-TTS source tree into: ${QWEN3_TTS_REPO}"
    copy_qwen3_tts_tree "${QWEN3_TTS_SOURCE_DIR}" "${QWEN3_TTS_REPO}"
  elif [[ -f "${QWEN3_TTS_SOURCE_ARCHIVE}" ]]; then
    echo "Extracting bundled Qwen3-TTS source archive into: ${QWEN3_TTS_REPO}"
    extract_qwen3_tts_archive "${QWEN3_TTS_SOURCE_ARCHIVE}" "${QWEN3_TTS_REPO}"
  else
    if ! command -v git >/dev/null 2>&1; then
      echo "No bundled Qwen3-TTS source found and git is unavailable." >&2
      echo "Provide ${QWEN3_TTS_SOURCE_ARCHIVE}, ${QWEN3_TTS_SOURCE_DIR}, or set QWEN3_TTS_REPO." >&2
      exit 1
    fi
    echo "Cloning Qwen3-TTS source into: ${QWEN3_TTS_REPO}"
    git clone "${QWEN3_TTS_GIT}" "${QWEN3_TTS_REPO}"
    git -C "${QWEN3_TTS_REPO}" fetch --depth 1 origin "${QWEN3_TTS_REF}"
    git -C "${QWEN3_TTS_REPO}" checkout FETCH_HEAD
  fi
else
  echo "${QWEN3_TTS_REPO} exists but is not a Qwen3-TTS source tree." >&2
  echo "Set QWEN3_TTS_REPO to a valid checkout or remove the directory and rerun." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating shared Qwen3-TTS + verl venv: ${VENV_DIR}"
  if command -v uv >/dev/null 2>&1; then
    uv venv "${VENV_DIR}" --python "${PYTHON_BIN}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m ensurepip --upgrade
"${PYTHON}" -m pip install -U pip

echo "Patching Qwen3-TTS source for this recipe"
"${PYTHON}" "${SCRIPT_DIR}/patch_qwen3_tts_source.py" --repo "${QWEN3_TTS_REPO}"

if [[ "${TORCH_SPEC}" != "skip" ]]; then
  echo "Installing Torch packages: ${TORCH_SPEC}"
  read -r -a TORCH_PACKAGES <<< "${TORCH_SPEC}"
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_PACKAGES[@]}"
  else
    "${PYTHON}" -m pip install "${TORCH_PACKAGES[@]}"
  fi
else
  "${PYTHON}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "torchaudio") if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("TORCH_SPEC=skip requires torch and torchaudio to already be installed in VENV_DIR")
PY
fi

echo "Installing shared runtime dependencies from: ${RUNTIME_REQUIREMENTS}"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT
"${PYTHON}" - <<'PY' > "${CONSTRAINTS}"
import importlib.metadata as md

for name in ("torch", "torchaudio"):
    print(f"{name}=={md.version(name)}")
print("numpy<2.0.0")
PY

"${PYTHON}" -m pip install -c "${CONSTRAINTS}" -r "${RUNTIME_REQUIREMENTS}"
echo "Installing qwen-tts editable package: ${QWEN3_TTS_REPO}"
"${PYTHON}" -m pip install --no-deps -e "${QWEN3_TTS_REPO}"
echo "Installing verl editable package: ${VERL_REPO}"
"${PYTHON}" -m pip install -e "${VERL_REPO}"

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
  mkdir -p "${MODEL_PATH}"
  HF_CLI="${VENV_DIR}/bin/huggingface-cli"
  if [[ ! -x "${HF_CLI}" ]]; then
    echo "huggingface-cli is missing after installing huggingface_hub" >&2
    exit 1
  fi
  echo "Downloading ${MODEL_ID} into: ${MODEL_PATH}"
  download_cmd=("${HF_CLI}" download "${MODEL_ID}" --local-dir "${MODEL_PATH}")
  if [[ -n "${HF_ENDPOINT}" ]]; then
    HF_ENDPOINT="${HF_ENDPOINT}" "${download_cmd[@]}"
  else
    "${download_cmd[@]}"
  fi
fi

echo
echo "Environment ready."
echo "Run:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export QWEN3_TTS_REPO=${QWEN3_TTS_REPO}"
echo "  export VERL_REPO=${VERL_REPO}"
echo "  export MODEL_PATH=${MODEL_PATH}"
echo "  cd ${VERL_REPO}"
echo "  python scripts/check_qwen3_tts_env.py"
