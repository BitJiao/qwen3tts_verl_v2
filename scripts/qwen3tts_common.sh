#!/usr/bin/env bash

qwen3tts_abs_path() {
  local path="$1"
  local base="${2:-$(pwd)}"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" == "~" || "${path}" == ~/* ]]; then
    printf '%s\n' "${path/#\~/${HOME}}"
  elif [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${base}/${path}"
  fi
}
