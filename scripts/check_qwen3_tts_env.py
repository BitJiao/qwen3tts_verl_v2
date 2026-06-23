#!/usr/bin/env python3
"""Check that the Qwen3-TTS + verl environment imports the intended code."""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from pathlib import Path


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def _module_file(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        _fail(f"{module_name} has no __file__; check your Python environment")
    return Path(module_file).resolve()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if path is None:
            _fail(f"{binary} is not on PATH")
        _ok(f"{binary}: {path}")

    verl_file = _module_file("verl")
    if repo_root not in verl_file.parents:
        _fail(f"verl imports from {verl_file}, not this repo: {repo_root}")
    _ok(f"verl imports from this repo: {verl_file}")

    qwen_file = _module_file("qwen_tts")
    _ok(f"qwen_tts imports from: {qwen_file}")

    modeling = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")
    required = [
        "Qwen3TTSForConditionalGeneration",
        "Qwen3TTSTalkerForConditionalGeneration",
        "Qwen3TTSTalkerResizeMLP",
    ]
    for name in required:
        if not hasattr(modeling, name):
            _fail(f"qwen_tts.core.models.modeling_qwen3_tts is missing {name}")
    _ok("Qwen3-TTS modeling classes are importable")

    from recipe.qwen3_tts.grpo_trainer import qwen3_tts_nll

    source = inspect.getsource(qwen3_tts_nll)
    if "talker.text_projection" not in source:
        _fail("recipe.qwen3_tts.grpo_trainer.qwen3_tts_nll does not project text embeddings")
    if "codec_0_loss = F.cross_entropy" not in source:
        _fail("recipe.qwen3_tts.grpo_trainer.qwen3_tts_nll does not use explicit codec_0 CE loss")
    _ok("Qwen3-TTS RL loss uses text_projection and explicit codec_0 CE")

    print(f"[OK] python: {sys.executable}")


if __name__ == "__main__":
    main()
