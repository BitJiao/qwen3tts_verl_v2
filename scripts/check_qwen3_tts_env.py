#!/usr/bin/env python3
"""Check that the Qwen3-TTS + verl environment imports the intended code."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def _ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def _fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)
    raise SystemExit(1)


def _module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        _fail(f"{module_name} is not installed. Run scripts/setup_qwen3tts_env.sh first.")
    return Path(spec.origin).resolve()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if path is None:
            _fail(f"{binary} is not on PATH")
        _ok(f"{binary}: {path}")

    verl_file = _module_path("verl")
    if repo_root not in verl_file.parents:
        _fail(f"verl imports from {verl_file}, not this repo: {repo_root}")
    _ok(f"verl imports from this repo: {verl_file}")

    qwen_file = _module_path("qwen_tts")
    _ok(f"qwen_tts is installed from: {qwen_file}")

    qwen_root = qwen_file.parent
    modeling_file = qwen_root / "core" / "models" / "modeling_qwen3_tts.py"
    if not modeling_file.is_file():
        _fail(f"Qwen3-TTS modeling file is missing: {modeling_file}")
    modeling_source = modeling_file.read_text()
    required = [
        "Qwen3TTSForConditionalGeneration",
        "Qwen3TTSTalkerForConditionalGeneration",
        "Qwen3TTSTalkerResizeMLP",
    ]
    for name in required:
        if f"class {name}" not in modeling_source:
            _fail(f"qwen_tts.core.models.modeling_qwen3_tts is missing {name}")
    _ok("Qwen3-TTS modeling source contains required classes")
    if "logits.reshape(-1, self.config.vocab_size)" not in modeling_source or "ignore_index=-100" not in modeling_source:
        _fail("Qwen3-TTS code predictor fine-tune loss is not patched. Run scripts/setup_qwen3tts_env.sh.")
    _ok("Qwen3-TTS code predictor fine-tune loss uses explicit CE")

    sft_source = (qwen_root.parent / "finetuning" / "sft_12hz.py").read_text()
    if "model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))" not in sft_source:
        _fail("Qwen3-TTS finetuning/sft_12hz.py does not project text embeddings")
    if "codec_0_loss = F.cross_entropy" not in sft_source:
        _fail("Qwen3-TTS finetuning/sft_12hz.py does not use explicit codec-0 CE loss")
    _ok("Qwen3-TTS upstream SFT script is patched")

    trainer_source = repo_root / "recipe" / "qwen3_tts" / "grpo_trainer.py"
    source = trainer_source.read_text()
    if "talker.text_projection" not in source:
        _fail("recipe.qwen3_tts.grpo_trainer does not project text embeddings")
    if "codec_0_loss = F.cross_entropy" not in source:
        _fail("recipe.qwen3_tts.grpo_trainer does not use explicit codec_0 CE loss")
    if "sub_talker_loss = F.cross_entropy" not in source:
        _fail("recipe.qwen3_tts.grpo_trainer does not use explicit sub-talker CE loss")
    _ok("Qwen3-TTS RL loss uses text_projection plus explicit codec/sub-talker CE")

    sft_script = (repo_root / "recipe" / "qwen3_tts" / "run_qwen3_tts_sft_fsdp.sh").read_text()
    if "engine.use_orig_params=true" not in sft_script:
        _fail("recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh must set engine.use_orig_params=true")
    _ok("Qwen3-TTS SFT script enables FSDP use_orig_params")

    print(f"[OK] python: {sys.executable}")


if __name__ == "__main__":
    main()
