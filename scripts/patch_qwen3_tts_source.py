#!/usr/bin/env python3
"""Patch Qwen3-TTS source files needed by this verl recipe."""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(source: str, old: str, new: str, path: Path, label: str) -> tuple[str, bool]:
    if new in source:
        return source, False
    if old not in source:
        raise SystemExit(f"{path}: could not find expected block for {label}")
    return source.replace(old, new, 1), True


def _patch_sft_12hz(repo: Path) -> bool:
    path = repo / "finetuning" / "sft_12hz.py"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    source = path.read_text()
    changed = False

    source, did_change = _replace_once(
        source,
        "import torch\n",
        "import torch\nimport torch.nn.functional as F\n",
        path,
        "torch.nn.functional import",
    )
    changed = changed or did_change

    source, did_change = _replace_once(
        source,
        "                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask\n",
        "                input_text_embedding = model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))\n"
        "                input_text_embedding = input_text_embedding * text_embedding_mask\n",
        path,
        "text projection",
    )
    changed = changed or did_change

    source, did_change = _replace_once(
        source,
        "                for i in range(1, 16):\n",
        "                for i in range(1, model.talker.config.num_code_groups):\n",
        path,
        "num_code_groups",
    )
    changed = changed or did_change

    source, did_change = _replace_once(
        source,
        "                outputs = model.talker(\n"
        "                    inputs_embeds=input_embeddings[:, :-1, :],\n"
        "                    attention_mask=attention_mask[:, :-1],\n"
        "                    labels=codec_0_labels[:, 1:],\n"
        "                    output_hidden_states=True\n"
        "                )\n"
        "\n"
        "                hidden_states = outputs.hidden_states[0][-1]\n",
        "                outputs = model.talker(\n"
        "                    inputs_embeds=input_embeddings[:, :-1, :],\n"
        "                    attention_mask=attention_mask[:, :-1],\n"
        "                    output_hidden_states=True\n"
        "                )\n"
        "                codec_loss_mask = codec_0_labels[:, 1:].ne(-100)\n"
        "                codec_0_loss = F.cross_entropy(\n"
        "                    outputs.logits[codec_loss_mask],\n"
        "                    codec_0_labels[:, 1:][codec_loss_mask],\n"
        "                )\n"
        "\n"
        "                hidden_states = outputs.hidden_states[0][-1]\n",
        path,
        "codec-0 explicit CE",
    )
    changed = changed or did_change

    source, did_change = _replace_once(
        source,
        "                loss = outputs.loss + 0.3 * sub_talker_loss\n",
        "                loss = codec_0_loss + 0.3 * sub_talker_loss\n",
        path,
        "total loss",
    )
    changed = changed or did_change

    if changed:
        path.write_text(source)
    return changed


def _patch_modeling(repo: Path) -> bool:
    path = repo / "qwen_tts" / "core" / "models" / "modeling_qwen3_tts.py"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    source = path.read_text()

    start = source.find("    def forward_finetune(\n")
    end = source.find("    @can_return_tuple\n    def forward(\n", start)
    if start == -1 or end == -1:
        raise SystemExit(f"{path}: could not locate code predictor forward_finetune")

    segment = source[start:end]
    original_loss = (
        "        loss = None\n"
        "        if labels is not None:\n"
        "            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)\n"
    )
    masked_loss = (
        "        loss = None\n"
        "        if labels is not None:\n"
        "            loss_mask = labels.ne(-100)\n"
        "            if loss_mask.any():\n"
        "                loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])\n"
        "            else:\n"
        "                loss = logits.sum() * 0.0\n"
    )
    user_loss = (
        "        loss = None\n"
        "        if labels is not None:\n"
        "            loss = F.cross_entropy(\n"
        "                logits.reshape(-1, self.config.vocab_size),\n"
        "                labels.reshape(-1),\n"
        "                ignore_index=-100,\n"
        "            )\n"
    )
    if user_loss in segment:
        return False
    if original_loss in segment:
        patched_segment = segment.replace(original_loss, user_loss, 1)
    elif masked_loss in segment:
        patched_segment = segment.replace(masked_loss, user_loss, 1)
    else:
        raise SystemExit(f"{path}: could not find code predictor loss block")

    path.write_text(source[:start] + patched_segment + source[end:])
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to a Qwen3-TTS source checkout")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "qwen_tts").is_dir():
        raise SystemExit(f"{repo} is not a Qwen3-TTS source checkout")

    changed = []
    if _patch_sft_12hz(repo):
        changed.append("finetuning/sft_12hz.py")
    if _patch_modeling(repo):
        changed.append("qwen_tts/core/models/modeling_qwen3_tts.py")

    if changed:
        print("Patched Qwen3-TTS source: " + ", ".join(changed))
    else:
        print("Qwen3-TTS source already patched")


if __name__ == "__main__":
    main()
