# Qwen3-TTS Recipe

This directory contains the Qwen3-TTS-specific training code. Use the repo root
README for the full uv environment setup.

## Files

- `dataset.py`: Qwen3-TTS SFT JSONL dataset and collator.
- `sft_trainer.py`: SFT entry point using `FSDPEngineWithQwen3TTS`.
- `grpo_trainer.py`: Qwen3-TTS RL trainer for GRPO/PPO/GSPO-style updates.
- `ray_grpo_trainer.py`: Ray multi-GPU rollout/loss worker runner.
- `combined_reward.py`: WER + speaker similarity + optional SpeechJudge reward.
- `speechjudge_server.py`: local HTTP server for SpeechJudge-GRM.
- `export_custom_voice.py`: export FSDP checkpoint to Qwen3-TTS custom voice layout.

## Environment

Verified uv environment:

```bash
source /opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623/bin/activate
export QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main
export MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base
cd /opt/data/private/jsj/qwen3tts_verl_v2
python scripts/check_qwen3_tts_env.py
```

Rebuild command used on this host:

```bash
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
PYTHON_BIN=/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
TORCH_SPEC="torch==2.3.1 torchaudio==2.3.1" \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
DOWNLOAD_MODEL=0 \
bash scripts/setup_qwen3tts_env.sh
```

SpeechJudge is not installed here. Keep it in a separate environment and use
`SPEECHJUDGE_SETUP.md` if needed.

## SFT

Input row:

```json
{"audio":"./utt.wav","text":"...","ref_audio":"./ref.wav","audio_codes":[[...]]}
```

Run:

```bash
CUDA_VISIBLE_DEVICES=1 \
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
TRAIN_BATCH_SIZE=1 \
MICRO_BATCH_SIZE_PER_GPU=1 \
TOTAL_EPOCHS=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

## GRPO

Input row:

```json
{"text":"She said she would be here by noon.","ref_audio":"./ref.wav","language":"Auto","target_duration":4.0}
```

Smoke run without ASR:

```bash
CUDA_VISIBLE_DEVICES=1 \
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
USE_RAY=0 \
DEVICE=cuda:0 \
ROLLOUT_DEVICES=cuda:0 \
GROUP_SIZE=2 \
PROMPT_BATCH_SIZE=1 \
MAX_STEPS=1 \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
REWARD_ASR_BACKEND=none \
ATTN_IMPLEMENTATION=eager \
MAX_NEW_TOKENS=64 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

For a full epoch, set `MAX_STEPS=-1`.

## Export Custom Voice

```bash
python -m recipe.qwen3_tts.export_custom_voice \
  --checkpoint_dir /path/to/checkpoints/global_step_8 \
  --base_model_dir "${MODEL_PATH}" \
  --output_dir /path/to/qwen3_tts_custom_voice \
  --speaker_name speaker_test \
  --train_jsonl /path/to/train_with_codes.jsonl \
  --overwrite
```

Use `--ref_audio /path/to/ref.wav` instead of `--train_jsonl` to choose a
specific 24 kHz reference audio.

## Loss Logic

The Qwen3-TTS loss logic is implemented directly in:

- `recipe/qwen3_tts/grpo_trainer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`

It uses `talker.text_projection(...)` and explicit CE losses with
`ignore_index=-100`, matching the local reference modeling file.
