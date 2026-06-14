# Qwen3-TTS verl Post-Training Adaptation

This tree adds Qwen3-TTS 12Hz Base post-training support to verl.

## What Is Included

- SFT with verl FSDP worker integration.
- Lightweight voice-clone RL runners for GRPO, PPO, and GSPO.
- WER + reference-audio similarity reward hook.
- Export utility for converting FSDP SFT checkpoints to Qwen3-TTS custom voice layout.

## Install

System dependency:

```bash
apt-get update
apt-get install -y ffmpeg
```

Python environment:

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main
uv venv .venv --python 3.11
source .venv/bin/activate

pip install -U pip
pip install -e /opt/data/private/jsj/Qwen3-TTS-main
pip install -e /opt/data/private/jsj/Qwen3-TTS-main/verl-main
pip install librosa soundfile
```

Optional ASR backend for reward:

```bash
pip install faster-whisper
```

## Data

SFT rows need official Qwen3-TTS audio codes:

```json
{"audio":"./utt.wav","text":"...","ref_audio":"./ref.wav","audio_codes":[[...]]}
```

RL rows need voice-clone prompts:

```json
{"text":"...","ref_audio":"./ref.wav","language":"Auto","target_duration":4.0}
```

## Run SFT

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

## Run RL

Smoke-test defaults use `MAX_STEPS=10`. Use `MAX_STEPS=-1` for a full epoch.

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main

# GRPO
MAX_STEPS=-1 bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh

# PPO
MAX_STEPS=-1 bash recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh

# GSPO
MAX_STEPS=-1 bash recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh
```

Common overrides:

```bash
TRAIN_JSONL=/path/to/rl.jsonl \
OUTPUT_DIR=/path/to/output \
ROLLOUT_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
REWARD_WER_WEIGHT=0.6 \
REWARD_SIM_WEIGHT=0.4 \
ASR_MODEL_PATH=/path/to/whisper-model \
MAX_STEPS=-1 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

The ready-to-run scripts default to the local ASR model at
`/opt/data/private/jsj/models/openai-whisper-small`. Set
`REWARD_ASR_BACKEND=none` to disable WER scoring.

`ATTN_IMPLEMENTATION=eager` is the default in the ready-to-run RL scripts
because the current stack can fail in speech-tokenizer decode with `sdpa`.

## GitHub Packaging

This directory is currently not a git repository. To publish:

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main
git init
git add QWEN3_TTS_VERL_ADAPTATION.md recipe/qwen3_tts recipe/__init__.py \
  verl/workers/engine/__init__.py \
  verl/workers/engine/fsdp/__init__.py \
  verl/workers/engine/fsdp/transformer_impl.py
git commit -m "Add Qwen3-TTS verl post-training adaptation"
git branch -M main
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```

Do not add generated artifacts such as `logs/`, `checkpoints/`, `__pycache__/`,
or `verl.egg-info/`.
