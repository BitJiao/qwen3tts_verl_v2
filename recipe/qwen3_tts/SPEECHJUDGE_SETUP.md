# SpeechJudge-GRM Setup

This recipe uses SpeechJudge-GRM as the naturalness reward for Qwen3-TTS GRPO.
Run SpeechJudge in its own environment and expose it through a local HTTP
server. Keep Qwen3-TTS training in the existing `.venv`.

## 1. Download SpeechJudge

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main
mkdir -p third_party pretrained

git clone https://github.com/AmphionTeam/SpeechJudge.git \
  third_party/SpeechJudge

HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/huggingface-cli download RMSnow/SpeechJudge-GRM \
  --local-dir pretrained/SpeechJudge-GRM \
  --local-dir-use-symlinks False
```

The completed model directory should contain:

```text
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
```

## 2. Create SpeechJudge Environment

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main
uv venv .speechjudge-venv --python 3.11
.speechjudge-venv/bin/python -m ensurepip --upgrade
.speechjudge-venv/bin/python -m pip install \
  "torch>=2.6" torchvision transformers==4.57.3 accelerate \
  qwen-omni-utils soundfile
```

Do not install these newer torch packages into `.venv`; that environment is
used by Qwen3-TTS training.

## 3. Start SpeechJudge Server

Use a GPU with enough free memory. Example:

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main

CUDA_VISIBLE_DEVICES=1 \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=/opt/data/private/jsj/Qwen3-TTS-main/verl-main:/opt/data/private/jsj/Qwen3-TTS-main \
/opt/data/private/jsj/Qwen3-TTS-main/.speechjudge-venv/bin/python \
  -m recipe.qwen3_tts.speechjudge_server \
  --host 127.0.0.1 \
  --port 8765 \
  --model_path /opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM \
  --speechjudge_repo /opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge \
  --attn_implementation eager
```

Check it:

```bash
curl http://127.0.0.1:8765/health
```

## 4. Run GRPO

From `verl-main`:

```bash
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=/opt/data/private/jsj/Qwen3-TTS-main/data/minds14_qwen3tts_all/all_grpo.jsonl \
REWARD_FN=recipe.qwen3_tts.combined_reward:compute_score \
REWARD_WER_WEIGHT=0.3 \
REWARD_SIM_WEIGHT=0.2 \
REWARD_JUDGE_WEIGHT=0.5 \
REWARD_DURATION_WEIGHT=0.0 \
REWARD_ASR_BACKEND=transformers \
ASR_MODEL_PATH=/opt/data/private/jsj/models/openai-whisper-small \
SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765 \
SPEECHJUDGE_REPO=/opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge \
SPEECHJUDGE_MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM \
MAX_STEPS=10 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

For a small single-GPU smoke test:

```bash
TRAIN_JSONL=/opt/data/private/jsj/Qwen3-TTS-main/data/minds14_qwen3tts/zh-CN_grpo.jsonl \
GROUP_SIZE=2 \
PROMPT_BATCH_SIZE=1 \
DEVICE=cuda:4 \
ROLLOUT_DEVICES=cuda:4 \
MAX_STEPS=2 \
REWARD_FN=recipe.qwen3_tts.combined_reward:compute_score \
REWARD_WER_WEIGHT=0.3 \
REWARD_SIM_WEIGHT=0.2 \
REWARD_JUDGE_WEIGHT=0.5 \
REWARD_DURATION_WEIGHT=0.0 \
REWARD_ASR_BACKEND=transformers \
ASR_MODEL_PATH=/opt/data/private/jsj/models/openai-whisper-small \
SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

The zh-CN smoke dataset has 8 samples, so `MAX_STEPS=10` runs 8 planned steps.
