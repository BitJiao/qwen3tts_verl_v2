# qwen3tts_verl_v2

This repository is a trimmed verl fork for Qwen3-TTS post-training. The
supported paths are:

- Qwen3-TTS SFT with verl FSDP.
- Qwen3-TTS GRPO/PPO/GSPO style RL.
- WER/speaker-sim reward, with SpeechJudge kept in a separate environment.

The environment used for Qwen3-TTS and verl is one shared Python venv. Do not
install SpeechJudge into this venv because its dependencies conflict.

## Verified Environment

The uv environment rebuilt and checked on this host is:

```bash
/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623
```

Important versions:

```text
Python 3.11.15
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
transformers==4.57.3
numpy==1.26.4
gradio==6.17.3
ray==2.55.1
qwen-tts==0.1.1
verl==0.9.0.dev0
```

The full setup/check logs are under:

```bash
logs/env_rebuild_20260623_1412/
```

These logs are local and intentionally ignored by git.

## System Packages

Install ffmpeg before running audio workflows:

```bash
apt-get update
apt-get install -y ffmpeg
```

The checks require both `/usr/bin/ffmpeg` and `/usr/bin/ffprobe` on `PATH`.

## Clean uv Install

From a clean clone:

```bash
git clone https://github.com/BitJiao/qwen3tts_verl_v2.git
cd qwen3tts_verl_v2
```

If `python3.11` is already on `PATH`, use:

```bash
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
DOWNLOAD_MODEL=0 \
bash scripts/setup_qwen3tts_env.sh
```

On this host, Python 3.11 came from uv, so the exact verified command was:

```bash
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
PYTHON_BIN=/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
TORCH_SPEC="torch==2.3.1 torchaudio==2.3.1" \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
DOWNLOAD_MODEL=0 \
bash scripts/setup_qwen3tts_env.sh
```

What the setup script does:

1. Reuses or clones Qwen3-TTS into `QWEN3_TTS_REPO`.
2. Creates `VENV_DIR` with `uv venv` when uv is available.
3. Installs `torch==2.3.1` and `torchaudio==2.3.1`.
4. Installs `requirements-qwen3tts-verl.txt` with Torch and NumPy constrained.
5. Installs Qwen3-TTS editable with `pip install --no-deps -e`.
6. Installs this verl fork editable with `pip install -e`.
7. Patches Qwen3-TTS source so the Qwen3-TTS fine-tune loss matches this recipe.

If you want the script to download the Base model:

```bash
DOWNLOAD_MODEL=1 bash scripts/setup_qwen3tts_env.sh
```

If Hugging Face is slow:

```bash
HF_ENDPOINT=https://hf-mirror.com DOWNLOAD_MODEL=1 bash scripts/setup_qwen3tts_env.sh
```

## Activate And Check

```bash
source /opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623/bin/activate
export QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main
export MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base
cd /opt/data/private/jsj/qwen3tts_verl_v2

python scripts/check_qwen3_tts_env.py
```

Expected key lines:

```text
[OK] verl imports from this repo
[OK] qwen_tts is installed from
[OK] Qwen3-TTS code predictor fine-tune loss uses explicit CE
[OK] Qwen3-TTS RL loss uses text_projection plus explicit codec/sub-talker CE
```

## SFT Data

SFT JSONL rows must contain Qwen3-TTS audio codes:

```json
{"audio":"./utt.wav","text":"...","ref_audio":"./ref.wav","audio_codes":[[...]]}
```

Generate codes with Qwen3-TTS:

```bash
python "${QWEN3_TTS_REPO}/finetuning/prepare_data.py" \
  --tokenizer_model_path "${MODEL_PATH}/speech_tokenizer" \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

## Run SFT

Smoke run:

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

The verified full local SFT run used 8 samples and completed 8 steps. Its local
log is:

```bash
logs/sft_full_20260623_1408/run_sft_full.log
```

## RL Data

RL JSONL rows need text plus a 24 kHz reference audio:

```json
{"text":"She said she would be here by noon.","ref_audio":"./ref.wav","language":"Auto","target_duration":4.0}
```

## Run GRPO

Single-GPU smoke run without ASR:

```bash
CUDA_VISIBLE_DEVICES=1 \
VENV_DIR=/opt/data/private/jsj/envs/qwen3tts_verl_uv_20260623 \
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
OUTPUT_DIR=/path/to/grpo_out \
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

Use `MAX_STEPS=-1` for a full epoch.

## SpeechJudge

SpeechJudge is intentionally separate. Use a separate environment and call it
through the HTTP flow in:

```bash
recipe/qwen3_tts/SPEECHJUDGE_SETUP.md
```

## Qwen3-TTS Loss Fixes

The supported training logic matches `/opt/data/private/jsj/modeling_qwen3_tts.py`:

- `talker.text_projection(...)` is applied to text embeddings.
- Codec-0 loss uses explicit `F.cross_entropy(..., ignore_index=-100)`.
- Sub-talker loss also uses explicit `F.cross_entropy(..., ignore_index=-100)`.
- The Qwen3-TTS code predictor fine-tune loss is patched to:

```python
F.cross_entropy(
    logits.reshape(-1, self.config.vocab_size),
    labels.reshape(-1),
    ignore_index=-100,
)
```

Implementation locations:

- `scripts/patch_qwen3_tts_source.py`
- `recipe/qwen3_tts/grpo_trainer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
