# qwen3tts_verl_v2

Qwen3-TTS 12Hz Base post-training recipe on top of verl.

This repository is a verl fork with one supported target: Qwen3-TTS SFT and
voice-clone RL. The Qwen3-TTS model package is not vendored here; install it
separately and expose it with `QWEN3_TTS_REPO`.

## What Works

- Qwen3-TTS SFT through the verl FSDP engine.
- GRPO/PPO/GSPO-style voice-clone RL runner.
- Ray multi-GPU rollout workers.
- WER/speaker-sim/SpeechJudge combined reward.
- FSDP checkpoint export to Qwen3-TTS `custom_voice` layout.

## Install

System dependency:

```bash
apt-get update
apt-get install -y ffmpeg
```

Python environment:

```bash
export QWEN3_TTS_REPO=/opt/data/private/jsj/Qwen3-TTS-main
export VERL_REPO=/opt/data/private/jsj/qwen3tts_verl_v2

cd "${QWEN3_TTS_REPO}"
uv venv .venv --python 3.11
source .venv/bin/activate

pip install -U pip
pip install -e "${QWEN3_TTS_REPO}"
pip install -e "${VERL_REPO}"
pip install librosa soundfile

cd "${VERL_REPO}"
python scripts/check_qwen3_tts_env.py
```

Do not install this repo from `Qwen3-TTS-main/verl-main`; that path was from an
older local checkout and is not valid for a clean clone of this repository.

## Data

SFT rows must contain Qwen3-TTS audio codes:

```json
{"audio":"./utt.wav","text":"...","ref_audio":"./ref.wav","audio_codes":[[...]]}
```

Generate codes with Qwen3-TTS:

```bash
python "${QWEN3_TTS_REPO}/finetuning/prepare_data.py" \
  --tokenizer_model_path /opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

RL rows need text plus a 24 kHz reference audio:

```json
{"text":"She said she would be here by noon.","ref_audio":"./ref.wav","language":"Auto","target_duration":4.0}
```

## SFT

```bash
cd "${VERL_REPO}"

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

The Qwen3-TTS loss is implemented in `FSDPEngineWithQwen3TTS`; it does not use
verl's generic LLM SFT loss.

## RL

Basic smoke run:

```bash
cd "${VERL_REPO}"

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
GROUP_SIZE=2 \
PROMPT_BATCH_SIZE=1 \
MAX_STEPS=1 \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
REWARD_ASR_BACKEND=none \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

8-rollout script:

```bash
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
MAX_STEPS=10 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

Use `MAX_STEPS=-1` for a full epoch. `ROLLOUT_DEVICES=auto` uses all visible
CUDA devices; set `CUDA_VISIBLE_DEVICES` or `ROLLOUT_DEVICES=cuda:0,cuda:1` to
control placement.

## Rewards

Default 8-rollout scripts use:

```text
recipe.qwen3_tts.combined_reward:compute_score
```

It combines WER, MFCC similarity, and optional SpeechJudge-GRM. For a first
environment check, set `REWARD_ASR_BACKEND=none` or use
`recipe.qwen3_tts.wer_sim_reward:compute_score`.

SpeechJudge requires a separate environment; see
`recipe/qwen3_tts/SPEECHJUDGE_SETUP.md`.

## Important Fixes

The Qwen3-TTS train loss in this repo explicitly:

- passes text embeddings through `talker.text_projection(...)` before adding
  codec embeddings;
- computes codec-0 CE from `outputs.logits` with `codec_0_labels[:, 1:]`
  directly, avoiding the extra shift in Transformers causal LM loss;
- uses `config.num_code_groups` instead of hard-coded `16`.

These fixes are in both:

- `recipe/qwen3_tts/grpo_trainer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`

## More

Detailed recipe notes are in `recipe/qwen3_tts/README.md`.
