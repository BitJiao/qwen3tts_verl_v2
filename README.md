# qwen3tts_verl_v2

Qwen3-TTS 12Hz Base post-training recipe on top of verl.

This repository is a verl fork with one supported target: Qwen3-TTS SFT and
voice-clone RL. A clean clone is expected to become one working directory:
Qwen3-TTS source is cloned into `third_party/Qwen3-TTS`, the shared Python
environment lives at `.venv`, and model weights live under `models/`.

## What Works

- Qwen3-TTS SFT through the verl FSDP engine.
- GRPO/PPO/GSPO-style voice-clone RL runner.
- Ray multi-GPU rollout workers.
- WER/speaker-sim reward, plus optional SpeechJudge through a separate server.
- FSDP checkpoint export to Qwen3-TTS `custom_voice` layout.

## Install From A Clean Clone

System dependency:

```bash
apt-get update
apt-get install -y ffmpeg
```

Clone this repo and let the setup script fetch Qwen3-TTS:

```bash
git clone https://github.com/BitJiao/qwen3tts_verl_v2.git
cd qwen3tts_verl_v2

bash scripts/setup_qwen3tts_env.sh
source .venv/bin/activate
export QWEN3_TTS_REPO="$(pwd)/third_party/Qwen3-TTS"
export VERL_REPO="$(pwd)"
export MODEL_PATH="$(pwd)/models/Qwen3-TTS-12Hz-1.7B-Base"

python scripts/check_qwen3_tts_env.py
```

The setup script installs `torch==2.3.1` and `torchaudio==2.3.1` before
installing Qwen3-TTS so pip does not resolve to an untested latest Torch/CUDA
stack. The remaining shared runtime dependencies are listed in
`requirements-qwen3tts-verl.txt` and installed into the same `.venv`. Override
with `TORCH_SPEC` and `TORCH_INDEX_URL` if your machine needs a different wheel
source. Set `TORCH_SPEC=skip` only if `.venv` already has compatible `torch`
and `torchaudio`.

The script also patches the cloned Qwen3-TTS source under `third_party/Qwen3-TTS`
so its 12 Hz fine-tuning path uses `talker.text_projection(...)` and explicit CE
loss. Existing Qwen3-TTS git checkouts are reused by default; set
`QWEN3_TTS_UPDATE=1` only when you intentionally want to update a clean checkout
before patching it again.

Optional model download:

```bash
DOWNLOAD_MODEL=1 bash scripts/setup_qwen3tts_env.sh
```

If Hugging Face is slow, pass your mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com DOWNLOAD_MODEL=1 bash scripts/setup_qwen3tts_env.sh
```

Do not install this repo from `Qwen3-TTS-main/verl-main`; that path was from an
older local checkout and is not valid for a clean clone of this repository.

For an existing Qwen3-TTS checkout, set `QWEN3_TTS_REPO=/path/to/Qwen3-TTS`
before running `scripts/setup_qwen3tts_env.sh`.

SpeechJudge is intentionally not installed by this script because it needs a
newer/conflicting stack. Run it in a separate environment and call it through
the local HTTP server described in `recipe/qwen3_tts/SPEECHJUDGE_SETUP.md`.

## Data

SFT rows must contain Qwen3-TTS audio codes:

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

RL rows need text plus a 24 kHz reference audio:

```json
{"text":"She said she would be here by noon.","ref_audio":"./ref.wav","language":"Auto","target_duration":4.0}
```

## SFT

```bash
cd "${VERL_REPO}"

MODEL_PATH="${MODEL_PATH}" \
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

MODEL_PATH="${MODEL_PATH}" \
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
