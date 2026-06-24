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

## What Is In Git

The repository includes the pieces needed to rebuild the working directory:

- Qwen3-TTS source archive: `models/sources/Qwen3-TTS-source.tar.gz`.
- Pinned runtime requirements: `requirements-qwen3tts-verl.txt`.
- Setup/check/smoke scripts under `scripts/`.
- Tiny smoke data under `data/smoke/`.

Large model weights are not committed. Put them under
`models/Qwen3-TTS-12Hz-1.7B-Base` or run setup with `DOWNLOAD_MODEL=1`.

## Install From A Clean Clone

All paths below are relative to the repository root unless explicitly noted.
Install the only required system package first:

```bash
apt-get update
apt-get install -y ffmpeg
```

Clone this repo, create one uv environment, and install the bundled
Qwen3-TTS source plus this verl fork:

```bash
git clone https://github.com/BitJiao/qwen3tts_verl_v2.git
cd qwen3tts_verl_v2

VENV_DIR=.venv \
PYTHON_BIN=3.11 \
TORCH_PROFILE=cu121-verified \
bash scripts/setup_qwen3tts_env.sh

source .venv/bin/activate
export QWEN3_TTS_REPO=third_party/Qwen3-TTS
export VERL_REPO="$(pwd)"
export MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base

python scripts/check_qwen3_tts_env.py
```

Generate and check the repo-local smoke data:

```bash
bash scripts/run_qwen3tts_smoke.sh
```

That command validates the installed environment and the uploaded tiny
`data/smoke` files. To run 1-GPU SFT and GRPO smoke as well, use:

```bash
RUN_TRAINING=1 \
MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
SMOKE_DEVICE=cuda:0 \
bash scripts/run_qwen3tts_smoke.sh
```

The training smoke requires local model weights and enough free GPU memory.

The repository includes a small Qwen3-TTS source archive at
`models/sources/Qwen3-TTS-source.tar.gz`. On machines where `git clone
https://github.com/QwenLM/Qwen3-TTS.git` is blocked, the setup script extracts
that archive into `third_party/Qwen3-TTS` automatically. You can also provide a
different local checkout or archive:

```bash
QWEN3_TTS_REPO=third_party/Qwen3-TTS bash scripts/setup_qwen3tts_env.sh
QWEN3_TTS_SOURCE_ARCHIVE=models/sources/Qwen3-TTS-source.tar.gz bash scripts/setup_qwen3tts_env.sh
```

The verified runtime stack is Python 3.11, `torch==2.3.1+cu121`,
`torchaudio==2.3.1+cu121`, and `numpy==1.26.4` on NVIDIA driver 535.104.05.
The setup script installs `torch==2.3.1` and `torchaudio==2.3.1` before the
rest of the pinned dependencies in `requirements-qwen3tts-verl.txt`, so pip
does not resolve to an untested Torch/CUDA 13 or NumPy 2.x stack.

For a newer server with NVIDIA driver 593 and CUDA 13.2 runtime support, use a
high-version PyTorch profile explicitly:

```bash
VENV_DIR=.venv-cu130 \
PYTHON_BIN=3.11 \
TORCH_PROFILE=cu130 \
bash scripts/setup_qwen3tts_env.sh
```

`TORCH_PROFILE=cu130` installs `torch==2.10.0` and `torchaudio==2.10.0` from
the PyTorch `cu130` wheel index. If you specifically need CUDA 13.2 nightly
wheels, use:

```bash
VENV_DIR=.venv-cu132 \
PYTHON_BIN=3.11 \
TORCH_PROFILE=cu132-nightly \
bash scripts/setup_qwen3tts_env.sh
```

The CUDA 13.2 profile uses the PyTorch nightly `cu132` wheel index and should be
treated as a preview stack. You can always override the wheel selection with
`TORCH_SPEC` and `TORCH_INDEX_URL`, or set `TORCH_PROFILE=skip` only when the
selected venv already has compatible `torch` and `torchaudio`.

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

For an existing Qwen3-TTS checkout, set `QWEN3_TTS_REPO=third_party/Qwen3-TTS`
before running `scripts/setup_qwen3tts_env.sh`.

Model weights should live under `models/`, for example
`models/Qwen3-TTS-12Hz-1.7B-Base`. Additional model weights can be added as
separate subdirectories under `models/`; keep source archives under
`models/sources/`.

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

Use a relative JSONL path:

```bash
cd "${VERL_REPO}"

MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=third_party/Qwen3-TTS \
TRAIN_JSONL=data/smoke/train_with_codes.jsonl \
N_GPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
TOTAL_EPOCHS=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

The Qwen3-TTS loss is implemented in `FSDPEngineWithQwen3TTS`; it does not use
verl's generic LLM SFT loss.

The SFT script sets `engine.use_orig_params=true` because Qwen3-TTS has frozen
code-predictor parameters mixed with trainable parameters. FSDP can otherwise
raise a mixed `requires_grad` flattening error.

## RL

Single-process GRPO smoke:

```bash
cd "${VERL_REPO}"

MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=third_party/Qwen3-TTS \
TRAIN_JSONL=data/smoke/train_grpo.jsonl \
OUTPUT_DIR=checkpoints/qwen3_tts_grpo_smoke \
USE_RAY=0 \
DEVICE=cuda:0 \
ROLLOUT_DEVICES=cuda:0 \
GROUP_SIZE=2 \
PROMPT_BATCH_SIZE=1 \
MAX_STEPS=1 \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
REWARD_ASR_BACKEND=none \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

8-GPU Ray rollout GRPO:

```bash
MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
QWEN3_TTS_REPO=third_party/Qwen3-TTS \
TRAIN_JSONL=data/smoke/train_grpo.jsonl \
ROLLOUT_DEVICES=auto \
REWARD_ASR_BACKEND=none \
REWARD_FN=recipe.qwen3_tts.wer_sim_reward:compute_score \
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

## SeedTTS Evaluation

Generate Qwen3-TTS outputs for a SeedTTS `meta.lst` or JSONL on all visible
GPUs:

```bash
QWEN3_TTS_REPO=third_party/Qwen3-TTS \
MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
INPUT_JSONL=data/smoke/seedtts_meta.lst \
DEVICES=auto \
OVERWRITE=1 \
OUTPUT_DIR=results/qwen3_tts_seedtts \
bash recipe/qwen3_tts/run_qwen3_tts_seedtts_eval_g8.sh
```

The input can be the official SeedTTS pipe format:

```text
filename|prompt_text|prompt_wav|target_text|ground_truth_wav
```

or JSONL with fields such as `sample_id`, `ref_text`, `ref_audio` or
`ref_audio_path`, and `text` or `target_text`. The script writes generated
audio to `OUTPUT_DIR/wav_res`, plus `manifest.jsonl`, `summary.json`,
`meta.lst`, `wav_res_ref_text`, and `wav_res_ref_text.txt`. The generated
`meta.lst` and `wav_res/` directory are compatible with the official
`seed-tts-eval` `cal_wer.sh` / `cal_sim.sh` workflow.
Set `OVERWRITE=1` only when replacing an existing output directory is intended.

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
