# Qwen3-TTS Recipe

This recipe is the only project-specific part of this verl fork.

## Files

- `dataset.py`: Qwen3-TTS SFT dataset and collate function.
- `sft_trainer.py`: FSDP SFT entry point.
- `grpo_trainer.py`: GRPO/PPO/GSPO-style Qwen3-TTS RL trainer.
- `ray_grpo_trainer.py`: Ray multi-GPU rollout/loss worker runner.
- `combined_reward.py`: WER + MFCC sim + SpeechJudge reward.
- `speechjudge_server.py`: local HTTP server for SpeechJudge-GRM.
- `export_custom_voice.py`: FSDP checkpoint export.
- `seedtts_eval.py`: SeedTTS-format Qwen3-TTS generation and manifest export.
- `run_qwen3_tts_seedtts_eval_g8.sh`: shell wrapper for multi-GPU SeedTTS eval.

## Environment

Use the repo-level setup script from a clean clone:

```bash
apt-get update
apt-get install -y ffmpeg

bash scripts/setup_qwen3tts_env.sh
source .venv/bin/activate
export QWEN3_TTS_REPO="$(pwd)/third_party/Qwen3-TTS"
export VERL_REPO="$(pwd)"
export MODEL_PATH="$(pwd)/models/Qwen3-TTS-12Hz-1.7B-Base"

python scripts/check_qwen3_tts_env.py
```

Set `DOWNLOAD_MODEL=1` when running the setup script if the Base checkpoint is
not already available locally.

The setup script creates one shared `.venv` for Qwen3-TTS and verl. It pins
`torch==2.3.1` and `torchaudio==2.3.1` by default and installs the fully pinned
runtime in `requirements-qwen3tts-verl.txt`, including `librosa==0.11.0` and
`numpy==1.26.4`. It also supports offline Qwen3-TTS source installation from
`models/sources/Qwen3-TTS-source.tar.gz`. Use `TORCH_SPEC` and
`TORCH_INDEX_URL` to override the Torch wheel selection. Set `TORCH_SPEC=skip`
only when the selected `.venv` already has compatible `torch` and `torchaudio`.
It also patches the cloned Qwen3-TTS source so the upstream 12 Hz SFT script and
code-predictor fine-tune loss match this recipe's loss implementation.

SpeechJudge is not part of this environment. Keep it separate and use the HTTP
server flow in `SPEECHJUDGE_SETUP.md`.

## SFT Data

Each JSONL row:

```json
{"audio":"./data/utt0001.wav","text":"...","ref_audio":"./data/ref.wav","audio_codes":[[...]]}
```

Generate `audio_codes` with Qwen3-TTS:

```bash
python "${QWEN3_TTS_REPO}/finetuning/prepare_data.py" \
  --tokenizer_model_path "${MODEL_PATH}/speech_tokenizer" \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

## SFT Train

```bash
cd "${VERL_REPO}"

MODEL_PATH="${MODEL_PATH}" \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

## Export SFT Checkpoint

```bash
cd "${VERL_REPO}"

python -m recipe.qwen3_tts.export_custom_voice \
  --checkpoint_dir checkpoints/qwen3-tts-sft/qwen3_tts_12hz_base/global_step_100 \
  --base_model_dir "${MODEL_PATH}" \
  --output_dir /path/to/qwen3_tts_custom_voice \
  --speaker_name speaker_test \
  --train_jsonl /path/to/train_with_codes.jsonl \
  --overwrite
```

Use `--ref_audio /path/to/ref.wav` instead of `--train_jsonl` to choose the
reference audio explicitly. Reference audio must be 24 kHz.

## RL Data

Each JSONL row:

```json
{"text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"Auto","target_duration":4.0}
```

If `--icl_mode` is used, include `ref_text`.

## RL Train

Single-GPU smoke test without ASR:

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

8-rollout GRPO:

```bash
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
MAX_STEPS=10 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

PPO/GSPO wrappers:

```bash
bash recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh
bash recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh
```

## SeedTTS Eval

Generate SeedTTS-format audio on all visible GPUs:

```bash
MODEL_PATH="${MODEL_PATH}" \
QWEN3_TTS_REPO="${QWEN3_TTS_REPO}" \
INPUT_JSONL=/path/to/seedtts/meta.lst \
OUTPUT_DIR=results/qwen3_tts_seedtts \
DEVICES=auto \
OVERWRITE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_seedtts_eval_g8.sh
```

The script accepts official SeedTTS `meta.lst` rows
`filename|prompt_text|prompt_wav|target_text|ground_truth_wav` and JSONL rows
with `sample_id`, `ref_text`, `ref_audio`/`ref_audio_path`, and
`text`/`target_text`. It writes generated wavs to `OUTPUT_DIR/wav_res` and
produces `manifest.jsonl`, `summary.json`, `meta.lst`, `wav_res_ref_text`, and
`wav_res_ref_text.txt`. Use `--devices 0,1,2,3,4,5,6,7` to force 8 GPUs. Set
`OVERWRITE=1` only when replacing an existing output directory is intended.

## Rewards

The ready-to-run script defaults to:

```text
recipe.qwen3_tts.combined_reward:compute_score
```

Weights:

```text
REWARD_WER_WEIGHT=0.3
REWARD_SIM_WEIGHT=0.2
REWARD_JUDGE_WEIGHT=0.5
REWARD_DURATION_WEIGHT=0.0
```

Set `COMBINED_REWARD_LOG_COMPONENTS=1` to print reward component means.

SpeechJudge runs in a separate environment; see `SPEECHJUDGE_SETUP.md`.

## Notes

- `ATTN_IMPLEMENTATION=eager` is the default for the ready-to-run scripts
  because `sdpa` can fail during speech-tokenizer decode on this stack.
- `MAX_STEPS=10` is a smoke-test setting. Use `MAX_STEPS=-1` for a full epoch.
- These PPO/GSPO scripts are Qwen3-TTS-specific wrappers and do not use verl's
  generic LLM `main_ppo` path.
