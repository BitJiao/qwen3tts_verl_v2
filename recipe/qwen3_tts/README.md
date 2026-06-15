# Qwen3-TTS verl Recipe

This recipe keeps Qwen3-TTS changes isolated under `recipe/qwen3_tts`.

- `sft_trainer.py`: Qwen3-TTS 12Hz SFT on verl `TrainingWorker` + FSDP.
- `grpo_trainer.py`: lightweight Qwen3-TTS GRPO/PPO/GSPO-style runner for Base voice-clone RL.
- `export_custom_voice.py`: export verl FSDP SFT shards to Qwen3-TTS `custom_voice`.

## Environment

The host needs `ffmpeg` and `ffprobe` on `PATH` for audio preprocessing:

```bash
apt-get update
apt-get install -y ffmpeg
```

Create or reuse the Qwen3-TTS virtualenv, then install both projects editable:

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main
uv venv .venv --python 3.11
source .venv/bin/activate

pip install -U pip
pip install -e /opt/data/private/jsj/Qwen3-TTS-main
pip install -e /opt/data/private/jsj/Qwen3-TTS-main/verl-main
pip install librosa soundfile
```

SpeechJudge-GRM runs in a separate Python environment because its
Qwen2.5-Omni dependencies require a newer torch than the Qwen3-TTS training
stack. Follow [SPEECHJUDGE_SETUP.md](SPEECHJUDGE_SETUP.md) before GRPO.

## SFT Data

Use the Qwen3-TTS official data format. Each JSONL row must contain:

```json
{"audio":"./data/utt0001.wav","text":"...","ref_audio":"./data/ref.wav","audio_codes":[[...]]}
```

Generate `audio_codes` with Qwen3-TTS first:

```bash
python /opt/data/private/jsj/Qwen3-TTS-main/finetuning/prepare_data.py \
  --tokenizer_model_path /opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

## SFT Train

Install or expose the local Qwen3-TTS package, then run from the `verl-main` root:

```bash
pip install -e /opt/data/private/jsj/Qwen3-TTS-main

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

The training checkpoint is saved by verl under `checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}`.

This recipe currently targets Qwen3-TTS 12Hz Base checkpoints. It uses the
official Qwen3-TTS finetuning loss internally, so it does not go through verl's
generic language-model SFT loss.

## SFT Export

verl saves FSDP shards. Convert one `global_step_*` checkpoint to the
Qwen3-TTS `custom_voice` inference layout after training:

```bash
PYTHONPATH="$(pwd):/opt/data/private/jsj/Qwen3-TTS-main:${PYTHONPATH:-}" \
python -m recipe.qwen3_tts.export_custom_voice \
  --checkpoint_dir checkpoints/qwen3-tts-sft/qwen3_tts_12hz_base/global_step_100 \
  --base_model_dir /opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
  --output_dir /path/to/qwen3_tts_custom_voice \
  --speaker_name speaker_test \
  --train_jsonl /path/to/train_with_codes.jsonl \
  --overwrite
```

Pass `--ref_audio /path/to/ref.wav` instead of `--train_jsonl` if you want to
choose the reference audio explicitly. The reference audio must be 24kHz.

## GRPO Data

The GRPO runner samples speech from a Qwen3-TTS Base checkpoint in voice-clone
mode. Each JSONL row must contain `text` and `ref_audio`; optional fields are
passed through to the reward function.

```json
{"text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"Auto","target_duration":4.0}
```

If you run with `--icl_mode`, also provide `ref_text`.

## RL Data

The RL runners sample speech from a Qwen3-TTS Base checkpoint in voice-clone
mode. Each JSONL row must contain `text` and `ref_audio`; optional fields are
passed through to the reward function.

```json
{"text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"Auto","target_duration":4.0}
```

If you run with `--icl_mode`, also provide `ref_text`.

## GRPO / PPO / GSPO Train

Run from the `verl-main` root:

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=/path/to/train_grpo.jsonl \
REWARD_FN=recipe.qwen3_tts.speechjudge_reward:compute_score \
GROUP_SIZE=4 \
MAX_STEPS=10 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

Ready-to-run 8-rollout scripts:

```bash
# GRPO smoke/full run. Set MAX_STEPS=-1 for a full epoch.
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh

# GRPO full-epoch run. This defaults to MAX_STEPS=-1.
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager_full.sh

# PPO-style clipped policy objective.
bash recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh

# GSPO-style narrow clipped policy objective.
bash recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh
```

Common overrides:

```bash
MAX_STEPS=-1 \
ROLLOUT_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765 \
SPEECHJUDGE_REPO=/opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge \
SPEECHJUDGE_MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM \
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

The ready-to-run scripts default to four rollout devices
`cuda:0,cuda:1,cuda:2,cuda:3`, `PROMPT_BATCH_SIZE=4`, and SpeechJudge-GRM
naturalness reward via `recipe.qwen3_tts.speechjudge_reward:compute_score`.
They also default to `SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765`.
The SpeechJudge repository is expected at
`/opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge`, and the model
checkpoint is expected at
`/opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM`.
Start `recipe.qwen3_tts.speechjudge_server` before launching GRPO.

The runner saves checkpoints to `OUTPUT_DIR`, defaulting to
`checkpoints/qwen3_tts_${ALGORITHM}_...`. It saves Base-style checkpoints, so
inference uses `generate_voice_clone`.

Reward functions can be either `module:function` or `/path/to/file.py:function`.
They may use this signature:

```python
def compute_score(sample, wav, sample_rate, audio_codes) -> float:
    ...
```

The default `speechjudge_reward.py` calls SpeechJudge-GRM to judge speech
naturalness and normalizes its 1-10 score to 0-1.

## Notes

- `ATTN_IMPLEMENTATION=eager` is the default for the ready-to-run RL scripts
  because `sdpa` can fail in the speech tokenizer decode path on this stack.
- `MAX_STEPS=10` is a smoke-test setting. Use `MAX_STEPS=-1` for the full
  configured epoch count.
- PPO/GSPO here are lightweight Qwen3-TTS voice-clone RL adaptations. They do
  not yet route through verl's full Ray `main_ppo` rollout stack.
