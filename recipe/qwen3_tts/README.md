# Qwen3-TTS verl Recipe

This recipe keeps Qwen3-TTS changes isolated under `recipe/qwen3_tts`.

- `sft_trainer.py`: Qwen3-TTS 12Hz SFT on verl `TrainingWorker` + FSDP.
- `grpo_trainer.py`: Qwen3-TTS GRPO/PPO/GSPO-style runner for Base voice-clone RL.
- `ray_grpo_trainer.py`: Ray multi-GPU worker runner used by the GRPO/PPO/GSPO shell scripts.
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
REWARD_FN=recipe.qwen3_tts.combined_reward:compute_score \
REWARD_WER_WEIGHT=0.3 \
REWARD_SIM_WEIGHT=0.2 \
REWARD_JUDGE_WEIGHT=0.5 \
GROUP_SIZE=4 \
MAX_STEPS=10 \
bash recipe/qwen3_tts/run_qwen3_tts_grpo.sh
```

`run_qwen3_tts_grpo.sh` now enables Ray by default. It starts one Ray worker
per visible GPU, or per `ROLLOUT_DEVICES` entry when that is set. Rollout and
loss run across the workers, then gradients are summed with torch distributed
before every optimizer step. Set `USE_RAY=0` only when you explicitly want the
old single-process fallback.

By default the driver computes rewards on the complete GRPO group, preserving
batch reward semantics. For purely independent reward functions, set
`RAY_REWARD_ON_WORKER=1` to score on rollout workers and reduce audio transfer.

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
ROLLOUT_DEVICES=auto \
REWARD_WER_WEIGHT=0.3 \
REWARD_SIM_WEIGHT=0.2 \
REWARD_JUDGE_WEIGHT=0.5 \
REWARD_DURATION_WEIGHT=0.0 \
REWARD_ASR_BACKEND=transformers \
ASR_MODEL_PATH=/opt/data/private/jsj/models/openai-whisper-small \
SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765 \
SPEECHJUDGE_REPO=/opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge \
SPEECHJUDGE_MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM \
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

The ready-to-run scripts default to `ROLLOUT_DEVICES=auto`,
`PROMPT_BATCH_SIZE=4`, and a combined reward via
`recipe.qwen3_tts.combined_reward:compute_score`:

```text
reward = (
  REWARD_WER_WEIGHT * wer_score
  + REWARD_SIM_WEIGHT * sim_score
  + REWARD_JUDGE_WEIGHT * speechjudge_score
  + REWARD_DURATION_WEIGHT * duration_score
) / positive_weight_sum
```

Default weights are `REWARD_WER_WEIGHT=0.3`, `REWARD_SIM_WEIGHT=0.2`,
`REWARD_JUDGE_WEIGHT=0.5`, and `REWARD_DURATION_WEIGHT=0.0`.
`duration_score` is optional and uses `target_duration` when present.
Set `COMBINED_REWARD_LOG_COMPONENTS=1` to print component means from the
reward function. The scripts default to
`SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765`.
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

The default `combined_reward.py` combines text faithfulness, reference-audio
similarity, and SpeechJudge-GRM naturalness. `wer_sim_reward.py` and
`speechjudge_reward.py` remain available as standalone reward functions for
ablation runs.

## Notes

- `ATTN_IMPLEMENTATION=eager` is the default for the ready-to-run RL scripts
  because `sdpa` can fail in the speech tokenizer decode path on this stack.
- `MAX_STEPS=10` is a smoke-test setting. Use `MAX_STEPS=-1` for the full
  configured epoch count.
- PPO/GSPO here are Qwen3-TTS voice-clone RL adaptations with a custom Ray
  runner. They do not route through verl's generic LLM `main_ppo` data path.
