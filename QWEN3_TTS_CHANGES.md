# Qwen3-TTS verl Change Summary

This repository is based on upstream `verl-main` and adds Qwen3-TTS 12Hz Base
post-training support.

## New Qwen3-TTS Recipe Files

- `recipe/qwen3_tts/dataset.py`
  - Qwen3-TTS SFT dataset wrapper for official `audio_codes` JSONL data.
- `recipe/qwen3_tts/sft_trainer.py`
  - SFT entry point using verl `SFTTrainer` and `TrainingWorker`.
- `recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh`
  - FSDP SFT launch script.
- `recipe/qwen3_tts/export_custom_voice.py`
  - Converts verl FSDP SFT checkpoints to Qwen3-TTS `custom_voice` layout.
- `recipe/qwen3_tts/grpo_trainer.py`
  - Lightweight Qwen3-TTS voice-clone RL runner.
  - Supports `grpo`, `ppo`, and `gspo` through `--algorithm`.
  - Generates rollouts, computes reward, builds Qwen3-TTS training batches,
    applies policy loss, and saves Base-style checkpoints.
- `recipe/qwen3_tts/wer_sim_reward.py`
  - Default WER + reference-audio similarity reward hook.
  - Uses edit-distance text score plus MFCC cosine similarity by default.
  - Defaults to `REWARD_ASR_BACKEND=none` to avoid network downloads; enable
    local ASR explicitly for WER scoring.
- `recipe/qwen3_tts/example_reward.py`
  - Simple audio-validity example reward.
- `recipe/qwen3_tts/smoke_reward.py`
  - Deterministic smoke-test reward.
- `recipe/qwen3_tts/register.py`
  - Registers local Qwen3-TTS model/config classes with Transformers.

## Launch Scripts

- `recipe/qwen3_tts/run_qwen3_tts_grpo.sh`
  - Generic RL launch script.
  - Exposes `ALGORITHM`, `REWARD_FN`, `GROUP_SIZE`, `PROMPT_BATCH_SIZE`,
    `MAX_STEPS`, `POLICY_EPOCHS`, and clip parameters.
- `recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh`
  - Ready-to-run GRPO smoke script.
  - Defaults to `MAX_STEPS=10`, `GROUP_SIZE=8`, `PROMPT_BATCH_SIZE=2`,
    `ATTN_IMPLEMENTATION=eager`.
- `recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager_full.sh`
  - Ready-to-run full-GRPO script.
  - Defaults to `MAX_STEPS=-1`, so it runs the configured full epoch.
- `recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh`
  - PPO-style clipped policy objective wrapper.
- `recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh`
  - GSPO-style narrow clipped policy objective wrapper.

## Modified Upstream verl Files

- `verl/workers/engine/fsdp/transformer_impl.py`
  - Added Qwen3-TTS finetune forward path.
  - Added `FSDPEngineWithQwen3TTS`.
  - Registered `model_type="qwen3_tts"` for FSDP/FSDP2.
  - Added Qwen3-TTS metrics: `codec_0_loss`, `sub_talker_loss`.
  - Made `disable_adapter()` safe for models without adapters.
- `verl/workers/engine/__init__.py`
  - Exports `FSDPEngineWithQwen3TTS`.
- `verl/workers/engine/fsdp/__init__.py`
  - Exports `FSDPEngineWithQwen3TTS`.

## Documentation

- `recipe/qwen3_tts/README.md`
  - Environment setup, data formats, SFT, GRPO, PPO, GSPO, reward usage.
- `QWEN3_TTS_VERL_ADAPTATION.md`
  - Top-level usage and GitHub packaging notes.
- `QWEN3_TTS_CHANGES.md`
  - This change summary.

## Runtime Artifacts Not Tracked

The repository intentionally excludes generated files such as:

- `logs/`
- `checkpoints/`
- `__pycache__/`
- `*.pyc`
- `*.egg-info/`
