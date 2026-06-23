# Qwen3-TTS verl Adaptation Notes

This file records the actual adaptation points. The runnable instructions are
in `README.md`.

## Code Paths

- `recipe/qwen3_tts/dataset.py`: builds the two-channel Qwen3-TTS SFT batch.
- `verl/workers/engine/fsdp/transformer_impl.py`: registers
  `FSDPEngineWithQwen3TTS` and computes the Qwen3-TTS SFT loss.
- `recipe/qwen3_tts/grpo_trainer.py`: lightweight GRPO/PPO/GSPO runner for
  Qwen3-TTS voice-clone RL.
- `recipe/qwen3_tts/ray_grpo_trainer.py`: Ray worker wrapper for multi-GPU
  rollout and loss.
- `recipe/qwen3_tts/combined_reward.py`: WER + sim + SpeechJudge reward.
- `recipe/qwen3_tts/export_custom_voice.py`: exports FSDP shards to Qwen3-TTS
  `custom_voice`.

## Loss Fixes

The original local draft was wrong in two places:

- text embeddings were added to codec embeddings without
  `talker.text_projection`, which is invalid when text and talker hidden sizes
  differ and still wrong semantically when they match;
- `codec_0_labels[:, 1:]` was passed into the Transformers causal LM loss,
  whose implementation shifts labels again.

The current code builds embeddings as:

```python
input_text_embedding = model.talker.text_projection(
    model.talker.model.text_embedding(input_text_ids)
)
input_embeddings = input_text_embedding + input_codec_embedding
```

It then computes codec-0 loss explicitly:

```python
codec_loss_mask = codec_0_labels[:, 1:].ne(-100)
codec_0_loss = F.cross_entropy(
    outputs.logits[codec_loss_mask],
    codec_0_labels[:, 1:][codec_loss_mask],
)
```

The sub-talker loss remains:

```python
_, sub_talker_loss = model.talker.forward_sub_talker_finetune(
    talker_codec_ids,
    talker_hidden_states,
)
loss = codec_0_loss + sub_talker_loss_coef * sub_talker_loss
```

## Environment Check

After installing editable packages, run:

```bash
cd /opt/data/private/jsj/qwen3tts_verl_v2
python scripts/check_qwen3_tts_env.py
```

This verifies `ffmpeg`, `ffprobe`, `qwen_tts`, this repo's `verl`, and the
patched Qwen3-TTS loss path.
