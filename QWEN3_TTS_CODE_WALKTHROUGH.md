# Qwen3-TTS verl Code Walkthrough

这份文档解释本仓库为了适配 Qwen3-TTS 后训练，对原始 verl 做了哪些代码修改，以及每个新增文件在训练链路里的作用。

## 总体结构

Qwen3-TTS 相关代码集中放在：

```text
recipe/qwen3_tts/
```

原始 verl 代码只改了 FSDP engine 注册相关的少量文件：

```text
verl/workers/engine/__init__.py
verl/workers/engine/fsdp/__init__.py
verl/workers/engine/fsdp/transformer_impl.py
```

这样做的目的：

- SFT 走 verl 的 FSDP worker/trainer 体系。
- RL/GRPO/PPO/GSPO 先用一个轻量 Qwen3-TTS 专用 runner 跑通 voice-clone 后训练。
- Qwen3-TTS 特有的 speech token、speaker encoder、sub-talker loss 不强行塞进 verl 通用 LLM loss。

## SFT 适配

### `recipe/qwen3_tts/dataset.py`

核心类：

```python
Qwen3TTSSFTDataset
```

作用：

- 读取 Qwen3-TTS 官方格式 JSONL。
- 每条样本需要包含：
  - `text`
  - `audio`
  - `ref_audio`
  - `audio_codes`
- 构造 Qwen3-TTS forward 需要的 batch 字段：
  - `input_ids`
  - `codec_ids`
  - `ref_mels`
  - `text_embedding_mask`
  - `codec_embedding_mask`
  - `attention_mask`
  - `codec_0_labels`
  - `codec_mask`
  - `loss_mask`

这些字段不是普通 LLM SFT 的 `input_ids/labels`，所以需要单独 dataset。

### `recipe/qwen3_tts/sft_trainer.py`

核心类：

```python
Qwen3TTSSFTTrainer
```

作用：

- 继承 verl 的 `SFTTrainer`。
- 使用 Qwen3-TTS dataset。
- 调用 verl `TrainingWorker` 进行 FSDP 训练。
- 不使用 verl 默认语言模型 cross entropy loss，而是交给自定义 engine forward。

### `recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh`

SFT 启动脚本。

常用命令：

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main

MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=/path/to/train_with_codes.jsonl \
N_GPUS_PER_NODE=1 \
bash recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

## verl FSDP Engine 修改

### `verl/workers/engine/fsdp/transformer_impl.py`

新增函数：

```python
_qwen3_tts_finetune_forward(...)
```

它复刻 Qwen3-TTS 官方 finetuning loss 逻辑：

1. 用 `speaker_encoder(ref_mels)` 得到 speaker embedding。
2. 把文本 embedding、codec embedding、speaker embedding 组合成输入 embedding。
3. 调用 `talker(...)` 预测 codec 0。
4. 调用 `forward_sub_talker_finetune(...)` 计算 sub-talker loss。
5. 总 loss：

```text
loss = codec_0_loss + sub_talker_loss_coef * sub_talker_loss
```

新增类：

```python
FSDPEngineWithQwen3TTS
```

注册方式：

```python
@EngineRegistry.register(model_type="qwen3_tts", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
```

作用：

- 让 verl 识别 `model_type=qwen3_tts`。
- 用 `AutoModel.from_pretrained(...)` 加载 Qwen3-TTS。
- 检查 checkpoint 必须带 `speaker_encoder`，也就是 Base checkpoint。
- monkey patch 模型 forward 为 `_qwen3_tts_finetune_forward`。
- 自定义 `forward_step`，读取 Qwen3-TTS batch 字段并返回 metrics。

额外修改：

```python
disable_adapter()
```

如果模型没有 LoRA/adapter，就返回 `nullcontext()`，避免无 adapter 模型报错。

### `verl/workers/engine/__init__.py`

导出：

```python
FSDPEngineWithQwen3TTS
```

### `verl/workers/engine/fsdp/__init__.py`

导出：

```python
FSDPEngineWithQwen3TTS
```

## RL / GRPO / PPO / GSPO 适配

### `recipe/qwen3_tts/grpo_trainer.py`

这是 Qwen3-TTS voice-clone RL 的主入口。

支持：

```text
--algorithm grpo
--algorithm ppo
--algorithm gspo
```

主要流程：

1. 加载 Qwen3-TTS Base 模型。
2. 在多个 GPU 上加载 rollout 模型。
3. 对每个 prompt 生成 `group_size` 条语音。
4. 解码 speech token 成 wav。
5. 调用 reward 函数打分。
6. 对同组 reward 做 advantage normalize。
7. 用生成出的 audio codes 构造 Qwen3-TTS 训练 batch。
8. 计算 Qwen3-TTS NLL。
9. 根据算法计算 policy loss。
10. 反传更新模型。
11. 定期保存 Base-style checkpoint。

#### rollout 多卡逻辑

关键函数：

```python
generate_prompt_batch_rollouts_parallel(...)
generate_voice_clone_rollouts_parallel(...)
```

当前 ready-to-run 脚本默认：

```text
ROLLOUT_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
PROMPT_BATCH_SIZE=4
GROUP_SIZE=8
```

因此每 step：

```text
4 prompts * 8 rollouts = 32 generated rollouts
```

四个 prompt 分到四张卡并行生成。

#### GRPO loss

代码：

```python
policy_loss_from_nll(...)
```

GRPO 当前使用：

```text
loss = advantage * nll
```

注意：这里是 lightweight GRPO-style runner，不是 verl Ray `main_ppo` 全栈 GRPO。

#### PPO / GSPO loss

PPO/GSPO 使用 old NLL 和 new NLL 构造 ratio：

```text
ratio = exp(old_nll - new_nll)
```

再做 clipped surrogate：

```text
min(ratio * advantage, clipped_ratio * advantage)
```

PPO 默认：

```text
CLIP_RATIO=0.2
```

GSPO 默认更窄：

```text
CLIP_RATIO_LOW=0.0003
CLIP_RATIO_HIGH=0.0004
```

#### 进度和 ETA

每 step 日志新增：

```text
total_steps
remaining_steps
progress_percent
elapsed
eta
eta_seconds
finish_at
avg_step_seconds
```

日志示例：

```json
{
  "step": 1,
  "total_steps": 2042,
  "progress_percent": 0.0489,
  "eta": "85h45m07s",
  "finish_at": "2026-06-17 22:54:46"
}
```

## Reward 适配

### `recipe/qwen3_tts/wer_sim_reward.py`

默认 reward：

```text
reward = 0.6 * WER_score + 0.4 * SIM_score
```

可调：

```bash
REWARD_WER_WEIGHT=0.6
REWARD_SIM_WEIGHT=0.4
```

#### WER_score

流程：

1. 用本地 Whisper ASR 转写生成音频。
2. 将转写文本与目标 `sample["text"]` 做归一化。
3. 中文按字算 edit distance，英文按词算 edit distance。
4. 得到：

```text
WER_score = 1 - edit_distance / reference_length
```

#### SIM_score

当前是轻量 proxy：

1. 读取 `sample["ref_audio"]`。
2. 提取生成音频和参考音频的 MFCC 均值/方差。
3. 计算 cosine similarity。

这不是最终 speaker verification 模型，只是简单可运行版本。后续可替换成 ECAPA/WavLM/CAM++ speaker encoder。

#### 本地 ASR

服务器不能直接访问 HuggingFace，所以 Whisper 已下载到：

```text
/opt/data/private/jsj/models/openai-whisper-small
```

ready-to-run 脚本默认：

```bash
REWARD_ASR_BACKEND=transformers
ASR_MODEL_PATH=/opt/data/private/jsj/models/openai-whisper-small
ASR_DEVICE_INDEX=4
ASR_BATCH_SIZE=8
```

也就是：

- 前四张卡 `cuda:0,1,2,3` 做 rollout/训练。
- `cuda:4` 专门跑 Whisper reward。
- ASR 批量转写，减少 pipeline 串行调用开销。

如果想关闭 WER，只保留 sim：

```bash
REWARD_ASR_BACKEND=none bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager_full.sh
```

## 启动脚本

### 完整 GRPO

```bash
cd /opt/data/private/jsj/Qwen3-TTS-main/verl-main
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager_full.sh
```

默认：

```text
ALGORITHM=grpo
MAX_STEPS=-1
NUM_EPOCHS=1
ROLLOUT_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
PROMPT_BATCH_SIZE=4
GROUP_SIZE=8
REWARD_ASR_BACKEND=transformers
ASR_DEVICE_INDEX=4
```

### 10-step smoke GRPO

```bash
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

默认：

```text
MAX_STEPS=10
```

### PPO

```bash
bash recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh
```

### GSPO

```bash
bash recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh
```

## Export

### `recipe/qwen3_tts/export_custom_voice.py`

作用：

- 读取 verl FSDP checkpoint。
- 合并 FSDP shard。
- 拷贝 Base 模型配置。
- 写出 Qwen3-TTS `custom_voice` 推理格式。

示例：

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

## 没有纳入 Git 的运行产物

这些文件不应上传：

```text
logs/
checkpoints/
__pycache__/
*.pyc
*.egg-info/
```
