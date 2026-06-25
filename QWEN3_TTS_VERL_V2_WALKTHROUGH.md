# Qwen3-TTS verl v2 魔改适配思路讲解

这份文档用于讲解我这个仓库 `BitJiao/qwen3tts_verl_v2` 是怎么把原本偏 LLM/VLM 后训练的 verl，改造成可以跑 Qwen3-TTS 12Hz Base 的 SFT、GRPO/PPO/GSPO 语音后训练、Ray 多卡 rollout、SpeechJudge 奖励和 SeedTTS 评测的。

仓库地址：

```text
https://github.com/BitJiao/qwen3tts_verl_v2
```

## 1. 背景：为什么不能直接用原版 verl

verl 默认训练链路面向的是文本自回归模型：batch 里主要是 `input_ids`、`attention_mask`、`labels`，loss 也是普通 LM cross entropy。Qwen3-TTS 不一样，它虽然也有 transformer talker，但训练目标不是“下一个文本 token”，而是生成语音 codec token。

Qwen3-TTS 的一次 forward 里至少有这些额外结构：

- 文本 token embedding；
- codec-0 token，也就是主 codec 轨；
- codec 1 到 codec 15 的 sub-talker 输入；
- 参考音频提出来的 speaker embedding；
- `text_embedding_mask`、`codec_embedding_mask`、`codec_mask` 这些位置掩码；
- codec-0 loss 和 sub-talker loss 两套损失。

所以这次适配的核心不是把 Qwen3-TTS 当成普通 `AutoModelForCausalLM` 塞进 verl，而是单独加一条 Qwen3-TTS recipe：

```text
recipe/qwen3_tts/
```

同时只在 verl 内部做必要的小切口：

```text
verl/workers/engine/fsdp/transformer_impl.py
verl/workers/engine/__init__.py
verl/workers/engine/fsdp/__init__.py
verl/utils/dtensor_compat.py
```

整体原则是：Qwen3-TTS 特有的音频数据、rollout、reward 和 loss 放到 recipe；verl 只负责 FSDP、checkpoint、分布式训练这些底座能力。

## 2. 总体架构

当前仓库支持的链路是：

```text
SFT JSONL / RL JSONL
        |
        v
recipe/qwen3_tts dataset / runner
        |
        +--> SFT: verl SFTTrainer + TrainingWorker + FSDPEngineWithQwen3TTS
        |
        +--> RL: Qwen3TTSModel rollout -> wav -> reward -> advantage -> Qwen3-TTS NLL -> policy loss
        |
        +--> eval: SeedTTS-style generation manifest + official eval-compatible outputs
```

仓库里和 Qwen3-TTS 相关的文件主要分几类：

```text
recipe/qwen3_tts/dataset.py               # SFT 数据拼 batch
recipe/qwen3_tts/sft_trainer.py           # 接 verl SFTTrainer
recipe/qwen3_tts/grpo_trainer.py          # 单进程 RL 主逻辑
recipe/qwen3_tts/ray_grpo_trainer.py      # Ray 多 GPU rollout / train worker
recipe/qwen3_tts/combined_reward.py       # WER + sim + SpeechJudge + duration
recipe/qwen3_tts/wer_sim_reward.py        # WER / MFCC speaker-sim 奖励
recipe/qwen3_tts/speechjudge_reward.py    # SpeechJudge-GRM 奖励客户端/本地调用
recipe/qwen3_tts/speechjudge_server.py    # 单独环境里的 SpeechJudge HTTP server
recipe/qwen3_tts/export_custom_voice.py   # FSDP checkpoint 导出 custom_voice
recipe/qwen3_tts/seedtts_eval.py          # SeedTTS 风格评测生成
scripts/setup_qwen3tts_env.sh             # 一键环境构建、源码补丁
scripts/patch_qwen3_tts_source.py         # 补丁 Qwen3-TTS 官方源码 loss
scripts/run_qwen3tts_smoke.sh             # smoke 检查
data/smoke/                               # 仓库内置极小测试数据
models/sources/Qwen3-TTS-source.tar.gz    # 仓库内置 Qwen3-TTS 源码包
```

## 3. 第一块修改：让 Transformers 能识别 Qwen3-TTS

文件：

```text
recipe/qwen3_tts/register.py
```

Qwen3-TTS 是自定义模型结构，不能完全依赖 Transformers 默认 auto class。因此我加了一个注册模块：

```python
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
```

然后在启动脚本里通过：

```bash
model.external_lib=recipe.qwen3_tts.register
```

让 verl 构造 `HFModelConfig` 之前先导入这个注册模块。这样 `AutoModel.from_pretrained(...)` 才能加载 Qwen3-TTS，而不是走普通 causal LM 路径。

## 4. 第二块修改：SFT 数据不再是普通 labels，而是 Qwen3-TTS codec batch

文件：

```text
recipe/qwen3_tts/dataset.py
```

新增核心类：

```python
Qwen3TTSSFTDataset
```

SFT 输入行格式是：

```json
{"audio":"./utt.wav","text":"...","ref_audio":"./ref.wav","audio_codes":[[...]]}
```

这里的 `audio_codes` 来自 Qwen3-TTS 官方 `finetuning/prepare_data.py`，不是训练时现算。dataset 做的事情包括：

- 支持 `.jsonl`、`.json`、`.parquet` 三种输入；
- 用 `AutoProcessor` 构造 assistant 文本 prompt；
- 读取 24 kHz `ref_audio`，调用 Qwen3-TTS 的 `mel_spectrogram` 得到 `ref_mels`；
- 把文本 token、codec-0、16 组 codec、speaker 位置拼到同一个 `input_ids` / `codec_ids` 结构；
- 生成 Qwen3-TTS forward 需要的 mask：
  - `text_embedding_mask`
  - `codec_embedding_mask`
  - `codec_mask`
  - `attention_mask`
  - `loss_mask`
- 生成 codec-0 监督标签 `codec_0_labels`，忽略位置填 `-100`。

这一块的关键是：我没有把 Qwen3-TTS 数据强行改成 `input_ids/labels`，而是保留 Qwen3-TTS 官方 finetune batch 形态，再让后面的 engine 专门处理这套字段。

## 5. 第三块修改：SFT trainer 接入 verl FSDP，但 loss 交给 Qwen3-TTS engine

文件：

```text
recipe/qwen3_tts/sft_trainer.py
recipe/qwen3_tts/run_qwen3_tts_sft_fsdp.sh
```

`Qwen3TTSSFTTrainer` 继承了 verl 的 `SFTTrainer`，但改了三件事：

1. `_build_dataset()` 使用 `Qwen3TTSSFTDataset`。
2. `_build_engine()` 创建 `TrainingWorkerConfig(model_type="qwen3_tts")`。
3. `loss_fn` 设置成 `_noop_loss`，因为真正的 loss 在 `FSDPEngineWithQwen3TTS.forward_step()` 里算。

启动脚本里比较重要的配置是：

```bash
model.external_lib=recipe.qwen3_tts.register
engine.strategy=fsdp
engine.model_dtype=bf16
engine.use_orig_params=true
data.pad_mode=no_padding
```

其中 `engine.use_orig_params=true` 是 v2 里专门加的坑点修复。Qwen3-TTS 里存在 frozen 参数和 trainable 参数混在同一个 FSDP flatten group 的情况，如果不打开 `use_orig_params`，FSDP 可能报 mixed `requires_grad` flattening error。

## 6. 第四块修改：在 verl 里注册 Qwen3-TTS FSDP Engine

文件：

```text
verl/workers/engine/fsdp/transformer_impl.py
verl/workers/engine/__init__.py
verl/workers/engine/fsdp/__init__.py
```

新增注册：

```python
@EngineRegistry.register(model_type="qwen3_tts", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithQwen3TTS(FSDPEngine):
    ...
```

这个 engine 做了几件关键事：

- 用 `AutoModel.from_pretrained(...)` 加载 Qwen3-TTS；
- 检查 checkpoint 必须带 `speaker_encoder`，也就是需要 Base 版本，不是已经导出的 `custom_voice` 版本；
- monkey patch 模型 `forward` 为 `_qwen3_tts_finetune_forward`；
- 在 `forward_step()` 里从 micro batch 读取：
  - `input_ids`
  - `codec_ids`
  - `ref_mels`
  - `text_embedding_mask`
  - `codec_embedding_mask`
  - `attention_mask`
  - `codec_0_labels`
  - `codec_mask`
- 输出 metrics：
  - `raw_loss`
  - `codec_0_loss`
  - `sub_talker_loss`

### 6.1 Qwen3-TTS loss 的核心修正

`_qwen3_tts_finetune_forward(...)` 里最重要的逻辑是：

```python
speaker_embedding = self.speaker_encoder(ref_mels.to(self.device).to(self.dtype)).detach()

input_text_embedding = self.talker.text_projection(
    self.talker.model.text_embedding(input_text_ids)
)
input_codec_embedding = self.talker.model.codec_embedding(input_codec_ids)
input_codec_embedding[:, 6, :] = speaker_embedding
```

也就是文本 embedding 不能直接和 codec embedding 相加，必须先过：

```text
talker.text_projection(...)
```

这个点在原先调试里很关键。如果不投影，维度/语义和 Qwen3-TTS 官方训练逻辑不一致。

codec-0 loss 也没有继续用 Transformers 默认 `labels=` 路径，而是显式写：

```python
codec_loss_mask = codec_0_labels[:, 1:].ne(-100)
codec_0_loss = F.cross_entropy(
    outputs.logits[codec_loss_mask],
    codec_0_labels[:, 1:][codec_loss_mask],
)
```

这样可以避免 causal LM 内部再次 shift 导致监督位置错位。

sub-talker loss 同样显式 CE：

```python
sub_talker_logits, _ = self.talker.forward_sub_talker_finetune(...)
sub_talker_loss = F.cross_entropy(
    sub_talker_logits.reshape(-1, sub_talker_logits.size(-1)),
    sub_talker_labels.reshape(-1),
    ignore_index=-100,
)
```

最后总 loss：

```text
loss = codec_0_loss + 0.3 * sub_talker_loss
```

v2 还把原来硬编码的 `range(1, 16)` 改成：

```python
range(1, self.talker.config.num_code_groups)
```

这样不会把 codec group 数写死。

## 7. 第五块修改：补丁 Qwen3-TTS 官方源码，保证 SFT/RL loss 口径一致

文件：

```text
scripts/patch_qwen3_tts_source.py
scripts/setup_qwen3tts_env.sh
scripts/check_qwen3_tts_env.py
```

我没有假设用户本地的 Qwen3-TTS 源码永远是已经改好的，所以环境脚本会在安装时自动 patch：

```bash
python scripts/patch_qwen3_tts_source.py --repo "${QWEN3_TTS_REPO}"
```

这个 patch 做了几件事：

- 给 Qwen3-TTS 官方 `finetuning/sft_12hz.py` 加 `torch.nn.functional as F`；
- 文本 embedding 加 `talker.text_projection(...)`；
- `range(1, 16)` 改成 `range(1, model.talker.config.num_code_groups)`；
- codec-0 loss 改成显式 `F.cross_entropy`；
- Qwen3-TTS code predictor 的 `forward_finetune` 改成 `ignore_index=-100` 的显式 CE。

`scripts/check_qwen3_tts_env.py` 会验证这些补丁是否真的生效，包括：

- `ffmpeg` / `ffprobe` 是否在 PATH；
- `verl` 是否从当前仓库导入；
- `qwen_tts` 是否可导入；
- Qwen3-TTS modeling 文件是否包含关键类；
- Qwen3-TTS 官方 SFT 和本 repo RL loss 是否都用了 `text_projection` 和显式 CE。

这一层的意义是把“能跑”变成“可复现能跑”，避免换机器以后因为源码版本或依赖解析不一致复现失败。

## 8. 第六块修改：RL 不是文本 rollout，而是语音生成 rollout

文件：

```text
recipe/qwen3_tts/grpo_trainer.py
recipe/qwen3_tts/run_qwen3_tts_grpo.sh
recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
recipe/qwen3_tts/run_qwen3_tts_ppo_all_g8_eager.sh
recipe/qwen3_tts/run_qwen3_tts_gspo_all_g8_eager.sh
```

RL 输入行格式：

```json
{"text":"hello world","ref_audio":"./ref.wav","ref_text":"hello","language":"en","target_duration":0.5}
```

整个 GRPO/PPO/GSPO runner 的流程是：

1. 加载 Qwen3-TTS Base 模型；
2. 对每条 prompt 生成 `group_size` 条 voice-clone 音频；
3. 得到每条音频对应的 `audio_codes`；
4. 用 speech tokenizer 把 codes decode 成 wav；
5. 对 wav 调 reward；
6. 同组 reward 做归一化 advantage；
7. 用生成出来的 `audio_codes` 反向构造训练 batch；
8. 调 `qwen3_tts_nll(...)` 得到当前模型 NLL；
9. 根据算法类型算 policy loss；
10. 反传、clip grad、更新参数、保存 checkpoint。

### 8.1 rollout 生成

核心函数：

```python
generate_voice_clone_rollouts(...)
generate_voice_clone_rollouts_parallel(...)
generate_prompt_batch_rollouts_parallel(...)
```

它调用的是 Qwen3-TTS 推理接口：

```python
tts.model.generate(...)
tts.model.speech_tokenizer.decode(...)
```

注意 voice clone 的 prompt 里可能包含参考音频 codec。decode 后我会把参考音频那一段裁掉，只保留生成目标语音：

```python
cut = int(ref_len / max(total_len, 1) * wav.shape[0])
wavs.append(wav[cut:])
```

### 8.2 RL loss

`qwen3_tts_nll(...)` 和 FSDP engine 里的 SFT loss 保持同一套逻辑：speaker encoder、text projection、codec-0 CE、sub-talker CE。

GRPO 当前实现是 lightweight GRPO-style：

```text
policy_loss = advantage * nll
```

PPO / GSPO 用 old NLL 和 new NLL 构造 ratio：

```text
ratio = exp(old_nll - new_nll)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

脚本里暴露了这些参数：

```text
ALGORITHM=grpo|ppo|gspo
GROUP_SIZE
PROMPT_BATCH_SIZE
POLICY_EPOCHS
CLIP_RATIO
CLIP_RATIO_LOW
CLIP_RATIO_HIGH
MAX_NEW_TOKENS
TEMPERATURE
TOP_K
TOP_P
```

这条 RL runner 不是原版 verl `main_ppo` 的全套 actor/critic/ref/rollout 架构，而是一个 Qwen3-TTS 专用轻量 runner。这样做的原因是 Qwen3-TTS 的 rollout 需要生成音频、decode wav、打音频 reward，直接塞进通用文本 rollout 成本更高，调试也更难。

## 9. 第七块修改：Ray 多 GPU rollout / training worker

文件：

```text
recipe/qwen3_tts/ray_grpo_trainer.py
```

单进程多线程 rollout 能跑，但全量语音生成很慢。所以 v2 加了 Ray 版本：

```bash
USE_RAY=1
ROLLOUT_DEVICES=auto
bash recipe/qwen3_tts/run_qwen3_tts_grpo_all_g8_eager.sh
```

Ray 版本里每个 worker 做三件事：

- `setup()`：加载 Qwen3-TTS、初始化 torch distributed、加载 reward；
- `rollout()`：拿到分配的 prompt，生成若干条 wav/codes；
- `train_step()`：根据 driver 下发的 advantages 做本 worker 的反传。

driver 负责：

- 根据 `prompt_batch_size` 和 `group_size` 把 rollout 任务切给多个 worker；
- 收集各 worker 的 codes/wavs/rewards；
- 按 sample 聚合 group reward；
- 算 group advantage；
- 把每个 worker 自己生成的 codes 和对应 advantage 发回去训练；
- 汇总日志和保存 checkpoint。

为了避免不同 worker 梯度不同步，`Qwen3TTSRayWorker._sync_gradients()` 里手动 all-reduce 每个参数的 gradient。这个实现不复杂，但够支撑 Qwen3-TTS 这种“rollout 很重、训练步相对轻”的实验场景。

日志里会输出这些关键信息：

```text
reward_mean
reward_std
codec_0_loss
sub_talker_loss
advantage_abs_mean
ratio_mean
zero_advantage_groups
generated_rollouts
rollouts_per_second
rollout_seconds
train_seconds
eta
finish_at
```

## 10. 第八块修改：奖励函数从文本正确性扩展到语音质量

文件：

```text
recipe/qwen3_tts/wer_sim_reward.py
recipe/qwen3_tts/combined_reward.py
recipe/qwen3_tts/speechjudge_reward.py
recipe/qwen3_tts/speechjudge_server.py
recipe/qwen3_tts/SPEECHJUDGE_SETUP.md
```

### 10.1 WER + speaker similarity

`wer_sim_reward.py` 提供一个轻量 reward：

```text
reward = WER_WEIGHT * wer_score + SIM_WEIGHT * sim_score
```

WER 部分：

- 可选 `transformers` Whisper 或 `faster_whisper`；
- `REWARD_ASR_BACKEND=none` 时直接关闭 ASR，方便 smoke；
- 中英文做了不同 tokenize：中文按字，英文按词；
- score 是 `1 - edit_distance / reference_length`。

speaker similarity 部分：

- 读取 `ref_audio`；
- 对生成音频和参考音频提 MFCC 均值/方差；
- 做 cosine similarity；
- clamp 到 `[0, 1]`。

MFCC speaker similarity 不是最终严格的 speaker verification，只是一个依赖轻、容易复现的 proxy。后续可以替换成更强的 speaker encoder。

### 10.2 SpeechJudge-GRM

`combined_reward.py` 把 reward 做成组合：

```text
total = wer * w1 + sim * w2 + judge * w3 + duration * w4
```

默认权重在脚本中是：

```bash
REWARD_WER_WEIGHT=0.3
REWARD_SIM_WEIGHT=0.2
REWARD_JUDGE_WEIGHT=0.5
REWARD_DURATION_WEIGHT=0.0
```

SpeechJudge 依赖和 Qwen3-TTS/verl 当前环境冲突，所以我没有把它装进同一个 venv，而是单独做了 HTTP server：

```bash
python -m recipe.qwen3_tts.speechjudge_server \
  --host 127.0.0.1 \
  --port 8765 \
  --model_path /path/to/SpeechJudge-GRM \
  --speechjudge_repo /path/to/SpeechJudge
```

训练进程只通过：

```bash
SPEECHJUDGE_SERVER_URL=http://127.0.0.1:8765
```

去请求 `/score`。这样 Qwen3-TTS 的 torch/transformers 版本和 SpeechJudge 的 qwen-omni 依赖不会互相污染。

## 11. 第九块修改：checkpoint 导出到 Qwen3-TTS custom_voice

文件：

```text
recipe/qwen3_tts/export_custom_voice.py
```

SFT/RL 训练时用的是 Qwen3-TTS Base checkpoint，因为要有 `speaker_encoder`。但是推理部署时，Qwen3-TTS 常用的是 `custom_voice` layout。因此我加了导出脚本：

```bash
python -m recipe.qwen3_tts.export_custom_voice \
  --checkpoint_dir /path/to/checkpoints/global_step_100 \
  --base_model_dir models/Qwen3-TTS-12Hz-1.7B-Base \
  --output_dir /path/to/qwen3_tts_custom_voice \
  --speaker_name speaker_test \
  --train_jsonl data/smoke/train_with_codes.jsonl \
  --overwrite
```

它做的事：

- 合并 verl FSDP shard；
- 复制 base model 配置和非模型文件；
- 从参考音频提 speaker embedding；
- 删除训练时的 `speaker_encoder` 权重；
- 把 speaker embedding 写入 `talker.model.codec_embedding.weight[speaker_id]`；
- 修改 `config.json`：
  - `tts_model_type = custom_voice`
  - 写入 `spk_id`
  - 写入 `spk_is_dialect`

这样导出的目录就能按 Qwen3-TTS custom voice 形式使用。

## 12. 第十块修改：SeedTTS 风格评测

文件：

```text
recipe/qwen3_tts/seedtts_eval.py
recipe/qwen3_tts/run_qwen3_tts_seedtts_eval_g8.sh
```

为了评测生成质量，我加了 SeedTTS-style generation 脚本。它支持两种输入：

pipe 格式：

```text
filename|prompt_text|prompt_wav|target_text|ground_truth_wav
```

或者 JSONL：

```json
{"sample_id":"xxx","ref_text":"...","ref_audio":"...","text":"...","gt_audio":"..."}
```

运行示例：

```bash
QWEN3_TTS_REPO=third_party/Qwen3-TTS \
MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
INPUT_JSONL=data/smoke/seedtts_meta.lst \
DEVICES=auto \
OVERWRITE=1 \
OUTPUT_DIR=results/qwen3_tts_seedtts \
bash recipe/qwen3_tts/run_qwen3_tts_seedtts_eval_g8.sh
```

输出包括：

```text
wav_res/                 # 生成音频
manifest.jsonl           # 每条样本的生成状态、时长、RTF
summary.json             # 吞吐、延迟、错误数
meta.lst                 # SeedTTS official eval 兼容格式
wav_res_ref_text
wav_res_ref_text.txt
```

`seedtts_eval.py` 还支持 `--score_only`，可以对已有 manifest 计算 WER/MFCC sim summary。

## 13. 第十一块修改：环境和可复现性

文件：

```text
scripts/setup_qwen3tts_env.sh
scripts/qwen3tts_common.sh
requirements-qwen3tts-verl.txt
models/sources/Qwen3-TTS-source.tar.gz
data/smoke/
scripts/create_qwen3tts_smoke_data.py
scripts/run_qwen3tts_smoke.sh
```

v2 的一个重点是 clean clone 后尽量能复现。仓库里放了：

- `models/sources/Qwen3-TTS-source.tar.gz`：如果机器不能 clone Qwen3-TTS，就从这个 archive 解压；
- `data/smoke/`：极小的 ref/target wav、SFT jsonl、GRPO jsonl、SeedTTS meta；
- `scripts/run_qwen3tts_smoke.sh`：先跑环境检查和数据生成，必要时再跑 1-GPU SFT/GRPO smoke；
- `requirements-qwen3tts-verl.txt`：固定 runtime 依赖，避免 pip 拉到不兼容的 torch / numpy。

安装脚本支持不同 torch profile：

```bash
TORCH_PROFILE=cu121-verified    # 默认验证栈: torch 2.3.1 + cu121
TORCH_PROFILE=cu130             # 高版本服务器: torch 2.10.0 + cu130
TORCH_PROFILE=cu132-nightly     # PyTorch nightly cu132
TORCH_PROFILE=skip              # venv 里已经有 torch/torchaudio
```

默认推荐 clean clone：

```bash
git clone https://github.com/BitJiao/qwen3tts_verl_v2.git
cd qwen3tts_verl_v2

VENV_DIR=.venv \
PYTHON_BIN=3.11 \
TORCH_PROFILE=cu121-verified \
bash scripts/setup_qwen3tts_env.sh

source .venv/bin/activate
python scripts/check_qwen3_tts_env.py
bash scripts/run_qwen3tts_smoke.sh
```

如果要跑训练 smoke：

```bash
RUN_TRAINING=1 \
MODEL_PATH=models/Qwen3-TTS-12Hz-1.7B-Base \
SMOKE_DEVICE=cuda:0 \
bash scripts/run_qwen3tts_smoke.sh
```

## 14. 和原版 verl 相比，我具体改了什么

可以按“新增 recipe、改 verl engine、环境补丁、评测导出”四类理解。

### 14.1 新增 Qwen3-TTS recipe

```text
recipe/qwen3_tts/
```

新增了完整的 Qwen3-TTS 专用训练闭环：

- SFT dataset；
- SFT trainer；
- GRPO/PPO/GSPO runner；
- Ray 多 GPU runner；
- reward 组件；
- SpeechJudge server；
- checkpoint export；
- SeedTTS eval；
- 启动脚本和说明文档。

### 14.2 修改 verl FSDP engine

```text
verl/workers/engine/fsdp/transformer_impl.py
```

新增：

- `_qwen3_tts_finetune_forward(...)`
- `FSDPEngineWithQwen3TTS`
- `model_type="qwen3_tts"` 注册
- Qwen3-TTS 专用 `forward_step()`

并把 `FSDPEngineWithQwen3TTS` 从：

```text
verl/workers/engine/__init__.py
verl/workers/engine/fsdp/__init__.py
```

导出。

### 14.3 修复 PyTorch DTensor 兼容

```text
verl/utils/dtensor_compat.py
verl/utils/fsdp_utils.py
verl/utils/checkpoint/fsdp_checkpoint_manager.py
verl/models/transformers/qwen3_5.py
verl/third_party/torch/distributed/*
```

不同 PyTorch 版本里 DTensor import 路径有变化，有的在：

```python
torch.distributed.tensor
```

有的在：

```python
torch.distributed._tensor
```

所以我加了 `verl/utils/dtensor_compat.py`，统一导入 `DTensor`、`Shard`、`Replicate`、`DeviceMesh` 等对象。这样 cu121-verified 和更高版本 torch profile 都更稳。

另外 `fsdp_checkpoint_manager.py` 对 custom config 的 `dtype` KeyError 做了 fallback：如果 `save_pretrained(... use_diff=True)` 因 `dtype` 失败，就用 `to_json_file(..., use_diff=False)` 保存完整 config。这是为了兼容 Qwen3-TTS 自定义 config。

### 14.4 修改启动和依赖

```text
requirements-qwen3tts-verl.txt
scripts/setup_qwen3tts_env.sh
scripts/check_qwen3_tts_env.py
scripts/run_qwen3tts_smoke.sh
```

主要目标是：

- 不让 pip 自动升级到未验证的 torch/CUDA；
- 固定 `numpy<2.0.0`；
- 自动 patch Qwen3-TTS 源码；
- 支持相对路径；
- 支持 bundled Qwen3-TTS source archive；
- 支持 smoke data；
- 把 SpeechJudge 和主环境隔离。

## 15. 讲解时可以按这个顺序讲

如果我要做一次口头讲解，我会按这个顺序展开：

1. 先讲为什么 Qwen3-TTS 不能直接套 verl 文本 SFT/RL。
2. 讲 Qwen3-TTS 的 batch 结构：文本、codec、speaker embedding、mask。
3. 讲 SFT 怎么接入 verl：dataset 自定义，engine 自定义，trainer 尽量复用。
4. 重点讲 loss 修正：`text_projection`、显式 codec-0 CE、显式 sub-talker CE、`num_code_groups`。
5. 讲 RL rollout 为什么要专门写 runner：生成 audio codes、decode wav、音频 reward。
6. 讲 reward：WER、speaker sim、SpeechJudge 分离环境。
7. 讲 Ray 多卡：每个 worker 生成和训练，driver 算 advantage。
8. 讲工程化：setup、patch、smoke data、SeedTTS eval、custom_voice export。
9. 最后讲局限：当前 RL 是 lightweight runner，不是完整 verl PPO 架构；MFCC sim 是 proxy；SpeechJudge 依赖需要单独环境。

## 16. 当前实现的局限和后续方向

当前 v2 已经能覆盖 Qwen3-TTS SFT、RL smoke、多卡 rollout、reward、导出和评测，但还有一些可以继续升级的地方：

- RL runner 目前是 Qwen3-TTS 专用 lightweight 实现，还没有完全接入 verl 标准 actor/critic/ref/rollout worker 架构；
- speaker similarity 现在用 MFCC cosine，只是低成本 proxy，可以换成专业 speaker verification 模型；
- SpeechJudge 通过 HTTP server 隔离依赖，稳定但有额外服务管理成本；
- PPO/GSPO 当前基于 sequence-level NLL ratio，不是 token-level logprob ratio；
- Ray worker 的 gradient sync 是手写 all-reduce，后续可以进一步封装成更标准的 distributed optimizer/engine。

但从工程目标看，这版已经完成了最关键的适配：Qwen3-TTS 能在 verl fork 上完成 SFT、语音 RL、奖励闭环、checkpoint 导出和 SeedTTS 风格评测，并且 clean clone 后有明确的环境构建和 smoke 验收路径。
