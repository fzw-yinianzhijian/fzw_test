# Qwen2.5-7B-Instruct 微调快速开始

## 📚 详细文档

请查看 [微调指导文档.md](./微调指导文档.md) 获取完整的微调指导，包括：
- 微调原理介绍
- LoRA 原理详解
- 详细步骤说明
- 时间估算
- 常见问题解答

## 🚀 快速开始（5步）

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes tensorboard
```

### 2. 设置 Hugging Face Token（如需要）

```bash
# 方式1：命令行登录
huggingface-cli login

# 方式2：环境变量
export HF_TOKEN="your_token_here"
```

**注意**：Qwen2.5-7B-Instruct 是公开模型，通常不需要 token。

### 3. 运行训练脚本

```bash
# 使用全量数据（81万条，预计 6-12 小时）
python train_qwen.py

# 或修改脚本中的 USE_SAMPLE_DATA = True，使用样本数据快速测试（约 10-30 分钟）
```

### 4. 监控训练

```bash
# 新开一个终端
tensorboard --logdir ./output/qwen2.5-7b-lora/logs
# 浏览器打开 http://localhost:6006
```

### 5. 测试模型

```bash
python test_model.py
```

## 📝 脚本说明

- **train_qwen.py**: 训练脚本，支持 LoRA 和 QLoRA
- **test_model.py**: 测试脚本，用于验证微调效果
- **微调指导文档.md**: 详细的微调指导文档

## ⚙️ 配置说明

在 `train_qwen.py` 中可以修改以下配置：

```python
# LoRA 配置
LORA_R = 16              # rank，越大表达能力越强，但参数越多
LORA_ALPHA = 32          # 缩放因子，通常设为 2*r
LORA_DROPOUT = 0.1       # Dropout

# 训练配置
BATCH_SIZE = 4           # 批次大小，显存不足时减小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积，增加有效 batch size
LEARNING_RATE = 2e-4     # 学习率
NUM_EPOCHS = 3           # 训练轮数
MAX_LENGTH = 512         # 最大序列长度
USE_QLORA = False        # 显存不足时设为 True（4-bit 量化）
USE_SAMPLE_DATA = False  # 快速测试时设为 True
```

## 💡 显存优化建议

**显存不足时：**
1. 设置 `USE_QLORA = True`（4-bit 量化）
2. 减小 `BATCH_SIZE`（如改为 1 或 2）
3. 增加 `GRADIENT_ACCUMULATION_STEPS`（如改为 8 或 16）
4. 减小 `MAX_LENGTH`（如改为 256）
5. 减小 `LORA_R`（如改为 8）

## 📊 时间估算

- **环境准备**: 8-15 分钟
- **数据准备**: 8-22 分钟
- **模型下载**: 11-32 分钟
- **训练（样本数据 1万条）**: 10-30 分钟
- **训练（全量数据 81万条）**: 6-12 小时（LoRA）或 9-18 小时（QLoRA）

## 🔗 相关链接

- **模型**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **数据集**: [Seikaijyu/Beautiful-Chinese](https://huggingface.co/datasets/Seikaijyu/Beautiful-Chinese)
- **Hugging Face**: https://huggingface.co/

## ❓ 遇到问题？

请查看 [微调指导文档.md](./微调指导文档.md) 中的"常见问题"部分。

