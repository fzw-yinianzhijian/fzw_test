#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-7B-Instruct LoRA 微调脚本
数据集：Beautiful-Chinese
"""

import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
import torch

# 多GPU训练时，设置每个进程使用的GPU
if torch.cuda.is_available():
    # 获取当前进程的local_rank（torchrun会自动设置）
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"进程 LOCAL_RANK={local_rank}, 使用设备: {device}")
else:
    device = torch.device("cpu")

# ==================== 配置 ====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = "./data/train_1w_split.json"  # 本地数据集路径
HF_TOKEN = os.getenv("HF_TOKEN", None)  # 从环境变量读取，或设置为 None

# 多 GPU 配置
USE_MULTI_GPU = True  # 是否使用多 GPU 训练（设为 False 可关闭多 GPU 训练）
# USE_MULTI_GPU = False  # 是否使用多 GPU 训练（设为 False 可关闭多 GPU 训练）
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
if USE_MULTI_GPU and NUM_GPUS > 1:
    print(f"多 GPU 训练已启用，检测到 {NUM_GPUS} 块 GPU")
elif USE_MULTI_GPU and NUM_GPUS == 1:
    print(f"多 GPU 训练已启用，但只检测到 1 块 GPU，将使用单 GPU 训练")
    USE_MULTI_GPU = False
else:
    print(f"单 GPU 训练模式，使用 GPU 0")
    NUM_GPUS = 1

# 根据配置设置输出目录
OUTPUT_DIR = f"./output/qwen2.5-7b-lora"

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# Qwen2.5 的 target_modules
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练配置
BATCH_SIZE = 2 # 每卡批次大小（多GPU时减少以避免OOM）
GRADIENT_ACCUMULATION_STEPS = 8 # 梯度累积步数（增加以保持有效批次大小）
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512
USE_QLORA = True  # 不使用 QLoRA（如果仍然OOM，可以设为True）
USE_SAMPLE_DATA = False  # 是否只使用部分数据（用于快速测试）
SAMPLE_SIZE = 10000  # 如果 USE_SAMPLE_DATA=True，使用的数据量

# ==================== 加载模型和 Tokenizer ====================
print("=" * 60)
print("加载模型和 Tokenizer...")
print("=" * 60)

# 检查是否使用离线模式（优先使用本地缓存）
# 如果网络有问题，可以设置为 True 强制使用本地缓存
# USE_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("FORCE_LOCAL_FILES", "0") == "1"
USE_OFFLINE = True

if USE_OFFLINE:
    print("离线模式：仅使用本地缓存文件")




tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=HF_TOKEN,
    local_files_only=USE_OFFLINE  # 离线模式时只使用本地文件
)

# Qwen tokenizer 通常已有 pad_token，如果没有则设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 配置量化（QLoRA）
quantization_config = None
if USE_QLORA:
    print("使用 QLoRA (4-bit 量化)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

# 加载模型
print(f"加载模型: {MODEL_NAME}")
# 多GPU训练时，明确指定设备，避免所有进程都在GPU 0上加载
if USE_MULTI_GPU:
    # 多GPU时，不使用device_map，让模型加载到当前进程指定的GPU
    # 然后由DDP自动分发到各个GPU
    device_map = None
    # 明确指定设备，确保模型加载到正确的GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map=None,  # DDP模式下不使用device_map
        torch_dtype=torch.bfloat16 if not USE_QLORA else None,
        trust_remote_code=True,
        token=HF_TOKEN,
        local_files_only=USE_OFFLINE
    )
    # 手动将模型移动到当前进程的GPU
    model = model.to(device)
else:
    # 单GPU时，使用device_map="auto"自动分配
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not USE_QLORA else None,
        trust_remote_code=True,
        token=HF_TOKEN,
        local_files_only=USE_OFFLINE
    )

# 准备 QLoRA 训练
if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

# ==================== 配置 LoRA ====================
print("\n配置 LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 确保模型处于训练模式
model.train()

print("\n可训练参数统计:")
model.print_trainable_parameters()

# ==================== 加载和预处理数据 ====================
print("\n" + "=" * 60)
print("加载数据集...")
print("=" * 60)

# 从本地 JSONL 文件加载数据集
print(f"从本地加载数据集: {DATASET_PATH}")
dataset = Dataset.from_json(DATASET_PATH)

# 如果使用样本数据（快速测试）
if USE_SAMPLE_DATA:
    print(f"使用样本数据: {SAMPLE_SIZE} 条")
    dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))

print(f"数据集大小: {len(dataset)}")
print(f"数据集字段: {dataset.column_names}")

# 显示示例
print("\n数据示例:")
print(dataset[0])

# Tokenize 函数
def tokenize_function(examples):
    """使用 Qwen 的 chat template 进行 tokenize"""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        # return_tensors="pt"
    )
    
    # Labels 就是 input_ids（因果语言模型）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("\n预处理数据集（tokenize）...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # 批处理大小
    remove_columns=dataset.column_names
)

print(f"预处理完成，样本数: {len(tokenized_dataset)}")

# ==================== 配置训练参数 ====================
print("\n" + "=" * 60)
print("配置训练参数...")
print("=" * 60)

# 构建 TrainingArguments，根据是否使用多GPU添加不同配置
training_args_dict = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": NUM_EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "fp16": not USE_QLORA,  # 不使用 QLoRA 时用 fp16
    "bf16": USE_QLORA,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "warmup_steps": 100,
    "report_to": "tensorboard",
    "remove_unused_columns": False,
    "optim": "adamw_torch",  # 使用 AdamW 优化器
    "lr_scheduler_type": "cosine",  # 余弦学习率调度
}

# 多 GPU 训练配置（仅在启用多GPU时添加）
if USE_MULTI_GPU:
    training_args_dict.update({
        "ddp_find_unused_parameters": False,  # DDP 优化，加快训练速度
        "dataloader_num_workers": 4,  # 数据加载并行数
        "dataloader_pin_memory": True,  # 固定内存，加速数据传输
    })
else:
    # 单 GPU 时的配置
    training_args_dict.update({
        "dataloader_num_workers": 2,  # 单GPU时减少worker数量
        "dataloader_pin_memory": True,
    })

training_args = TrainingArguments(**training_args_dict)

print(f"输出目录: {OUTPUT_DIR}")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"多 GPU 训练: {'是' if USE_MULTI_GPU else '否'}")
print(f"GPU 数量: {NUM_GPUS}")
print(f"每卡 Batch size: {BATCH_SIZE}")
print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
print(f"总有效 batch size: {effective_batch_size}")
print(f"学习率: {LEARNING_RATE}")
print(f"最大序列长度: {MAX_LENGTH}")
print(f"使用 QLoRA: {USE_QLORA}")

# ==================== 创建 Trainer ====================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言模型，不是掩码语言模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ==================== 开始训练 ====================
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)
total_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
estimated_steps = len(tokenized_dataset) // total_batch_size * NUM_EPOCHS
print(f"预计训练步数: {estimated_steps}")
print(f"使用 TensorBoard 查看训练日志: tensorboard --logdir {OUTPUT_DIR}/runs")
if USE_MULTI_GPU:
    print(f"\n多 GPU 训练模式已启用，使用 {NUM_GPUS} 块 GPU")
    print("运行命令: torchrun --nproc_per_node={} train_qwen.py".format(NUM_GPUS))
else:
    print(f"\n单 GPU 训练模式")
    print("运行命令: python train_qwen.py")
print("=" * 60 + "\n")

trainer.train()

# ==================== 保存模型 ====================
print("\n" + "=" * 60)
print(f"保存模型到 {OUTPUT_DIR}...")
print("=" * 60)

trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n训练完成！")
print(f"模型保存在: {OUTPUT_DIR}")
print(f"LoRA 权重文件: {OUTPUT_DIR}/adapter_model.bin")
print(f"配置文件: {OUTPUT_DIR}/adapter_config.json")

