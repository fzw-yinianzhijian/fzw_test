#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试微调后的 Qwen2.5-7B-Instruct 模型
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== 配置 ====================
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL_PATH = "./output/qwen2.5-7b-lora"  # LoRA 权重路径
HF_TOKEN = os.getenv("HF_TOKEN", None)

# 生成参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# ==================== 加载模型 ====================
print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

print("加载 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# 加载 tokenizer
print("加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    token=HF_TOKEN
)

# 设置为评估模式
model.eval()

# ==================== 测试函数 ====================
def generate_response(user_input, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P):
    """生成回复"""
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    # 使用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码（只取新生成的部分）
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response

# ==================== 测试 ====================
print("\n" + "=" * 60)
print("开始测试模型...")
print("=" * 60)

# 测试问题列表
test_questions = [
    "如何保障工作中遵循正确的安全准则？",
    "我想学习如何使用Python进行数据分析，有没有相关的在线课程可以推荐？",
    "你好，请介绍一下你自己。",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n【测试 {i}】")
    print(f"问题: {question}")
    print("-" * 60)
    
    response = generate_response(question)
    print(f"回答: {response}")
    print("=" * 60)

# 交互式测试
print("\n" + "=" * 60)
print("进入交互模式（输入 'quit' 退出）")
print("=" * 60)

while True:
    try:
        user_input = input("\n你: ")
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        response = generate_response(user_input)
        print(f"模型: {response}")
    except KeyboardInterrupt:
        print("\n\n退出...")
        break
    except Exception as e:
        print(f"错误: {e}")

print("\n测试结束！")

