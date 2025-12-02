from datasets import load_dataset, Dataset
import json

# 加载数据集
dataset = load_dataset("Seikaijyu/Beautiful-Chinese", split="train")

# 转换为 Qwen 格式
def convert_to_qwen_format(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
    }

# 转换数据
qwen_dataset = dataset.map(convert_to_qwen_format, remove_columns=["question", "answer"])

# 保存为 JSON 文件
# qwen_dataset.to_json("data/train.json")

qwen_dataset.select(range(10000)).to_json("data/train_1w.json")

qwen_dataset.select(range(100000)).to_json("data/train_10w.json")

# 可选：只使用部分数据（用于快速测试）
# qwen_dataset.select(range(1000)).to_json("data/train_sample.json")
