from datasets import Dataset
import json

# 加载转换后的数据
with open("data/train_10w.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# 划分训练集和验证集（90% / 10%）
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 保存
split_dataset["train"].to_json("data/train_10w_split.json")
split_dataset["test"].to_json("data/eval_10w_split.json")

print(f"训练集: {len(split_dataset['train'])} 条")
print(f"验证集: {len(split_dataset['test'])} 条")
