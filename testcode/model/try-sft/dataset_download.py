from datasets import load_dataset

# 加载 Beautiful-Chinese 数据集
dataset = load_dataset("Seikaijyu/Beautiful-Chinese", split="train")

# 查看数据集信息
print(f"数据集大小: {len(dataset)}")
print(f"数据集字段: {dataset.column_names}")
print("\n前3条数据示例:")
for i in range(3):
    print(dataset[i])
