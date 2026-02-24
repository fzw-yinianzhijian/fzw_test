from transformers import pipeline
import os

download_path = './model_weights/gemma-7b-it'


pipe = pipeline(
    "text-generation",
    model=download_path,
    device_map="auto",  # 自动分配设备
    dtype="auto"  # 自动选择数据类型
)
print("模型加载完成！开始交互（输入 'quit' 或 'exit' 退出）\n")

# 交互式循环
while True:
    try:

        user_input = input("你: ").strip()
        

        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        # 生成回复
        print("模型: ", end="", flush=True)
        outputs = pipe(user_input, max_new_tokens=200, temperature=0.7)
        generated_text = outputs[0]['generated_text']
        
        # 只打印新生成的部分（去掉原始输入）
        if generated_text.startswith(user_input):
            response = generated_text[len(user_input):].strip()
        else:
            response = generated_text.strip()
        
        print(response)
        print()  # 空行分隔
        
    except KeyboardInterrupt:
        print("\n\n再见！")
        break
    except Exception as e:
        print(f"错误: {e}\n")