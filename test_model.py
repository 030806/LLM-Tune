import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quick_test(model_path: str):
    """快速测试模型"""
    print("快速测试模型...")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试问题
    test_questions = [
        "我在睡眠方面遇到困难，有哪些方法可以改善我的睡眠质量",
        " 我正在与身体形象问题作斗争。如何提升我的自我形象",
        "我感觉我的焦虑正在掌控我的生活，我不知道该怎么办"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")

        # 格式化输入
        prompt = f"用户：{question}\n助手："

        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=4096,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"回答: {response}")
        print("-" * 50)


if __name__ == "__main__":
    model_path =  "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    quick_test(model_path)