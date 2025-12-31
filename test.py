import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

BASE_MODEL = "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LORA_MODEL = "/data/ljf/LLM/models/deepseek_r1_7b_lora_0.1/best_model"


def load_model_and_tokenizer():
    """同时加载模型和tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # 添加这一行，某些模型需要
    )

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, LORA_MODEL)

    # 合并LoRA权重到基础模型（可选，提高推理速度）
    model = model.merge_and_unload()

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, query, max_new_tokens=512):
    """生成回复"""
    prompt = query
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 使用GenerationConfig统一配置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy解码
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config
        )

    # 只提取新生成的文本
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    return response


def main():
    if len(sys.argv) < 2:
        # 如果没有命令行参数，使用示例问题
        query = "能否指导我如何设计一个程序来计算一个数的数字根？"
        print(f"使用示例问题: {query}")
    else:
        query = sys.argv[1]

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()

    print(f"\n输入问题: {query}")
    print("生成回答中...")

    response = generate_response(model, tokenizer, query)

    print("\n" + "=" * 50)
    print("📝 模型回答:")
    print("=" * 50)
    print(response)
    print("=" * 50)

    # 显示一些统计信息
    input_tokens = len(tokenizer.encode(query))
    output_tokens = len(tokenizer.encode(response))
    print(f"\n📊 统计: 输入 {input_tokens} tokens, 输出 {output_tokens} tokens")


if __name__ == "__main__":
    main()