import os
import torch
import readline
import json
from datetime import datetime
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel

# ========== 配置区域 ==========
BASE_MODEL = "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LORA_MODEL = "/data/ljf/LLM/models/deepseek_r1_7b_lora_0.1/best_model"


# ==============================

class ChatBot:
    def __init__(self, base_model_path: str = BASE_MODEL, adapter_path: str = LORA_MODEL):
        """初始化聊天机器人"""
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.conversation_history = []

        print("=" * 60)
        print("🤖 DeepSeek-R1 推理模型聊天机器人")
        print("=" * 60)
        print(f"🏗️  基座模型: {os.path.basename(base_model_path)}")
        if adapter_path and os.path.exists(adapter_path):
            print(f"🧩 LoRA权重: {os.path.basename(os.path.dirname(adapter_path))}")

        self.load_model()

        print("\n✅ 模型加载完成！")
        print("💡 命令提示:")
        print("  '退出'/'quit' - 结束对话")
        print("  '清除'/'clear' - 清除历史")
        print("  '保存'/'save' - 保存对话")
        print("  '统计'/'stats' - 查看统计")
        print("  '帮助'/'help' - 显示帮助")
        print("-" * 40)

    def load_model(self):
        """加载模型和tokenizer"""
        try:
            print("🚀 加载 Tokenizer...")
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=False
            )

            # 关键设置
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print("🚀 加载基座模型...")
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # 检查并加载LoRA权重
            if self.adapter_path and os.path.exists(self.adapter_path):
                print("🧩 挂载 LoRA 权重...")
                self.model = PeftModel.from_pretrained(model, self.adapter_path)
                print("🔄 合并LoRA权重到基础模型...")
                self.model = self.model.merge_and_unload()  # 合并以提高推理速度
                print("✅ LoRA权重合并完成")
            else:
                print("⚠️  未找到LoRA权重，使用基础模型")
                self.model = model

            self.model.eval()
            print("🎯 模型准备就绪")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def format_conversation(self, messages: List[Dict]) -> str:
        """格式化对话历史 - 简化版"""
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"用户：{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"助手：{msg['content']}\n"
        return formatted.strip()

    def generate_response(self, user_input: str, max_new_tokens: int = 2048) -> str:
        """生成回复"""
        try:
            # 添加用户输入到历史
            self.conversation_history.append({"role": "user", "content": user_input})

            # 格式化prompt
            prompt = self.format_conversation(self.conversation_history) + "\n助手："

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            ).to(self.model.device)

            # 显示token数量
            input_tokens = inputs.input_ids.shape[1]
            print(f"📝 输入Tokens: {input_tokens}")

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True
                )

            # 解码
            response = self.tokenizer.decode(
                outputs[0][input_tokens:],
                skip_special_tokens=True
            ).strip()

            # 统计
            output_tokens = len(self.tokenizer.encode(response))
            print(f"📤 输出Tokens: {output_tokens}")

            # 添加到历史
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，生成回复时出错。"

    def extract_reasoning_and_answer(self, response: str) -> tuple:
        """提取推理和答案 - 增强版"""
        reasoning = ""
        answer = ""

        # 1. 尝试提取推理部分
        reasoning_tags = [
            ("<reasoning>", "</reasoning>"),
            ("<think>", "</think>"),
            ("推理：", "答案："),
            ("思考：", "结论：")
        ]

        for start_tag, end_tag in reasoning_tags:
            if start_tag in response:
                start_idx = response.find(start_tag) + len(start_tag)
                if end_tag in response[start_idx:]:
                    end_idx = response.find(end_tag, start_idx)
                    reasoning = response[start_idx:end_idx].strip()
                    answer_start = response.find("答：", end_idx)
                    if answer_start != -1:
                        answer = response[answer_start + len("答："):].strip()
                    break

        # 2. 如果没有提取到，尝试其他方法
        if not reasoning or not answer:
            # 尝试按"答："分割
            if "答：" in response:
                parts = response.split("答：", 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            # 尝试按"答案："分割
            elif "答案：" in response:
                parts = response.split("答案：", 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                # 整个作为答案
                answer = response

        # 3. 清理答案中的特殊标记
        special_tokens = ["</s>", "<|endoftext|>", "<|im_end|>", "</response>"]
        for token in special_tokens:
            answer = answer.replace(token, "").strip()

        return reasoning.strip(), answer.strip()

    def save_conversation(self, filename: str = None):
        """保存对话记录"""
        if not self.conversation_history:
            print("⚠️  对话历史为空")
            return

        os.makedirs("./conversations", exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        filepath = os.path.join("./conversations", filename)

        conversation_data = {
            "model": {
                "base": self.base_model_path,
                "lora": self.adapter_path if os.path.exists(str(self.adapter_path)) else None
            },
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(self.conversation_history),
            "conversation": self.conversation_history
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            print(f"💾 对话已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return None

    def load_conversation(self, filepath: str):
        """加载对话记录"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.conversation_history = data.get("conversation", [])
            print(f"📂 已加载 {len(self.conversation_history)} 条消息")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def clear_conversation(self):
        """清除对话历史"""
        self.conversation_history = []
        print("🗑️  对话历史已清除")

    def print_stats(self):
        """打印统计信息"""
        if not self.conversation_history:
            print("📊 暂无对话历史")
            return

        user_msgs = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in self.conversation_history if msg["role"] == "assistant")

        # 计算token数
        total_tokens = 0
        for msg in self.conversation_history:
            total_tokens += len(self.tokenizer.encode(msg["content"]))

        print(f"\n📊 对话统计:")
        print(f"  用户消息: {user_msgs} 条")
        print(f"  助手消息: {assistant_msgs} 条")
        print(f"  总消息数: {len(self.conversation_history)} 条")
        print(f"  总Token数: {total_tokens}")
        print(f"  模型: {os.path.basename(self.base_model_path)}")

    def print_help(self):
        """显示帮助"""
        print("\n📖 命令列表:")
        print("  退出/quit     - 结束对话")
        print("  清除/clear    - 清除历史")
        print("  保存/save     - 保存对话")
        print("  统计/stats    - 查看统计")
        print("  帮助/help     - 显示此帮助")
        print("  加载/load <文件> - 加载对话")
        print("-" * 40)

    def start_chat(self):
        """开始聊天会话"""
        print("🎤 开始对话 (输入'帮助'查看命令)...\n")

        while True:
            try:
                user_input = input("👤 你: ").strip()

                if not user_input:
                    continue

                # 命令处理
                user_input_lower = user_input.lower()

                if user_input_lower in ['退出', 'quit', 'exit', 'q']:
                    print("👋 再见！")
                    break

                elif user_input_lower in ['清除', 'clear']:
                    self.clear_conversation()
                    continue

                elif user_input_lower in ['保存', 'save']:
                    self.save_conversation()
                    continue

                elif user_input_lower in ['统计', 'stats']:
                    self.print_stats()
                    continue

                elif user_input_lower in ['帮助', 'help']:
                    self.print_help()
                    continue

                elif user_input_lower.startswith('加载 ') or user_input_lower.startswith('load '):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        self.load_conversation(parts[1])
                    continue

                # 普通对话
                print("🤖 思考中...", end='', flush=True)
                response = self.generate_response(user_input)
                print("\r" + " " * 30 + "\r", end='')

                reasoning, answer = self.extract_reasoning_and_answer(response)

                # 显示结果
                print("🤖 助手:")
                if reasoning:
                    print(f"💭 推理过程:\n{reasoning}\n")
                    print("📝 最终答案:")
                print(f"{answer}\n")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\n⚠️  中断对话")
                save = input("是否保存当前对话？(y/n): ").strip().lower()
                if save == 'y':
                    self.save_conversation()
                print("👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 DeepSeek-R1 推理模型聊天系统")
    print("=" * 60)

    # 检查模型路径
    if not os.path.exists(BASE_MODEL):
        print(f"❌ 错误: 基座模型不存在\n{BASE_MODEL}")
        return

    if not os.path.exists(LORA_MODEL):
        print(f"⚠️  警告: LoRA模型不存在\n{LORA_MODEL}")
        print("将使用基础模型进行对话")
        use_lora = False
    else:
        use_lora = True

    try:
        # 创建聊天机器人
        if use_lora:
            chatbot = ChatBot(BASE_MODEL, LORA_MODEL)
        else:
            chatbot = ChatBot(BASE_MODEL)

        # 开始聊天
        chatbot.start_chat()

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()