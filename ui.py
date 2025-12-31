# streamlit_chatbot.py
import os
import torch
import json
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========== 配置区域 ==========
BASE_MODELS = {
    "DeepSeek-R1-Distill-Qwen-7B": "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "/data/ljf/LLM/models/deepseek-ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
}
LORA_MODELS = {
    "心理健康专家（7B微调版本）": "/data/ljf/LLM/models/deepseek_r1_7b_lora_psy/best_model",
    "心理健康专家（1.5B微调版本）": "/data/ljf/LLM/models/deepseek_r1_1.5b_lora/best_model"
}
# ==============================

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="DeepSeek-R1 模型对话系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------- 初始化Session State（核心，持久化数据） --------------------------
def init_session_state():
    """初始化Streamlit会话状态，避免重复加载模型/丢失数据"""
    default_state = {
        # 模型相关
        "model_loaded": False,
        "model": None,
        "tokenizer": None,
        "base_model": "",
        "lora_model": "",
        "use_lora": True,
        # 对话相关
        "conversation_history": [],
        "total_tokens": 0,
        # 生成参数
        "max_new_tokens": 10000,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        # 状态标记
        "latest_response": "",
        "latest_reasoning": "",
        "latest_answer": "",
        # 模型缓存
        "loaded_models": {},
        # 示例问题相关
        "example_questions": [
            "我最近感到非常孤独，身边没有朋友或家人。",
            "我感到非常沮丧和无望，不确定该怎么做。我能做些什么来让自己感觉更积极？",
            "我与父母之间存在一些问题，他们总是唠叨我，似乎从不信任我。",
            "我正在与身体形象问题作斗争。如何提升我的自我形象",
            "我在睡眠时间安排上遇到了困难。我晚睡晚起，但似乎无法在早上早起。"
        ],
        # 新增：跟踪示例问题状态
        "selected_example": None,
        "processing_example": False,
        # 新增：重新生成状态
        "regenerate_message": None,
        "need_regenerate": False
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------------- 核心功能函数 --------------------------
def load_model_with_lora(base_model_path: str, lora_model_path: str = None):
    """加载模型和Tokenizer（支持LoRA）"""
    cache_key = f"{base_model_path}_{lora_model_path if lora_model_path else 'base'}"

    # 检查缓存
    if cache_key in st.session_state.loaded_models:
        st.session_state.model = st.session_state.loaded_models[cache_key]["model"]
        st.session_state.tokenizer = st.session_state.loaded_models[cache_key]["tokenizer"]
        st.success(f"✅ 从缓存加载模型: {os.path.basename(base_model_path)}")
        if lora_model_path:
            st.success(f"✅ LoRA权重: {os.path.basename(os.path.dirname(lora_model_path))}")
        return True

    try:
        with st.spinner(f"🤖 正在加载模型..."):
            # 加载Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                use_fast=False
            )

            # 关键设置
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'left'

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # 加载LoRA权重
            if lora_model_path and os.path.exists(lora_model_path) and st.session_state.use_lora:
                model = PeftModel.from_pretrained(model, lora_model_path)
                model = model.merge_and_unload()  # 合并权重提高推理速度

            model.eval()

            # 缓存模型
            st.session_state.loaded_models[cache_key] = {
                "model": model,
                "tokenizer": tokenizer
            }

            # 更新Session State
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.base_model = base_model_path
            st.session_state.lora_model = lora_model_path
            st.session_state.model_loaded = True

            # 显示信息
            st.success(f"✅ 模型加载完成: {os.path.basename(base_model_path)}")
            if lora_model_path and os.path.exists(lora_model_path) and st.session_state.use_lora:
                st.success(f"✅ LoRA权重: {os.path.basename(os.path.dirname(lora_model_path))}")

        return True
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.session_state.model_loaded = False
        return False


def format_conversation(messages: List[Dict]) -> str:
    """格式化对话历史"""
    formatted = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"用户：{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"助手：{msg['content']}\n"
    return formatted.strip()


def generate_response_stream(user_input: str):
    """流式生成回复"""
    if not st.session_state.model_loaded:
        st.error("❌ 模型未加载，请先选择并加载模型！")
        return

    try:
        # 添加用户输入到对话历史
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        # 格式化对话（保留最近5轮对话以避免过长）
        recent_history = st.session_state.conversation_history[-10:]  # 最多保留10轮
        # 原代码（约 164 行）
        # prompt = format_conversation(recent_history) + "\n助手："

        # === 修改后的代码 ===
        # 定义系统提示词，强制要求输出思考标签
        system_instruction = "在回答之前，请务必先进行详细的思考，并将思考过程包裹在 <think> 标签中，然后输出最终回答。\n\n"

       # 拼接 Prompt
        prompt = system_instruction + format_conversation(recent_history) + "\n助手：<think>\n"

        # 编码输入
        inputs = st.session_state.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8000,
            padding=True
        ).to(st.session_state.model.device)

        input_tokens = inputs.input_ids.shape[1]

        # 生成回复（流式输出）
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=st.session_state.max_new_tokens,
                do_sample=st.session_state.do_sample,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                repetition_penalty=st.session_state.repetition_penalty,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )

        # 解码回复
        response = st.session_state.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True
        ).strip()

        # 提取推理和答案
        reasoning, answer = extract_reasoning_and_answer(response)

        # 更新状态
        st.session_state.latest_reasoning = reasoning
        st.session_state.latest_answer = answer
        st.session_state.latest_response = response

        # 计算token数
        response_tokens = len(st.session_state.tokenizer.encode(response))
        total_tokens = input_tokens + response_tokens
        st.session_state.total_tokens += total_tokens

        # 添加到对话历史
        st.session_state.conversation_history.append({"role": "assistant", "content": response})

        return response, reasoning, answer, input_tokens, response_tokens

    except Exception as e:
        st.error(f"生成回复时出错: {str(e)}")
        return None, "", "", 0, 0


# def extract_reasoning_and_answer(response: str) -> tuple:
#     """从回复中提取推理过程和最终答案 - 增强版"""
#     reasoning = ""
#     answer = ""
#
#     # 1. 尝试提取推理部分
#     reasoning_tags = [
#         ("<reasoning>", "</reasoning>"),
#         ("<think>", "</think>"),
#         ("推理：", "答案："),
#         ("思考：", "结论：")
#     ]
#
#     for start_tag, end_tag in reasoning_tags:
#         if start_tag in response:
#             start_idx = response.find(start_tag) + len(start_tag)
#             if end_tag in response[start_idx:]:
#                 end_idx = response.find(end_tag, start_idx)
#                 reasoning = response[start_idx:end_idx].strip()
#                 answer_start = response.find("答：", end_idx)
#                 if answer_start != -1:
#                     answer = response[answer_start + len("答："):].strip()
#                 break
#
#     # 2. 如果没有提取到，尝试其他方法
#     if not reasoning or not answer:
#         # 尝试按"答："分割
#         if "答：" in response:
#             parts = response.split("答：", 1)
#             reasoning = parts[0].strip()
#             answer = parts[1].strip()
#         # 尝试按"答案："分割
#         elif "答案：" in response:
#             parts = response.split("答案：", 1)
#             reasoning = parts[0].strip()
#             answer = parts[1].strip()
#         else:
#             # 整个作为答案
#             answer = response
#
#     # 3. 清理答案中的特殊标记
#     special_tokens = ["</s>", "<|endoftext|>", "<|im_end|>", "</response>"]
#     for token in special_tokens:
#         answer = answer.replace(token, "").strip()
#
#     return reasoning.strip(), answer.strip()
def extract_reasoning_and_answer(response: str) -> tuple:
    """
    优化版提取逻辑：
    专门处理通过Prompt诱导产生的无头标签输出。
    只要发现 </think>，之前的内容就是推理，之后的就是答案。
    """
    reasoning = ""
    answer = ""

    # 1. 优先处理标准 XML 标签 </think>
    if "</think>" in response:
        # 以 </think> 为界切分
        parts = response.split("</think>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()

        # 如果模型抽风自己又补了一个 <think> 开头，把它去掉
        reasoning = reasoning.replace("<think>", "").strip()

    # 2. 如果没有 </think>，尝试旧的匹配逻辑（兼容其他格式）
    elif "</reasoning>" in response:
        parts = response.split("</reasoning>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()

        # 如果模型抽风自己又补了一个 <think> 开头，把它去掉
        reasoning = reasoning.replace("<reasoning>", "").strip()


    # 3. 兜底：如果没有标签，尝试根据换行或关键字区分，或者全当答案
    else:
        # 有时候模型忘了写标签，但我们强制它思考了，
        # 如果 prompt 结尾是 <think>，那么整个开头可能都是思考。
        # 这里做一个简单的启发式判断：如果内容特别长且没有标签，可能出错了，
        # 但为了安全，暂时全当答案，或者你可以根据实际情况调整。
        answer = response

    # 4. 清理答案中的残留标记
    special_tokens = ["</s>", "<|endoftext|>", "<|im_end|>", "Assistant:", "Model:"]
    for token in special_tokens:
        answer = answer.replace(token, "").strip()

    return reasoning, answer

def save_conversation(filename: str = None) -> str:
    """保存对话记录"""
    if not st.session_state.conversation_history:
        st.warning("⚠️ 无对话记录可保存！")
        return ""

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"

    os.makedirs("./conversations", exist_ok=True)
    filepath = os.path.join("./conversations", filename)

    conversation_data = {
        "model": {
            "base": st.session_state.base_model,
            "lora": st.session_state.lora_model if st.session_state.lora_model else None
        },
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(st.session_state.conversation_history),
        "total_tokens": st.session_state.total_tokens,
        "conversation": st.session_state.conversation_history
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        return filepath
    except Exception as e:
        st.error(f"❌ 保存失败: {str(e)}")
        return ""


def load_conversation(filepath: str):
    """加载对话记录"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        st.session_state.conversation_history = data.get("conversation", [])
        st.session_state.total_tokens = data.get("total_tokens", 0)

        # 尝试加载对应的模型
        model_info = data.get("model", {})
        base_model = model_info.get("base", "")
        lora_model = model_info.get("lora", "")

        if base_model and os.path.exists(base_model):
            load_model_with_lora(base_model, lora_model if os.path.exists(str(lora_model)) else None)

        return True
    except Exception as e:
        st.error(f"❌ 加载失败: {str(e)}")
        return False


def clear_conversation():
    """清除对话历史"""
    st.session_state.conversation_history = []
    st.session_state.latest_response = ""
    st.session_state.latest_reasoning = ""
    st.session_state.latest_answer = ""
    st.session_state.total_tokens = 0
    st.session_state.selected_example = None
    st.session_state.processing_example = False
    st.session_state.regenerate_message = None
    st.session_state.need_regenerate = False


def get_conversation_stats() -> Dict:
    """获取对话统计信息"""
    user_msgs = len([msg for msg in st.session_state.conversation_history if msg["role"] == "user"])
    assistant_msgs = len([msg for msg in st.session_state.conversation_history if msg["role"] == "assistant"])
    return {
        "user_msgs": user_msgs,
        "assistant_msgs": assistant_msgs,
        "total_msgs": len(st.session_state.conversation_history),
        "total_tokens": st.session_state.total_tokens,
        "avg_tokens_per_msg": st.session_state.total_tokens / len(
            st.session_state.conversation_history) if st.session_state.conversation_history else 0
    }


# -------------------------- 页面布局与交互逻辑 --------------------------
def main():
    # 初始化Session State
    init_session_state()

    # 页面标题
    st.title("🤖 DeepSeek-R1 模型对话系统")
    st.markdown("基于 DeepSeek-R1-Distill-Qwen-7B + LoRA 微调的对话模型")
    st.divider()

    # -------------------------- 侧边栏：模型配置 + 功能按钮 --------------------------
    with st.sidebar:
        # 显示Logo
        local_img_path = "./image/1.png"
        if os.path.exists(local_img_path):
            st.sidebar.image(local_img_path, width=200, caption="DeepSeek-R1 SFT微调对话系统")
        st.sidebar.divider()

        st.header("⚙️ 模型配置")

        # 1. 基础模型选择
        st.subheader("基础模型")

        # --- 修改开始 ---
        # 复制配置字典并添加自定义选项
        base_model_options = BASE_MODELS.copy()
        base_model_options["自定义路径"] = "custom"

        selected_base = st.selectbox("选择基础模型", list(base_model_options.keys()))

        if selected_base == "自定义路径":
            custom_base = st.text_input("输入基础模型路径", placeholder="例如：/path/to/model")
            if custom_base:
                base_model_path = custom_base
            else:
                # 如果未输入，默认使用列表中的第一个模型
                base_model_path = list(BASE_MODELS.values())[0]
        else:
            base_model_path = base_model_options[selected_base]
        # --- 修改结束 ---

        # 2. LoRA模型选择
        st.subheader("LoRA微调模型")
        st.session_state.use_lora = st.checkbox("启用LoRA微调", value=True)

        if st.session_state.use_lora:
            lora_options = ["无"] + list(LORA_MODELS.keys()) + ["自定义路径"]
            selected_lora = st.selectbox("选择LoRA模型", lora_options)

            if selected_lora == "自定义路径":
                custom_lora = st.text_input("输入LoRA模型路径", placeholder="例如：/path/to/lora/best_model")
                lora_model_path = custom_lora if custom_lora else None
            elif selected_lora != "无":
                lora_model_path = LORA_MODELS[selected_lora]
            else:
                lora_model_path = None
        else:
            lora_model_path = None

        # 加载模型按钮
        if st.button("🚀 加载/重新加载模型", type="primary", use_container_width=True):
            if os.path.exists(base_model_path):
                if load_model_with_lora(base_model_path, lora_model_path):
                    st.rerun()
            else:
                st.error(f"❌ 基础模型路径不存在: {base_model_path}")

        st.divider()

        # 3. 生成参数配置
        st.subheader("📝 生成参数")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.max_new_tokens = st.slider("最大长度", 512, 4096, st.session_state.max_new_tokens, 128)
            st.session_state.temperature = st.slider("温度", 0.1, 1.5, st.session_state.temperature, 0.1)
        with col2:
            st.session_state.top_p = st.slider("Top P", 0.1, 1.0, st.session_state.top_p, 0.1)
            st.session_state.repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, st.session_state.repetition_penalty,
                                                            0.1)

        st.session_state.do_sample = st.checkbox("启用随机采样", value=True)

        st.divider()

        # 4. 功能按钮
        st.subheader("🔧 对话管理")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ 清除", use_container_width=True):
                clear_conversation()
                st.success("对话已清除！")
                st.rerun()
        with col2:
            if st.button("💾 保存", use_container_width=True):
                filepath = save_conversation()
                if filepath:
                    st.success(f"对话已保存到: {filepath}")
        with col3:
            if st.button("📊 统计", use_container_width=True):
                stats = get_conversation_stats()
                st.info(f"总消息: {stats['total_msgs']} | 总Token: {stats['total_tokens']}")

        # 加载对话
        st.subheader("📂 历史对话")
        conversation_dir = "./conversations"
        if os.path.exists(conversation_dir):
            conversation_files = [f for f in os.listdir(conversation_dir) if f.endswith('.json')]
            if conversation_files:
                selected_file = st.selectbox("选择对话文件", conversation_files)
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("📤 加载", use_container_width=True):
                        load_conversation(os.path.join(conversation_dir, selected_file))
                        st.rerun()
                with col2:
                    if st.button("🗑️", help="删除文件", use_container_width=True):
                        try:
                            os.remove(os.path.join(conversation_dir, selected_file))
                            st.success("文件已删除")
                            st.rerun()
                        except:
                            st.error("删除失败")
            else:
                st.info("暂无历史对话")
        else:
            st.info("对话目录不存在")

        st.divider()

        # 5. 对话统计
        st.subheader("📈 实时统计")
        stats = get_conversation_stats()
        st.metric("用户消息", stats['user_msgs'])
        st.metric("助手回复", stats['assistant_msgs'])
        st.metric("总Token数", f"{stats['total_tokens']:,}")

        if st.session_state.model_loaded:
            st.success("✅ 模型已加载")
        else:
            st.warning("⚠️ 模型未加载")

    # -------------------------- 主区域：聊天界面 --------------------------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 对话窗口")

        # 显示历史对话
        for i, msg in enumerate(st.session_state.conversation_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    # 提取推理和答案
                    reasoning, answer = extract_reasoning_and_answer(msg["content"])

                    # 如果是最后一条消息且已经提取过，使用缓存
                    if i == len(st.session_state.conversation_history) - 1:
                        reasoning = st.session_state.latest_reasoning
                        answer = st.session_state.latest_answer

                    if reasoning:
                        with st.expander("🧠 查看推理过程", expanded=True):
                            st.markdown(reasoning)
                    if answer:
                        st.markdown(answer)
                    if not reasoning and not answer:
                        st.markdown(msg["content"])

        # ========== 处理重新生成 ==========
        if st.session_state.need_regenerate:
            # 找到最后一条用户消息
            last_user_msg = None
            for msg in reversed(st.session_state.conversation_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                # 移除最后一条助手回复（如果有）
                if st.session_state.conversation_history and st.session_state.conversation_history[-1][
                    "role"] == "assistant":
                    st.session_state.conversation_history.pop()

                # 显示用户消息
                with st.chat_message("user", avatar="👤"):
                    st.markdown(last_user_msg)

                # 生成新的回复
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤖 重新生成中..."):
                        response, reasoning, answer, input_tokens, output_tokens = generate_response_stream(
                            last_user_msg)

                        if response:
                            # 显示推理和答案
                            if reasoning:
                                with st.expander("🧠 推理过程", expanded=True):
                                    st.markdown(reasoning)
                                st.markdown("**📝 最终答案:**")
                                st.markdown(answer)
                            elif answer:
                                st.markdown(answer)
                            else:
                                st.markdown(response)

                            # 显示统计信息
                            st.caption(
                                f"输入: {input_tokens}tokens | 输出: {output_tokens}tokens | 总计: {input_tokens + output_tokens}tokens")

                # 重置重新生成状态
                st.session_state.need_regenerate = False
                st.rerun()
            else:
                st.warning("没有找到用户消息来重新生成")
                st.session_state.need_regenerate = False

        # ========== 处理示例问题 ==========
        elif st.session_state.selected_example and not st.session_state.processing_example:
            example_question = st.session_state.selected_example

            # 设置正在处理标志
            st.session_state.processing_example = True

            # 显示用户输入的示例问题
            with st.chat_message("user", avatar="👤"):
                st.markdown(example_question)

            # 生成回复
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤖 思考中..."):
                    response, reasoning, answer, input_tokens, output_tokens = generate_response_stream(
                        example_question)

                    if response:
                        # 显示推理和答案
                        if reasoning:
                            with st.expander("🧠 推理过程", expanded=True):
                                st.markdown(reasoning)
                            st.markdown("**📝 最终答案:**")
                            st.markdown(answer)
                        elif answer:
                            st.markdown(answer)
                        else:
                            st.markdown(response)

                        # 显示统计信息
                        st.caption(
                            f"输入: {input_tokens}tokens | 输出: {output_tokens}tokens | 总计: {input_tokens + output_tokens}tokens")

            # 重置示例状态
            st.session_state.selected_example = None
            st.session_state.processing_example = False

            # 等待一下然后重新运行，以便显示结果
            st.rerun()

        # ========== 处理用户正常输入 ==========
        else:
            user_input = st.chat_input("请输入您的问题...")

            if user_input:
                # 处理退出命令
                if user_input.lower() in ["退出", "quit", "exit", "q"]:
                    st.info("👋 对话已结束！")
                    if st.session_state.conversation_history:
                        if st.button("退出前保存对话"):
                            save_conversation()
                    st.stop()

                # 显示用户输入
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)

                # 生成并显示回复
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤖 思考中..."):
                        response, reasoning, answer, input_tokens, output_tokens = generate_response_stream(user_input)

                        if response:
                            # 显示推理和答案
                            if reasoning:
                                with st.expander("🧠 推理过程", expanded=True):
                                    st.markdown(reasoning)
                                st.markdown("**📝 最终答案:**")
                                st.markdown(answer)
                            elif answer:
                                st.markdown(answer)
                            else:
                                st.markdown(response)

                            # 显示统计信息
                            st.caption(
                                f"输入: {input_tokens}tokens | 输出: {output_tokens}tokens | 总计: {input_tokens + output_tokens}tokens")

    with col2:
        st.header("📋 示例问题")
        st.markdown("点击下方问题快速测试：")

        # 显示示例问题按钮
        for question in st.session_state.example_questions:
            if st.button(question, key=f"example_{question[:20]}", use_container_width=True):
                # 设置选中的示例问题，并重置处理状态
                st.session_state.selected_example = question
                st.session_state.processing_example = False
                st.rerun()

        st.divider()

        st.header("⚡ 快捷操作")

        if st.button("🔄 重新生成", use_container_width=True):
            if st.session_state.conversation_history:
                # 检查是否有助手回复可以重新生成
                has_assistant = any(msg["role"] == "assistant" for msg in st.session_state.conversation_history)
                if has_assistant:
                    # 设置标记，主区域会检测到这个标记
                    st.session_state.need_regenerate = True
                    st.rerun()
                else:
                    st.warning("没有助手回复可以重新生成")
            else:
                st.warning("没有对话历史")

        if st.button("📝 导出对话", use_container_width=True):
            if st.session_state.conversation_history:
                # 导出为文本
                export_text = ""
                for msg in st.session_state.conversation_history:
                    role = "用户" if msg["role"] == "user" else "助手"
                    export_text += f"{role}: {msg['content']}\n\n"

                st.download_button(
                    label="📥 下载对话记录",
                    data=export_text,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        st.divider()

        # 模型状态
        st.header("🔍 模型状态")
        if st.session_state.model_loaded:
            st.success("✅ 运行正常")
            st.info(f"基础模型: {os.path.basename(st.session_state.base_model)}")
            if st.session_state.lora_model:
                st.info(f"LoRA模型: {os.path.basename(os.path.dirname(st.session_state.lora_model))}")
        else:
            st.warning("⚠️ 模型未加载")
            st.info("请在侧边栏加载模型")

    # 页面底部提示
    st.divider()
    st.caption("💡 提示：支持多轮对话 | 推理过程自动提取 | LoRA权重自动合并 | 对话历史持久化")


if __name__ == "__main__":
    main()