from modelscope import snapshot_download
# print("开始下载 Qwen2.5-1.5B-Instruct...")
# model_dir = snapshot_download(
#     'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
#     cache_dir='/data/LLM/models'
# )

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    cache_dir='/data/ljf/LLM/models/deepseek-ai'
)

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    cache_dir='/data/ljf/LLM/models/deepseek-ai'
)

print(f"模型下载到：{model_dir}")

# 2. 下载 Llama-3.1-8B-Instruct
# print("开始下载 Llama-3.1-8B-Instruct...")
# try:
#     # 注意：Llama 3.1 可能需要接受许可协议
#     llama_dir = snapshot_download(
#         'LLM-Research/Meta-Llama-3.1-8B-Instruct',
#         cache_dir=cache_dir,
#         revision='main'
#     )
#     print(f"Llama-3.1-8B-Instruct 下载完成，路径：{llama_dir}")
# except Exception as e:
#     print(f"下载 Llama-3.1-8B-Instruct 时出错：{e}")
#     print("可能需要先接受 HuggingFace 或 ModelScope 的许可协议")
#     print("可以尝试以下替代方案：")
#
#     # 尝试其他可用的 Llama 3 版本
#     try:
#         llama_dir = snapshot_download(
#             'skyline2006/Llama-3-8B-Instruct',
#             cache_dir=cache_dir
#         )
#         print(f"Llama-3-8B-Instruct 下载完成，路径：{llama_dir}")
#     except Exception as e2:
#         print(f"替代方案也失败：{e2}")

print("\n" + "=" * 50)
print("下载完成！")
print(f"所有模型都保存在：{cache_dir}")
# print(f"模型下载到：{model_dir}")