from modelscope import snapshot_download

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    cache_dir='/data/ljf/LLM/models/deepseek-ai'
)

print(f"模型下载到：{model_dir}")