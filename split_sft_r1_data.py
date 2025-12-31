import json
import os
import random
from collections import defaultdict


def stratified_split_dataset(input_file, output_dir, train_ratio=0.8, seed=42):
    """
    分层抽样划分数据集，确保训练集和验证集的分布相似
    """

    random.seed(seed)

    # 读取数据并按问题长度分组（简单分层策略）
    data_by_length = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question = data['input']
            # 按问题长度分组，每20个字符一个区间
            length_group = len(question) // 20
            data_by_length[length_group].append(data)

    print(f"数据按长度分组: {len(data_by_length)} 个组")

    train_data = []
    val_data = []

    # 对每个组进行分层抽样
    for group, group_data in data_by_length.items():
        random.shuffle(group_data)
        split_index = int(len(group_data) * train_ratio)

        train_data.extend(group_data[:split_index])
        val_data.extend(group_data[split_index:])

        print(f"组 {group}: 总数 {len(group_data)}, 训练 {split_index}, 验证 {len(group_data) - split_index}")

    # 再次打乱以确保随机性
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"\n最终划分结果:")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    # 保存文件
    train_file = os.path.join(output_dir, "sft_r1_train.jsonl")
    val_file = os.path.join(output_dir, "sft_r1_val.jsonl")

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n文件已保存:")
    print(f"训练集: {train_file}")
    print(f"验证集: {val_file}")

    return train_data, val_data


def create_dataset_card(output_dir):
    """
    创建数据集说明文件
    """
    card_content = {
        "name": "SFT_R1_Finance_Dataset",
        "description": "金融领域SFT训练数据集，包含问题和带推理过程的回答",
        "version": "1.0",
        "splits": {
            "train": "sft_r1_train.jsonl",
            "validation": "sft_r1_val.jsonl"
        },
        "features": {
            "input": "用户问题",
            "output": "包含推理过程和最终答案的完整回复"
        },
        "format": {
            "reasoning": "使用<reasoning>标签包裹编号的推理步骤",
            "answer": "推理后的最终答案"
        }
    }

    card_file = os.path.join(output_dir, "README.md")
    with open(card_file, 'w', encoding='utf-8') as f:
        f.write("# SFT R1 Finance Dataset\n\n")
        f.write("## 数据集说明\n\n")
        f.write("这是一个用于金融领域SFT（Supervised Fine-Tuning）训练的数据集。\n\n")
        f.write("## 文件结构\n\n")
        f.write("- `sft_r1_train.jsonl`: 训练集\n")
        f.write("- `sft_r1_val.jsonl`: 验证集\n")
        f.write("- `dataset_info.json`: 数据集统计信息\n")
        f.write("- `README.md`: 本说明文件\n\n")
        f.write("## 数据格式\n\n")
        f.write("每条数据包含两个字段：\n\n")
        f.write("```json\n")
        f.write(
            '{\n  "input": "用户：问题内容",\n  "output": "<reasoning>\\n1. 推理步骤1\\n2. 推理步骤2\\n...</reasoning>\\n答：最终答案"\n}\n')
        f.write("```\n")

    print(f"数据集说明文件已创建: {card_file}")


# 主执行程序
if __name__ == "__main__":
    input_file = "data/dataset/psychology/stf/sft_r1_data.jsonl"
    output_dir = "data/dataset/spilt"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_file):
        print("使用分层抽样划分数据集...")
        train_data, val_data = stratified_split_dataset(input_file, output_dir)

        # 创建数据集说明
        create_dataset_card(output_dir)

        print("\n数据集划分完成！")
    else:
        print(f"输入文件不存在: {input_file}")
        print("请先运行数据转换脚本")