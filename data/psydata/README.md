# SFT R1 Finance Dataset

## 数据集说明

这是一个用于金融领域SFT（Supervised Fine-Tuning）训练的数据集。

## 文件结构

- `sft_r1_train.jsonl`: 训练集
- `sft_r1_val.jsonl`: 验证集
- `dataset_info.json`: 数据集统计信息
- `README.md`: 本说明文件

## 数据格式

每条数据包含两个字段：

```json
{
  "input": "用户：问题内容",
  "output": "<reasoning>\n1. 推理步骤1\n2. 推理步骤2\n...</reasoning>\n答：最终答案"
}
```
