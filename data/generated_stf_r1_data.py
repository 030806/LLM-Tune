import json
import os
import re


def format_reasoning_content(reasoning_list):
    """
    格式化推理内容，保持原有的结构

    Args:
        reasoning_list: 推理内容列表

    Returns:
        str: 格式化后的推理内容
    """
    formatted_lines = []

    for i, reasoning in enumerate(reasoning_list, 1):
        if isinstance(reasoning, str):
            # 按段落分割
            paragraphs = reasoning.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # 按句子分割
                    sentences = re.split(r'(?<=[。！？])', para)
                    for sentence in sentences:
                        if sentence.strip():
                            formatted_lines.append(f"{len(formatted_lines) + 1}. {sentence.strip()}")
        else:
            formatted_lines.append(f"{len(formatted_lines) + 1}. {str(reasoning)}")

    return "\n".join(formatted_lines)


def convert_to_target_format_enhanced(input_file, output_file):
    """
    将数据转换为目标格式（增强版）
    """

    converted_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                # 解析原始数据
                data = json.loads(line.strip())

                # 构建输入部分
                input_text = f"用户：{data['question']}"

                # 构建推理部分
                reasoning_formatted = format_reasoning_content(data['reasoning'])

                # 构建输出部分
                output_text = f"<reasoning>\n{reasoning_formatted}\n</reasoning>\n答：{data['answer']}"

                # 构建最终格式
                target_data = {
                    "input": input_text,
                    "output": output_text
                }

                # 写入转换后的数据
                outfile.write(json.dumps(target_data, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}")
                continue
            except KeyError as e:
                print(f"缺少必要字段 {e}: {line}")
                continue
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue

    print(f"转换完成！共转换 {converted_count} 条数据")
    print(f"输出文件: {output_file}")


# 文件路径
input_file = "./dataset/raw_questions/converted_data.jsonl"
output_file = "./dataset/stf/sft_r1_data.jsonl"

# 执行转换
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(input_file):
        print("使用增强版本转换...")
        convert_to_target_format_enhanced(input_file, output_file)
    else:
        print(f"输入文件不存在: {input_file}")
        print("请检查文件路径是否正确")