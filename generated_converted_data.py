import json
import os


def convert_data_format(input_file, output_dir):
    """
    将input/content/reasoning_content格式转换为question/reasoning/answer格式

    Args:
        input_file: 输入文件路径
        output_dir: 输出目录路径
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 输出文件路径
    output_file = os.path.join(output_dir, "converted_data.jsonl")

    converted_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # 解析原始JSON数据
                original_data = json.loads(line.strip())

                # 提取关键字段
                question = original_data.get("input", "")
                answer = original_data.get("content", "")
                reasoning_content = original_data.get("reasoning_content", "")

                # 检查必要字段是否存在
                if not question:
                    print(f"警告: 第{line_num}行缺少input字段")
                    question = ""

                if not answer:
                    print(f"警告: 第{line_num}行缺少content字段")
                    answer = ""

                # 构建目标格式
                target_format = {
                    "question": question,
                    "reasoning": [],
                    "answer": answer
                }

                # 处理reasoning_content
                if reasoning_content:
                    # 如果reasoning_content是字符串，尝试按段落分割
                    if isinstance(reasoning_content, str):
                        # 按句号、分号或换行符分割
                        import re
                        # 分割成句子，保留标点
                        sentences = re.split(r'(?<=[。；！？])', reasoning_content)
                        # 过滤空字符串和空白字符
                        sentences = [s.strip() for s in sentences if s.strip()]

                        if sentences:
                            target_format["reasoning"] = sentences
                        else:
                            # 如果无法分割，直接作为单个推理段落
                            target_format["reasoning"] = [reasoning_content]
                    elif isinstance(reasoning_content, list):
                        target_format["reasoning"] = reasoning_content
                    else:
                        # 其他类型转换为字符串
                        target_format["reasoning"] = [str(reasoning_content)]
                else:
                    # 如果没有推理内容，创建一个简单的推理
                    target_format["reasoning"] = [f"分析问题：{question}"]

                # 写入转换后的数据
                outfile.write(json.dumps(target_format, ensure_ascii=False) + '\n')
                converted_count += 1

                # 每处理1000条数据打印一次进度
                if converted_count % 1000 == 0:
                    print(f"已处理 {converted_count} 条数据...")

            except json.JSONDecodeError as e:
                error_count += 1
                print(f"解析JSON时出错 (第{line_num}行): {e}")
                print(f"问题行内容: {line[:200]}...")
                continue
            except Exception as e:
                error_count += 1
                print(f"处理数据时出错 (第{line_num}行): {e}")
                print(f"问题行内容: {line[:200]}...")
                continue

    print(f"转换完成！")
    print(f"成功转换: {converted_count} 条数据")
    print(f"转换失败: {error_count} 条数据")
    print(f"输出文件: {output_file}")

    # 显示前几条转换后的数据作为示例
    if converted_count > 0:
        print("\n转换后的数据示例（前3条）:")
        print("-" * 50)
        with open(output_file, 'r', encoding='utf-8') as sample_file:
            for i, sample_line in enumerate(sample_file):
                if i >= 3:
                    break
                sample_data = json.loads(sample_line.strip())
                print(f"示例 {i + 1}:")
                print(f"  问题: {sample_data['question'][:100]}...")
                print(f"  推理段落数: {len(sample_data['reasoning'])}")
                print(f"  答案: {sample_data['answer'][:100]}...")
                print()


# 文件路径
input_file = "data/distill_psychology-10k-r1.json"
output_dir = "data/dataset/psychology"

# 执行转换
if __name__ == "__main__":
    if os.path.exists(input_file):
        # 统计输入文件行数
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            print(f"输入文件行数: {total_lines}")
        except:
            print("无法统计输入文件行数")

        convert_data_format(input_file, output_dir)
    else:
        print(f"输入文件不存在: {input_file}")
        print("请检查文件路径是否正确")