import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
# 新增：导入PeftModel（之前代码中使用但未导入）
from peft import PeftModel


class ModelEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str = None, is_base_model: bool = False):
        """初始化评测器
        Args:
            model_path: 模型路径（基础模型或LoRA模型）
            tokenizer_path: tokenizer路径
            is_base_model: 是否为基础模型（跳过LoRA加载）
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_base_model = is_base_model

        print(f"加载模型: {model_path}")
        self.load_model()

        # 初始化指标计算器（避免依赖问题）
        self.metrics_initialized = False
        self.init_metrics()

    def init_metrics(self):
        """初始化评估指标，处理依赖问题"""
        try:
            import evaluate
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
            self.metrics_initialized = True
            print("✅ 评估指标加载成功")
        except ImportError as e:
            print(f"⚠️  评估指标依赖缺失: {e}")
            print("🔧 请安装依赖: pip install evaluate rouge-score nltk absl-py")
            self.metrics_initialized = False
        except Exception as e:
            print(f"⚠️  评估指标初始化失败: {e}")
            self.metrics_initialized = False

    def load_model(self):
        """加载模型和tokenizer - 修复版"""
        try:
            # 1. 加载tokenizer（从基础模型）
            print(f"加载tokenizer: {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=False
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 如果是基础模型，直接加载
            if self.is_base_model:
                print("加载基础模型...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model.eval()
                print("✅ 基础模型加载成功")
                return

            # 2. 加载基础模型（用于LoRA）
            print("加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.tokenizer_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # 3. 加载LoRA权重
            print(f"加载LoRA权重: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            # 4. 合并LoRA权重（提高推理速度，可选）
            print("合并LoRA权重...")
            self.model = self.model.merge_and_unload()

            self.model.eval()
            print("✅ 微调模型加载成功")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

            # 如果LoRA加载失败，尝试直接加载基础模型
            print("尝试直接加载基础模型...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True,
                    use_fast=False
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.tokenizer_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

                self.model.eval()
                print("✅ 基础模型加载成功")
            except Exception as e2:
                print(f"❌ 基础模型加载也失败: {e2}")
                raise

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """生成回复"""
        try:
            # 直接使用prompt，不应用chat模板
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # 解码回复
            response = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"生成回复时出错: {e}")
            return ""

    def extract_reasoning_and_answer(self, response: str) -> tuple:
        """从回复中提取推理过程和最终答案"""
        reasoning = ""
        answer = ""

        # 尝试多种格式
        formats = [
            ("<reasoning>", "</reasoning>", "答："),
            ("推理：", "答案：", "答案："),
            ("思考：", "结论：", "结论："),
            ("分析：", "结果：", "结果：")
        ]

        for start_tag, end_tag, answer_tag in formats:
            if start_tag in response:
                # 提取推理部分
                start_idx = response.find(start_tag) + len(start_tag)
                if end_tag in response[start_idx:]:
                    end_idx = response.find(end_tag, start_idx)
                    reasoning = response[start_idx:end_idx].strip()

                    # 提取答案部分
                    answer_start = response.find(answer_tag, end_idx)
                    if answer_start != -1:
                        answer = response[answer_start + len(answer_tag):].strip()
                        break
                else:
                    # 如果没有结束标签，取开始标签后的所有内容
                    reasoning = response[start_idx:].strip()

        # 如果没有找到特定格式
        if not reasoning and not answer:
            # 尝试按段落分割
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if any(tag in line.lower() for tag in ['答：', '答案：', '结论：', '结果：']):
                    reasoning = '\n'.join(lines[:i]).strip()
                    answer = '\n'.join(lines[i:]).strip()
                    break

            # 如果还是没找到，全部作为答案
            if not reasoning:
                answer = response

        return reasoning, answer

    def calculate_similarity_metrics(self, reference: str, prediction: str) -> Dict:
        """计算相似度指标（简化版，避免依赖问题）"""
        metrics = {}

        if not self.metrics_initialized:
            # 使用简单的字符串匹配作为备选
            metrics.update(self.calculate_basic_metrics(reference, prediction))
            return metrics

        try:
            # BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=[prediction],
                references=[[reference]]
            )
            metrics["bleu"] = float(bleu_result["bleu"])  # 转换为Python float

            # ROUGE score
            rouge_result = self.rouge_metric.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["rouge1"] = float(rouge_result["rouge1"])  # 转换为Python float
            metrics["rouge2"] = float(rouge_result["rouge2"])  # 转换为Python float
            metrics["rougeL"] = float(rouge_result["rougeL"])  # 转换为Python float

        except Exception as e:
            print(f"计算相似度指标时出错，使用基础指标: {e}")
            metrics.update(self.calculate_basic_metrics(reference, prediction))

        return metrics

    def calculate_basic_metrics(self, reference: str, prediction: str) -> Dict:
        """计算基础相似度指标（不依赖外部包）"""
        ref_words = set(reference.split())
        pred_words = set(prediction.split())

        # 计算Jaccard相似度
        intersection = len(ref_words.intersection(pred_words))
        union = len(ref_words.union(pred_words))
        jaccard_similarity = float(intersection / union if union > 0 else 0)

        # 计算重叠率
        overlap_ratio = float(
            len([w for w in prediction.split() if w in reference]) / len(prediction.split()) if prediction else 0)

        return {
            "jaccard_similarity": jaccard_similarity,
            "overlap_ratio": overlap_ratio,
            "bleu": jaccard_similarity,  # 用Jaccard近似BLEU
            "rouge1": overlap_ratio,  # 用重叠率近似ROUGE-1
            "rouge2": 0.0,  # 简化处理
            "rougeL": overlap_ratio  # 用重叠率近似ROUGE-L
        }

    def evaluate_single_example(self, test_case: Dict) -> Dict:
        """评估单个测试用例"""
        question = test_case["input"].replace("用户：", "").strip()
        expected_output = test_case["output"]

        # 生成回复
        start_time = datetime.now()
        generated_response = self.generate_response(question)
        generation_time = (datetime.now() - start_time).total_seconds()

        # 提取推理和答案
        reasoning, answer = self.extract_reasoning_and_answer(generated_response)

        # 计算指标
        similarity_metrics = self.calculate_similarity_metrics(expected_output, generated_response)

        # 计算响应长度
        response_length = len(generated_response)

        return {
            "question": question,
            "expected_output": expected_output,
            "generated_response": generated_response,
            "reasoning": reasoning,
            "answer": answer,
            "generation_time": float(generation_time),  # 转换为Python float
            "response_length": int(response_length),  # 转换为Python int
            "similarity_metrics": similarity_metrics,
            "has_reasoning_format": "<reasoning>" in generated_response and "</reasoning>" in generated_response,
            "has_answer_format": "答：" in generated_response,
            "is_empty_response": len(generated_response.strip()) == 0
        }

    def evaluate_on_dataset(self, test_file: str, num_samples: int = None) -> Dict:
        """在测试集上进行评估"""
        print(f"开始评估，测试文件: {test_file}")

        # 检查测试文件是否存在
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在: {test_file}")

        # 加载测试数据
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))

        if num_samples and num_samples < len(test_data):
            test_data = test_data[:num_samples]

        print(f"测试样本数量: {len(test_data)}")

        results = []
        total_metrics = {
            "generation_time": [],
            "response_length": [],
            "has_reasoning_format": 0,
            "has_answer_format": 0,
            "empty_responses": 0
        }

        # 初始化指标累计
        metric_keys = ["bleu", "rouge1", "rouge2", "rougeL", "jaccard_similarity", "overlap_ratio"]
        for key in metric_keys:
            total_metrics[key] = []

        for i, test_case in enumerate(test_data):
            print(f"处理样本 {i + 1}/{len(test_data)}")

            try:
                result = self.evaluate_single_example(test_case)
                results.append(result)

                # 累计指标
                metrics = result["similarity_metrics"]
                for key in metrics:
                    if key in total_metrics:
                        total_metrics[key].append(metrics[key])

                # 累计其他指标
                total_metrics["generation_time"].append(result["generation_time"])
                total_metrics["response_length"].append(result["response_length"])
                total_metrics["has_reasoning_format"] += int(result["has_reasoning_format"])
                total_metrics["has_answer_format"] += int(result["has_answer_format"])
                total_metrics["empty_responses"] += int(result["is_empty_response"])

            except Exception as e:
                print(f"评估样本 {i + 1} 时出错: {e}")
                continue

        # 计算平均指标 - 确保所有值都是Python原生类型
        avg_metrics = {}
        for key, values in total_metrics.items():
            if isinstance(values, list) and values:
                # 转换为Python原生类型
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"std_{key}"] = float(np.std(values))
                avg_metrics[f"min_{key}"] = float(np.min(values))
                avg_metrics[f"max_{key}"] = float(np.max(values))
            else:
                # 对于非列表值（计数类型），直接使用
                avg_metrics[key] = values

        # 计算格式正确率
        if results:
            avg_metrics["reasoning_format_rate"] = float(total_metrics["has_reasoning_format"] / len(results))
            avg_metrics["answer_format_rate"] = float(total_metrics["has_answer_format"] / len(results))
            avg_metrics["empty_response_rate"] = float(total_metrics["empty_responses"] / len(results))
            avg_metrics["total_samples"] = int(len(results))  # 转换为Python int
        else:
            avg_metrics.update({
                "reasoning_format_rate": 0.0,
                "answer_format_rate": 0.0,
                "empty_response_rate": 0.0,
                "total_samples": 0
            })

        return {
            "results": results,
            "summary": avg_metrics,
            "total_samples": len(results),
            "metrics_available": self.metrics_initialized
        }

    def convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def run_comprehensive_evaluation(self, test_file: str, output_dir: str, num_samples: int = None,
                                     model_name: str = "model"):
        """运行综合评估
        Args:
            test_file: 测试文件路径
            output_dir: 输出目录
            num_samples: 评估样本数量
            model_name: 模型名称（用于区分基础/微调模型）
        """
        print(f"开始{model_name}的综合评估...")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 评估结果
        evaluation_result = self.evaluate_on_dataset(test_file, num_samples)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 转换为可序列化的格式
        serializable_result = self.convert_to_serializable(evaluation_result)

        # 保存详细结果
        detailed_file = os.path.join(output_dir, f"{model_name}_detailed_results_{timestamp}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        # 保存摘要结果
        summary_file = os.path.join(output_dir, f"{model_name}_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result["summary"], f, indent=2, ensure_ascii=False)

        # 生成CSV报告
        self.generate_csv_report(serializable_result["results"], output_dir, timestamp, model_name)

        # 打印评估摘要
        self.print_evaluation_summary(serializable_result["summary"], serializable_result["metrics_available"],
                                      model_name)

        print(f"✅ {model_name}评估完成!")
        print(f"📊 详细结果: {detailed_file}")
        print(f"📈 评估摘要: {summary_file}")

        return evaluation_result

    def generate_csv_report(self, results: List[Dict], output_dir: str, timestamp: str, model_name: str = "model"):
        """生成CSV格式的报告"""
        csv_data = []

        for i, result in enumerate(results):
            row = {
                "id": i + 1,
                "question": result["question"],
                "expected_output": result["expected_output"],
                "generated_response": result["generated_response"],
                "reasoning": result["reasoning"],
                "answer": result["answer"],
                "generation_time": result["generation_time"],
                "response_length": result["response_length"],
                "has_reasoning_format": result["has_reasoning_format"],
                "has_answer_format": result["has_answer_format"],
                "is_empty_response": result["is_empty_response"]
            }

            # 添加相似度指标
            for metric_name, value in result["similarity_metrics"].items():
                row[metric_name] = value

            csv_data.append(row)

        csv_file = os.path.join(output_dir, f"{model_name}_evaluation_report_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"📋 CSV报告: {csv_file}")

    def print_evaluation_summary(self, summary: Dict, metrics_available: bool, model_name: str = "model"):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print(f"📊 {model_name} 模型评估摘要")
        print("=" * 60)

        print(f"📈 评估样本数: {summary.get('total_samples', 'N/A')}")
        print(f"📊 指标状态: {'完整指标' if metrics_available else '基础指标'}")
        print()

        if metrics_available:
            print("🎯 相似度指标:")
            print(f"   BLEU Score: {summary.get('avg_bleu', 0):.4f} ± {summary.get('std_bleu', 0):.4f}")
            print(f"   ROUGE-1:    {summary.get('avg_rouge1', 0):.4f} ± {summary.get('std_rouge1', 0):.4f}")
            print(f"   ROUGE-2:    {summary.get('avg_rouge2', 0):.4f} ± {summary.get('std_rouge2', 0):.4f}")
            print(f"   ROUGE-L:    {summary.get('avg_rougeL', 0):.4f} ± {summary.get('std_rougeL', 0):.4f}")
        else:
            print("🎯 基础相似度指标:")
            print(f"   Jaccard相似度: {summary.get('avg_jaccard_similarity', 0):.4f}")
            print(f"   重叠率:        {summary.get('avg_overlap_ratio', 0):.4f}")

        print()
        print("⏱️  性能指标:")
        print(f"   平均生成时间: {summary.get('avg_generation_time', 0):.2f}秒")
        print(f"   平均响应长度: {summary.get('avg_response_length', 0):.1f}字符")
        print()

        print("📝 格式正确率:")
        print(f"   推理格式正确率: {summary.get('reasoning_format_rate', 0) * 100:.1f}%")
        print(f"   答案格式正确率: {summary.get('answer_format_rate', 0) * 100:.1f}%")
        print(f"   空响应率:       {summary.get('empty_response_rate', 0) * 100:.1f}%")
        print("=" * 60)


def generate_comparison_report(base_summary: Dict, finetuned_summary: Dict, output_dir: str, timestamp: str):
    """生成基础模型和微调模型的对比报告"""
    print("\n" + "=" * 80)
    print("📊 基础模型 vs 微调模型 对比报告")
    print("=" * 80)

    # 创建对比数据
    comparison = {
        "模型对比": {
            "基础模型": "Base Model",
            "微调模型": "Finetuned Model"
        },
        "核心指标对比": {}
    }

    # 相似度指标对比
    if "avg_bleu" in base_summary and "avg_bleu" in finetuned_summary:
        comparison["核心指标对比"]["BLEU Score"] = {
            "基础模型": f"{base_summary['avg_bleu']:.4f}",
            "微调模型": f"{finetuned_summary['avg_bleu']:.4f}",
            "提升率": f"{((finetuned_summary['avg_bleu'] - base_summary['avg_bleu']) / base_summary['avg_bleu'] * 100):+.1f}%" if
            base_summary['avg_bleu'] > 0 else "N/A"
        }

    if "avg_rouge1" in base_summary and "avg_rouge1" in finetuned_summary:
        comparison["核心指标对比"]["ROUGE-1"] = {
            "基础模型": f"{base_summary['avg_rouge1']:.4f}",
            "微调模型": f"{finetuned_summary['avg_rouge1']:.4f}",
            "提升率": f"{((finetuned_summary['avg_rouge1'] - base_summary['avg_rouge1']) / base_summary['avg_rouge1'] * 100):+.1f}%" if
            base_summary['avg_rouge1'] > 0 else "N/A"
        }

    # 性能指标对比
    comparison["性能指标对比"] = {
        "平均生成时间(秒)": {
            "基础模型": f"{base_summary.get('avg_generation_time', 0):.2f}",
            "微调模型": f"{finetuned_summary.get('avg_generation_time', 0):.2f}",
            "变化率": f"{((finetuned_summary.get('avg_generation_time', 0) - base_summary.get('avg_generation_time', 0)) / base_summary.get('avg_generation_time', 1) * 100):+.1f}%"
        },
        "平均响应长度(字符)": {
            "基础模型": f"{base_summary.get('avg_response_length', 0):.1f}",
            "微调模型": f"{finetuned_summary.get('avg_response_length', 0):.1f}",
            "变化率": f"{((finetuned_summary.get('avg_response_length', 0) - base_summary.get('avg_response_length', 0)) / base_summary.get('avg_response_length', 1) * 100):+.1f}%"
        }
    }

    # 格式指标对比
    comparison["格式指标对比"] = {
        "推理格式正确率": {
            "基础模型": f"{base_summary.get('reasoning_format_rate', 0) * 100:.1f}%",
            "微调模型": f"{finetuned_summary.get('reasoning_format_rate', 0) * 100:.1f}%",
            "提升率": f"{((finetuned_summary.get('reasoning_format_rate', 0) - base_summary.get('reasoning_format_rate', 0)) / max(base_summary.get('reasoning_format_rate', 0.01), 0.01) * 100):+.1f}%"
        },
        "答案格式正确率": {
            "基础模型": f"{base_summary.get('answer_format_rate', 0) * 100:.1f}%",
            "微调模型": f"{finetuned_summary.get('answer_format_rate', 0) * 100:.1f}%",
            "提升率": f"{((finetuned_summary.get('answer_format_rate', 0) - base_summary.get('answer_format_rate', 0)) / max(base_summary.get('answer_format_rate', 0.01), 0.01) * 100):+.1f}%"
        },
        "空响应率": {
            "基础模型": f"{base_summary.get('empty_response_rate', 0) * 100:.1f}%",
            "微调模型": f"{finetuned_summary.get('empty_response_rate', 0) * 100:.1f}%",
            "变化率": f"{((finetuned_summary.get('empty_response_rate', 0) - base_summary.get('empty_response_rate', 0)) / max(base_summary.get('empty_response_rate', 0.01), 0.01) * 100):+.1f}%"
        }
    }

    # 打印对比报告
    for section, metrics in comparison.items():
        if section == "模型对比":
            continue
        print(f"\n{section}:")
        for metric, values in metrics.items():
            print(f"  {metric}:")
            print(f"    基础模型: {values['基础模型']}")
            print(f"    微调模型: {values['微调模型']}")
            print(f"    变化率:   {values['提升率' if '提升率' in values else '变化率']}")

    # 保存对比报告
    comparison_file = os.path.join(output_dir, f"model_comparison_{timestamp}.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 生成CSV格式的对比摘要
    comparison_csv_data = []
    for section, metrics in comparison.items():
        if section == "模型对比":
            continue
        for metric, values in metrics.items():
            row = {
                "指标类别": section,
                "具体指标": metric,
                "基础模型值": values['基础模型'],
                "微调模型值": values['微调模型'],
                "变化率": values['提升率' if '提升率' in values else '变化率']
            }
            comparison_csv_data.append(row)

    comparison_csv_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    df = pd.DataFrame(comparison_csv_data)
    df.to_csv(comparison_csv_file, index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print(f"✅ 对比报告已保存: {comparison_file}")
    print(f"✅ 对比CSV已保存: {comparison_csv_file}")
    print("=" * 80)

    return comparison


def main():
    """主函数"""
    # 配置参数 - 更新路径
    BASE_MODEL = "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    FINETUNED_MODEL_PATH = "/data/ljf/LLM/models/deepseek_r1_7b_lora_psy/best_model"  # 微调模型路径
    TEST_FILE = "/data/ljf/LLM/psydata/sft_r1_val.jsonl"  # 测试数据文件
    OUTPUT_DIR = "./scripts/compare/evaluation_results"  # 输出目录
    NUM_SAMPLES = 10  # 评估样本数量 (None表示全部)

    # 检查测试文件是否存在
    if not os.path.exists(TEST_FILE):
        print(f"❌ 测试文件不存在: {TEST_FILE}")
        print("请检查文件路径是否正确")
        return

    # 检查输出目录是否存在，不存在则创建
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 记录时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 评估基础模型
    print("\n" + "=" * 80)
    print("开始评估基础模型...")
    print("=" * 80)
    try:
        base_evaluator = ModelEvaluator(BASE_MODEL, tokenizer_path=BASE_MODEL, is_base_model=True)
        base_result = base_evaluator.run_comprehensive_evaluation(
            TEST_FILE, OUTPUT_DIR, NUM_SAMPLES, model_name="base_model"
        )
    except Exception as e:
        print(f"基础模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 评估微调模型
    print("\n" + "=" * 80)
    print("开始评估微调模型...")
    print("=" * 80)
    try:
        finetuned_evaluator = ModelEvaluator(FINETUNED_MODEL_PATH, tokenizer_path=BASE_MODEL)
        finetuned_result = finetuned_evaluator.run_comprehensive_evaluation(
            TEST_FILE, OUTPUT_DIR, NUM_SAMPLES, model_name="finetuned_model"
        )
    except Exception as e:
        print(f"微调模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 生成对比报告
    print("\n" + "=" * 80)
    print("生成模型对比报告...")
    print("=" * 80)
    comparison_report = generate_comparison_report(
        base_result["summary"],
        finetuned_result["summary"],
        OUTPUT_DIR,
        timestamp
    )

    # 显示几个对比示例
    print("\n🔍 基础模型 vs 微调模型 输出对比:")
    for i in range(min(3, len(base_result["results"]), len(finetuned_result["results"]))):
        base_res = base_result["results"][i]
        finetuned_res = finetuned_result["results"][i]

        print(f"\n示例 {i + 1}:")
        print(f"问题: {base_res['question']}")
        print(f"\n基础模型回复: {base_res['generated_response']}")
        print(f"\n微调模型回复: {finetuned_res['generated_response']}")
        print("-" * 80)

    print("\n✅ 所有评估完成！")
    print(f"📁 所有结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()