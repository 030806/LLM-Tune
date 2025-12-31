# scripts/train_distill.py
import os
import json
import math
import time
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------- CONFIG --------------
MODEL_NAME = "/data/ljf/LLM/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "/data/ljf/LLM/models/deepseek_r1_7b_lora_0.1"
TRAIN_FILE = "/data//ljf/LLM/data/sft_r1_train.jsonl"
VAL_FILE = "/data//ljf/LLM/data/sft_r1_val.jsonl"

# Hardware / train config - 针对16GB显存优化
MICRO_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
SAVE_STRATEGY = "steps"
LOGGING_STEPS = 20
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

USE_4BIT = False
USE_TEACHER_LOGITS = False
LAMBDA_KL = 0.5

# Checkpoint配置
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
LOGS_DIR = os.path.join(OUTPUT_DIR, "training_logs")
# ------------------------------------

device = "cuda:1" if torch.cuda.is_available() else "cpu"


# ---------------- 训练日志管理器 ----------------
class TrainingLogger:
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)

        # 创建日志文件
        self.training_log_file = os.path.join(logs_dir, "training_log.jsonl")
        self.checkpoint_log_file = os.path.join(logs_dir, "checkpoint_log.csv")
        self.training_summary_file = os.path.join(logs_dir, "training_summary.json")

        # 初始化checkpoint日志
        if not os.path.exists(self.checkpoint_log_file):
            checkpoint_df = pd.DataFrame(columns=[
                'checkpoint_name', 'global_step', 'epoch', 'train_loss',
                'eval_loss', 'learning_rate', 'timestamp', 'is_best'
            ])
            checkpoint_df.to_csv(self.checkpoint_log_file, index=False)

        # 初始化训练摘要
        self.training_summary = {
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_epochs': 0,
            'best_eval_loss': float('inf'),
            'best_checkpoint': None,
            'final_checkpoint': None,
            'training_history': []
        }

    def log_training_step(self, log_data):
        """记录训练步骤日志"""
        log_data['timestamp'] = datetime.now().isoformat()
        with open(self.training_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

    def log_checkpoint(self, checkpoint_info):
        """记录checkpoint信息"""
        # 更新CSV文件
        df = pd.read_csv(self.checkpoint_log_file)
        new_row = pd.DataFrame([checkpoint_info])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.checkpoint_log_file, index=False)

        # 更新训练摘要
        self.training_summary['training_history'].append(checkpoint_info)

        # 如果是最佳模型，更新最佳记录
        if checkpoint_info.get('is_best', False):
            self.training_summary['best_eval_loss'] = checkpoint_info['eval_loss']
            self.training_summary['best_checkpoint'] = checkpoint_info['checkpoint_name']

    def save_training_summary(self, final_info=None):
        """保存训练摘要"""
        self.training_summary['end_time'] = datetime.now().isoformat()
        self.training_summary['total_training_time'] = str(
            datetime.fromisoformat(self.training_summary['end_time']) -
            datetime.fromisoformat(self.training_summary['start_time'])
        )

        if final_info:
            self.training_summary['final_checkpoint'] = final_info.get('checkpoint_name')
            self.training_summary.update(final_info)

        with open(self.training_summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_summary, f, indent=2, ensure_ascii=False)

    def get_training_progress(self):
        """获取训练进度"""
        if os.path.exists(self.training_summary_file):
            with open(self.training_summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self.training_summary


# ---------------- 智能保存回调 ----------------
class SmartSaveCallback(TrainerCallback):
    def __init__(self, output_dir, save_steps=100, save_total_limit=3, logger=None):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.best_eval_loss = float('inf')
        self.checkpoint_history = []
        self.logger = logger or TrainingLogger(LOGS_DIR)

        # 创建目录
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)

        # 加载之前的训练状态
        self.load_training_state()

    def load_training_state(self):
        """加载之前的训练状态"""
        summary = self.logger.get_training_progress()
        if summary['best_eval_loss'] < float('inf'):
            self.best_eval_loss = summary['best_eval_loss']
            print(f"📊 加载之前的最佳评估损失: {self.best_eval_loss:.4f}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """记录训练日志"""
        if logs:
            log_data = {
                'global_step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.logger.log_training_step(log_data)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """每save_steps步保存一次checkpoint"""
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            checkpoint_name = f"checkpoint-{state.global_step}"
            checkpoint_dir = os.path.join(CHECKPOINT_DIR, checkpoint_name)

            # 保存模型和tokenizer
            kwargs['model'].save_pretrained(checkpoint_dir)
            # kwargs['tokenizer'].save_pretrained(checkpoint_dir)

            # 保存训练状态
            training_state = {
                'global_step': state.global_step,
                'epoch': state.epoch,
                'train_loss': state.log_history[-1]['loss'] if state.log_history else None,
                'learning_rate': state.log_history[-1].get('learning_rate', 0) if state.log_history else 0,
            }
            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

            # 保存checkpoint信息
            checkpoint_info = {
                'checkpoint_name': checkpoint_name,
                'global_step': state.global_step,
                'epoch': state.epoch,
                'train_loss': training_state['train_loss'],
                'eval_loss': None,  # 将在评估时更新
                'learning_rate': training_state['learning_rate'],
                'timestamp': datetime.now().isoformat(),
                'is_best': False
            }

            # 记录checkpoint日志
            self.logger.log_checkpoint(checkpoint_info)
            self.checkpoint_history.append((state.global_step, training_state['train_loss'], checkpoint_dir))

            print(f"✅ 检查点已保存: {checkpoint_dir}")
            print(f"📝 训练损失: {training_state['train_loss']:.4f}")

            # 管理checkpoint数量
            self.manage_checkpoints()

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """评估完成后保存最佳模型"""
        if metrics and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']

            # 更新最近checkpoint的评估损失
            if self.checkpoint_history:
                latest_checkpoint = self.checkpoint_history[-1]
                checkpoint_name = f"checkpoint-{latest_checkpoint[0]}"

                # 更新checkpoint日志中的评估损失
                self.update_checkpoint_eval_loss(checkpoint_name, current_loss)

            # 如果当前loss更好，保存最佳模型
            if current_loss < self.best_eval_loss:
                self.best_eval_loss = current_loss

                # 保存最佳模型
                best_checkpoint_dir = BEST_MODEL_DIR
                if os.path.exists(best_checkpoint_dir):
                    import shutil
                    shutil.rmtree(best_checkpoint_dir)

                kwargs['model'].save_pretrained(best_checkpoint_dir)
                # kwargs['tokenizer'].save_pretrained(best_checkpoint_dir)

                # 保存评估信息
                eval_info = {
                    'eval_loss': current_loss,
                    'global_step': state.global_step,
                    'epoch': state.epoch,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                }
                with open(os.path.join(best_checkpoint_dir, "eval_info.json"), 'w') as f:
                    json.dump(eval_info, f, indent=2, ensure_ascii=False)

                # 标记为最佳checkpoint
                if self.checkpoint_history:
                    latest_checkpoint_name = f"checkpoint-{self.checkpoint_history[-1][0]}"
                    self.mark_checkpoint_as_best(latest_checkpoint_name)

                print(f"🎯 新的最佳模型已保存! eval_loss: {current_loss:.4f}")

    def update_checkpoint_eval_loss(self, checkpoint_name, eval_loss):
        """更新checkpoint的评估损失"""
        df = pd.read_csv(self.logger.checkpoint_log_file)
        mask = df['checkpoint_name'] == checkpoint_name
        if mask.any():
            df.loc[mask, 'eval_loss'] = eval_loss
            df.to_csv(self.logger.checkpoint_log_file, index=False)

    def mark_checkpoint_as_best(self, checkpoint_name):
        """标记checkpoint为最佳"""
        df = pd.read_csv(self.logger.checkpoint_log_file)
        # 重置所有的最佳标记
        df['is_best'] = False
        # 设置当前为最佳
        df.loc[df['checkpoint_name'] == checkpoint_name, 'is_best'] = True
        df.to_csv(self.logger.checkpoint_log_file, index=False)

    def manage_checkpoints(self):
        """管理checkpoint数量，只保留最好的几个"""
        if len(self.checkpoint_history) <= self.save_total_limit:
            return

        # 按loss排序，保留最好的
        self.checkpoint_history.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))

        # 删除多余的checkpoint
        while len(self.checkpoint_history) > self.save_total_limit:
            step, loss, path = self.checkpoint_history.pop()
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)
                print(f"🗑️  删除检查点: {path} (loss: {loss:.4f})")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时保存最终摘要"""
        final_info = {
            'checkpoint_name': 'final_model',
            'global_step': state.global_step,
            'epoch': state.epoch,
            'final_train_loss': state.log_history[-1]['loss'] if state.log_history else None,
            'best_eval_loss': self.best_eval_loss
        }
        self.logger.save_training_summary(final_info)


# ---------------- util ----------------
def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def find_latest_checkpoint():
    """查找最新的checkpoint"""
    if not os.path.exists(CHECKPOINT_DIR):
        return None

    checkpoints = []
    for item in os.listdir(CHECKPOINT_DIR):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, item)
        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, checkpoint_path))
            except:
                continue

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]
    return None


# ---------------- dataset loading & tokenization ----------------
def preprocess(dataset_items, tokenizer, max_length=1024):
    inputs = []
    for it in dataset_items:
        inp = it["input"]
        out = it["output"]
        text = inp + "\n" + out
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        input_ids = tok["input_ids"]
        inputs.append({"input_ids": input_ids, "attention_mask": tok["attention_mask"]})
    return inputs


# ---------------- collator ----------------
@dataclass
class DataCollatorForCausal:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id or 0)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


# ---------------- main ----------------
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化日志系统
    logger = TrainingLogger(LOGS_DIR)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model -- this may take a while")

    # 检查是否有之前的checkpoint
    resume_from_checkpoint = find_latest_checkpoint()
    if resume_from_checkpoint:
        print(f"🎯 发现检查点，从 {resume_from_checkpoint} 恢复训练")

    # load with 4bit if requested
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME if not resume_from_checkpoint else resume_from_checkpoint,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME if not resume_from_checkpoint else resume_from_checkpoint,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # apply LoRA via PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load data
    train_items = read_jsonl(TRAIN_FILE)
    val_items = read_jsonl(VAL_FILE)

    print(f"训练数据: {len(train_items)} 条")
    print(f"验证数据: {len(val_items)} 条")

    train_tok = preprocess(train_items, tokenizer)
    val_tok = preprocess(val_items, tokenizer)

    collator = DataCollatorForCausal(tokenizer=tokenizer)

    # 计算总步数和保存步数
    total_steps = (len(train_tok) * EPOCHS) // (MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
    save_steps = max(100, len(train_tok) // (MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS))

    print(f"总训练步数: {total_steps}")
    print(f"保存步数: {save_steps}")

    # training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,
        save_total_limit=1,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # 初始化callback
    smart_save_callback = SmartSaveCallback(
        OUTPUT_DIR,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logger=logger
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[smart_save_callback]
    )

    # 训练
    print("开始训练...")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    training_time = time.time() - start_time

    # 保存最终模型
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # 记录最终训练信息
    final_checkpoint_info = {
        'checkpoint_name': 'final_model',
        'global_step': trainer.state.global_step,
        'epoch': trainer.state.epoch,
        'train_loss': trainer.state.log_history[-1]['loss'] if trainer.state.log_history else None,
        'eval_loss': smart_save_callback.best_eval_loss,
        'learning_rate': trainer.state.log_history[-1].get('learning_rate', 0) if trainer.state.log_history else 0,
        'timestamp': datetime.now().isoformat(),
        'is_best': False,
        'training_time_seconds': training_time
    }
    logger.log_checkpoint(final_checkpoint_info)
    logger.save_training_summary(final_checkpoint_info)

    print("🎉 训练完成!")
    print(f"训练总时间: {training_time:.2f} 秒")
    print(f"最终模型保存至: {final_model_dir}")
    print(f"最佳模型保存至: {BEST_MODEL_DIR}")
    print(f"训练日志保存至: {LOGS_DIR}")


if __name__ == "__main__":
    main()  # scripts/train_distill.py
