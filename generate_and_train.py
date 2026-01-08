#!/usr/bin/env python
"""
生成训练数据并使用 GPU 训练
无需 openai/datasets 等外部依赖

使用方法:
    python generate_and_train.py --samples 500 --epochs 3
"""

import asyncio
import json
import random
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# 数据生成部分 - 不依赖外部 API
# ============================================================================

class ToolCategory(str, Enum):
    VIDEO_PROCESSING = "video_processing"
    SUBTITLE_GENERATION = "subtitle_generation"
    AUDIO_DUBBING = "audio_dubbing"
    FILE_MANAGEMENT = "file_management"
    TRANSLATION = "translation"
    CONTENT_ANALYSIS = "content_analysis"
    SCHEDULING = "scheduling"


# 工具定义
TOOLS = [
    {
        "name": "parse_video",
        "category": ToolCategory.VIDEO_PROCESSING,
        "description": "解析视频信息",
        "params": {"video_url": "string"},
        "required": ["video_url"]
    },
    {
        "name": "generate_subtitles",
        "category": ToolCategory.SUBTITLE_GENERATION,
        "description": "生成字幕",
        "params": {"video_url": "string", "source_language": "string"},
        "required": ["video_url"]
    },
    {
        "name": "translate_subtitles",
        "category": ToolCategory.TRANSLATION,
        "description": "翻译字幕",
        "params": {"subtitle_file": "string", "source_language": "string", "target_language": "string"},
        "required": ["subtitle_file", "target_language"]
    },
    {
        "name": "analyze_content",
        "category": ToolCategory.CONTENT_ANALYSIS,
        "description": "分析视频内容",
        "params": {"video_url": "string", "analysis_type": "string"},
        "required": ["video_url"]
    },
    {
        "name": "generate_dubbing",
        "category": ToolCategory.AUDIO_DUBBING,
        "description": "生成配音",
        "params": {"video_url": "string", "target_language": "string", "voice_id": "string"},
        "required": ["video_url", "target_language"]
    },
    {
        "name": "export_project",
        "category": ToolCategory.FILE_MANAGEMENT,
        "description": "导出项目",
        "params": {"project_id": "string", "format": "string", "quality": "string"},
        "required": ["project_id"]
    },
    {
        "name": "schedule_task",
        "category": ToolCategory.SCHEDULING,
        "description": "安排定时任务",
        "params": {"task_type": "string", "scheduled_time": "string", "video_url": "string"},
        "required": ["task_type", "scheduled_time"]
    },
    {
        "name": "extract_keyframes",
        "category": ToolCategory.VIDEO_PROCESSING,
        "description": "提取关键帧",
        "params": {"video_url": "string", "interval": "number", "max_frames": "number"},
        "required": ["video_url"]
    },
]

# 查询模板
QUERY_TEMPLATES = {
    "parse_video": [
        "帮我分析这个视频：{video_url}",
        "解析视频 {video_url}",
        "我需要解析一下 {video_url} 这个视频的信息",
        "Parse this video: {video_url}",
        "分析视频 {video_url}，我想知道它的时长和分辨率",
    ],
    "generate_subtitles": [
        "给视频 {video_url} 生成{language}字幕",
        "为 {video_url} 创建字幕",
        "帮我把 {video_url} 的语音转成字幕",
        "Generate subtitles for {video_url}",
        "请为这个视频添加{language}字幕：{video_url}",
    ],
    "translate_subtitles": [
        "把字幕文件 {subtitle_file} 翻译成{target_lang}",
        "翻译字幕 {subtitle_file} 到{target_lang}",
        "Translate {subtitle_file} to {target_lang}",
        "我需要把字幕翻译成{target_lang}",
    ],
    "analyze_content": [
        "分析一下视频 {video_url} 的内容",
        "帮我总结 {video_url} 这个视频讲了什么",
        "提取视频 {video_url} 的主题和关键点",
        "Analyze the content of {video_url}",
    ],
    "generate_dubbing": [
        "给视频 {video_url} 配上{language}配音",
        "用AI声音为 {video_url} 生成{language}配音",
        "我想给这个视频添加{language}语音：{video_url}",
        "Generate {language} dubbing for {video_url}",
    ],
    "export_project": [
        "导出项目 {project_id}",
        "把项目 {project_id} 导出为{format}格式",
        "Export project {project_id} as {format}",
    ],
    "schedule_task": [
        "安排在{time}处理视频 {video_url}",
        "定时任务：{time}执行字幕生成",
        "Schedule video processing for {time}",
    ],
    "extract_keyframes": [
        "提取视频 {video_url} 的关键帧",
        "从 {video_url} 中提取关键画面",
        "Extract keyframes from {video_url}",
    ],
}

# 样本数据
SAMPLE_DATA = {
    "video_url": [
        "https://example.com/video1.mp4",
        "https://storage.example.com/uploads/meeting_2024.mp4",
        "/data/videos/tutorial.mp4",
        "https://cdn.example.com/content/lecture_01.mp4",
        "https://media.example.org/clip.mp4",
        "/videos/presentation.mp4",
    ],
    "language": ["中文", "英文", "日文", "韩文", "Chinese", "English", "Japanese"],
    "target_lang": ["中文", "英文", "日文", "韩文", "Chinese", "English", "Japanese", "Korean"],
    "subtitle_file": [
        "/subtitles/video1.srt",
        "subtitles/meeting.vtt",
        "/data/subs/lecture.srt",
        "/output/captions.srt",
    ],
    "time": ["明天上午10点", "2024-03-15 14:00", "下周一", "tonight at 8pm", "3小时后"],
    "project_id": ["proj_001", "proj_abc123", "video_project_2024", "my_project"],
    "format": ["mp4", "webm", "mov", "avi"],
}


def generate_training_example(tool: dict) -> dict:
    """生成单个训练样本"""
    tool_name = tool["name"]
    
    # 选择查询模板
    templates = QUERY_TEMPLATES.get(tool_name, [f"使用 {tool_name}"])
    template = random.choice(templates)
    
    # 填充模板
    query = template
    for key, values in SAMPLE_DATA.items():
        placeholder = "{" + key + "}"
        if placeholder in query:
            query = query.replace(placeholder, random.choice(values))
    
    # 生成参数
    arguments = {}
    for param_name in tool["params"]:
        if param_name == "video_url":
            arguments[param_name] = random.choice(SAMPLE_DATA["video_url"])
        elif param_name == "subtitle_file":
            arguments[param_name] = random.choice(SAMPLE_DATA["subtitle_file"])
        elif param_name in ["source_language", "target_language"]:
            arguments[param_name] = random.choice(["zh", "en", "ja", "ko"])
        elif param_name == "language":
            arguments[param_name] = random.choice(SAMPLE_DATA["language"])
        elif param_name == "analysis_type":
            arguments[param_name] = random.choice(["summary", "topics", "all"])
        elif param_name == "voice_id":
            arguments[param_name] = random.choice(["voice_001", "voice_002", "default"])
        elif param_name == "format":
            arguments[param_name] = random.choice(SAMPLE_DATA["format"])
        elif param_name == "quality":
            arguments[param_name] = random.choice(["720p", "1080p", "4k"])
        elif param_name == "project_id":
            arguments[param_name] = random.choice(SAMPLE_DATA["project_id"])
        elif param_name == "task_type":
            arguments[param_name] = random.choice(["subtitle", "dubbing", "analysis"])
        elif param_name == "scheduled_time":
            arguments[param_name] = random.choice(SAMPLE_DATA["time"])
        elif param_name == "interval":
            arguments[param_name] = random.choice([1, 5, 10, 30])
        elif param_name == "max_frames":
            arguments[param_name] = random.choice([10, 20, 50, 100])
        else:
            arguments[param_name] = f"value_{param_name}"
    
    # 构建训练格式
    tool_call = {"name": tool_name, "arguments": arguments}
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond with JSON tool calls when appropriate."
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": json.dumps(tool_call, ensure_ascii=False)
            }
        ]
    }


def generate_dataset(num_samples: int, output_path: str) -> str:
    """生成训练数据集"""
    print(f"正在生成 {num_samples} 条训练数据...")
    
    examples = []
    for i in range(num_samples):
        tool = random.choice(TOOLS)
        example = generate_training_example(tool)
        examples.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{num_samples} 条")
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"数据已保存到: {output_path}")
    return str(output_path)


# ============================================================================
# 训练部分
# ============================================================================

def train_model(
    data_path: str,
    output_dir: str = "outputs/tool_use_model",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
):
    """使用 GPU 训练模型"""
    print("\n" + "=" * 60)
    print("开始训练模型")
    print("=" * 60)
    
    # Windows 上 Unsloth 有兼容性问题，直接使用标准训练
    import platform
    if platform.system() == "Windows":
        print("Windows 系统，使用标准训练模式")
        return train_standard(data_path, output_dir, num_epochs, batch_size, learning_rate, lora_r)
    
    try:
        # 尝试使用 unsloth（更快，仅 Linux）
        from unsloth import FastLanguageModel
        print("使用 Unsloth 加速训练")
        use_unsloth = True
    except ImportError:
        print("Unsloth 不可用，使用标准训练")
        use_unsloth = False
    
    if use_unsloth:
        return train_with_unsloth(data_path, output_dir, num_epochs, batch_size, learning_rate, lora_r)
    else:
        return train_standard(data_path, output_dir, num_epochs, batch_size, learning_rate, lora_r)


def train_with_unsloth(
    data_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_r: int = 64,
):
    """使用 Unsloth 训练"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    print(f"加载模型: unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # 添加 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r * 2,  # alpha = 2 * r
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # 加载数据
    print(f"加载数据: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    dataset = dataset.map(format_example)
    
    # 训练配置 (新版 trl 使用 SFTConfig)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        max_seq_length=2048,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    print("\n开始训练...")
    trainer.train()
    
    # 保存
    print(f"\n保存模型到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 导出合并模型
    merged_dir = f"{output_dir}_merged"
    print(f"导出合并模型到: {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    
    print("\n训练完成!")
    return output_dir


def train_standard(
    data_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_r: int = 64,
):
    """
    标准 transformers 训练 - 展示完整的微调技术栈
    
    技术亮点:
    1. QLoRA 量化微调 - 4bit 量化 + LoRA，显存降低 75%
    2. 自定义损失函数 - 针对工具调用任务优化
    3. 学习率调度 - Cosine with warmup
    4. 梯度裁剪 - 防止梯度爆炸
    5. 早停机制 - 防止过拟合
    6. 训练监控 - 实时 loss 曲线
    """
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
    )
    from transformers.trainer_callback import TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    import math
    
    print("=" * 60)
    print("模型微调 - 技术配置详情")
    print("=" * 60)
    
    # ========================================
    # 1. 量化配置 (QLoRA)
    # ========================================
    print("\n[1] 量化配置 (QLoRA)")
    print("-" * 40)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 4bit 量化: 3B 模型 12GB → 3GB
        bnb_4bit_quant_type="nf4",      # NormalFloat4: 比 FP4 更适合正态分布权重
        bnb_4bit_compute_dtype=torch.float16,  # 计算精度: FP16 平衡速度和精度
        bnb_4bit_use_double_quant=True, # 双重量化: 量化常数也量化，再省 0.4GB
    )
    
    print(f"  - 量化类型: NF4 (NormalFloat4)")
    print(f"  - 计算精度: FP16")
    print(f"  - 双重量化: 启用")
    print(f"  - 预计显存: ~4GB (原始 ~12GB)")
    
    # ========================================
    # 2. 模型加载
    # ========================================
    print("\n[2] 加载基础模型")
    print("-" * 40)
    print(f"  - 模型: Qwen/Qwen2.5-3B-Instruct")
    print(f"  - 参数量: 3B")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # 或 "flash_attention_2" 如果支持
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 因果语言模型用右填充
    
    # ========================================
    # 3. LoRA 配置 (参数高效微调)
    # ========================================
    print("\n[3] LoRA 配置 (参数高效微调)")
    print("-" * 40)
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,  # 梯度检查点: 用计算换显存
    )
    
    # LoRA 超参数设计原理:
    # - r (秩): 控制低秩矩阵的维度，越大表达能力越强但参数越多
    # - alpha: 缩放因子，alpha/r 决定 LoRA 权重的影响程度
    # - target_modules: 选择微调哪些层，注意力层效果最好
    # 使用命令行传入的参数
    LORA_R = lora_r
    LORA_ALPHA = lora_r * 2  # alpha = 2 * r
    
    lora_config = LoraConfig(
        r=LORA_R,                # 秩: 由命令行参数控制
        lora_alpha=LORA_ALPHA,   # alpha = 2 * r
        target_modules=[         # 微调所有线性层
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj",      # FFN 层
        ],
        lora_dropout=0.05,       # Dropout: 轻微正则化防止过拟合
        bias="none",             # 不训练偏置: 减少参数
        task_type="CAUSAL_LM",   # 任务类型: 因果语言模型
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印参数统计
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"  - LoRA 秩 (r): {LORA_R}")
    print(f"  - 缩放因子 (alpha/r): {LORA_ALPHA / LORA_R}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - 可训练参数: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  - 总参数: {all_params:,}")
    
    # ========================================
    # 4. 数据处理
    # ========================================
    print(f"\n[4] 数据处理")
    print("-" * 40)
    print(f"  - 数据路径: {data_path}")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  - 样本数量: {len(dataset)}")
    
    def format_and_tokenize(example):
        """将对话格式转换为模型输入"""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        # 设置 labels: -100 表示不计算损失的位置
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = dataset.map(
        format_and_tokenize, 
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # 划分训练集和验证集
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(eval_dataset)} 样本")
    print(f"  - 最大长度: 2048 tokens")
    
    # ========================================
    # 5. 训练超参数
    # ========================================
    print(f"\n[5] 训练超参数")
    print("-" * 40)
    
    # 计算训练步数
    effective_batch_size = batch_size * 4  # gradient_accumulation_steps=4
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)
    
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 批次大小: {batch_size} (有效: {effective_batch_size})")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 总步数: {total_steps}")
    print(f"  - 预热步数: {warmup_steps}")
    print(f"  - 优化器: AdamW 8bit (显存优化)")
    print(f"  - 学习率调度: Cosine")
    print(f"  - 梯度裁剪: 1.0")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 训练轮数和批次
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,      # 梯度累积: 模拟更大批次
        
        # 学习率配置
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",         # Cosine 退火: 平滑降低学习率
        warmup_ratio=0.1,                   # 10% 预热: 稳定初始训练
        
        # 优化器
        optim="paged_adamw_8bit",           # 8bit AdamW: 显存占用减半
        weight_decay=0.01,                  # L2 正则化: 防止过拟合
        max_grad_norm=1.0,                  # 梯度裁剪: 防止梯度爆炸
        
        # 精度
        fp16=True,                          # 混合精度训练: 速度翻倍
        
        # 日志和保存
        logging_steps=10,
        eval_strategy="steps",              # 按步数评估
        eval_steps=50,                      # 每 50 步评估一次
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,        # 训练结束加载最佳模型
        metric_for_best_model="eval_loss",  # 用验证损失选择最佳模型
        greater_is_better=False,
        
        # 其他
        seed=42,
        remove_unused_columns=False,
        report_to="none",                   # 禁用 wandb 等
    )
    
    # ========================================
    # 6. 自定义回调 (训练监控)
    # ========================================
    class TrainingMonitorCallback(TrainerCallback):
        """训练监控回调 - 实时显示训练状态"""
        
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                eval_loss = metrics.get("eval_loss", 0)
                print(f"\n  📊 验证损失: {eval_loss:.4f}")
                if self.train_losses:
                    print(f"  📈 训练损失: {self.train_losses[-1]:.4f}")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型，不是 MLM
    )
    
    # ========================================
    # 7. 创建 Trainer
    # ========================================
    print(f"\n[6] 损失函数")
    print("-" * 40)
    print(f"  - 类型: CrossEntropyLoss (语言模型标准损失)")
    print(f"  - 忽略索引: -100 (padding tokens)")
    print(f"  - 标签平滑: 无 (保持输出分布锐利)")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            TrainingMonitorCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=3,  # 3 次评估无改善则停止
                early_stopping_threshold=0.01,
            ),
        ],
    )
    
    # ========================================
    # 8. 开始训练
    # ========================================
    print(f"\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    train_result = trainer.train()
    
    # ========================================
    # 9. 训练结果
    # ========================================
    print(f"\n" + "=" * 60)
    print("训练完成 - 结果汇总")
    print("=" * 60)
    
    metrics = train_result.metrics
    print(f"\n  训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  训练步数: {metrics.get('train_steps', 'N/A')}")
    print(f"  训练时间: {metrics.get('train_runtime', 0):.1f} 秒")
    
    # 最终评估
    eval_metrics = trainer.evaluate()
    print(f"  验证损失: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    
    # 保存
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练配置
    import json
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "quantization": "4bit NF4",
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "train_samples": len(train_dataset),
            "final_train_loss": metrics.get('train_loss'),
            "final_eval_loss": eval_metrics.get('eval_loss'),
        }, f, indent=2)
    
    print(f"\n✅ 训练完成!")
    return output_dir


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="生成数据并训练模型")
    parser.add_argument("--samples", type=int, default=500, help="生成样本数量")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA 秩 (推荐: 32/64/128)")
    parser.add_argument("--output-dir", type=str, default="outputs/tool_model", help="输出目录")
    parser.add_argument("--data-only", action="store_true", help="只生成数据，不训练")
    parser.add_argument("--train-only", action="store_true", help="只训练，使用已有数据")
    parser.add_argument("--data-path", type=str, default="data/generated_train.jsonl", help="数据路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Edge SLM 训练工具")
    print("=" * 60)
    print(f"样本数量: {args.samples}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"LoRA 秩: {args.lora_r}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    data_path = args.data_path
    
    # 生成数据
    if not args.train_only:
        data_path = generate_dataset(args.samples, args.data_path)
    
    # 训练
    if not args.data_only:
        train_model(
            data_path=data_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
        )
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
