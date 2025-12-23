"""
Local Model Distillation - 使用本地部署的 Qwen2.5 替代 OpenAI API
支持多种本地部署方式：Ollama、vLLM、Transformers 直接加载

使用方法:
1. Ollama: ollama run qwen2.5:14b
2. vLLM: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct
3. 直接加载: 使用 transformers 本地推理
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import logging

from rich.progress import Progress
from rich.console import Console

from edge_slm.data.schema import (
    ToolDefinition,
    ToolCall,
    ToolUseExample,
    ToolCategory,
    LIGHT_ON_TOOLS,
)
from edge_slm.data.distiller import QUERY_TEMPLATES, SAMPLE_DATA

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class LocalDistillationConfig:
    """本地模型蒸馏配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"  # 或 "qwen2.5:14b" for Ollama
    backend: Literal["ollama", "vllm", "transformers", "openai_compatible"] = "ollama"
    
    # API 端点 (用于 Ollama/vLLM)
    api_base: str = "http://localhost:11434/v1"  # Ollama 默认
    # api_base: str = "http://localhost:8000/v1"  # vLLM 默认
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # 数据集配置
    num_samples: int = 100
    batch_size: int = 5  # 本地模型建议小批次
    output_dir: str = "data/local_distilled"
    include_reasoning: bool = True
    
    # 复杂度分布
    complexity_distribution: dict = field(default_factory=lambda: {
        "simple": 0.5,
        "medium": 0.35,
        "complex": 0.15,
    })


class LocalModelDistiller:
    """
    使用本地部署的大模型生成训练数据
    
    支持的后端:
    - ollama: 使用 Ollama 本地服务
    - vllm: 使用 vLLM OpenAI 兼容 API
    - transformers: 直接加载模型
    - openai_compatible: 任何 OpenAI 兼容的 API
    """
    
    def __init__(self, config: LocalDistillationConfig):
        self.config = config
        self.tools = LIGHT_ON_TOOLS
        self.client = None
        self.model = None
        self.tokenizer = None
        
        self._setup_backend()
    
    def _setup_backend(self):
        """根据配置初始化后端"""
        if self.config.backend in ["ollama", "vllm", "openai_compatible"]:
            self._setup_openai_compatible()
        elif self.config.backend == "transformers":
            self._setup_transformers()
    
    def _setup_openai_compatible(self):
        """设置 OpenAI 兼容的客户端"""
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI(
            base_url=self.config.api_base,
            api_key="not-needed",  # 本地服务通常不需要 key
        )
        
        console.print(f"[green]已连接到本地服务: {self.config.api_base}[/green]")
    
    def _setup_transformers(self):
        """直接加载 transformers 模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        console.print(f"[cyan]正在加载模型: {self.config.model_name}[/cyan]")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        console.print("[green]模型加载完成[/green]")
    
    def _generate_query(self, category: ToolCategory) -> str:
        """生成随机查询"""
        templates = QUERY_TEMPLATES.get(category, QUERY_TEMPLATES[ToolCategory.VIDEO_PROCESSING])
        template = random.choice(templates)
        
        filled = template
        for key, values in SAMPLE_DATA.items():
            if "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}", random.choice(values))
        
        return filled
    
    def _select_tools_for_query(self, complexity: str) -> list[ToolDefinition]:
        """根据复杂度选择工具"""
        if complexity == "simple":
            return [random.choice(self.tools)]
        elif complexity == "medium":
            category = random.choice(list(ToolCategory))
            related = [t for t in self.tools if t.category == category]
            if len(related) < 2:
                related = random.sample(self.tools, min(3, len(self.tools)))
            return related[:3]
        else:
            return random.sample(self.tools, min(5, len(self.tools)))
    
    def _build_prompt(self, query: str, tools: list[ToolDefinition], complexity: str) -> tuple[str, str]:
        """构建提示词"""
        tools_json = json.dumps(
            [t.to_openai_format() for t in tools],
            ensure_ascii=False,
            indent=2
        )
        
        system_prompt = """你是一个专业的工具调用数据生成专家。
给定用户查询和可用工具，你需要生成：
1. 合适的工具调用（JSON格式）
2. 简短的推理说明

请严格按照以下JSON格式回复：
{
    "reasoning": "简要说明意图识别和工具选择的原因",
    "tool_calls": [
        {"name": "工具名称", "arguments": {"参数1": "值1", ...}}
    ]
}

规则：
- 参数值必须符合工具定义的类型
- 简单查询只使用1个工具
- 复杂查询可以使用多个工具
- 使用真实合理的参数值
- 只输出JSON，不要有其他内容"""

        user_prompt = f"""用户查询: {query}

可用工具:
{tools_json}

复杂度级别: {complexity}

请生成合适的工具调用。"""

        return system_prompt, user_prompt
    
    async def _generate_with_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[dict]:
        """使用 API 生成"""
        try:
            # 根据后端选择模型名称
            if self.config.backend == "ollama":
                model = self.config.model_name.replace("/", ":")  # Qwen/Qwen2.5-14B -> qwen2.5:14b
                if "Qwen" in model:
                    model = "qwen2.5:14b"  # Ollama 模型名
            else:
                model = self.config.model_name
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            content = response.choices[0].message.content
            
            # 提取 JSON
            return self._extract_json(content)
            
        except Exception as e:
            logger.warning(f"API 调用失败: {e}")
            return None
    
    def _generate_with_transformers(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[dict]:
        """使用 transformers 直接生成"""
        import torch
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return self._extract_json(generated)
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON"""
        import re
        
        # 方法1: 直接解析
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        # 方法2: 修复常见错误
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                json_str = match.group()
                # 修复单引号
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                # 修复尾随逗号
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None
    
    async def _generate_single_example(
        self,
        complexity: str,
        category: Optional[ToolCategory] = None,
    ) -> Optional[ToolUseExample]:
        """生成单个训练样本"""
        if category is None:
            category = random.choice(list(ToolCategory))
        
        query = self._generate_query(category)
        tools = self._select_tools_for_query(complexity)
        system_prompt, user_prompt = self._build_prompt(query, tools, complexity)
        
        # 根据后端选择生成方式
        if self.config.backend == "transformers":
            result = self._generate_with_transformers(system_prompt, user_prompt)
        else:
            result = await self._generate_with_api(system_prompt, user_prompt)
        
        if not result or "tool_calls" not in result:
            return None
        
        try:
            tool_calls = [
                ToolCall(name=tc["name"], arguments=tc["arguments"])
                for tc in result.get("tool_calls", [])
            ]
            
            if not tool_calls:
                return None
            
            return ToolUseExample(
                user_query=query,
                available_tools=tools,
                tool_calls=tool_calls,
                reasoning=result.get("reasoning"),
                category=category,
                complexity=complexity,
                source=f"local_{self.config.backend}",
            )
        except Exception as e:
            logger.warning(f"解析失败: {e}")
            return None
    
    async def generate_dataset(self) -> list[ToolUseExample]:
        """生成完整数据集"""
        all_examples = []
        
        samples_per_complexity = {
            k: int(v * self.config.num_samples)
            for k, v in self.config.complexity_distribution.items()
        }
        
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]使用本地模型生成训练数据...",
                total=self.config.num_samples
            )
            
            for complexity, num_samples in samples_per_complexity.items():
                generated = 0
                retries = 0
                max_retries = num_samples * 2
                
                while generated < num_samples and retries < max_retries:
                    example = await self._generate_single_example(complexity)
                    
                    if example:
                        all_examples.append(example)
                        generated += 1
                        progress.update(task, advance=1)
                    
                    retries += 1
                    
                    # 避免过快请求
                    if self.config.backend != "transformers":
                        await asyncio.sleep(0.1)
        
        console.print(f"[green]生成了 {len(all_examples)} 条训练样本[/green]")
        return all_examples
    
    def save_dataset(self, examples: list[ToolUseExample]) -> Path:
        """保存数据集"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSONL 格式
        jsonl_path = output_path / "tool_use_train.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in examples:
                training_format = example.to_training_format(
                    include_reasoning=self.config.include_reasoning
                )
                f.write(json.dumps(training_format, ensure_ascii=False) + "\n")
        
        # 保存原始数据
        raw_path = output_path / "tool_use_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                [json.loads(e.to_json()) for e in examples],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 保存统计信息
        stats = {
            "total_examples": len(examples),
            "model": self.config.model_name,
            "backend": self.config.backend,
            "by_complexity": {},
            "by_category": {},
        }
        for e in examples:
            stats["by_complexity"][e.complexity] = stats["by_complexity"].get(e.complexity, 0) + 1
            stats["by_category"][e.category.value] = stats["by_category"].get(e.category.value, 0) + 1
        
        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]数据集已保存到 {output_path}[/green]")
        console.print(f"  - 训练数据: {jsonl_path}")
        console.print(f"  - 原始数据: {raw_path}")
        console.print(f"  - 统计信息: {stats_path}")
        
        return output_path


async def run_local_distillation(
    num_samples: int = 100,
    output_dir: str = "data/local_distilled",
    backend: str = "ollama",
    api_base: str = "http://localhost:11434/v1",
    model_name: str = "qwen2.5:14b",
) -> Path:
    """
    使用本地模型运行数据蒸馏
    
    Args:
        num_samples: 生成样本数量
        output_dir: 输出目录
        backend: 后端类型 (ollama/vllm/transformers)
        api_base: API 地址
        model_name: 模型名称
    
    Returns:
        输出目录路径
    """
    config = LocalDistillationConfig(
        num_samples=num_samples,
        output_dir=output_dir,
        backend=backend,
        api_base=api_base,
        model_name=model_name,
    )
    
    distiller = LocalModelDistiller(config)
    examples = await distiller.generate_dataset()
    return distiller.save_dataset(examples)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用本地模型生成训练数据")
    parser.add_argument("--num-samples", type=int, default=100, help="生成样本数量")
    parser.add_argument("--output-dir", type=str, default="data/local_distilled", help="输出目录")
    parser.add_argument("--backend", type=str, default="ollama", 
                       choices=["ollama", "vllm", "transformers"], help="后端类型")
    parser.add_argument("--api-base", type=str, default="http://localhost:11434/v1", help="API 地址")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="模型名称")
    
    args = parser.parse_args()
    
    asyncio.run(run_local_distillation(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        backend=args.backend,
        api_base=args.api_base,
        model_name=args.model,
    ))
