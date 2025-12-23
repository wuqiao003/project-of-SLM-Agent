"""
GPT-4 Distillation module for generating high-quality Tool-Use training data.
Uses GPT-4 as a teacher model to generate diverse, high-quality examples.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

from openai import AsyncOpenAI
from rich.progress import Progress, TaskID
from rich.console import Console

from edge_slm.data.schema import (
    ToolDefinition,
    ToolCall,
    ToolUseExample,
    ToolCategory,
    LIGHT_ON_TOOLS,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DistillationConfig:
    """Configuration for GPT-4 distillation."""
    teacher_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1024
    num_samples: int = 1000
    batch_size: int = 10
    output_dir: str = "data/distilled"
    include_reasoning: bool = True
    
    # Diversity settings
    complexity_distribution: dict = None
    
    def __post_init__(self):
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                "simple": 0.4,
                "medium": 0.4,
                "complex": 0.2,
            }


# Query templates for different scenarios
QUERY_TEMPLATES = {
    ToolCategory.VIDEO_PROCESSING: [
        "帮我分析这个视频：{video_url}",
        "我需要解析一下 {video_url} 这个视频的信息",
        "请提取 {video_url} 视频的关键帧",
        "分析视频 {video_url}，我想知道它的时长和分辨率",
        "Parse this video for me: {video_url}",
    ],
    ToolCategory.SUBTITLE_GENERATION: [
        "给视频 {video_url} 生成{language}字幕",
        "我需要为 {video_url} 创建字幕，视频是{language}的",
        "帮我把 {video_url} 的语音转成字幕",
        "Generate subtitles for {video_url} in {language}",
        "请为这个视频添加{language}字幕：{video_url}",
    ],
    ToolCategory.TRANSLATION: [
        "把这个字幕文件翻译成{target_lang}：{subtitle_file}",
        "翻译字幕 {subtitle_file}，从{source_lang}到{target_lang}",
        "Translate {subtitle_file} to {target_lang}",
        "我需要把{source_lang}字幕翻译成{target_lang}",
    ],
    ToolCategory.AUDIO_DUBBING: [
        "给视频 {video_url} 配上{language}配音",
        "用AI声音为 {video_url} 生成{language}配音",
        "我想给这个视频添加{language}语音：{video_url}",
        "Generate {language} dubbing for {video_url}",
    ],
    ToolCategory.CONTENT_ANALYSIS: [
        "分析一下视频 {video_url} 的内容",
        "帮我总结 {video_url} 这个视频讲了什么",
        "提取视频 {video_url} 的主题和关键点",
        "Analyze the content of {video_url}",
    ],
    ToolCategory.SCHEDULING: [
        "安排在{time}处理视频 {video_url}",
        "定时任务：{time}执行字幕生成",
        "Schedule video processing for {time}",
    ],
    ToolCategory.FILE_MANAGEMENT: [
        "导出项目 {project_id}，格式{format}",
        "把项目 {project_id} 导出为{quality}的{format}",
        "Export project {project_id} as {format}",
    ],
}

# Sample data for template filling
SAMPLE_DATA = {
    "video_url": [
        "https://example.com/video1.mp4",
        "https://storage.example.com/uploads/meeting_2024.mp4",
        "/data/videos/tutorial.mp4",
        "https://cdn.example.com/content/lecture_01.mp4",
    ],
    "language": ["中文", "英文", "日文", "韩文", "Chinese", "English", "Japanese"],
    "source_lang": ["中文", "英文", "日文", "Chinese", "English"],
    "target_lang": ["中文", "英文", "日文", "韩文", "Chinese", "English", "Japanese", "Korean"],
    "subtitle_file": [
        "/subtitles/video1.srt",
        "subtitles/meeting.vtt",
        "/data/subs/lecture.srt",
    ],
    "time": [
        "明天上午10点",
        "2024-03-15 14:00",
        "下周一",
        "tonight at 8pm",
    ],
    "project_id": ["proj_001", "proj_abc123", "video_project_2024"],
    "format": ["mp4", "webm", "mov"],
    "quality": ["720p", "1080p", "4k"],
}


class GPT4Distiller:
    """
    Distiller that uses GPT-4 to generate high-quality Tool-Use training examples.
    """
    
    def __init__(
        self,
        config: DistillationConfig,
        api_key: Optional[str] = None,
    ):
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key)
        self.tools = LIGHT_ON_TOOLS
        
    def _generate_query(self, category: ToolCategory) -> str:
        """Generate a random query from templates."""
        templates = QUERY_TEMPLATES.get(category, QUERY_TEMPLATES[ToolCategory.VIDEO_PROCESSING])
        template = random.choice(templates)
        
        # Fill in template variables
        filled = template
        for key, values in SAMPLE_DATA.items():
            if "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}", random.choice(values))
        
        return filled
    
    def _select_tools_for_query(self, complexity: str) -> list[ToolDefinition]:
        """Select relevant tools based on complexity."""
        if complexity == "simple":
            # Single tool scenario
            return [random.choice(self.tools)]
        elif complexity == "medium":
            # 2-3 related tools
            category = random.choice(list(ToolCategory))
            related = [t for t in self.tools if t.category == category]
            if len(related) < 2:
                related = random.sample(self.tools, min(3, len(self.tools)))
            return related[:3]
        else:
            # Complex: multiple tools, potentially chained
            return random.sample(self.tools, min(5, len(self.tools)))
    
    async def _generate_single_example(
        self,
        complexity: str,
        category: Optional[ToolCategory] = None,
    ) -> Optional[ToolUseExample]:
        """Generate a single training example using GPT-4."""
        
        if category is None:
            category = random.choice(list(ToolCategory))
        
        # Generate query and select tools
        query = self._generate_query(category)
        tools = self._select_tools_for_query(complexity)
        
        # Build prompt for GPT-4
        tools_json = json.dumps(
            [t.to_openai_format() for t in tools],
            ensure_ascii=False,
            indent=2
        )
        
        system_prompt = """You are an expert at generating training data for tool-use AI models.
Given a user query and available tools, generate:
1. The appropriate tool call(s) as JSON
2. Brief reasoning explaining why this tool was chosen

Respond in this exact JSON format:
{
    "reasoning": "Brief explanation of intent recognition and tool selection",
    "tool_calls": [
        {"name": "tool_name", "arguments": {"param1": "value1", ...}}
    ]
}

Rules:
- Always use valid parameter values matching the tool schema
- For simple queries, use exactly 1 tool call
- For complex queries, you may chain multiple tool calls
- Arguments must match the parameter types defined in the schema
- Use realistic, plausible values for arguments"""

        user_prompt = f"""User Query: {query}

Available Tools:
{tools_json}

Generate the appropriate tool call(s) for this query. Complexity level: {complexity}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Parse tool calls
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
                source="gpt4_distillation",
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate example: {e}")
            return None
    
    async def generate_batch(
        self,
        batch_size: int,
        complexity: str,
    ) -> list[ToolUseExample]:
        """Generate a batch of examples concurrently."""
        tasks = [
            self._generate_single_example(complexity)
            for _ in range(batch_size)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        examples = []
        for r in results:
            if isinstance(r, ToolUseExample):
                examples.append(r)
        
        return examples
    
    async def generate_dataset(
        self,
        progress_callback: Optional[callable] = None,
    ) -> list[ToolUseExample]:
        """
        Generate the full dataset according to configuration.
        
        Returns:
            List of ToolUseExample objects
        """
        all_examples = []
        
        # Calculate samples per complexity level
        samples_per_complexity = {
            k: int(v * self.config.num_samples)
            for k, v in self.config.complexity_distribution.items()
        }
        
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Generating training data...",
                total=self.config.num_samples
            )
            
            for complexity, num_samples in samples_per_complexity.items():
                num_batches = (num_samples + self.config.batch_size - 1) // self.config.batch_size
                
                for _ in range(num_batches):
                    batch_size = min(self.config.batch_size, num_samples - len([
                        e for e in all_examples if e.complexity == complexity
                    ]))
                    
                    if batch_size <= 0:
                        break
                    
                    examples = await self.generate_batch(batch_size, complexity)
                    all_examples.extend(examples)
                    
                    progress.update(task, advance=len(examples))
                    
                    if progress_callback:
                        progress_callback(len(all_examples), self.config.num_samples)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
        
        console.print(f"[green]Generated {len(all_examples)} training examples[/green]")
        return all_examples
    
    def save_dataset(
        self,
        examples: list[ToolUseExample],
        output_path: Optional[str] = None,
    ) -> Path:
        """Save generated dataset to disk."""
        output_path = Path(output_path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL for training
        jsonl_path = output_path / "tool_use_train.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in examples:
                training_format = example.to_training_format(
                    include_reasoning=self.config.include_reasoning
                )
                f.write(json.dumps(training_format, ensure_ascii=False) + "\n")
        
        # Save raw examples for reference
        raw_path = output_path / "tool_use_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                [json.loads(e.to_json()) for e in examples],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # Save statistics
        stats = {
            "total_examples": len(examples),
            "by_complexity": {},
            "by_category": {},
        }
        for e in examples:
            stats["by_complexity"][e.complexity] = stats["by_complexity"].get(e.complexity, 0) + 1
            stats["by_category"][e.category.value] = stats["by_category"].get(e.category.value, 0) + 1
        
        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        console.print(f"[green]Dataset saved to {output_path}[/green]")
        console.print(f"  - Training data: {jsonl_path}")
        console.print(f"  - Raw examples: {raw_path}")
        console.print(f"  - Statistics: {stats_path}")
        
        return output_path


async def run_distillation(
    num_samples: int = 1000,
    output_dir: str = "data/distilled",
    api_key: Optional[str] = None,
) -> Path:
    """
    Convenience function to run the full distillation pipeline.
    
    Args:
        num_samples: Number of training examples to generate
        output_dir: Output directory for the dataset
        api_key: OpenAI API key (uses env var if not provided)
    
    Returns:
        Path to the output directory
    """
    config = DistillationConfig(
        num_samples=num_samples,
        output_dir=output_dir,
    )
    
    distiller = GPT4Distiller(config, api_key=api_key)
    examples = await distiller.generate_dataset()
    return distiller.save_dataset(examples)
