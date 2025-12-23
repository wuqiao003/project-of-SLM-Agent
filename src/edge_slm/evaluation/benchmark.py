"""
Benchmark suite for evaluating tool-use models.

Compares:
- Local SLM vs Cloud API performance
- With vs without structured decoding
- Different quantization levels
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from edge_slm.evaluation.metrics import ToolUseMetrics, compute_metrics, format_metrics_report
from edge_slm.data.schema import LIGHT_ON_TOOLS, ToolDefinition

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    
    # Test data
    test_data_path: Optional[str] = None
    num_samples: int = 100
    
    # Models to compare
    local_model_path: Optional[str] = None
    cloud_model: str = "gpt-4-turbo-preview"
    
    # Test settings
    use_structured_decoding: bool = True
    warmup_samples: int = 5
    
    # Output
    output_dir: str = "benchmark_results"
    save_predictions: bool = True


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    
    model_name: str
    metrics: ToolUseMetrics
    config: dict = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "metrics": {
                "tool_selection_accuracy": self.metrics.tool_selection_accuracy,
                "argument_accuracy": self.metrics.argument_accuracy,
                "exact_match_accuracy": self.metrics.exact_match_accuracy,
                "json_validity_rate": self.metrics.json_validity_rate,
                "schema_compliance_rate": self.metrics.schema_compliance_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "p50_latency_ms": self.metrics.p50_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "p99_latency_ms": self.metrics.p99_latency_ms,
                "error_rate": self.metrics.error_rate,
                "total_samples": self.metrics.total_samples,
            },
            "config": self.config,
            "num_errors": len(self.errors),
        }


class ToolUseBenchmark:
    """
    Comprehensive benchmark suite for tool-use models.
    
    Evaluates:
    1. Accuracy: Tool selection and argument generation
    2. Format compliance: JSON validity and schema adherence
    3. Latency: End-to-end inference time
    4. Reliability: Error rates and retry requirements
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.test_data: list[dict] = []
        self.results: list[BenchmarkResult] = []
    
    def load_test_data(self) -> None:
        """Load test data from file or generate synthetic data."""
        if self.config.test_data_path:
            path = Path(self.config.test_data_path)
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    self.test_data = [json.loads(line) for line in f if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self.test_data = json.load(f)
        else:
            # Generate synthetic test data
            self.test_data = self._generate_synthetic_data()
        
        # Limit to num_samples
        self.test_data = self.test_data[:self.config.num_samples]
        
        console.print(f"[green]Loaded {len(self.test_data)} test samples[/green]")
    
    def _generate_synthetic_data(self) -> list[dict]:
        """Generate synthetic test data for benchmarking."""
        import random
        
        test_cases = []
        tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS]
        
        # Sample queries for each tool
        queries = {
            "parse_video": [
                ("分析视频 https://example.com/video.mp4", {"video_url": "https://example.com/video.mp4"}),
                ("帮我解析这个视频的信息：https://cdn.test.com/v1.mp4", {"video_url": "https://cdn.test.com/v1.mp4"}),
            ],
            "generate_subtitles": [
                ("给视频生成中文字幕：https://example.com/video.mp4", {"video_url": "https://example.com/video.mp4", "source_language": "zh"}),
                ("为 https://test.com/v.mp4 创建英文字幕", {"video_url": "https://test.com/v.mp4", "source_language": "en"}),
            ],
            "translate_subtitles": [
                ("把字幕 /subs/v1.srt 翻译成日文", {"subtitle_file": "/subs/v1.srt", "target_language": "ja", "source_language": "zh"}),
            ],
            "generate_dubbing": [
                ("给视频配上中文配音：https://example.com/v.mp4", {"video_url": "https://example.com/v.mp4", "target_language": "zh"}),
            ],
            "analyze_content": [
                ("分析视频内容：https://example.com/v.mp4", {"video_url": "https://example.com/v.mp4", "analysis_type": "all"}),
            ],
        }
        
        for tool_name, examples in queries.items():
            for query, expected_args in examples:
                test_cases.append({
                    "query": query,
                    "tools": tools,
                    "expected": {
                        "name": tool_name,
                        "arguments": expected_args,
                    }
                })
        
        # Duplicate to reach num_samples
        while len(test_cases) < self.config.num_samples:
            test_cases.extend(test_cases[:self.config.num_samples - len(test_cases)])
        
        random.shuffle(test_cases)
        return test_cases[:self.config.num_samples]
    
    async def benchmark_local(
        self,
        engine: Any,
        name: str = "Local SLM",
    ) -> BenchmarkResult:
        """
        Benchmark local model performance.
        
        Args:
            engine: Inference engine
            name: Model name for reporting
        
        Returns:
            BenchmarkResult
        """
        console.print(f"\n[bold blue]Benchmarking: {name}[/bold blue]")
        
        predictions = []
        references = []
        latencies = []
        errors = []
        
        # Warmup
        console.print(f"Warming up with {self.config.warmup_samples} samples...")
        for i in range(min(self.config.warmup_samples, len(self.test_data))):
            sample = self.test_data[i]
            _ = engine.generate(
                self._build_prompt(sample["query"], sample["tools"]),
                tools=sample["tools"],
            )
        
        # Benchmark
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Running {name}...", total=len(self.test_data))
            
            for sample in self.test_data:
                try:
                    result = engine.generate(
                        self._build_prompt(sample["query"], sample["tools"]),
                        tools=sample["tools"],
                    )
                    
                    latencies.append(result.latency_ms)
                    
                    if result.is_valid and result.parsed:
                        predictions.append(result.parsed)
                    else:
                        predictions.append({})
                        errors.append(result.error or "Invalid output")
                    
                    references.append(sample["expected"])
                    
                except Exception as e:
                    predictions.append({})
                    references.append(sample["expected"])
                    errors.append(str(e))
                
                progress.update(task, advance=1)
        
        # Compute metrics
        metrics = compute_metrics(predictions, references, latencies)
        
        result = BenchmarkResult(
            model_name=name,
            metrics=metrics,
            config={
                "structured_decoding": self.config.use_structured_decoding,
                "num_samples": len(self.test_data),
            },
            predictions=predictions if self.config.save_predictions else [],
            errors=errors,
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_cloud(
        self,
        client: Any,
        name: str = "Cloud API",
    ) -> BenchmarkResult:
        """
        Benchmark cloud API performance.
        
        Args:
            client: OpenAI client
            name: Model name for reporting
        
        Returns:
            BenchmarkResult
        """
        console.print(f"\n[bold blue]Benchmarking: {name}[/bold blue]")
        
        predictions = []
        references = []
        latencies = []
        errors = []
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Running {name}...", total=len(self.test_data))
            
            for sample in self.test_data:
                try:
                    start = time.time()
                    
                    response = await client.chat.completions.create(
                        model=self.config.cloud_model,
                        messages=[{"role": "user", "content": sample["query"]}],
                        tools=[{"type": "function", "function": t.get("function", t)} 
                               for t in sample["tools"]],
                        tool_choice="auto",
                    )
                    
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)
                    
                    if response.choices[0].message.tool_calls:
                        tc = response.choices[0].message.tool_calls[0]
                        predictions.append({
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        })
                    else:
                        predictions.append({})
                        errors.append("No tool call in response")
                    
                    references.append(sample["expected"])
                    
                except Exception as e:
                    predictions.append({})
                    references.append(sample["expected"])
                    errors.append(str(e))
                    latencies.append(0)
                
                progress.update(task, advance=1)
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        metrics = compute_metrics(predictions, references, latencies)
        
        result = BenchmarkResult(
            model_name=name,
            metrics=metrics,
            config={
                "model": self.config.cloud_model,
                "num_samples": len(self.test_data),
            },
            predictions=predictions if self.config.save_predictions else [],
            errors=errors,
        )
        
        self.results.append(result)
        return result
    
    def _build_prompt(self, query: str, tools: list[dict]) -> str:
        """Build prompt for inference."""
        tools_desc = "\n".join([
            f"- {t.get('function', t)['name']}: {t.get('function', t)['description']}"
            for t in tools
        ])
        
        return f"""<|im_start|>system
You are a helpful AI assistant. Use the following tools:

{tools_desc}

Respond with JSON: {{"name": "tool_name", "arguments": {{...}}}}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    
    def compare_results(self) -> None:
        """Print comparison table of all benchmark results."""
        if not self.results:
            console.print("[yellow]No results to compare[/yellow]")
            return
        
        table = Table(title="Benchmark Comparison")
        
        table.add_column("Model", style="cyan")
        table.add_column("Tool Acc", justify="right")
        table.add_column("Exact Match", justify="right")
        table.add_column("JSON Valid", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("P95 Latency", justify="right")
        table.add_column("Error Rate", justify="right")
        
        for result in self.results:
            m = result.metrics
            table.add_row(
                result.model_name,
                f"{m.tool_selection_accuracy:.1%}",
                f"{m.exact_match_accuracy:.1%}",
                f"{m.json_validity_rate:.1%}",
                f"{m.avg_latency_ms:.1f}ms",
                f"{m.p95_latency_ms:.1f}ms",
                f"{m.error_rate:.1%}",
            )
        
        console.print(table)
        
        # Calculate improvements
        if len(self.results) >= 2:
            local = self.results[0]
            cloud = self.results[1]
            
            latency_improvement = (cloud.metrics.avg_latency_ms - local.metrics.avg_latency_ms) / cloud.metrics.avg_latency_ms * 100
            
            console.print(f"\n[bold green]Latency Improvement: {latency_improvement:.1f}%[/bold green]")
    
    def save_results(self) -> None:
        """Save benchmark results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "config": {
                "num_samples": self.config.num_samples,
                "use_structured_decoding": self.config.use_structured_decoding,
            },
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        for result in self.results:
            name = result.model_name.lower().replace(" ", "_")
            with open(output_dir / f"{name}_result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        
        console.print(f"[green]Results saved to {output_dir}[/green]")


async def run_benchmark(
    local_model_path: str,
    test_data_path: Optional[str] = None,
    num_samples: int = 100,
    compare_cloud: bool = False,
    cloud_api_key: Optional[str] = None,
) -> list[BenchmarkResult]:
    """
    Run the benchmark suite.
    
    Args:
        local_model_path: Path to local model
        test_data_path: Path to test data
        num_samples: Number of samples to test
        compare_cloud: Whether to compare with cloud API
        cloud_api_key: OpenAI API key for cloud comparison
    
    Returns:
        List of BenchmarkResult objects
    """
    from edge_slm.inference import create_engine
    
    config = BenchmarkConfig(
        test_data_path=test_data_path,
        num_samples=num_samples,
        local_model_path=local_model_path,
    )
    
    benchmark = ToolUseBenchmark(config)
    benchmark.load_test_data()
    
    # Benchmark local model
    engine = create_engine(local_model_path, use_structured_decoding=True)
    engine.load_model()
    
    await benchmark.benchmark_local(engine, "Local SLM (Structured)")
    
    # Optionally compare with cloud
    if compare_cloud and cloud_api_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=cloud_api_key)
        await benchmark.benchmark_cloud(client, "GPT-4 Turbo")
    
    benchmark.compare_results()
    benchmark.save_results()
    
    return benchmark.results
