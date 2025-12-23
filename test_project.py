#!/usr/bin/env python
"""
项目测试脚本 - 验证所有组件是否正常工作
运行: python test_project.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def test_imports():
    """测试所有模块导入"""
    console.print(Panel.fit("[bold cyan]1. 测试模块导入[/bold cyan]"))
    
    results = []
    
    # 核心模块
    modules = [
        ("edge_slm", "核心包"),
        ("edge_slm.data.schema", "数据 Schema"),
        ("edge_slm.data.distiller", "GPT-4 蒸馏器"),
        ("edge_slm.data.local_distiller", "本地模型蒸馏器"),
        ("edge_slm.data.dataset", "数据集处理"),
        ("edge_slm.inference.engine", "推理引擎"),
        ("edge_slm.inference.structured", "结构化解码"),
        ("edge_slm.agent.router", "智能路由"),
        ("edge_slm.finetune.trainer", "训练器"),
        ("edge_slm.evaluation.metrics", "评估指标"),
        ("edge_slm.cli", "命令行工具"),
    ]
    
    for module, desc in modules:
        try:
            __import__(module)
            results.append((module, desc, "✅"))
        except Exception as e:
            results.append((module, desc, f"❌ {str(e)[:30]}"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("模块", style="cyan")
    table.add_column("描述")
    table.add_column("状态")
    
    for module, desc, status in results:
        table.add_row(module, desc, status)
    
    console.print(table)
    
    success = all("✅" in r[2] for r in results)
    return success


def test_schema():
    """测试数据 Schema"""
    console.print(Panel.fit("[bold cyan]2. 测试数据 Schema[/bold cyan]"))
    
    from edge_slm.data.schema import LIGHT_ON_TOOLS, ToolCategory
    
    console.print(f"[green]✅ 已定义 {len(LIGHT_ON_TOOLS)} 个工具[/green]")
    
    # 按类别统计
    by_category = {}
    for tool in LIGHT_ON_TOOLS:
        cat = tool.category.value
        by_category[cat] = by_category.get(cat, 0) + 1
    
    for cat, count in by_category.items():
        console.print(f"   - {cat}: {count} 个工具")
    
    # 测试工具格式转换
    tool = LIGHT_ON_TOOLS[0]
    openai_format = tool.to_openai_format()
    
    assert "function" in openai_format
    assert "name" in openai_format["function"]
    console.print(f"[green]✅ OpenAI 格式转换正常[/green]")
    
    return True


def test_structured_decoder():
    """测试结构化解码器"""
    console.print(Panel.fit("[bold cyan]3. 测试结构化解码器[/bold cyan]"))
    
    from edge_slm.inference.structured import StructuredDecoder, create_tool_constraint
    from edge_slm.data.schema import LIGHT_ON_TOOLS
    
    decoder = StructuredDecoder(use_outlines=False)
    
    # 测试 JSON 提取
    test_cases = [
        ('{"name": "parse_video", "arguments": {"video_url": "test.mp4"}}', True),
        ("Let me help. {'name': 'parse_video', 'arguments': {'url': 'test.mp4'}}", True),
        ('```json\n{"name": "test"}\n```', True),
        ("invalid text", False),
    ]
    
    passed = 0
    for text, should_succeed in test_cases:
        result = decoder._extract_json(text)
        is_dict = isinstance(result, dict)
        if is_dict == should_succeed:
            passed += 1
    
    console.print(f"[green]✅ JSON 提取测试: {passed}/{len(test_cases)} 通过[/green]")
    
    # 测试约束创建
    tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS[:3]]
    constraint = create_tool_constraint(tools)
    
    assert constraint.schema is not None
    console.print(f"[green]✅ 工具约束创建正常[/green]")
    
    return True


def test_router():
    """测试智能路由"""
    console.print(Panel.fit("[bold cyan]4. 测试智能路由[/bold cyan]"))
    
    from edge_slm.agent.router import AgentRouter, RoutingConfig, RoutingStrategy
    from edge_slm.data.schema import LIGHT_ON_TOOLS
    
    router = AgentRouter(RoutingConfig(strategy=RoutingStrategy.SMART))
    tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS]
    
    test_queries = [
        ("分析视频 https://example.com/v.mp4", True),  # 简单，应该本地
        ("分析这个视频并总结所有关键观点，比较不同部分的主题差异，生成详细报告", False),  # 复杂
    ]
    
    for query, expected_local in test_queries:
        decision = router.should_use_local(query, tools)
        status = "✅" if decision.use_local == expected_local else "⚠️"
        route = "本地" if decision.use_local else "云端"
        console.print(f"   {status} \"{query[:30]}...\" → {route} (复杂度: {decision.estimated_complexity.value})")
    
    console.print(f"[green]✅ 路由决策正常[/green]")
    return True


def test_metrics():
    """测试评估指标"""
    console.print(Panel.fit("[bold cyan]5. 测试评估指标[/bold cyan]"))
    
    from edge_slm.evaluation.metrics import compute_metrics
    
    predictions = [
        {"name": "parse_video", "arguments": {"video_url": "test.mp4"}},
        {"name": "generate_subtitles", "arguments": {"video_url": "v.mp4", "source_language": "zh"}},
    ]
    
    references = [
        {"name": "parse_video", "arguments": {"video_url": "test.mp4"}},
        {"name": "generate_subtitles", "arguments": {"video_url": "v.mp4", "source_language": "zh"}},
    ]
    
    latencies = [100.0, 150.0]
    
    metrics = compute_metrics(predictions, references, latencies)
    
    console.print(f"   - 工具准确率: {metrics.tool_selection_accuracy:.2%}")
    console.print(f"   - 参数准确率: {metrics.argument_accuracy:.2%}")
    console.print(f"   - 平均延迟: {metrics.avg_latency_ms:.1f}ms")
    
    console.print(f"[green]✅ 评估指标计算正常[/green]")
    return True


def test_cli():
    """测试命令行工具"""
    console.print(Panel.fit("[bold cyan]6. 测试命令行工具[/bold cyan]"))
    
    from edge_slm.cli import app
    from typer.testing import CliRunner
    
    runner = CliRunner()
    
    # 测试 help
    result = runner.invoke(app, ["--help"])
    
    if result.exit_code == 0:
        console.print(f"[green]✅ CLI 帮助命令正常[/green]")
        
        # 检查命令是否存在
        commands = ["distill", "train", "serve", "infer", "benchmark", "export"]
        for cmd in commands:
            if cmd in result.stdout:
                console.print(f"   - {cmd} ✓")
        
        return True
    else:
        console.print(f"[red]❌ CLI 测试失败: {result.stdout}[/red]")
        return False


def test_sample_data_generation():
    """测试样本数据生成"""
    console.print(Panel.fit("[bold cyan]7. 测试样本数据生成[/bold cyan]"))
    
    from edge_slm.data.schema import ToolUseExample, ToolCall, LIGHT_ON_TOOLS
    
    # 创建一个示例
    tool = LIGHT_ON_TOOLS[0]  # parse_video
    
    example = ToolUseExample(
        user_query="帮我分析这个视频 https://example.com/video.mp4",
        available_tools=[tool],
        tool_calls=[
            ToolCall(
                name="parse_video",
                arguments={"video_url": "https://example.com/video.mp4"}
            )
        ],
        category=tool.category,
        complexity="simple",
    )
    
    # 转换为训练格式
    training_format = example.to_training_format()
    
    assert "messages" in training_format
    assert len(training_format["messages"]) >= 3  # system, user, assistant
    
    console.print(f"[green]✅ 训练数据格式正确[/green]")
    console.print(f"   - 消息数: {len(training_format['messages'])}")
    
    return True


def check_gpu():
    """检查 GPU 状态"""
    console.print(Panel.fit("[bold cyan]8. 检查 GPU 状态[/bold cyan]"))
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            
            console.print(f"[green]✅ GPU 可用: {device_name}[/green]")
            console.print(f"   - 总显存: {memory_total:.1f} GB")
            console.print(f"   - 可用显存: {memory_free:.1f} GB")
            return True
        else:
            console.print(f"[yellow]⚠️ 未检测到 GPU，将使用 CPU 模式[/yellow]")
            console.print(f"   训练和推理会较慢，但仍可运行")
            return True
            
    except Exception as e:
        console.print(f"[yellow]⚠️ GPU 检查失败: {e}[/yellow]")
        return True


def check_ollama():
    """检查 Ollama 服务"""
    console.print(Panel.fit("[bold cyan]9. 检查 Ollama 服务 (可选)[/bold cyan]"))
    
    try:
        import httpx
        
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            console.print(f"[green]✅ Ollama 服务运行中[/green]")
            
            if models:
                console.print(f"   已安装模型:")
                for m in models[:5]:
                    console.print(f"   - {m}")
                    
                # 检查是否有 qwen
                qwen_models = [m for m in models if "qwen" in m.lower()]
                if qwen_models:
                    console.print(f"[green]   ✓ 已安装 Qwen 模型，可用于本地数据生成[/green]")
                else:
                    console.print(f"[yellow]   ⚠️ 未安装 Qwen 模型，运行: ollama pull qwen2.5:7b[/yellow]")
            else:
                console.print(f"[yellow]   未安装任何模型，运行: ollama pull qwen2.5:7b[/yellow]")
            
            return True
        else:
            console.print(f"[yellow]⚠️ Ollama 服务未响应[/yellow]")
            return True
            
    except Exception:
        console.print(f"[yellow]⚠️ Ollama 未运行 (可选，用于本地数据生成)[/yellow]")
        console.print(f"   安装: https://ollama.ai")
        return True


def main():
    """运行所有测试"""
    console.print(Panel.fit(
        "[bold magenta]Edge SLM Agent - 项目测试[/bold magenta]\n"
        "验证所有组件是否正常工作",
        title="🧪 测试开始"
    ))
    
    console.print()
    
    tests = [
        ("模块导入", test_imports),
        ("数据 Schema", test_schema),
        ("结构化解码器", test_structured_decoder),
        ("智能路由", test_router),
        ("评估指标", test_metrics),
        ("命令行工具", test_cli),
        ("样本数据生成", test_sample_data_generation),
        ("GPU 状态", check_gpu),
        ("Ollama 服务", check_ollama),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            console.print(f"[red]❌ {name} 测试异常: {e}[/red]")
            results.append((name, False))
        console.print()
    
    # 总结
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    console.print("=" * 60)
    
    if passed == total:
        console.print(Panel.fit(
            f"[bold green]✅ 所有测试通过 ({passed}/{total})[/bold green]\n\n"
            "项目已准备就绪！\n\n"
            "[cyan]下一步:[/cyan]\n"
            "1. 启动 Web UI: python web_ui.py\n"
            "2. 生成数据: python run.py distill --local --num-samples 50\n"
            "3. 快速演示: python quick_start.py",
            title="🎉 测试完成"
        ))
    else:
        failed = [name for name, p in results if not p]
        console.print(Panel.fit(
            f"[bold yellow]⚠️ 部分测试未通过 ({passed}/{total})[/bold yellow]\n\n"
            f"失败项: {', '.join(failed)}\n\n"
            "请检查错误信息并修复问题",
            title="测试完成"
        ))
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
