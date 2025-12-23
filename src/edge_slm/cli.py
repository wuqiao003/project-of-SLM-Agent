"""
Command-line interface for Edge SLM Agent.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="edge-slm",
    help="Edge-side SLM Fine-tuning & Structured Inference for Agents",
)
console = Console()


@app.command()
def distill(
    output_dir: str = typer.Option("data/distilled", help="Output directory"),
    num_samples: int = typer.Option(1000, help="Number of samples to generate"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
    local: bool = typer.Option(False, help="Use local model instead of OpenAI"),
    backend: str = typer.Option("ollama", help="Local backend: ollama/vllm/transformers"),
    api_base: str = typer.Option("http://localhost:11434/v1", help="Local API base URL"),
    model: str = typer.Option("qwen2.5:14b", help="Local model name"),
):
    """Generate training data using GPT-4 or local model distillation."""
    
    if local:
        from edge_slm.data.local_distiller import run_local_distillation
        
        console.print(f"[bold blue]使用本地模型生成数据 ({backend}: {model})...[/bold blue]")
        
        asyncio.run(run_local_distillation(
            num_samples=num_samples,
            output_dir=output_dir,
            backend=backend,
            api_base=api_base,
            model_name=model,
        ))
    else:
        from edge_slm.data.distiller import run_distillation
        
        console.print("[bold blue]Starting GPT-4 distillation...[/bold blue]")
        
        asyncio.run(run_distillation(
            num_samples=num_samples,
            output_dir=output_dir,
            api_key=api_key,
        ))
    
    console.print(f"[green]Dataset saved to {output_dir}[/green]")


@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to training data (JSONL)"),
    output_dir: str = typer.Option("outputs/model", help="Output directory"),
    model_name: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", help="Base model"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    use_unsloth: bool = typer.Option(True, help="Use Unsloth for acceleration"),
):
    """Fine-tune a model on tool-use data."""
    console.print(f"[bold blue]Starting training on {data_path}...[/bold blue]")
    
    if use_unsloth:
        try:
            from edge_slm.finetune.unsloth_trainer import train_with_unsloth
            train_with_unsloth(
                data_path=data_path,
                output_dir=output_dir,
                model_name=f"unsloth/{model_name.split('/')[-1]}-bnb-4bit",
                num_epochs=epochs,
            )
        except ImportError:
            console.print("[yellow]Unsloth not available, falling back to standard trainer[/yellow]")
            from edge_slm.finetune.trainer import train_tool_use_model
            train_tool_use_model(
                data_path=data_path,
                output_dir=output_dir,
                model_name=model_name,
                num_epochs=epochs,
                batch_size=batch_size,
            )
    else:
        from edge_slm.finetune.trainer import train_tool_use_model
        train_tool_use_model(
            data_path=data_path,
            output_dir=output_dir,
            model_name=model_name,
            num_epochs=epochs,
            batch_size=batch_size,
        )
    
    console.print(f"[green]Model saved to {output_dir}[/green]")


@app.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to model"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
    backend: str = typer.Option("transformers", help="Inference backend"),
    cloud_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="Cloud API key"),
):
    """Start the inference server."""
    from edge_slm.agent.service import run_server
    
    console.print(f"[bold blue]Starting server at {host}:{port}...[/bold blue]")
    console.print(f"Model: {model_path}")
    console.print(f"Backend: {backend}")
    
    run_server(
        model_path=model_path,
        host=host,
        port=port,
        backend=backend,
        cloud_api_key=cloud_key,
    )


@app.command()
def benchmark(
    model_path: str = typer.Argument(..., help="Path to model"),
    test_data: Optional[str] = typer.Option(None, help="Path to test data"),
    num_samples: int = typer.Option(100, help="Number of test samples"),
    compare_cloud: bool = typer.Option(False, help="Compare with cloud API"),
    cloud_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="Cloud API key"),
):
    """Run benchmark evaluation."""
    from edge_slm.evaluation.benchmark import run_benchmark
    
    console.print("[bold blue]Running benchmark...[/bold blue]")
    
    asyncio.run(run_benchmark(
        local_model_path=model_path,
        test_data_path=test_data,
        num_samples=num_samples,
        compare_cloud=compare_cloud,
        cloud_api_key=cloud_key,
    ))


@app.command()
def infer(
    model_path: str = typer.Argument(..., help="Path to model"),
    query: str = typer.Argument(..., help="User query"),
    structured: bool = typer.Option(True, help="Use structured decoding"),
):
    """Run single inference."""
    from edge_slm.inference import create_engine
    from edge_slm.data.schema import LIGHT_ON_TOOLS
    
    console.print(f"[bold blue]Loading model: {model_path}[/bold blue]")
    
    engine = create_engine(model_path, use_structured_decoding=structured)
    engine.load_model()
    
    tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS]
    
    # Build prompt
    tools_desc = "\n".join([f"- {t['function']['name']}: {t['function']['description']}" for t in tools])
    prompt = f"""<|im_start|>system
You are a helpful AI assistant. Use the following tools:

{tools_desc}

Respond with JSON: {{"name": "tool_name", "arguments": {{...}}}}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    
    console.print(f"\n[cyan]Query:[/cyan] {query}")
    
    result = engine.generate(prompt, tools=tools)
    
    console.print(f"\n[green]Result:[/green]")
    console.print(f"  Text: {result.text}")
    console.print(f"  Parsed: {result.parsed}")
    console.print(f"  Latency: {result.latency_ms:.1f}ms")
    console.print(f"  Valid: {result.is_valid}")


@app.command()
def export(
    model_path: str = typer.Argument(..., help="Path to fine-tuned model"),
    output_path: str = typer.Argument(..., help="Output path"),
    format: str = typer.Option("merged", help="Export format: merged, gguf, vllm"),
    quantization: str = typer.Option("q4_k_m", help="GGUF quantization method"),
):
    """Export model for deployment."""
    console.print(f"[bold blue]Exporting model to {format} format...[/bold blue]")
    
    try:
        from edge_slm.finetune.unsloth_trainer import UnslothTrainer, UnslothConfig
        
        config = UnslothConfig(model_name=model_path)
        trainer = UnslothTrainer(config)
        trainer.load_model()
        
        if format == "gguf":
            trainer.export_to_gguf(output_path, quantization)
        elif format == "vllm":
            trainer.export_to_vllm(output_path)
        else:
            trainer.save_merged_model(output_path)
        
        console.print(f"[green]Model exported to {output_path}[/green]")
        
    except ImportError:
        console.print("[red]Unsloth required for export. Install with pip install unsloth[/red]")


if __name__ == "__main__":
    app()
