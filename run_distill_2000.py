#!/usr/bin/env python
"""
使用本地 Ollama qwen2.5:14b 生成 2000 条训练数据
不使用 OpenAI API，完全本地运行

运行方式:
    python run_distill_2000.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from edge_slm.data.local_distiller import run_local_distillation

async def main():
    print("=" * 60)
    print("本地数据生成 - 使用 Ollama qwen2.5:14b")
    print("=" * 60)
    print(f"目标: 2000 条训练数据")
    print(f"后端: Ollama 原生 API (不使用 OpenAI)")
    print(f"模型: qwen2.5:14b")
    print(f"输出: data/distilled_2000/")
    print("=" * 60)
    print()
    
    output_path = await run_local_distillation(
        num_samples=2000,
        output_dir="data/distilled_2000",
        backend="ollama",
        model_name="qwen2.5:14b",
    )
    
    print()
    print("=" * 60)
    print(f"完成! 数据已保存到: {output_path}")
    print("=" * 60)
    
    # 显示下一步
    print("\n下一步 - 开始训练:")
    print("  python run.py train data/distilled_2000/tool_use_train.jsonl --epochs 5")

if __name__ == "__main__":
    asyncio.run(main())
