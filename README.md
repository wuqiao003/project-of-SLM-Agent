# Edge SLM Agent

<p align="center">
  <b>端侧轻量化模型微调与结构化推理优化框架</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-RTX%203060%206GB-brightgreen.svg" alt="GPU">
</p>

---

## 📖 项目简介

**Edge SLM Agent** 是一个专为消费级 GPU（如 RTX 3060 6GB）设计的端侧 AI Agent 框架，实现了高效的工具调用（Tool-Use）能力。通过创新的**语法约束解码**和**智能路由**技术，在保证 100% 有效 JSON 输出的同时，大幅降低推理成本。

### 🎯 核心特性

| 特性 | 描述 |
|------|------|
| **语法约束解码** | 基于有限状态机（FSM）的结构化输出，确保 100% 有效 JSON |
| **智能路由** | 自动判断使用本地模型或云端 API，节省 60%+ 成本 |
| **高效微调** | QLoRA + Unsloth 加速，6GB 显存训练 3B 模型 |
| **多后端支持** | Transformers / vLLM / Ollama 灵活切换 |
| **完整工具链** | 数据生成 → 训练 → 评估 → 部署一站式解决方案 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Edge SLM Agent                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Web UI    │  │  REST API   │  │         CLI             │  │
│  │  (Gradio)   │  │  (FastAPI)  │  │       (Typer)           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                │
│         └────────────────┼─────────────────────┘                │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    智能路由器 (Router)                     │  │
│  │   ┌─────────────┐              ┌─────────────────────┐    │  │
│  │   │ 复杂度评估  │──────────────▶│    路由策略决策     │    │  │
│  │   └─────────────┘              └─────────────────────┘    │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  本地推理   │  │  云端 API   │  │   混合模式  │              │
│  │ (3B Model)  │  │  (GPT-4)    │  │  (Fallback) │              │
│  └──────┬──────┘  └─────────────┘  └─────────────┘              │
│         │                                                       │
│         ▼                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              结构化解码器 (Structured Decoder)             │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │  │
│  │   │ JSON Schema │───▶│  FSM 约束   │───▶│ 有效输出    │   │  │
│  │   └─────────────┘    └─────────────┘    └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 项目结构

```
edge-slm-agent/
├── src/edge_slm/              # 核心源代码
│   ├── agent/                 # Agent 路由和服务
│   │   ├── router.py          # 智能路由器
│   │   └── service.py         # FastAPI 服务
│   ├── data/                  # 数据处理
│   │   ├── schema.py          # 工具定义 Schema
│   │   ├── dataset.py         # 数据集加载
│   │   ├── distiller.py       # GPT-4 数据蒸馏
│   │   └── local_distiller.py # 本地模型蒸馏
│   ├── inference/             # 推理引擎
│   │   ├── engine.py          # 基础推理引擎
│   │   ├── vllm_engine.py     # vLLM 高性能引擎
│   │   └── structured.py      # 结构化解码器
│   ├── finetune/              # 模型微调
│   │   ├── trainer.py         # 标准 QLoRA 训练器
│   │   └── unsloth_trainer.py # Unsloth 加速训练器
│   └── evaluation/            # 评估模块
│       ├── benchmark.py       # 基准测试
│       └── metrics.py         # 评估指标
├── configs/                   # 配置文件
│   └── rtx3060_config.yaml    # RTX 3060 优化配置
├── data/                      # 数据目录
│   ├── sample_train.jsonl     # 示例训练数据
│   └── distilled/             # 蒸馏数据
├── outputs/                   # 输出目录
│   └── model/                 # 训练模型检查点
├── scripts/                   # 示例脚本
├── tests/                     # 测试文件
├── run.py                     # CLI 入口
├── quick_start.py             # 快速演示
├── web_ui.py                  # Gradio Web UI
└── production_server.py       # 生产环境 API
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+ (推荐)
- GPU: RTX 3060 6GB 或更高
- 内存: 16GB+ RAM

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/edge-slm-agent.git
cd edge-slm-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：安装 Unsloth 加速（推荐）
pip install unsloth
```

### 快速体验

```bash
# 运行快速演示
python quick_start.py
```

---

## 📚 使用指南

### 1. 数据蒸馏

从大模型生成高质量训练数据：

```bash
# 使用 GPT-4 蒸馏（需要 API Key）
python run.py distill --num-samples 1000 --api-key YOUR_API_KEY

# 使用本地模型蒸馏（推荐 Ollama）
python run.py distill --num-samples 500 --local --backend ollama --model qwen2.5:14b
```

**支持的本地后端：**
- `ollama` - 推荐，易于使用
- `vllm` - 高性能，适合批量生成
- `transformers` - 通用后端

### 2. 模型训练

使用 QLoRA 微调模型：

```bash
# 标准训练
python run.py train data/distilled/train.jsonl --output-dir outputs/model

# 使用 Unsloth 加速（2-5x 速度提升）
python run.py train data/distilled/train.jsonl --use-unsloth

# 自定义参数
python run.py train data/train.jsonl \
    --output-dir outputs/my_model \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch-size 4
```

**训练配置（RTX 3060 6GB 优化）：**

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA Rank | 32 | 平衡性能和显存 |
| Batch Size | 2 | 适配 6GB 显存 |
| Gradient Accumulation | 8 | 有效批次 = 16 |
| Max Seq Length | 1536 | 覆盖大部分场景 |
| Quantization | 4-bit NF4 | 显存优化 |

### 3. 模型推理

单次推理测试：

```bash
python run.py infer outputs/model/final_adapter "帮我分析视频 https://example.com/video.mp4"
```

### 4. 启动服务

```bash
# 开发环境
python run.py serve outputs/model --port 8000

# 生产环境（带智能路由）
python production_server.py

# Web UI
python web_ui.py  # 访问 http://localhost:7860
```

### 5. 基准测试

```bash
# 本地模型测试
python run.py benchmark outputs/model --num-samples 100

# 对比云端 API
python run.py benchmark outputs/model --compare-cloud --cloud-key YOUR_API_KEY
```

---

## 🔧 预定义工具

系统内置 8 种视频处理相关工具：

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `parse_video` | 视频解析 | video_url, extract_audio, extract_frames |
| `generate_subtitles` | 字幕生成 | video_id, language, style |
| `translate_subtitles` | 字幕翻译 | subtitle_id, target_language, preserve_timing |
| `generate_dubbing` | AI 配音 | video_id, voice_id, language, emotion |
| `analyze_content` | 内容分析 | video_id, analysis_type |
| `schedule_task` | 任务调度 | task_type, video_id, scheduled_time, priority |
| `export_project` | 项目导出 | project_id, format, quality, include_subtitles |
| `list_voices` | 语音列表 | language, gender |

### 自定义工具

```python
from edge_slm.data.schema import ToolDefinition, ToolParameter

my_tool = ToolDefinition(
    name="my_custom_tool",
    description="我的自定义工具",
    parameters=[
        ToolParameter(name="param1", type="string", description="参数1", required=True),
        ToolParameter(name="param2", type="integer", description="参数2", default=10),
    ]
)
```

---

## 🎛️ 智能路由

### 路由策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `LOCAL_FIRST` | 本地优先，失败回退云端 | 成本敏感，延迟要求低 |
| `CLOUD_FIRST` | 云端优先，失败回退本地 | 质量优先 |
| `LOCAL_ONLY` | 仅本地 | 离线环境 |
| `CLOUD_ONLY` | 仅云端 | 高质量要求 |
| `SMART` | 智能路由 | 平衡成本和质量 |

### 智能路由原理

```
用户查询 → 复杂度评估 → 路由决策
                ↓
    ┌─────────────────────────┐
    │ 复杂度 < 阈值 (0.7)     │ → 本地模型
    │ 复杂度 >= 阈值          │ → 云端 API
    │ 本地置信度 < 0.8        │ → 云端验证
    └─────────────────────────┘
```

---

## 📊 性能指标

### 结构化解码对比

| 指标 | 无约束 | 语法约束 | 提升 |
|------|--------|----------|------|
| JSON 有效率 | 85-90% | **100%** | +10-15% |
| Schema 合规率 | 75-85% | **100%** | +15-25% |
| 重试率 | 10-15% | **0%** | -100% |
| 有效延迟 | 基准 | **-60%** | 显著降低 |

### 推理性能（RTX 3060 6GB）

| 模型 | 延迟 | 吞吐量 | 显存占用 |
|------|------|--------|----------|
| Qwen2.5-3B (4-bit) | ~150ms | ~7 req/s | ~4.5GB |
| Qwen2.5-3B + vLLM | ~80ms | ~15 req/s | ~5GB |

---

## 🔌 API 接口

### REST API

```bash
# 健康检查
curl http://localhost:8000/health

# 工具调用推理
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我分析视频 https://example.com/video.mp4"}'

# 带工具列表
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "生成中文字幕",
    "tools": ["generate_subtitles", "translate_subtitles"]
  }'
```

### 响应格式

```json
{
  "tool_call": {
    "name": "parse_video",
    "arguments": {
      "video_url": "https://example.com/video.mp4",
      "extract_audio": true
    }
  },
  "confidence": 0.95,
  "latency_ms": 145,
  "source": "local"
}
```

---

## ⚙️ 配置说明

### 完整配置示例 (rtx3060_config.yaml)

```yaml
# 模型配置
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  quantization: "int4"
  max_memory_mb: 5500

# LoRA 配置
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 1536

# 推理配置
inference:
  engine: "transformers"  # transformers / vllm
  use_guided_decoding: true
  guided_decoding_backend: "outlines"
  max_new_tokens: 512
  temperature: 0.1

# Agent 配置
agent:
  routing_strategy: "local_first"
  complexity_threshold: 0.7
  confidence_threshold: 0.8
  fallback_enabled: true
```

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_router.py -v
pytest tests/test_structured_decoding.py -v

# 项目完整测试
python test_project.py
```

---

## 📦 模型导出

```bash
# 导出合并模型
python run.py export outputs/model/final_adapter outputs/merged --format merged

# 导出 GGUF 格式（用于 llama.cpp）
python run.py export outputs/model/final_adapter outputs/model.gguf --format gguf --quantization q4_k_m

# 导出 vLLM 格式
python run.py export outputs/model/final_adapter outputs/vllm_model --format vllm
```

---

## 🔍 常见问题

### Q: 显存不足怎么办？

1. 减小 `batch_size` 到 1
2. 减小 `max_seq_length` 到 1024
3. 启用梯度检查点 `gradient_checkpointing: true`
4. 使用更激进的量化

### Q: 训练速度慢？

1. 安装 Unsloth: `pip install unsloth`
2. 使用 `--use-unsloth` 参数
3. 增加 `gradient_accumulation_steps`

### Q: JSON 输出不稳定？

启用结构化解码：
```yaml
inference:
  use_guided_decoding: true
  guided_decoding_backend: "outlines"
```

### Q: 如何添加新工具？

参考 `src/edge_slm/data/schema.py` 中的 `LIGHT_ON_TOOLS` 定义。

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Transformers](https://github.com/huggingface/transformers) - 模型加载和推理
- [PEFT](https://github.com/huggingface/peft) - LoRA 微调
- [Unsloth](https://github.com/unslothai/unsloth) - 训练加速
- [Outlines](https://github.com/outlines-dev/outlines) - 结构化解码
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理

---

<p align="center">
  <b>🌟 如果这个项目对你有帮助，请给个 Star！</b>
</p>
