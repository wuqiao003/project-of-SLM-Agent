#!/usr/bin/env python
"""
Edge SLM Agent - Web UI
=======================
A beautiful web interface for Edge SLM Agent.

Run: python web_ui.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json
import gradio as gr
from typing import Optional
import asyncio

# Import project modules
from edge_slm.data.schema import LIGHT_ON_TOOLS, ToolCategory
from edge_slm.inference.structured import StructuredDecoder
from edge_slm.agent.router import AgentRouter, RoutingConfig, RoutingStrategy


# ============================================================================
# Global State
# ============================================================================
class AppState:
    """Application state manager."""
    def __init__(self):
        self.model_loaded = False
        self.model_path = None
        self.engine = None
        self.decoder = StructuredDecoder(use_outlines=False)
        self.router = AgentRouter(RoutingConfig(strategy=RoutingStrategy.SMART))
        
    def get_tools_info(self):
        """Get formatted tools information."""
        tools_by_category = {}
        for tool in LIGHT_ON_TOOLS:
            cat = tool.category.value
            if cat not in tools_by_category:
                tools_by_category[cat] = []
            tools_by_category[cat].append(tool)
        return tools_by_category


app_state = AppState()


# ============================================================================
# UI Functions
# ============================================================================

def get_tools_display():
    """Generate tools display HTML."""
    tools_by_category = app_state.get_tools_info()
    
    html = "<div style='max-height: 400px; overflow-y: auto;'>"
    
    category_icons = {
        "video_processing": "🎬",
        "subtitle_generation": "📝",
        "audio_dubbing": "🎙️",
        "file_management": "📁",
        "translation": "🌐",
        "content_analysis": "📊",
        "scheduling": "⏰",
        "general": "⚙️",
    }
    
    for category, tools in tools_by_category.items():
        icon = category_icons.get(category, "📦")
        html += f"<h4>{icon} {category.replace('_', ' ').title()}</h4>"
        html += "<ul>"
        for tool in tools:
            params = ", ".join([p.name for p in tool.parameters[:3]])
            if len(tool.parameters) > 3:
                params += "..."
            html += f"<li><b>{tool.name}</b>({params})<br/>"
            html += f"<small style='color: #666;'>{tool.description[:80]}...</small></li>"
        html += "</ul>"
    
    html += "</div>"
    return html


def run_inference(query: str, use_structured: bool = True) -> tuple:
    """Run inference on user query."""
    if not query.strip():
        return "请输入查询内容", "", ""
    
    # Get routing decision
    tools = [t.to_openai_format() for t in LIGHT_ON_TOOLS]
    decision = app_state.router.should_use_local(query, tools)
    
    routing_info = f"""**路由决策:**
- 使用本地模型: {'✅ 是' if decision.use_local else '❌ 否 (建议使用云端)'}
- 复杂度评估: {decision.estimated_complexity.value}
- 置信度: {decision.confidence:.2f}
- 原因: {decision.reason}
"""
    
    # Simulate model output (since model might not be loaded)
    # In real scenario, this would call the actual model
    simulated_output = simulate_tool_call(query)
    
    # Parse the output
    parsed = app_state.decoder._extract_json(simulated_output)
    
    if isinstance(parsed, dict):
        formatted_result = json.dumps(parsed, indent=2, ensure_ascii=False)
        status = "✅ 解析成功"
    else:
        formatted_result = str(parsed)
        status = "⚠️ 解析结果"
    
    return status, formatted_result, routing_info


def simulate_tool_call(query: str) -> str:
    """Simulate a tool call based on query keywords."""
    query_lower = query.lower()
    
    if "视频" in query or "video" in query_lower:
        if "字幕" in query or "subtitle" in query_lower:
            return json.dumps({
                "name": "generate_subtitles",
                "arguments": {
                    "video_url": extract_url(query) or "https://example.com/video.mp4",
                    "source_language": "zh",
                    "output_format": "srt"
                }
            }, ensure_ascii=False)
        elif "分析" in query or "analyze" in query_lower:
            return json.dumps({
                "name": "analyze_content",
                "arguments": {
                    "video_url": extract_url(query) or "https://example.com/video.mp4",
                    "analysis_type": "all",
                    "detail_level": "detailed"
                }
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "name": "parse_video",
                "arguments": {
                    "video_url": extract_url(query) or "https://example.com/video.mp4",
                    "extract_frames": False
                }
            }, ensure_ascii=False)
    
    elif "翻译" in query or "translate" in query_lower:
        return json.dumps({
            "name": "translate_subtitles",
            "arguments": {
                "subtitle_file": "subtitles.srt",
                "source_language": "zh",
                "target_language": "en"
            }
        }, ensure_ascii=False)
    
    elif "配音" in query or "dubbing" in query_lower:
        return json.dumps({
            "name": "generate_dubbing",
            "arguments": {
                "video_url": extract_url(query) or "https://example.com/video.mp4",
                "subtitle_file": "subtitles.srt",
                "voice_id": "voice_001",
                "target_language": "en"
            }
        }, ensure_ascii=False)
    
    elif "导出" in query or "export" in query_lower:
        return json.dumps({
            "name": "export_project",
            "arguments": {
                "project_id": "proj_001",
                "output_format": "mp4",
                "quality": "1080p"
            }
        }, ensure_ascii=False)
    
    else:
        return json.dumps({
            "name": "parse_video",
            "arguments": {
                "video_url": "https://example.com/video.mp4"
            }
        }, ensure_ascii=False)


def extract_url(text: str) -> Optional[str]:
    """Extract URL from text."""
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None


def test_json_extraction(raw_output: str) -> tuple:
    """Test JSON extraction from raw model output."""
    if not raw_output.strip():
        return "请输入模型输出", ""
    
    result = app_state.decoder._extract_json(raw_output)
    
    if isinstance(result, dict):
        return "✅ 提取成功", json.dumps(result, indent=2, ensure_ascii=False)
    else:
        return "❌ 提取失败", str(result)


def generate_sample_data(num_samples: int, categories: list) -> str:
    """Generate sample training data preview."""
    from edge_slm.data.schema import ToolUseExample, ToolCall
    
    samples = []
    
    # Sample queries for each category
    sample_queries = {
        "video_processing": [
            ("帮我解析这个视频 https://example.com/v.mp4", "parse_video", {"video_url": "https://example.com/v.mp4"}),
        ],
        "subtitle_generation": [
            ("为视频生成中文字幕", "generate_subtitles", {"video_url": "video.mp4", "source_language": "zh"}),
        ],
        "translation": [
            ("把字幕翻译成英文", "translate_subtitles", {"subtitle_file": "sub.srt", "source_language": "zh", "target_language": "en"}),
        ],
        "audio_dubbing": [
            ("给视频配音", "generate_dubbing", {"video_url": "v.mp4", "subtitle_file": "s.srt", "voice_id": "v1", "target_language": "en"}),
        ],
        "content_analysis": [
            ("分析视频内容", "analyze_content", {"video_url": "v.mp4", "analysis_type": "all"}),
        ],
    }
    
    count = 0
    for cat in categories:
        if cat in sample_queries and count < num_samples:
            for query, tool_name, args in sample_queries[cat]:
                if count >= num_samples:
                    break
                    
                tool = next((t for t in LIGHT_ON_TOOLS if t.name == tool_name), None)
                if tool:
                    example = ToolUseExample(
                        user_query=query,
                        available_tools=[tool],
                        tool_calls=[ToolCall(name=tool_name, arguments=args)],
                        category=tool.category,
                    )
                    samples.append(example.to_training_format())
                    count += 1
    
    if not samples:
        return "请选择至少一个类别"
    
    return json.dumps(samples[:3], indent=2, ensure_ascii=False) + f"\n\n... 共 {len(samples)} 条样本"


def get_model_status() -> str:
    """Get current model status."""
    if app_state.model_loaded:
        return f"✅ 模型已加载: {app_state.model_path}"
    return "⚠️ 模型未加载 (使用模拟模式)"


# ============================================================================
# Build UI
# ============================================================================

def create_ui():
    """Create the Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .tool-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .status-box {
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .success { background-color: #d4edda; }
    .warning { background-color: #fff3cd; }
    .error { background-color: #f8d7da; }
    """
    
    with gr.Blocks(
        title="Edge SLM Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 Edge SLM Agent
        ### 端侧轻量化模型微调与结构化推理优化
        
        ---
        """)
        
        # Status bar
        with gr.Row():
            status_display = gr.Markdown(get_model_status())
            refresh_btn = gr.Button("🔄 刷新状态", size="sm")
            refresh_btn.click(fn=get_model_status, outputs=status_display)
        
        # Main tabs
        with gr.Tabs():
            
            # ================================================================
            # Tab 1: Inference
            # ================================================================
            with gr.TabItem("💬 推理测试", id="inference"):
                gr.Markdown("### 测试工具调用推理")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="用户查询",
                            placeholder="例如: 帮我分析视频 https://example.com/video.mp4",
                            lines=3
                        )
                        
                        with gr.Row():
                            structured_check = gr.Checkbox(
                                label="使用结构化解码",
                                value=True
                            )
                            infer_btn = gr.Button("🚀 运行推理", variant="primary")
                        
                        # Example queries
                        gr.Examples(
                            examples=[
                                ["帮我解析视频 https://example.com/video.mp4"],
                                ["为这个视频生成中文字幕"],
                                ["把字幕翻译成英文"],
                                ["分析视频内容并提取关键信息"],
                                ["给视频配上英文配音"],
                            ],
                            inputs=query_input,
                            label="示例查询"
                        )
                    
                    with gr.Column(scale=2):
                        infer_status = gr.Markdown("等待输入...")
                        result_output = gr.Code(
                            label="工具调用结果",
                            language="json",
                            lines=10
                        )
                        routing_output = gr.Markdown(label="路由信息")
                
                infer_btn.click(
                    fn=run_inference,
                    inputs=[query_input, structured_check],
                    outputs=[infer_status, result_output, routing_output]
                )
            
            # ================================================================
            # Tab 2: Tools Browser
            # ================================================================
            with gr.TabItem("🛠️ 工具浏览", id="tools"):
                gr.Markdown("### 可用工具列表")
                
                tools_html = gr.HTML(get_tools_display())
                
                gr.Markdown("---")
                gr.Markdown("### 工具 Schema 预览")
                
                tool_selector = gr.Dropdown(
                    choices=[t.name for t in LIGHT_ON_TOOLS],
                    label="选择工具",
                    value=LIGHT_ON_TOOLS[0].name
                )
                
                schema_output = gr.Code(
                    label="JSON Schema",
                    language="json",
                    lines=15
                )
                
                def show_tool_schema(tool_name):
                    tool = next((t for t in LIGHT_ON_TOOLS if t.name == tool_name), None)
                    if tool:
                        return json.dumps(tool.to_openai_format(), indent=2, ensure_ascii=False)
                    return "{}"
                
                tool_selector.change(
                    fn=show_tool_schema,
                    inputs=tool_selector,
                    outputs=schema_output
                )
                
                # Initialize with first tool
                demo.load(
                    fn=lambda: show_tool_schema(LIGHT_ON_TOOLS[0].name),
                    outputs=schema_output
                )
            
            # ================================================================
            # Tab 3: JSON Extraction Test
            # ================================================================
            with gr.TabItem("🔍 JSON 提取测试", id="extraction"):
                gr.Markdown("""
                ### 测试 JSON 提取功能
                
                模拟从模型输出中提取有效 JSON，支持处理各种格式问题。
                """)
                
                with gr.Row():
                    with gr.Column():
                        raw_input = gr.Textbox(
                            label="模型原始输出",
                            placeholder='例如: Let me help you. {"name": "parse_video", "arguments": {...}}',
                            lines=5
                        )
                        extract_btn = gr.Button("🔍 提取 JSON", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                ['{"name": "parse_video", "arguments": {"video_url": "https://example.com/v.mp4"}}'],
                                ["Let me help you. {'name': 'parse_video', 'arguments': {'video_url': 'test.mp4'}}"],
                                ['The tool to use is: {"name": "generate_subtitles", "arguments": {"video_url": "v.mp4", "source_language": "zh",}}'],
                                ['```json\n{"name": "analyze_content", "arguments": {"video_url": "v.mp4"}}\n```'],
                            ],
                            inputs=raw_input,
                            label="测试用例"
                        )
                    
                    with gr.Column():
                        extract_status = gr.Markdown("等待输入...")
                        extracted_output = gr.Code(
                            label="提取结果",
                            language="json",
                            lines=8
                        )
                
                extract_btn.click(
                    fn=test_json_extraction,
                    inputs=raw_input,
                    outputs=[extract_status, extracted_output]
                )
            
            # ================================================================
            # Tab 4: Data Generation
            # ================================================================
            with gr.TabItem("📊 数据生成", id="datagen"):
                gr.Markdown("""
                ### 训练数据生成
                
                生成用于微调的训练数据样本预览。
                """)
                
                with gr.Row():
                    with gr.Column():
                        num_samples_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1,
                            label="样本数量"
                        )
                        
                        category_select = gr.CheckboxGroup(
                            choices=[
                                "video_processing",
                                "subtitle_generation", 
                                "translation",
                                "audio_dubbing",
                                "content_analysis"
                            ],
                            value=["video_processing", "subtitle_generation"],
                            label="选择类别"
                        )
                        
                        gen_btn = gr.Button("📝 生成样本预览", variant="primary")
                    
                    with gr.Column():
                        sample_output = gr.Code(
                            label="样本预览",
                            language="json",
                            lines=20
                        )
                
                gen_btn.click(
                    fn=generate_sample_data,
                    inputs=[num_samples_slider, category_select],
                    outputs=sample_output
                )
            
            # ================================================================
            # Tab 5: Settings & Help
            # ================================================================
            with gr.TabItem("⚙️ 设置与帮助", id="settings"):
                gr.Markdown("""
                ### 使用说明
                
                #### 命令行工具
                
                ```bash
                # 生成训练数据
                python run.py distill --num-samples 100
                
                # 训练模型
                python run.py train data/sample_train.jsonl
                
                # 启动推理服务
                python run.py serve outputs/model --port 8000
                
                # 运行推理
                python run.py infer outputs/model "帮我分析视频"
                
                # 基准测试
                python run.py benchmark outputs/model
                ```
                
                #### 项目特性
                
                - **结构化解码**: 使用 Grammar-Constrained Decoding 确保 100% 有效 JSON 输出
                - **智能路由**: 自动判断使用本地模型或云端 API
                - **轻量化微调**: 支持 LoRA/QLoRA 高效微调
                - **多工具支持**: 8 种视频处理相关工具
                
                #### 系统要求
                
                - Python 3.10+
                - PyTorch 2.1+
                - (可选) CUDA 11.8+ 用于 GPU 加速
                - (可选) Rust 用于 Outlines 结构化解码
                
                ---
                
                ### 关于
                
                Edge SLM Agent 是一个端侧轻量化模型微调与结构化推理优化项目，
                专注于在资源受限的环境下实现高效的工具调用能力。
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>Edge SLM Agent v1.0 | Built with Gradio</small>
        </center>
        """)
    
    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Edge SLM Agent Web UI...")
    print("=" * 50)
    
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
