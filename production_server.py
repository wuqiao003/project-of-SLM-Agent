"""
生产环境部署示例 - Edge SLM Agent API 服务

使用方法:
1. 训练模型: python run.py train data/distilled/tool_use_train.jsonl --output-dir outputs/model
2. 启动服务: python production_server.py
3. 调用 API: curl -X POST http://localhost:8000/api/process -d '{"query": "分析视频 xxx.mp4"}'
"""

import sys
sys.path.insert(0, "src")

import json
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# ==================== 配置 ====================

MODEL_PATH = "outputs/model"  # 训练好的模型路径
USE_GPU = True
PORT = 8000


# ==================== 数据模型 ====================

class ProcessRequest(BaseModel):
    """用户请求"""
    query: str
    context: Optional[Dict[str, Any]] = None

class ProcessResponse(BaseModel):
    """处理响应"""
    success: bool
    tool_call: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float


# ==================== 工具执行器 ====================

class ToolExecutor:
    """
    真实的工具执行器
    在生产环境中，这里会调用真实的 API
    """
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        
        # 这里是模拟实现，实际应用中替换为真实 API 调用
        handlers = {
            "parse_video": self._parse_video,
            "generate_subtitles": self._generate_subtitles,
            "translate_subtitles": self._translate_subtitles,
            "generate_dubbing": self._generate_dubbing,
            "analyze_content": self._analyze_content,
            "schedule_task": self._schedule_task,
            "export_project": self._export_project,
            "list_voices": self._list_voices,
        }
        
        handler = handlers.get(tool_name)
        if handler:
            return await handler(arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _parse_video(self, args: Dict) -> Dict:
        """解析视频 - 实际应用中调用 FFmpeg 或视频处理 API"""
        video_url = args.get("video_url", "")
        # 模拟返回
        return {
            "status": "success",
            "video_url": video_url,
            "duration": "00:05:30",
            "resolution": "1920x1080",
            "fps": 30,
            "codec": "h264",
            "size_mb": 125.5
        }
    
    async def _generate_subtitles(self, args: Dict) -> Dict:
        """生成字幕 - 实际应用中调用 Whisper 或语音识别 API"""
        return {
            "status": "success",
            "subtitle_file": "/output/subtitles.srt",
            "language": args.get("source_language", "auto"),
            "segments": 45
        }
    
    async def _translate_subtitles(self, args: Dict) -> Dict:
        """翻译字幕 - 实际应用中调用翻译 API"""
        return {
            "status": "success",
            "output_file": "/output/translated.srt",
            "source_language": args.get("source_language"),
            "target_language": args.get("target_language"),
            "segments_translated": 45
        }
    
    async def _generate_dubbing(self, args: Dict) -> Dict:
        """生成配音 - 实际应用中调用 TTS API"""
        return {
            "status": "success",
            "audio_file": "/output/dubbing.mp3",
            "voice_id": args.get("voice_id"),
            "duration": "00:05:30"
        }
    
    async def _analyze_content(self, args: Dict) -> Dict:
        """分析内容 - 实际应用中调用视频分析 API"""
        return {
            "status": "success",
            "topics": ["技术教程", "编程"],
            "sentiment": "positive",
            "key_moments": [
                {"time": "00:01:30", "description": "介绍部分"},
                {"time": "00:03:00", "description": "核心内容"}
            ]
        }
    
    async def _schedule_task(self, args: Dict) -> Dict:
        """调度任务 - 实际应用中写入任务队列"""
        return {
            "status": "scheduled",
            "task_id": f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "task_type": args.get("task_type"),
            "scheduled_time": args.get("scheduled_time")
        }
    
    async def _export_project(self, args: Dict) -> Dict:
        """导出项目 - 实际应用中调用视频编码器"""
        return {
            "status": "success",
            "output_file": f"/output/{args.get('project_id')}.{args.get('output_format', 'mp4')}",
            "format": args.get("output_format", "mp4"),
            "quality": args.get("quality", "1080p")
        }
    
    async def _list_voices(self, args: Dict) -> Dict:
        """列出可用语音"""
        voices = {
            "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
            "en-US": ["en-US-JennyNeural", "en-US-GuyNeural"],
            "ja-JP": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
        }
        lang = args.get("language", "zh-CN")
        return {
            "status": "success",
            "language": lang,
            "voices": voices.get(lang, voices["en-US"])
        }


# ==================== 推理引擎 ====================

class ProductionEngine:
    """生产环境推理引擎"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tool_executor = ToolExecutor()
        self._loaded = False
    
    def load(self):
        """加载模型"""
        if self._loaded:
            return
        
        model_dir = Path(self.model_path)
        
        if not model_dir.exists():
            print(f"⚠️ 模型目录不存在: {model_path}")
            print("📝 将使用模拟模式运行（用于演示）")
            self._loaded = True
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"🔄 加载模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if USE_GPU else torch.float32,
                device_map="auto" if USE_GPU else "cpu",
                trust_remote_code=True
            )
            
            print("✅ 模型加载完成")
            self._loaded = True
            
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("📝 将使用模拟模式运行")
            self._loaded = True
    
    async def process(self, query: str) -> Dict[str, Any]:
        """处理用户请求"""
        import time
        start_time = time.time()
        
        # 1. 意图识别（调用模型）
        tool_call = await self._infer(query)
        
        # 2. 执行工具
        if tool_call and "name" in tool_call:
            result = await self.tool_executor.execute(
                tool_call["name"],
                tool_call.get("arguments", {})
            )
        else:
            result = None
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "tool_call": tool_call,
            "result": result,
            "latency_ms": latency
        }
    
    async def _infer(self, query: str) -> Optional[Dict]:
        """模型推理"""
        
        if self.model is None:
            # 模拟模式：简单的关键词匹配
            return self._mock_infer(query)
        
        # 真实模型推理
        try:
            from edge_slm.data.schema import LIGHT_ON_TOOLS
            
            # 构建 prompt
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in LIGHT_ON_TOOLS
            ])
            
            prompt = f"""You are an AI assistant that helps users by calling appropriate tools.
Available tools:
{tools_desc}

User: {query}
Assistant: """
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if USE_GPU:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            
            # 提取 JSON
            import re
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            print(f"推理错误: {e}")
        
        return None
    
    def _mock_infer(self, query: str) -> Dict:
        """模拟推理（用于演示）"""
        query_lower = query.lower()
        
        if "分析" in query or "parse" in query_lower or "解析" in query:
            return {
                "name": "parse_video",
                "arguments": {"video_url": self._extract_url(query)}
            }
        elif "字幕" in query or "subtitle" in query_lower:
            if "翻译" in query or "translate" in query_lower:
                return {
                    "name": "translate_subtitles",
                    "arguments": {
                        "subtitle_file": "/input/subtitle.srt",
                        "source_language": "zh",
                        "target_language": "en"
                    }
                }
            else:
                return {
                    "name": "generate_subtitles",
                    "arguments": {
                        "video_url": self._extract_url(query),
                        "source_language": "zh",
                        "output_format": "srt"
                    }
                }
        elif "配音" in query or "dubbing" in query_lower or "语音" in query:
            return {
                "name": "generate_dubbing",
                "arguments": {
                    "video_url": self._extract_url(query),
                    "subtitle_file": "/input/subtitle.srt",
                    "voice_id": "zh-CN-XiaoxiaoNeural",
                    "target_language": "zh"
                }
            }
        elif "导出" in query or "export" in query_lower:
            return {
                "name": "export_project",
                "arguments": {
                    "project_id": "project_001",
                    "output_format": "mp4",
                    "quality": "1080p"
                }
            }
        elif "安排" in query or "schedule" in query_lower:
            return {
                "name": "schedule_task",
                "arguments": {
                    "task_type": "parse",
                    "task_params": {"video_url": self._extract_url(query)},
                    "scheduled_time": "2024-12-25T10:00:00Z",
                    "priority": "normal"
                }
            }
        else:
            return {
                "name": "analyze_content",
                "arguments": {
                    "video_url": self._extract_url(query),
                    "analysis_type": "all"
                }
            }
    
    def _extract_url(self, text: str) -> str:
        """从文本中提取 URL"""
        import re
        url_match = re.search(r'https?://\S+|/\S+\.\w+', text)
        return url_match.group() if url_match else "https://example.com/video.mp4"


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="Edge SLM Agent API",
    description="端侧轻量化模型推理服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局引擎实例
engine: Optional[ProductionEngine] = None


@app.on_event("startup")
async def startup():
    """启动时加载模型"""
    global engine
    engine = ProductionEngine(MODEL_PATH)
    engine.load()


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "service": "Edge SLM Agent",
        "model_loaded": engine._loaded if engine else False
    }


@app.post("/api/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    """
    处理用户请求
    
    示例:
    - "帮我分析视频 https://example.com/video.mp4"
    - "给视频添加中文字幕"
    - "把字幕翻译成英文"
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = await engine.process(request.query)
        return ProcessResponse(
            success=True,
            tool_call=result["tool_call"],
            result=result["result"],
            latency_ms=result["latency_ms"]
        )
    except Exception as e:
        return ProcessResponse(
            success=False,
            error=str(e),
            latency_ms=0
        )


@app.get("/api/tools")
async def list_tools():
    """列出所有可用工具"""
    try:
        from edge_slm.data.schema import LIGHT_ON_TOOLS
        return {
            "tools": [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"]
                }
                for t in LIGHT_ON_TOOLS
            ]
        }
    except:
        return {
            "tools": [
                {"name": "parse_video", "description": "解析视频文件"},
                {"name": "generate_subtitles", "description": "生成字幕"},
                {"name": "translate_subtitles", "description": "翻译字幕"},
                {"name": "generate_dubbing", "description": "生成配音"},
                {"name": "analyze_content", "description": "分析内容"},
                {"name": "schedule_task", "description": "调度任务"},
                {"name": "export_project", "description": "导出项目"},
                {"name": "list_voices", "description": "列出语音"},
            ]
        }


# ==================== 主入口 ====================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Edge SLM Agent - 生产环境 API 服务                  ║
╠══════════════════════════════════════════════════════════════╣
║  端点:                                                        ║
║  • GET  /           - 健康检查                                ║
║  • POST /api/process - 处理用户请求                           ║
║  • GET  /api/tools   - 列出可用工具                           ║
╠══════════════════════════════════════════════════════════════╣
║  示例请求:                                                    ║
║  curl -X POST http://localhost:8000/api/process \\            ║
║       -H "Content-Type: application/json" \\                  ║
║       -d '{"query": "分析视频 https://example.com/v.mp4"}'    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
