"""
Schema definitions for Tool-Use dataset.
Defines the structure of tools, tool calls, and training examples.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum
import json


class ToolCategory(str, Enum):
    """Categories of tools for the Light-On project scenarios."""
    VIDEO_PROCESSING = "video_processing"
    SUBTITLE_GENERATION = "subtitle_generation"
    AUDIO_DUBBING = "audio_dubbing"
    FILE_MANAGEMENT = "file_management"
    TRANSLATION = "translation"
    CONTENT_ANALYSIS = "content_analysis"
    SCHEDULING = "scheduling"
    GENERAL = "general"


@dataclass
class ParameterSchema:
    """Schema for a single parameter in a tool."""
    name: str
    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: Optional[Any] = None
    
    def to_json_schema(self) -> dict:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Definition of a tool/function that can be called."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ParameterSchema] = field(default_factory=list)
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def to_json_schema(self) -> dict:
        """Generate JSON Schema for tool output validation."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": self.name},
                "arguments": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }


@dataclass
class ToolCall:
    """A single tool call with name and arguments."""
    name: str
    arguments: dict[str, Any]
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "name": self.name,
            "arguments": self.arguments
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ToolCall":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(name=data["name"], arguments=data["arguments"])


@dataclass
class ToolUseExample:
    """A complete training example for tool-use fine-tuning."""
    # Input
    user_query: str
    available_tools: list[ToolDefinition]
    conversation_history: list[dict] = field(default_factory=list)
    
    # Output
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None  # Chain-of-thought reasoning
    
    # Metadata
    category: ToolCategory = ToolCategory.GENERAL
    complexity: Literal["simple", "medium", "complex"] = "simple"
    source: str = "synthetic"
    
    def to_training_format(self, include_reasoning: bool = False) -> dict:
        """
        Convert to training format for SFT.
        
        Returns a dict with 'messages' in chat format.
        """
        # Build system message with tool definitions
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.available_tools
        ])
        
        system_msg = f"""You are an AI assistant that helps users by calling appropriate tools.
Available tools:
{tools_desc}

When the user makes a request, analyze their intent and call the appropriate tool(s).
Always respond with a valid JSON object containing the tool call.
Format: {{"name": "tool_name", "arguments": {{"param1": "value1", ...}}}}"""

        messages = [{"role": "system", "content": system_msg}]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add user query
        messages.append({"role": "user", "content": self.user_query})
        
        # Build assistant response
        if include_reasoning and self.reasoning:
            response = f"<thinking>\n{self.reasoning}\n</thinking>\n\n"
        else:
            response = ""
        
        # Add tool calls
        if len(self.tool_calls) == 1:
            response += self.tool_calls[0].to_json()
        else:
            # Multiple tool calls
            calls = [tc.to_json() for tc in self.tool_calls]
            response += json.dumps(calls, ensure_ascii=False)
        
        messages.append({"role": "assistant", "content": response})
        
        return {"messages": messages}
    
    def to_json(self) -> str:
        """Serialize example to JSON for storage."""
        return json.dumps({
            "user_query": self.user_query,
            "available_tools": [t.to_openai_format() for t in self.available_tools],
            "conversation_history": self.conversation_history,
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls],
            "reasoning": self.reasoning,
            "category": self.category.value,
            "complexity": self.complexity,
            "source": self.source,
        }, ensure_ascii=False, indent=2)


# =============================================================================
# Predefined Tools for Light-On Scenarios
# =============================================================================

LIGHT_ON_TOOLS = [
    ToolDefinition(
        name="parse_video",
        description="Parse and analyze a video file, extracting metadata, duration, resolution, and key frames.",
        category=ToolCategory.VIDEO_PROCESSING,
        parameters=[
            ParameterSchema("video_url", "string", "URL or path to the video file"),
            ParameterSchema("extract_frames", "boolean", "Whether to extract key frames", required=False, default=False),
            ParameterSchema("frame_interval", "integer", "Interval in seconds for frame extraction", required=False, default=10),
        ]
    ),
    ToolDefinition(
        name="generate_subtitles",
        description="Generate subtitles for a video using speech recognition.",
        category=ToolCategory.SUBTITLE_GENERATION,
        parameters=[
            ParameterSchema("video_url", "string", "URL or path to the video file"),
            ParameterSchema("source_language", "string", "Source language code (e.g., 'en', 'zh')", enum=["en", "zh", "ja", "ko", "es", "fr", "de"]),
            ParameterSchema("output_format", "string", "Subtitle format", required=False, enum=["srt", "vtt", "ass"], default="srt"),
        ]
    ),
    ToolDefinition(
        name="translate_subtitles",
        description="Translate existing subtitles to another language.",
        category=ToolCategory.TRANSLATION,
        parameters=[
            ParameterSchema("subtitle_file", "string", "Path to the subtitle file"),
            ParameterSchema("source_language", "string", "Source language code"),
            ParameterSchema("target_language", "string", "Target language code"),
            ParameterSchema("preserve_timing", "boolean", "Keep original timing", required=False, default=True),
        ]
    ),
    ToolDefinition(
        name="generate_dubbing",
        description="Generate AI voice dubbing for a video based on subtitles.",
        category=ToolCategory.AUDIO_DUBBING,
        parameters=[
            ParameterSchema("video_url", "string", "URL or path to the video file"),
            ParameterSchema("subtitle_file", "string", "Path to the subtitle file for dubbing"),
            ParameterSchema("voice_id", "string", "ID of the voice to use for dubbing"),
            ParameterSchema("target_language", "string", "Target language for dubbing"),
            ParameterSchema("speed", "number", "Speech speed multiplier", required=False, default=1.0),
        ]
    ),
    ToolDefinition(
        name="analyze_content",
        description="Analyze video content to extract topics, sentiment, and key moments.",
        category=ToolCategory.CONTENT_ANALYSIS,
        parameters=[
            ParameterSchema("video_url", "string", "URL or path to the video file"),
            ParameterSchema("analysis_type", "string", "Type of analysis", enum=["topics", "sentiment", "summary", "all"]),
            ParameterSchema("detail_level", "string", "Level of detail", required=False, enum=["brief", "detailed"], default="brief"),
        ]
    ),
    ToolDefinition(
        name="schedule_task",
        description="Schedule a video processing task for later execution.",
        category=ToolCategory.SCHEDULING,
        parameters=[
            ParameterSchema("task_type", "string", "Type of task to schedule", enum=["parse", "subtitle", "dubbing", "translate"]),
            ParameterSchema("task_params", "object", "Parameters for the task"),
            ParameterSchema("scheduled_time", "string", "ISO 8601 datetime for execution"),
            ParameterSchema("priority", "string", "Task priority", required=False, enum=["low", "normal", "high"], default="normal"),
        ]
    ),
    ToolDefinition(
        name="export_project",
        description="Export a completed video project with all assets.",
        category=ToolCategory.FILE_MANAGEMENT,
        parameters=[
            ParameterSchema("project_id", "string", "ID of the project to export"),
            ParameterSchema("output_format", "string", "Export format", enum=["mp4", "webm", "mov"]),
            ParameterSchema("quality", "string", "Output quality", required=False, enum=["720p", "1080p", "4k"], default="1080p"),
            ParameterSchema("include_subtitles", "boolean", "Burn subtitles into video", required=False, default=False),
        ]
    ),
    ToolDefinition(
        name="list_voices",
        description="List available AI voices for dubbing.",
        category=ToolCategory.AUDIO_DUBBING,
        parameters=[
            ParameterSchema("language", "string", "Filter by language", required=False),
            ParameterSchema("gender", "string", "Filter by gender", required=False, enum=["male", "female", "neutral"]),
        ]
    ),
]


def get_tools_by_category(category: ToolCategory) -> list[ToolDefinition]:
    """Get all tools in a specific category."""
    return [t for t in LIGHT_ON_TOOLS if t.category == category]


def get_tool_by_name(name: str) -> Optional[ToolDefinition]:
    """Get a tool by its name."""
    for tool in LIGHT_ON_TOOLS:
        if tool.name == name:
            return tool
    return None


def generate_tool_call_schema() -> dict:
    """
    Generate a comprehensive JSON Schema for validating tool calls.
    Used for Grammar-Constrained Decoding.
    """
    tool_schemas = []
    for tool in LIGHT_ON_TOOLS:
        tool_schemas.append(tool.to_json_schema())
    
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ToolCall",
        "description": "A tool call with name and arguments",
        "oneOf": tool_schemas,
    }
