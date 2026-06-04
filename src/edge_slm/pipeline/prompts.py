"""
Unified prompts for training, inference, and evaluation.
"""

from edge_slm.data.schema import LIGHT_ON_TOOLS, ToolDefinition
from edge_slm.data.dataset import format_tools_for_prompt


def build_system_prompt(tools: list[ToolDefinition] | None = None) -> str:
    """System message aligned with training data and benchmark."""
    tools = tools or LIGHT_ON_TOOLS
    tools_desc = format_tools_for_prompt(tools)
    return f"""You are an AI assistant that helps users by calling appropriate tools.
Available tools:
{tools_desc}

When the user makes a request, analyze their intent and call the appropriate tool.
Respond with a valid JSON object only, no markdown:
{{"name": "tool_name", "arguments": {{"param1": "value1", ...}}}}"""


def build_tool_use_messages(
    user_query: str,
    tool_call_json: str,
    tools: list[ToolDefinition] | None = None,
) -> list[dict]:
    """Build chat messages for SFT (matches ToolUseExample.to_training_format)."""
    return [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": tool_call_json},
    ]


def build_inference_prompt(
    user_query: str,
    tools: list[ToolDefinition] | None = None,
    model_type: str = "qwen",
) -> str:
    """Build a single-string prompt for engines that expect raw text."""
    from edge_slm.data.dataset import get_prompt_template, format_tools_for_prompt as fmt

    tools = tools or LIGHT_ON_TOOLS
    template = get_prompt_template(model_type)
    return template.format(tools=fmt(tools), query=user_query)
