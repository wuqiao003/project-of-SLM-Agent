"""
Tests for structured decoding functionality.
"""

import pytest
import json

from edge_slm.inference.structured import (
    StructuredDecoder,
    GrammarConstraint,
    OutputFormat,
    TOOL_CALL_SCHEMA,
    create_tool_constraint,
)


class TestGrammarConstraint:
    """Tests for GrammarConstraint class."""
    
    def test_json_schema_constraint(self):
        """Test JSON schema constraint creation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["name", "value"],
        }
        
        constraint = GrammarConstraint(
            format=OutputFormat.JSON_SCHEMA,
            schema=schema,
        )
        
        assert constraint.format == OutputFormat.JSON_SCHEMA
        assert constraint.schema == schema
    
    def test_to_outlines_config(self):
        """Test conversion to Outlines config."""
        constraint = GrammarConstraint(
            format=OutputFormat.JSON_SCHEMA,
            schema=TOOL_CALL_SCHEMA,
        )
        
        config = constraint.to_outlines_config()
        assert "json_schema" in config
        assert config["json_schema"] == TOOL_CALL_SCHEMA
    
    def test_to_vllm_config(self):
        """Test conversion to vLLM config."""
        constraint = GrammarConstraint(
            format=OutputFormat.JSON_SCHEMA,
            schema=TOOL_CALL_SCHEMA,
        )
        
        config = constraint.to_vllm_config()
        assert "guided_json" in config
        assert "guided_decoding_backend" in config
    
    def test_choice_constraint(self):
        """Test choice constraint."""
        choices = ["option1", "option2", "option3"]
        constraint = GrammarConstraint(
            format=OutputFormat.CHOICE,
            choices=choices,
        )
        
        config = constraint.to_outlines_config()
        assert config["choice"] == choices


class TestStructuredDecoder:
    """Tests for StructuredDecoder class."""
    
    def test_create_tool_call_schema(self):
        """Test tool call schema generation."""
        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string"},
                        },
                        "required": ["param1"],
                    }
                }
            }
        ]
        
        decoder = StructuredDecoder()
        schema = decoder.create_tool_call_schema(tools)
        
        assert "$schema" in schema
        assert "oneOf" in schema
        assert len(schema["oneOf"]) == 1
        
        tool_schema = schema["oneOf"][0]
        assert tool_schema["properties"]["name"]["const"] == "test_tool"
    
    def test_create_constraint(self):
        """Test constraint creation from tools."""
        tools = [
            {
                "function": {
                    "name": "parse_video",
                    "description": "Parse video",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    }
                }
            }
        ]
        
        decoder = StructuredDecoder()
        constraint = decoder.create_constraint(tools=tools)
        
        assert constraint.format == OutputFormat.JSON_SCHEMA
        assert constraint.schema is not None
    
    def test_validate_output_valid(self):
        """Test output validation with valid output."""
        decoder = StructuredDecoder()
        
        output = {
            "name": "test_tool",
            "arguments": {"param1": "value1"}
        }
        
        constraint = GrammarConstraint(
            format=OutputFormat.JSON_SCHEMA,
            schema=TOOL_CALL_SCHEMA,
        )
        
        is_valid, error = decoder.validate_output(output, constraint)
        assert is_valid
        assert error is None
    
    def test_validate_output_invalid(self):
        """Test output validation with invalid output."""
        decoder = StructuredDecoder()
        
        output = "not a dict"
        
        constraint = GrammarConstraint(
            format=OutputFormat.JSON_SCHEMA,
            schema=TOOL_CALL_SCHEMA,
        )
        
        is_valid, error = decoder.validate_output(output, constraint)
        assert not is_valid
        assert error is not None


class TestCreateToolConstraint:
    """Tests for create_tool_constraint helper."""
    
    def test_single_tool(self):
        """Test with single tool."""
        tools = [
            {
                "function": {
                    "name": "single_tool",
                    "description": "Single tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ]
        
        constraint = create_tool_constraint(tools)
        assert constraint.format == OutputFormat.JSON_SCHEMA
    
    def test_multiple_tools(self):
        """Test with multiple tools."""
        tools = [
            {"function": {"name": f"tool_{i}", "description": f"Tool {i}", "parameters": {}}}
            for i in range(5)
        ]
        
        constraint = create_tool_constraint(tools)
        assert len(constraint.schema["oneOf"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
