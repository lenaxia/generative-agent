"""
Unit tests for Hybrid Role Lifecycle Architecture.

Tests the enhanced RoleRegistry, RequestRouter, and UniversalAgent
with lifecycle hooks and parameter extraction capabilities.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import yaml

from llm_provider.role_registry import RoleRegistry, RoleDefinition
from llm_provider.request_router import RequestRouter
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from common.task_context import TaskContext
from common.task_graph import TaskGraph, TaskDescription


class TestHybridRoleRegistry:
    """Test enhanced RoleRegistry with lifecycle function support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.roles_dir = Path(self.temp_dir) / "roles"
        self.roles_dir.mkdir()
        
        # Create test role directory structure
        self.weather_dir = self.roles_dir / "weather"
        self.weather_dir.mkdir()
        
        # Create enhanced weather role definition
        self.weather_definition = {
            "role": {
                "name": "weather",
                "version": "2.0.0",
                "description": "Weather role with pre-processing data fetching",
                "execution_type": "hybrid",
                "fast_reply": True
            },
            "parameters": {
                "location": {
                    "type": "string",
                    "required": True,
                    "description": "City, state, country, or coordinates for weather lookup",
                    "examples": ["Seattle", "New York, NY", "90210", "47.6062,-122.3321"]
                },
                "timeframe": {
                    "type": "string",
                    "required": False,
                    "description": "When to get weather for",
                    "enum": ["current", "today", "tomorrow", "this week", "next week"],
                    "default": "current"
                },
                "format": {
                    "type": "string",
                    "required": False,
                    "description": "Output format preference",
                    "enum": ["brief", "detailed", "forecast"],
                    "default": "brief"
                }
            },
            "lifecycle": {
                "pre_processing": {
                    "enabled": True,
                    "functions": [
                        {
                            "name": "fetch_weather_data",
                            "uses_parameters": ["location", "timeframe"]
                        }
                    ]
                },
                "post_processing": {
                    "enabled": True,
                    "functions": [
                        {"name": "format_for_tts"}
                    ]
                }
            },
            "prompts": {
                "system": "You are a weather specialist. Weather data: {weather_current}"
            },
            "model_config": {
                "temperature": 0.1,
                "max_tokens": 2048
            },
            "tools": {
                "automatic": False,
                "shared": []
            }
        }
        
        # Write definition file
        with open(self.weather_dir / "definition.yaml", "w") as f:
            yaml.dump(self.weather_definition, f)
        
        # Create lifecycle functions file
        lifecycle_content = '''
async def fetch_weather_data(instruction, context, parameters):
    """Pre-processor: Fetch weather data."""
    location = parameters.get("location")
    return {
        "weather_current": f"Sunny 72°F in {location}",
        "location_resolved": location,
        "data_timestamp": "2024-01-01T12:00:00Z"
    }

async def format_for_tts(llm_result, context, pre_data):
    """Post-processor: Format for TTS."""
    return llm_result.replace("°F", " degrees Fahrenheit")
'''
        
        with open(self.weather_dir / "lifecycle.py", "w") as f:
            f.write(lifecycle_content)
        
        self.registry = RoleRegistry(str(self.roles_dir))
    
    def test_get_role_parameters(self):
        """Test parameter schema extraction from role definition."""
        parameters = self.registry.get_role_parameters("weather")
        
        assert "location" in parameters
        assert parameters["location"]["type"] == "string"
        assert parameters["location"]["required"] is True
        assert "Seattle" in parameters["location"]["examples"]
        
        assert "timeframe" in parameters
        assert parameters["timeframe"]["required"] is False
        assert "current" in parameters["timeframe"]["enum"]
        assert parameters["timeframe"]["default"] == "current"
    
    def test_get_role_execution_type(self):
        """Test execution type detection for hybrid roles."""
        execution_type = self.registry.get_role_execution_type("weather")
        assert execution_type == "hybrid"
    
    def test_lifecycle_functions_loading(self):
        """Test lifecycle function loading from Python modules."""
        lifecycle_functions = self.registry.get_lifecycle_functions("weather")
        
        assert "fetch_weather_data" in lifecycle_functions
        assert "format_for_tts" in lifecycle_functions
        assert callable(lifecycle_functions["fetch_weather_data"])
        assert callable(lifecycle_functions["format_for_tts"])
    
    def test_register_lifecycle_functions(self):
        """Test manual registration of lifecycle functions."""
        async def test_function(instruction, context, parameters):
            return {"test": "data"}
        
        functions = {"test_function": test_function}
        self.registry.register_lifecycle_functions("test_role", functions)
        
        retrieved = self.registry.get_lifecycle_functions("test_role")
        assert "test_function" in retrieved
        assert retrieved["test_function"] == test_function
    
    def test_enhanced_role_loading(self):
        """Test enhanced role loading with lifecycle support."""
        role_def = self.registry.get_role("weather")
        
        assert role_def is not None
        assert role_def.name == "weather"
        assert role_def.config["role"]["execution_type"] == "hybrid"
        
        # Check lifecycle functions were loaded
        lifecycle_functions = self.registry.get_lifecycle_functions("weather")
        assert len(lifecycle_functions) == 2


class TestEnhancedRequestRouter:
    """Test enhanced RequestRouter with parameter extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm_factory = Mock(spec=LLMFactory)
        self.role_registry = Mock(spec=RoleRegistry)
        self.universal_agent = Mock(spec=UniversalAgent)
        
        # Mock fast-reply roles with parameter schemas
        mock_weather_role = Mock()
        mock_weather_role.name = "weather"
        mock_weather_role.config = {
            "role": {"description": "Weather information specialist"}
        }
        
        self.role_registry.get_fast_reply_roles.return_value = [mock_weather_role]
        self.role_registry.get_role_parameters.return_value = {
            "location": {
                "type": "string",
                "required": True,
                "description": "Location for weather lookup",
                "examples": ["Seattle", "New York"]
            },
            "timeframe": {
                "type": "string",
                "required": False,
                "enum": ["current", "today", "tomorrow"],
                "default": "current"
            }
        }
        
        self.router = RequestRouter(
            self.llm_factory, 
            self.role_registry, 
            self.universal_agent
        )
    
    def test_enhanced_routing_prompt_generation(self):
        """Test enhanced routing prompt with parameter schemas."""
        fast_reply_roles = self.role_registry.get_fast_reply_roles()
        
        prompt = self.router._build_enhanced_routing_prompt(
            "What's the weather in Seattle?", 
            fast_reply_roles
        )
        
        assert "weather" in prompt
        assert "location" in prompt
        assert "timeframe" in prompt
        assert "Seattle" in prompt
        assert "enum" in prompt
        assert "JSON" in prompt
    
    def test_routing_and_parameter_parsing(self):
        """Test parsing of routing result with parameters."""
        llm_response = '''
        {
            "route": "weather",
            "confidence": 0.95,
            "parameters": {
                "location": "Seattle",
                "timeframe": "current"
            }
        }
        '''
        
        result = self.router._parse_routing_and_parameters(llm_response)
        
        assert result["route"] == "weather"
        assert result["confidence"] == 0.95
        assert result["parameters"]["location"] == "Seattle"
        assert result["parameters"]["timeframe"] == "current"
    
    def test_routing_with_invalid_json(self):
        """Test fallback handling for invalid JSON responses."""
        invalid_response = "This is not valid JSON"
        
        result = self.router._parse_routing_and_parameters(invalid_response)
        
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.0
        assert "error" in result
    
    def test_enhanced_route_request(self):
        """Test complete enhanced routing with parameter extraction."""
        # Mock LLM response - return string directly, not coroutine
        self.universal_agent.execute_task = Mock(return_value='''
        {
            "route": "weather",
            "confidence": 0.92,
            "parameters": {
                "location": "New York",
                "timeframe": "today"
            }
        }
        ''')
        
        result = self.router.route_request("What's the weather today in New York?")
        
        assert result["route"] == "weather"
        assert result["confidence"] == 0.92
        assert result["parameters"]["location"] == "New York"
        assert result["parameters"]["timeframe"] == "today"
        assert "execution_time_ms" in result


class TestHybridUniversalAgent:
    """Test UniversalAgent with hybrid execution support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm_factory = Mock(spec=LLMFactory)
        self.role_registry = Mock(spec=RoleRegistry)
        self.mcp_manager = Mock()
        
        self.agent = UniversalAgent(
            self.llm_factory,
            self.role_registry,
            self.mcp_manager
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_execution_detection(self):
        """Test detection of hybrid execution type."""
        # Mock role definition
        mock_role_def = Mock()
        mock_role_def.config = {
            "role": {"execution_type": "hybrid"},
            "lifecycle": {
                "pre_processing": {
                    "enabled": True,
                    "functions": [{"name": "fetch_weather_data"}]
                },
                "post_processing": {
                    "enabled": True,
                    "functions": [{"name": "format_for_tts"}]
                }
            }
        }
        
        self.role_registry.get_role.return_value = mock_role_def
        self.role_registry.get_role_execution_type.return_value = "hybrid"
        
        # Mock lifecycle functions
        async def mock_pre_processor(instruction, context, parameters):
            return {"weather_data": "sunny"}
        
        async def mock_post_processor(result, context, pre_data):
            return result + " (processed)"
        
        lifecycle_functions = {
            "fetch_weather_data": mock_pre_processor,
            "format_for_tts": mock_post_processor
        }
        
        self.role_registry.get_lifecycle_functions.return_value = lifecycle_functions
        
        # Mock LLM execution
        with patch.object(self.agent, '_execute_llm_with_context', return_value="Weather is sunny"):
            result = self.agent.execute_task(
                instruction="What's the weather?",
                role="weather",
                extracted_parameters={"location": "Seattle"}
            )
        
        # Should have executed hybrid path
        assert "processed" in result
    
    @pytest.mark.asyncio
    async def test_pre_processing_execution(self):
        """Test pre-processing function execution."""
        mock_role_def = Mock()
        mock_role_def.config = {
            "lifecycle": {
                "pre_processing": {
                    "enabled": True,
                    "functions": [
                        {
                            "name": "fetch_weather_data",
                            "uses_parameters": ["location", "timeframe"]
                        }
                    ]
                }
            }
        }
        
        async def mock_fetch_weather(instruction, context, parameters):
            return {
                "weather_current": f"Sunny in {parameters['location']}",
                "location_resolved": parameters["location"]
            }
        
        lifecycle_functions = {"fetch_weather_data": mock_fetch_weather}
        
        # Create a proper TaskContext with TaskGraph
        task_desc = TaskDescription(
            task_name="test_task",
            agent_id="test_agent",
            task_type="test_type",
            prompt="test instruction"
        )
        task_graph = TaskGraph([task_desc], request_id="test_request")
        context = TaskContext(task_graph)
        
        result = await self.agent._run_pre_processors(
            mock_role_def,
            lifecycle_functions,
            "What's the weather?",
            context,
            {"location": "Seattle", "timeframe": "current"}
        )
        
        assert "fetch_weather_data" in result
        assert "Sunny in Seattle" in result["fetch_weather_data"]["weather_current"]
    
    @pytest.mark.asyncio
    async def test_post_processing_execution(self):
        """Test post-processing function execution."""
        mock_role_def = Mock()
        mock_role_def.config = {
            "lifecycle": {
                "post_processing": {
                    "enabled": True,
                    "functions": [{"name": "format_for_tts"}]
                }
            }
        }
        
        async def mock_format_tts(result, context, pre_data):
            return result.replace("°F", " degrees Fahrenheit")
        
        lifecycle_functions = {"format_for_tts": mock_format_tts}
        
        # Create a proper TaskContext with TaskGraph
        task_desc = TaskDescription(
            task_name="test_task",
            agent_id="test_agent",
            task_type="test_type",
            prompt="test instruction"
        )
        task_graph = TaskGraph([task_desc], request_id="test_request")
        context = TaskContext(task_graph)
        
        result = await self.agent._run_post_processors(
            mock_role_def,
            lifecycle_functions,
            "Temperature is 72°F",
            context,
            {}
        )
        
        assert "72 degrees Fahrenheit" in result
    
    def test_data_injection_for_llm_context(self):
        """Test injection of pre-processed data into LLM context."""
        mock_role_def = Mock()
        mock_role_def.config = {
            "prompts": {
                "system": "Weather data: {weather_current} in {location_resolved}"
            }
        }
        
        pre_data = {
            "fetch_weather_data": {
                "weather_current": "Sunny 72°F",
                "location_resolved": "Seattle, WA"
            }
        }
        
        result = self.agent._inject_pre_data(
            mock_role_def,
            "What's the weather?",
            pre_data
        )
        
        assert "Sunny 72°F" in result
        assert "Seattle, WA" in result
        assert "What's the weather?" in result
    
    def test_lifecycle_phase_detection(self):
        """Test detection of lifecycle phases."""
        # Test pre-processing detection
        role_def_with_pre = Mock()
        role_def_with_pre.config = {
            "lifecycle": {
                "pre_processing": {"enabled": True}
            }
        }
        
        assert self.agent._has_pre_processing(role_def_with_pre) is True
        
        # Test post-processing detection
        role_def_with_post = Mock()
        role_def_with_post.config = {
            "lifecycle": {
                "post_processing": {"enabled": True}
            }
        }
        
        assert self.agent._has_post_processing(role_def_with_post) is True
        
        # Test no lifecycle
        role_def_no_lifecycle = Mock()
        role_def_no_lifecycle.config = {}
        
        assert self.agent._has_pre_processing(role_def_no_lifecycle) is False
        assert self.agent._has_post_processing(role_def_no_lifecycle) is False


class TestParameterValidation:
    """Test parameter validation and enum constraint handling."""
    
    def test_enum_parameter_validation(self):
        """Test validation of enum parameters."""
        parameter_schema = {
            "timeframe": {
                "type": "string",
                "enum": ["current", "today", "tomorrow"],
                "default": "current"
            }
        }
        
        # Valid enum value
        assert self._validate_parameter("today", parameter_schema["timeframe"]) is True
        
        # Invalid enum value
        assert self._validate_parameter("next_week", parameter_schema["timeframe"]) is False
    
    def test_required_parameter_validation(self):
        """Test validation of required parameters."""
        parameter_schema = {
            "location": {
                "type": "string",
                "required": True
            }
        }
        
        # Required parameter present
        parameters = {"location": "Seattle"}
        assert self._validate_required_parameters(parameters, {"location": parameter_schema["location"]}) is True
        
        # Required parameter missing
        parameters = {}
        assert self._validate_required_parameters(parameters, {"location": parameter_schema["location"]}) is False
    
    def _validate_parameter(self, value, schema):
        """Helper method to validate a single parameter."""
        if "enum" in schema:
            return value in schema["enum"]
        return True
    
    def _validate_required_parameters(self, parameters, schema):
        """Helper method to validate required parameters."""
        for param_name, param_schema in schema.items():
            if param_schema.get("required", False) and param_name not in parameters:
                return False
        return True


if __name__ == "__main__":
    pytest.main([__file__])