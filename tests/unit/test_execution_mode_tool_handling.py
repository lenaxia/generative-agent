"""Unit tests for execution-mode specific tool handling.

Tests that different execution modes (fast_reply, complex_workflow, etc.)
get appropriate tool configurations based on role definitions.
"""

from unittest.mock import Mock

import pytest

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleDefinition
from llm_provider.universal_agent import ExecutionMode, UniversalAgent


class TestExecutionModeToolHandling:
    """Test execution-mode specific tool handling."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        mock_agent = Mock()
        factory.get_agent.return_value = mock_agent
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Mock role registry with weather role."""
        registry = Mock()

        # Mock weather role definition with execution-specific tool config
        weather_role_config = {
            "role": {
                "name": "weather",
                "description": "Weather role with execution-specific tools",
            },
            "tools": {
                "automatic": False,
                "shared": [],
                "fast_reply": {"enabled": False, "available": []},
                "execution_modes": {
                    "fast_reply": [],
                    "workflow": ["get_weather_forecast"],
                },
            },
            "prompts": {"system": "You are a weather specialist."},
        }

        weather_role_def = Mock(spec=RoleDefinition)
        weather_role_def.name = "weather"
        weather_role_def.config = weather_role_config
        weather_role_def.custom_tools = [
            Mock(__name__="get_weather"),
            Mock(__name__="get_weather_forecast"),
        ]

        registry.get_role.return_value = weather_role_def
        return registry

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create UniversalAgent with mocked dependencies."""
        return UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=mock_role_registry
        )

    def test_fast_reply_mode_no_tools(self, universal_agent, mock_role_registry):
        """Test that fast-reply mode gets no custom tools by default."""
        role_def = mock_role_registry.get_role("weather")

        # Test fast-reply mode
        tools = universal_agent._assemble_role_tools(
            role_def, [], ExecutionMode.FAST_REPLY
        )

        # Should only have built-in tools (calculator, file_read, shell)
        tool_names = [getattr(tool, "__name__", str(tool)) for tool in tools]
        assert any("calculator" in name for name in tool_names)
        assert any("file_read" in name for name in tool_names)
        assert any("shell" in name for name in tool_names)

        # Should NOT have custom weather tools
        assert "get_weather" not in tool_names
        assert "get_weather_forecast" not in tool_names

        # Should have exactly 3 tools (built-ins only)
        assert len(tools) == 3

    def test_complex_workflow_mode_has_tools(self, universal_agent, mock_role_registry):
        """Test that complex workflow mode gets configured tools."""
        role_def = mock_role_registry.get_role("weather")

        # Test workflow mode
        tools = universal_agent._assemble_role_tools(
            role_def, [], ExecutionMode.WORKFLOW
        )

        # Should have built-in tools
        tool_names = [getattr(tool, "__name__", str(tool)) for tool in tools]
        assert any("calculator" in name for name in tool_names)
        assert any("file_read" in name for name in tool_names)
        assert any("shell" in name for name in tool_names)

        # Should have custom weather tools (since execution_modes allows them for complex_workflow)
        assert "get_weather" in tool_names
        assert "get_weather_forecast" in tool_names

        # Should have 5 tools (3 built-ins + 2 custom)
        assert len(tools) == 5

    def test_workflow_mode_with_execution_modes_config(
        self, universal_agent, mock_role_registry
    ):
        """Test that workflow mode uses execution_modes config when available."""
        role_def = mock_role_registry.get_role("weather")

        # Test workflow mode with execution_modes configuration
        tools = universal_agent._assemble_role_tools(
            role_def, [], ExecutionMode.WORKFLOW
        )

        # Should have built-in tools
        tool_names = [getattr(tool, "__name__", str(tool)) for tool in tools]
        assert any("calculator" in name for name in tool_names)
        assert any("file_read" in name for name in tool_names)
        assert any("shell" in name for name in tool_names)

        # Should have custom weather tools (execution_modes.workflow allows them)
        assert "get_weather" in tool_names
        assert "get_weather_forecast" in tool_names

        # Should have 5 tools (3 built-ins + 2 custom from execution_modes)
        assert len(tools) == 5

    def test_should_include_custom_tools_logic(self, universal_agent):
        """Test the _should_include_custom_tools decision logic."""
        # Test fast-reply mode with no configuration
        tools_config = {"automatic": False}
        result = universal_agent._should_include_custom_tools(
            tools_config, ExecutionMode.FAST_REPLY, "test_role"
        )
        assert result is False  # Fast-reply default: no tools

        # Test workflow mode with no configuration
        result = universal_agent._should_include_custom_tools(
            tools_config, ExecutionMode.WORKFLOW, "test_role"
        )
        assert result is False  # automatic: false

        # Test with automatic: true
        tools_config = {"automatic": True}
        result = universal_agent._should_include_custom_tools(
            tools_config, ExecutionMode.WORKFLOW, "test_role"
        )
        assert result is True  # automatic: true

        # Test with execution_modes configuration
        tools_config = {
            "automatic": False,
            "execution_modes": {"fast_reply": [], "workflow": ["some_tool"]},
        }

        # Fast-reply should have no tools
        result = universal_agent._should_include_custom_tools(
            tools_config, ExecutionMode.FAST_REPLY, "test_role"
        )
        assert result is False

        # Workflow should have tools
        result = universal_agent._should_include_custom_tools(
            tools_config, ExecutionMode.WORKFLOW, "test_role"
        )
        assert result is True

    def test_execution_mode_enum_values(self):
        """Test that ExecutionMode enum has expected values."""
        assert ExecutionMode.FAST_REPLY == "fast_reply"
        assert ExecutionMode.WORKFLOW == "workflow"

        # Test enum iteration
        modes = list(ExecutionMode)
        assert len(modes) == 2
        assert ExecutionMode.FAST_REPLY in modes
        assert ExecutionMode.WORKFLOW in modes
