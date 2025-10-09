"""Unit tests for UniversalAgent core functionality.

Tests the UniversalAgent class which provides a unified interface for creating
role-specific agents using the StrandsAgent framework with semantic model types
and tool integration.
"""

from unittest.mock import Mock, patch

import pytest

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from llm_provider.tool_registry import ToolRegistry
from llm_provider.universal_agent import UniversalAgent


class TestUniversalAgentUnit:
    """Unit tests for UniversalAgent core functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        return factory

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create mock MCP client manager."""
        manager = Mock(spec=MCPClientManager)
        manager.get_tools_for_role.return_value = []
        return manager

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.get_tools.return_value = []
        return registry

    @pytest.fixture
    def mock_strands_agent(self):
        """Create mock StrandsAgent Agent."""
        agent = Mock()
        agent.return_value = "Mock agent response"
        return agent

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_mcp_manager):
        """Create UniversalAgent instance for testing."""
        with patch("llm_provider.universal_agent.ToolRegistry") as mock_registry_class:
            mock_registry_instance = Mock(spec=ToolRegistry)
            mock_registry_instance.get_tools.return_value = []
            mock_registry_class.return_value = mock_registry_instance

            agent = UniversalAgent(
                llm_factory=mock_llm_factory, mcp_manager=mock_mcp_manager
            )

            # Replace the tool registry with our mock
            agent.tool_registry = mock_registry_instance
            return agent

    @pytest.fixture
    def sample_task_context(self):
        """Create sample TaskContext for testing."""
        from common.task_graph import TaskDescription

        # Create sample tasks
        task1 = TaskDescription(
            task_name="Test Task 1",
            agent_id="planning_agent",
            task_type="Planning",
            prompt="Plan the first task",
        )

        tasks = [task1]
        dependencies = []

        return TaskContext.from_tasks(
            tasks=tasks, dependencies=dependencies, request_id="test_request_123"
        )

    def test_assume_role_planning(self, universal_agent, mock_strands_agent):
        """Test Universal Agent can assume planning role with STRONG LLM."""
        with patch("llm_provider.universal_agent.Agent") as mock_agent_class:
            # Mock the LLM factory to return a mock agent
            mock_agent_instance = Mock()
            universal_agent.llm_factory.get_agent.return_value = mock_agent_instance
            mock_agent_class.return_value = mock_strands_agent

            # Execute
            agent = universal_agent.assume_role("planning", LLMType.STRONG)

            # Verify agent retrieval through factory
            universal_agent.llm_factory.get_agent.assert_called_once_with(
                LLMType.STRONG
            )

            # Verify agent creation with planning prompt
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert call_args[1]["model"] == mock_agent_instance.model
            assert "planning specialist agent" in call_args[1]["system_prompt"]
            assert "Break down complex tasks" in call_args[1]["system_prompt"]

            # Verify tools were included
            tools = call_args[1]["tools"]
            assert len(tools) >= 3  # At least calculator, file_read, shell

            # Verify state tracking
            assert universal_agent.current_agent == mock_strands_agent
            assert universal_agent.current_role == "planning"
            assert universal_agent.current_llm_type == LLMType.STRONG

            assert agent == mock_strands_agent

    def test_assume_role_search(self, universal_agent, mock_strands_agent):
        """Test Universal Agent can assume search role with WEAK LLM."""
        with patch("llm_provider.universal_agent.Agent") as mock_agent_class:
            # Mock the LLM factory to return a mock agent
            mock_agent_instance = Mock()
            universal_agent.llm_factory.get_agent.return_value = mock_agent_instance
            mock_agent_class.return_value = mock_strands_agent

            # Execute
            universal_agent.assume_role("search", LLMType.WEAK)

            # Verify agent retrieval through factory
            universal_agent.llm_factory.get_agent.assert_called_once_with(LLMType.WEAK)

            # Verify agent creation with search prompt
            call_args = mock_agent_class.call_args
            assert "web search specialist" in call_args[1]["system_prompt"]
            assert "search" in call_args[1]["system_prompt"]

            # Verify state tracking
            assert universal_agent.current_role == "search"
            assert universal_agent.current_llm_type == LLMType.WEAK

    def test_tool_registry_integration(self, universal_agent, mock_strands_agent):
        """Test tool registry is properly integrated."""
        # Setup mock tools from registry
        mock_tool_1 = Mock()
        mock_tool_1.__name__ = "mock_tool_1"
        mock_tool_2 = Mock()
        mock_tool_2.__name__ = "mock_tool_2"

        universal_agent.tool_registry.get_tools.return_value = [
            mock_tool_1,
            mock_tool_2,
        ]

        with (
            patch("llm_provider.universal_agent.Agent") as mock_agent_class,
            patch("llm_provider.universal_agent.BedrockModel") as mock_model_class,
        ):
            mock_agent_class.return_value = mock_strands_agent
            mock_model_class.return_value = Mock()

            # Execute with specific tools
            universal_agent.assume_role("coding", tools=["tool1", "tool2"])

            # Verify tool registry was called with correct tools
            universal_agent.tool_registry.get_tools.assert_called_once_with(
                ["tool1", "tool2"]
            )

            # Verify tools were passed to agent
            call_args = mock_agent_class.call_args
            tools = call_args[1]["tools"]

            # Should include registry tools + common tools (calculator, file_read, shell)
            assert mock_tool_1 in tools
            assert mock_tool_2 in tools
            assert len(tools) >= 5  # 2 registry + 3 common + role-specific

    def test_mcp_client_integration(self, universal_agent, mock_strands_agent):
        """Test MCP client integration works correctly."""
        # Setup mock MCP tools
        mock_mcp_tool_1 = Mock()
        mock_mcp_tool_1.__name__ = "mcp_tool_1"
        mock_mcp_tool_2 = Mock()
        mock_mcp_tool_2.__name__ = "mcp_tool_2"

        universal_agent.mcp_manager.get_tools_for_role.return_value = [
            mock_mcp_tool_1,
            mock_mcp_tool_2,
        ]

        with (
            patch("llm_provider.universal_agent.Agent") as mock_agent_class,
            patch("llm_provider.universal_agent.BedrockModel") as mock_model_class,
        ):
            mock_agent_class.return_value = mock_strands_agent
            mock_model_class.return_value = Mock()

            # Execute
            universal_agent.assume_role("analysis")

            # Verify MCP manager was called for role-specific tools
            universal_agent.mcp_manager.get_tools_for_role.assert_called_once_with(
                "analysis"
            )

            # Verify MCP tools were included
            call_args = mock_agent_class.call_args
            tools = call_args[1]["tools"]

            assert mock_mcp_tool_1 in tools
            assert mock_mcp_tool_2 in tools

    def test_execute_task_success(
        self, universal_agent, mock_strands_agent, sample_task_context
    ):
        """Test successful task execution with role assumption."""
        with (
            patch("llm_provider.universal_agent.Agent") as mock_agent_class,
            patch("llm_provider.universal_agent.BedrockModel") as mock_model_class,
        ):
            mock_agent_class.return_value = mock_strands_agent
            mock_model_class.return_value = Mock()

            # Setup mock response
            mock_strands_agent.return_value = "Task completed successfully"

            # Execute
            result = universal_agent.execute_task(
                instruction="Analyze the data and provide insights",
                role="analysis",
                llm_type=LLMType.DEFAULT,
                context=sample_task_context,
            )

            # Verify role was assumed
            mock_agent_class.assert_called_once()

            # Verify agent was called with instruction
            mock_strands_agent.assert_called_once_with(
                "Analyze the data and provide insights"
            )

            # Verify result
            assert result == "Task completed successfully"

            # Verify state tracking
            assert universal_agent.current_role == "analysis"
            assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_execute_task_with_exception_handling(
        self, universal_agent, mock_strands_agent
    ):
        """Test task execution handles exceptions gracefully."""
        with (
            patch("llm_provider.universal_agent.Agent") as mock_agent_class,
            patch("llm_provider.universal_agent.BedrockModel") as mock_model_class,
        ):
            mock_agent_class.return_value = mock_strands_agent
            mock_model_class.return_value = Mock()

            # Setup mock to raise exception
            mock_strands_agent.side_effect = Exception("Model API error")

            # Execute
            result = universal_agent.execute_task(
                instruction="This will fail", role="default"
            )

            # Verify error was handled and returned as string
            assert "Error executing task" in result
            assert "Model API error" in result

    def test_role_specific_prompts(self, universal_agent):
        """Test that different roles get appropriate system prompts."""
        test_cases = [
            ("planning", "planning specialist agent", "Break down complex tasks"),
            ("search", "search specialist agent", "Perform web searches"),
            ("weather", "weather information specialist", "Retrieve current weather"),
            ("summarizer", "text summarization specialist", "Create concise summaries"),
            ("slack", "Slack integration specialist", "Send messages to Slack"),
            ("coding", "coding specialist agent", "Write clean, efficient code"),
            ("analysis", "analysis specialist agent", "Analyze data and information"),
            (
                "unknown_role",
                "helpful AI assistant",
                "Provide accurate, helpful responses",
            ),
        ]

        for role, expected_phrase_1, expected_phrase_2 in test_cases:
            prompt = universal_agent._get_role_prompt(role)
            assert expected_phrase_1 in prompt
            assert expected_phrase_2 in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
