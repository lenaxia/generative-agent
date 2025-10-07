"""
Unit tests for role fallback functionality in UniversalAgent.

Tests the behavior when "None" is specified or when a requested role
is not found in the registry - should fall back to default role or basic agent.
"""

from unittest.mock import Mock, patch

import pytest

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRoleFallbackBehavior:
    """Test suite for role fallback functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content='{"selected_tools": ["search_tools", "data_processing"], "system_prompt": "You are a specialized search and data processing agent."}'
        )
        factory.create_strands_model.return_value = mock_model
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Create a mock role registry with default role."""
        registry = Mock(spec=RoleRegistry)

        # Mock default role definition
        default_role = Mock(spec=RoleDefinition)
        default_role.name = "default"
        default_role.config = {
            "prompts": {"system": "You are a helpful AI assistant."},
            "tools": {"shared": ["basic_tool"]},
        }
        default_role.custom_tools = []

        # Mock role lookup - return default for "default", None for others
        def mock_get_role(role_name):
            if role_name == "default":
                return default_role
            return None

        registry.get_role.side_effect = mock_get_role
        registry.get_shared_tool.return_value = Mock()

        return registry

    @pytest.fixture
    def mock_task_context(self):
        """Create a mock task context with task information."""
        context = Mock(spec=TaskContext)

        # Mock task graph with pending task
        mock_task_graph = Mock()
        mock_task_node = Mock()
        mock_task_node.task_name = "Search for weather data"
        mock_task_node.prompt = "Find current weather information for Seattle"
        mock_task_node.task_type = "search"
        mock_task_node.status = Mock(value="PENDING")

        mock_task_graph.nodes = {"task_1": mock_task_node}
        context.task_graph = mock_task_graph

        return context

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create UniversalAgent instance with mocked dependencies."""
        return UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=mock_role_registry
        )

    def test_assume_role_with_none_falls_back_to_default(
        self, universal_agent, mock_task_context
    ):
        """Test that role="None" falls back to default role."""
        with patch("strands.Agent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Test with role="None"
            result = universal_agent.assume_role(
                "None", LLMType.DEFAULT, mock_task_context
            )

            # Verify fallback to default role
            assert result is not None
            assert universal_agent.current_role == "default"
            assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_assume_role_with_missing_role_falls_back_to_default(
        self, universal_agent, mock_task_context
    ):
        """Test that missing role falls back to default role."""
        with patch("strands.Agent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Test with non-existent role
            result = universal_agent.assume_role(
                "non_existent_role", LLMType.DEFAULT, mock_task_context
            )

            # Verify fallback to default role
            assert result is not None
            assert universal_agent.current_role == "default"
            assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_assume_role_with_missing_default_creates_basic_agent(
        self, mock_llm_factory
    ):
        """Test that missing default role creates basic agent."""
        # Create registry with no roles at all
        empty_registry = Mock(spec=RoleRegistry)
        empty_registry.get_role.return_value = None

        agent = UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=empty_registry
        )

        with patch("strands.Agent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Call with None role when no default exists
            result = agent.assume_role("None", LLMType.DEFAULT, None)

            # Should create basic agent
            assert result is not None
            assert agent.current_role == "basic"
            assert agent.current_llm_type == LLMType.DEFAULT

    def test_basic_agent_creation_with_tools(self, universal_agent):
        """Test that basic agent creation works with additional tools."""
        with patch("strands.Agent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Test basic agent creation
            result = universal_agent._create_basic_agent(
                LLMType.DEFAULT, None, ["test_tool"]
            )

            # Verify basic agent was created
            assert result is not None
            assert universal_agent.current_role == "basic"
            assert universal_agent.current_llm_type == LLMType.DEFAULT

    def test_no_invoke_method_called(self, universal_agent, mock_task_context):
        """Test that no invoke method is called on models (the original bug)."""
        with patch("strands.Agent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # This should not trigger any model.invoke() calls
            result = universal_agent.assume_role(
                "None", LLMType.DEFAULT, mock_task_context
            )

            # Verify no invoke was called on the factory's models
            mock_model = universal_agent.llm_factory.create_strands_model.return_value
            assert not hasattr(mock_model, "invoke") or not mock_model.invoke.called

            # Should still create an agent successfully
            assert result is not None
