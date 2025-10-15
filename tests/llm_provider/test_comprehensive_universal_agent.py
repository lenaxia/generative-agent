from unittest.mock import Mock, patch

import pytest

from common.communication_manager import DeliveryGuarantee
from common.task_context import TaskContext
from common.task_graph import TaskDescription, TaskGraph
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from llm_provider.universal_agent import UniversalAgent


class TestComprehensiveUniversalAgent:
    """Comprehensive tests for UniversalAgent with actual implementation."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value="strands")
        return factory

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPClientManager)
        manager.get_tools_for_role = Mock(return_value=[])
        manager.get_registered_servers = Mock(return_value=[])
        manager.get_all_tools = Mock(return_value=[])
        manager.get_server_configs = Mock(return_value={})
        return manager

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_mcp_manager):
        """Create a UniversalAgent for testing."""
        return UniversalAgent(mock_llm_factory, mcp_manager=mock_mcp_manager)

    @pytest.fixture
    def sample_task_context(self):
        """Create a sample TaskContext for testing."""
        tasks = [
            TaskDescription(
                task_name="test_task",
                agent_id="planning_agent",
                task_type="planning",
                prompt="Create a comprehensive plan",
            )
        ]
        # Create task graph but don't store unused variable
        TaskGraph(tasks=tasks, dependencies=[])
        return TaskContext.from_tasks(
            tasks=tasks, dependencies=[], request_id="test_123"
        )

    def test_universal_agent_initialization(
        self, universal_agent, mock_llm_factory, mock_mcp_manager
    ):
        """Test Universal Agent initialization with all components."""
        assert universal_agent is not None
        assert universal_agent.llm_factory == mock_llm_factory
        assert universal_agent.mcp_manager == mock_mcp_manager
        assert hasattr(universal_agent, "tool_registry")
        assert hasattr(universal_agent, "current_agent")
        assert hasattr(universal_agent, "current_role")

    def test_role_assumption_and_model_selection(self, universal_agent):
        """Test role assumption with semantic model selection."""
        # Test planning role (should use STRONG model)
        with patch("llm_provider.universal_agent.Agent") as mock_agent_class:
            mock_model = Mock()
            universal_agent.llm_factory.create_strands_model.return_value = mock_model
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            # Assume role but don't store unused agent variable
            universal_agent.assume_role(
                role="planning", llm_type=LLMType.STRONG, context=None, tools=[]
            )

            # Verify agent was obtained from pool with STRONG model type
            universal_agent.llm_factory.get_agent.assert_called_with(LLMType.STRONG)

        # Test search role (should use WEAK model)
        with patch("llm_provider.universal_agent.Agent") as mock_agent_class:
            mock_model = Mock()
            universal_agent.llm_factory.create_strands_model.return_value = mock_model
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            # Assume role but don't store unused agent variable
            universal_agent.assume_role(
                role="search", llm_type=LLMType.WEAK, context=None, tools=[]
            )

            # Verify agent was obtained from pool with WEAK model type
            universal_agent.llm_factory.get_agent.assert_called_with(LLMType.WEAK)

    def test_task_execution_with_context(self, universal_agent, sample_task_context):
        """Test task execution with TaskContext integration."""
        # Mock the execution
        with patch.object(universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Task completed successfully"
            mock_assume.return_value = mock_agent

            result = universal_agent.execute_task(
                instruction="Complete this task",
                role="planning",
                llm_type=LLMType.STRONG,
                context=sample_task_context,
            )

            # Should work without errors
            assert isinstance(result, str)

    def test_mcp_integration_and_tool_usage(self, universal_agent):
        """Test MCP integration and tool usage."""
        # Test that MCP manager is integrated
        assert universal_agent.mcp_manager is not None

        # Test MCP status (mock to avoid server dependency)
        with patch.object(
            universal_agent.mcp_manager, "get_registered_servers", return_value=[]
        ):
            status = universal_agent.get_mcp_status()
            assert isinstance(status, dict)
            assert "mcp_available" in status

    def test_tool_registry_functionality(self, universal_agent):
        """Test tool registry functionality."""
        # Test that tool registry exists and works
        assert universal_agent.tool_registry is not None

        # Test getting tools (should work without errors)
        tools = universal_agent.tool_registry.get_tools([])
        assert isinstance(tools, list)

    def test_conversation_history_integration(
        self, universal_agent, sample_task_context
    ):
        """Test conversation history integration."""
        # Test that context is properly handled
        with patch.object(universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Conversation processed"
            mock_assume.return_value = mock_agent

            result = universal_agent.execute_task(
                instruction="Process conversation",
                role="planning",
                context=sample_task_context,
            )

            # Should work without errors
            assert isinstance(result, str)

    def test_multi_step_task_execution(self, universal_agent):
        """Test multi-step task execution."""
        # Test that multiple tasks can be executed
        with patch.object(universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Step completed"
            mock_assume.return_value = mock_agent

            # Execute multiple tasks
            results = []
            for i in range(3):
                result = universal_agent.execute_task(f"Step {i}")
                results.append(result)

            # All should complete
            assert len(results) == 3
            for result in results:
                assert isinstance(result, str)

    def test_performance_and_metrics(self, universal_agent):
        """Test performance and metrics."""
        # Test status reporting (mock MCP manager to avoid server dependency)
        with patch.object(
            universal_agent.mcp_manager, "get_registered_servers", return_value=[]
        ):
            status = universal_agent.get_status()
            assert isinstance(status, dict)

        # Test available roles
        roles = universal_agent.get_available_roles()
        assert isinstance(roles, list)
        assert len(roles) > 0

    def test_mcp_status_and_health_check(self, universal_agent):
        """Test MCP status and health check."""
        # Test MCP status (mock to avoid server dependency)
        with patch.object(
            universal_agent.mcp_manager, "get_registered_servers", return_value=[]
        ):
            mcp_status = universal_agent.get_mcp_status()
            assert isinstance(mcp_status, dict)
            assert "mcp_available" in mcp_status

    def test_framework_compatibility(self, universal_agent):
        """Test framework compatibility."""
        # Test that the agent works with the Strands framework
        assert universal_agent.llm_factory is not None

        # Test role assumption works
        with (
            patch.object(universal_agent, "_create_strands_model") as mock_create,
            patch("llm_provider.universal_agent.Agent") as mock_agent_class,
        ):
            mock_model = Mock()
            mock_create.return_value = mock_model
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            agent = universal_agent.assume_role("planning")
            assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__])
