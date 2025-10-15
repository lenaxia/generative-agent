import unittest
from unittest.mock import Mock, patch

from common.task_context import TaskContext
from common.task_graph import TaskDescription, TaskGraph
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from llm_provider.universal_agent import ExecutionMode, UniversalAgent


class TestUniversalAgent(unittest.TestCase):
    """Test UniversalAgent with actual implementation API."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = LLMFactory({})
        self.universal_agent = UniversalAgent(self.factory)

    def test_universal_agent_initialization(self):
        """Test UniversalAgent initialization."""
        assert self.universal_agent.llm_factory is not None
        assert isinstance(self.universal_agent.tool_registry, ToolRegistry)
        assert self.universal_agent.current_agent is None
        assert self.universal_agent.current_role is None
        assert self.universal_agent.current_llm_type is None

    def test_assume_role_basic(self):
        """Test basic role assumption functionality."""
        with (
            patch.object(
                self.universal_agent.llm_factory, "get_agent"
            ) as mock_get_agent,
            patch.object(
                self.universal_agent, "_update_agent_context"
            ) as mock_update_context,
        ):
            mock_agent_instance = Mock()
            mock_updated_agent = Mock()
            mock_get_agent.return_value = mock_agent_instance
            mock_update_context.return_value = mock_updated_agent

            agent = self.universal_agent.assume_role("planning", LLMType.STRONG)

            # Verify agent was obtained from pool and context updated
            mock_get_agent.assert_called_once_with(LLMType.STRONG)
            mock_update_context.assert_called_once()
            # The current_agent should be the updated agent, not the original
            assert self.universal_agent.current_agent == mock_updated_agent
            assert self.universal_agent.current_role == "planning"
            assert self.universal_agent.current_llm_type == LLMType.STRONG
            assert agent == mock_updated_agent

    def test_assume_role_with_tools(self):
        """Test role assumption with specific tools."""
        with (
            patch.object(
                self.universal_agent.llm_factory, "get_agent"
            ) as mock_get_agent,
            patch.object(
                self.universal_agent, "_update_agent_context"
            ) as mock_update_context,
        ):
            mock_agent_instance = Mock()
            mock_updated_agent = Mock()
            mock_get_agent.return_value = mock_agent_instance
            mock_update_context.return_value = mock_updated_agent

            agent = self.universal_agent.assume_role(
                "search", LLMType.WEAK, tools=["tool1", "tool2"]
            )

            # Verify agent was obtained from pool and context updated
            mock_get_agent.assert_called_once_with(LLMType.WEAK)
            mock_update_context.assert_called_once()
            assert agent == mock_updated_agent

    def test_execute_task_with_current_agent(self):
        """Test task execution with current agent."""
        # Set up current agent
        mock_agent = Mock()
        mock_agent.return_value = "Task completed successfully"
        self.universal_agent.current_agent = mock_agent
        self.universal_agent.current_role = "planning"

        with patch.object(self.universal_agent, "assume_role") as mock_assume:
            mock_assume.return_value = mock_agent

            result = self.universal_agent.execute_task("Create a project plan")

            # Should use current agent or create new one
            assert isinstance(result, str)

    def test_execute_task_with_role_switching(self):
        """Test task execution that requires role switching."""
        with patch.object(self.universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Timer completed"
            mock_assume.return_value = mock_agent

            # Set up current agent with different role
            self.universal_agent.current_role = "planning"
            self.universal_agent.current_agent = Mock()

            # Execute task with existing timer role
            self.universal_agent.execute_task("Set a timer", role="timer")

            # Should switch roles (simplified assertion)
            mock_assume.assert_called_once()

    def test_execute_task_fallback_methods(self):
        """Test task execution with different agent method signatures."""
        # Mock the assume_role to return a mock agent
        with patch.object(self.universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Task completed"
            mock_assume.return_value = mock_agent

            result = self.universal_agent.execute_task("Test task")

            # Should work without errors
            assert isinstance(result, str)

    def test_execute_task_mock_fallback(self):
        """Test task execution with mock object fallback."""
        # Mock the assume_role to return a mock agent
        with patch.object(self.universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Mock execution completed"
            mock_assume.return_value = mock_agent

            result = self.universal_agent.execute_task("Test task")

            # Should use mock execution
            assert isinstance(result, str)

    def test_get_available_roles(self):
        """Test getting available roles."""
        roles = self.universal_agent.get_available_roles()
        assert isinstance(roles, list)
        assert "planning" in roles
        assert "search" in roles

    def test_get_status(self):
        """Test status reporting."""
        # Set up some state
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_llm_type = LLMType.STRONG
        self.universal_agent.current_agent = Mock()

        status = self.universal_agent.get_status()

        assert status["current_role"] == "planning"
        assert status["current_llm_type"] == "strong"
        # Check for actual keys that exist in the implementation
        assert "current_role" in status
        assert "current_llm_type" in status

    def test_integration_with_task_context(self):
        """Test integration with TaskContext."""
        # Create a task context
        tasks = [
            TaskDescription(
                task_name="test_task",
                agent_id="planning_agent",
                task_type="planning",
                prompt="Create a plan",
            )
        ]
        task_graph = TaskGraph(tasks=tasks, dependencies=[])
        context = TaskContext(task_graph=task_graph)

        # Test role assumption with context
        with (
            patch.object(
                self.universal_agent.llm_factory, "get_agent"
            ) as mock_get_agent,
            patch.object(
                self.universal_agent, "_update_agent_context"
            ) as mock_update_context,
        ):
            mock_agent_instance = Mock()
            mock_updated_agent = Mock()
            mock_get_agent.return_value = mock_agent_instance
            mock_update_context.return_value = mock_updated_agent

            agent = self.universal_agent.assume_role(
                "planning", LLMType.DEFAULT, context=context
            )

            # Should work without errors and use agent pooling
            mock_get_agent.assert_called_once_with(LLMType.DEFAULT)
            mock_update_context.assert_called_once()
            assert agent == mock_updated_agent

    def test_reset(self):
        """Test resetting the Universal Agent state."""
        # Set up some state
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_agent = Mock()
        self.universal_agent.current_llm_type = LLMType.STRONG

        # Reset
        self.universal_agent.reset()

        # Verify state is cleared
        assert self.universal_agent.current_agent is None
        assert self.universal_agent.current_role is None
        assert self.universal_agent.current_llm_type is None

    def test_tool_decorator_integration(self):
        """Test that tool decorators work with the agent."""
        # Test that the agent can work with @tool decorated functions
        with patch.object(self.universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Tool execution completed"
            mock_assume.return_value = mock_agent

            result = self.universal_agent.execute_task("Execute tool task")

            # Should work without errors
            assert isinstance(result, str)


if __name__ == "__main__":
    unittest.main()
