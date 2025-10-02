import unittest
from unittest.mock import Mock, patch, MagicMock
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from common.task_context import TaskContext
from common.task_graph import TaskGraph, TaskDescription


class TestUniversalAgent(unittest.TestCase):
    """Test UniversalAgent with actual implementation API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = LLMFactory({})
        self.universal_agent = UniversalAgent(self.factory)
    
    def test_universal_agent_initialization(self):
        """Test UniversalAgent initialization."""
        self.assertIsNotNone(self.universal_agent.llm_factory)
        self.assertIsInstance(self.universal_agent.tool_registry, ToolRegistry)
        self.assertIsNone(self.universal_agent.current_agent)
        self.assertIsNone(self.universal_agent.current_role)
        self.assertIsNone(self.universal_agent.current_llm_type)
    
    def test_assume_role_basic(self):
        """Test basic role assumption functionality."""
        with patch.object(self.universal_agent, '_create_strands_model') as mock_model, \
             patch('llm_provider.universal_agent.Agent') as mock_agent_class:
            
            mock_model.return_value = Mock()
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            agent = self.universal_agent.assume_role("planning", LLMType.STRONG)
            
            # Verify agent was created and stored
            self.assertEqual(self.universal_agent.current_agent, mock_agent_instance)
            self.assertEqual(self.universal_agent.current_role, "planning")
            self.assertEqual(self.universal_agent.current_llm_type, LLMType.STRONG)
            self.assertEqual(agent, mock_agent_instance)
    
    def test_assume_role_with_tools(self):
        """Test role assumption with specific tools."""
        with patch.object(self.universal_agent, '_create_strands_model') as mock_model, \
             patch('llm_provider.universal_agent.Agent') as mock_agent_class:
            
            mock_model.return_value = Mock()
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            agent = self.universal_agent.assume_role(
                "search",
                LLMType.WEAK,
                tools=["tool1", "tool2"]
            )
            
            # Verify agent was created
            self.assertEqual(agent, mock_agent_instance)
            mock_agent_class.assert_called_once()
    
    def test_execute_task_with_current_agent(self):
        """Test task execution with current agent."""
        # Set up current agent
        mock_agent = Mock()
        mock_agent.return_value = "Task completed successfully"
        self.universal_agent.current_agent = mock_agent
        self.universal_agent.current_role = "planning"
        
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_assume.return_value = mock_agent
            
            result = self.universal_agent.execute_task("Create a project plan")
            
            # Should use current agent or create new one
            self.assertIsInstance(result, str)
    
    def test_execute_task_with_role_switching(self):
        """Test task execution that requires role switching."""
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Search completed"
            mock_assume.return_value = mock_agent
            
            # Set up current agent with different role
            self.universal_agent.current_role = "planning"
            self.universal_agent.current_agent = Mock()
            
            result = self.universal_agent.execute_task(
                "Search for information",
                role="search"
            )
            
            # Should switch roles (only 3 parameters in actual implementation)
            mock_assume.assert_called_once_with("search", LLMType.DEFAULT, None)
    
    def test_execute_task_fallback_methods(self):
        """Test task execution with different agent method signatures."""
        # Mock the assume_role to return a mock agent
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Task completed"
            mock_assume.return_value = mock_agent
            
            result = self.universal_agent.execute_task("Test task")
            
            # Should work without errors
            self.assertIsInstance(result, str)
    
    def test_execute_task_mock_fallback(self):
        """Test task execution with mock object fallback."""
        # Mock the assume_role to return a mock agent
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Mock execution completed"
            mock_assume.return_value = mock_agent
            
            result = self.universal_agent.execute_task("Test task")
            
            # Should use mock execution
            self.assertIsInstance(result, str)
    
    def test_get_available_roles(self):
        """Test getting available roles."""
        roles = self.universal_agent.get_available_roles()
        self.assertIsInstance(roles, list)
        self.assertIn("planning", roles)
        self.assertIn("search", roles)
    
    def test_get_status(self):
        """Test status reporting."""
        # Set up some state
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_llm_type = LLMType.STRONG
        self.universal_agent.current_agent = Mock()
        
        status = self.universal_agent.get_status()
        
        self.assertEqual(status["current_role"], "planning")
        self.assertEqual(status["current_llm_type"], "strong")
        # Check for actual keys that exist in the implementation
        self.assertIn("current_role", status)
        self.assertIn("current_llm_type", status)
    
    def test_integration_with_task_context(self):
        """Test integration with TaskContext."""
        # Create a task context
        tasks = [
            TaskDescription(
                task_name="test_task",
                agent_id="planning_agent", 
                task_type="planning",
                prompt="Create a plan"
            )
        ]
        task_graph = TaskGraph(tasks=tasks, dependencies=[])
        context = TaskContext(task_graph=task_graph)
        
        # Test role assumption with context
        with patch.object(self.universal_agent, '_create_strands_model') as mock_model, \
             patch('llm_provider.universal_agent.Agent') as mock_agent_class:
            
            mock_model.return_value = Mock()
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            agent = self.universal_agent.assume_role(
                "planning",
                LLMType.DEFAULT,
                context=context
            )
            
            # Should work without errors
            self.assertEqual(agent, mock_agent_instance)
    
    def test_reset(self):
        """Test resetting the Universal Agent state."""
        # Set up some state
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_agent = Mock()
        self.universal_agent.current_llm_type = LLMType.STRONG
        
        # Reset
        self.universal_agent.reset()
        
        # Verify state is cleared
        self.assertIsNone(self.universal_agent.current_agent)
        self.assertIsNone(self.universal_agent.current_role)
        self.assertIsNone(self.universal_agent.current_llm_type)
    
    def test_tool_decorator_integration(self):
        """Test that tool decorators work with the agent."""
        # Test that the agent can work with @tool decorated functions
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Tool execution completed"
            mock_assume.return_value = mock_agent
            
            result = self.universal_agent.execute_task("Execute tool task")
            
            # Should work without errors
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()