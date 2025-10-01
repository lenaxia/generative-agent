import unittest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock dependencies before importing
sys.modules['langchain'] = Mock()
sys.modules['langchain.prompts'] = Mock()
sys.modules['langchain.tools'] = Mock()
sys.modules['langchain_openai'] = Mock()
sys.modules['langchain_aws'] = Mock()
sys.modules['langchain_core'] = Mock()
sys.modules['langchain_core.output_parsers'] = Mock()

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from llm_provider.tool_registry import ToolRegistry, tool
from common.task_context import TaskContext, TaskGraph, TaskDescription
from config.base_config import BaseConfig


class TestUniversalAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for Universal Agent."""
        # Mock configurations
        self.mock_config = Mock(spec=BaseConfig)
        self.mock_config.name = "test_config"
        self.mock_config.provider_name = "bedrock"
        self.mock_config.provider_type = "bedrock"
        self.mock_config.model_id = "us.amazon.nova-pro-v1:0"
        self.mock_config.temperature = 0.3
        self.mock_config.additional_params = {}
        
        self.configs = {
            LLMType.DEFAULT: [self.mock_config],
            LLMType.STRONG: [self.mock_config],
            LLMType.WEAK: [self.mock_config]
        }
        
        # Create factory and universal agent
        self.factory = LLMFactory(configs=self.configs, framework="strands")
        self.universal_agent = UniversalAgent(self.factory)

    def test_universal_agent_initialization(self):
        """Test Universal Agent initialization."""
        self.assertIsNotNone(self.universal_agent.llm_factory)
        self.assertIsNotNone(self.universal_agent.tool_registry)
        self.assertIsNone(self.universal_agent.current_agent)
        self.assertIsNone(self.universal_agent.current_role)
        self.assertIsNone(self.universal_agent.current_llm_type)

    def test_assume_role_basic(self):
        """Test basic role assumption functionality."""
        with patch.object(self.factory, 'create_universal_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            agent = self.universal_agent.assume_role("planning", LLMType.STRONG)
            
            # Verify factory was called correctly
            mock_create.assert_called_once_with(
                llm_type=LLMType.STRONG,
                role="planning",
                tools=[]
            )
            
            # Verify state was updated
            self.assertEqual(self.universal_agent.current_role, "planning")
            self.assertEqual(self.universal_agent.current_llm_type, LLMType.STRONG)
            self.assertEqual(self.universal_agent.current_agent, mock_agent)
            self.assertEqual(agent, mock_agent)

    def test_assume_role_with_tools(self):
        """Test role assumption with specific tools."""
        # Add some tools to registry
        self.universal_agent.tool_registry.add_tool("tool1", lambda x: x, "Test tool 1")
        self.universal_agent.tool_registry.add_tool("tool2", lambda x: x, "Test tool 2")
        
        with patch.object(self.factory, 'create_universal_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            agent = self.universal_agent.assume_role(
                "search", 
                LLMType.WEAK, 
                tools=["tool1", "tool2"]
            )
            
            # Verify tools were passed
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            self.assertEqual(call_args[1]['llm_type'], LLMType.WEAK)
            self.assertEqual(call_args[1]['role'], "search")
            self.assertEqual(len(call_args[1]['tools']), 2)

    def test_execute_task_with_current_agent(self):
        """Test task execution with current agent."""
        # Set up current agent
        mock_agent = Mock()
        mock_agent.return_value = "Task completed successfully"
        self.universal_agent.current_agent = mock_agent
        self.universal_agent.current_role = "planning"
        
        result = self.universal_agent.execute_task("Create a project plan")
        
        mock_agent.assert_called_once_with("Create a project plan")
        self.assertEqual(result, "Task completed successfully")

    def test_execute_task_with_role_switching(self):
        """Test task execution that requires role switching."""
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "Search completed"
            mock_assume.return_value = mock_agent
            
            # Set up current agent with different role
            self.universal_agent.current_role = "planning"
            self.universal_agent.current_agent = Mock()
            
            # Mock the agent to be callable
            self.universal_agent.current_agent = mock_agent
            
            result = self.universal_agent.execute_task(
                "Search for information",
                role="search"
            )
            
            # Should switch roles
            mock_assume.assert_called_once_with("search", LLMType.DEFAULT, None, None)

    def test_execute_task_fallback_methods(self):
        """Test task execution with different agent method signatures."""
        # Test that the execute_task method can handle different agent interfaces
        # This test verifies the fallback mechanism works
        
        # Create a simple non-callable object
        class SimpleAgent:
            def run(self, prompt):
                return f"Run method called with: {prompt}"
        
        simple_agent = SimpleAgent()
        self.universal_agent.current_agent = simple_agent
        self.universal_agent.current_role = "test"
        
        result = self.universal_agent.execute_task("Test task")
        
        # Should call the run method
        self.assertEqual(result, "Run method called with: Test task")

    def test_execute_task_mock_fallback(self):
        """Test task execution with mock object fallback."""
        # Create a simple object that's not callable
        class NonCallableAgent:
            pass
        
        non_callable_agent = NonCallableAgent()
        
        self.universal_agent.current_agent = non_callable_agent
        self.universal_agent.current_role = "test_role"
        
        result = self.universal_agent.execute_task("Test task")
        
        # Should use fallback mock execution
        self.assertIsInstance(result, str)
        self.assertIn("Mock execution", result)
        self.assertIn("Test task", result)
        self.assertIn("test_role", result)

    def test_get_available_roles(self):
        """Test getting available roles from prompt library."""
        roles = self.universal_agent.get_available_roles()
        
        # Should include default roles from prompt library
        self.assertIn("default", roles)
        self.assertIn("planning", roles)
        self.assertIn("search", roles)
        self.assertIsInstance(roles, list)

    def test_get_available_tools(self):
        """Test getting available tools from tool registry."""
        # Add some tools
        self.universal_agent.add_tool("test_tool", lambda x: x, "Test tool")
        
        tools = self.universal_agent.get_available_tools()
        
        self.assertIn("test_tool", tools)
        self.assertIsInstance(tools, list)

    def test_add_remove_tool(self):
        """Test adding and removing tools."""
        def test_func(x):
            return x * 2
        
        # Add tool
        self.universal_agent.add_tool("multiply", test_func, "Multiply by 2")
        
        tools = self.universal_agent.get_available_tools()
        self.assertIn("multiply", tools)
        
        # Remove tool
        self.universal_agent.remove_tool("multiply")
        
        tools = self.universal_agent.get_available_tools()
        self.assertNotIn("multiply", tools)

    def test_get_role_configuration(self):
        """Test getting role configuration."""
        config = self.universal_agent.get_role_configuration("planning")
        
        self.assertEqual(config["role"], "planning")
        self.assertIn("prompt", config)
        self.assertIn("available_tools", config)
        self.assertIn("recommended_llm_type", config)
        self.assertEqual(config["recommended_llm_type"], LLMType.STRONG)

    def test_recommended_llm_type_mapping(self):
        """Test LLM type recommendations for different roles."""
        # Test complex roles get STRONG models
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("planning"),
            LLMType.STRONG
        )
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("analysis"),
            LLMType.STRONG
        )
        
        # Test simple roles get WEAK models
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("search"),
            LLMType.WEAK
        )
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("weather"),
            LLMType.WEAK
        )
        
        # Test balanced roles get DEFAULT models
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("summarizer"),
            LLMType.DEFAULT
        )
        
        # Test unknown roles get DEFAULT
        self.assertEqual(
            self.universal_agent._get_recommended_llm_type("unknown_role"),
            LLMType.DEFAULT
        )

    def test_optimize_for_cost(self):
        """Test cost optimization."""
        # Set up current role
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            self.universal_agent.current_role = "planning"
            
            self.universal_agent.optimize_for_cost()
            
            mock_assume.assert_called_once_with("planning", LLMType.WEAK)

    def test_optimize_for_performance(self):
        """Test performance optimization."""
        # Set up current role
        with patch.object(self.universal_agent, 'assume_role') as mock_assume:
            self.universal_agent.current_role = "search"
            
            self.universal_agent.optimize_for_performance()
            
            mock_assume.assert_called_once_with("search", LLMType.STRONG)

    def test_reset(self):
        """Test agent reset functionality."""
        # Set up some state
        self.universal_agent.current_agent = Mock()
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_llm_type = LLMType.STRONG
        
        # Reset
        self.universal_agent.reset()
        
        # Verify state is cleared
        self.assertIsNone(self.universal_agent.current_agent)
        self.assertIsNone(self.universal_agent.current_role)
        self.assertIsNone(self.universal_agent.current_llm_type)

    def test_get_status(self):
        """Test status reporting."""
        # Set up some state
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_llm_type = LLMType.STRONG
        self.universal_agent.current_agent = Mock()
        
        status = self.universal_agent.get_status()
        
        self.assertEqual(status["current_role"], "planning")
        self.assertEqual(status["current_llm_type"], "strong")
        self.assertTrue(status["has_active_agent"])
        self.assertGreater(status["available_roles"], 0)
        self.assertEqual(status["framework"], "strands")

    def test_string_representations(self):
        """Test string and repr methods."""
        self.universal_agent.current_role = "planning"
        self.universal_agent.current_llm_type = LLMType.STRONG
        
        str_repr = str(self.universal_agent)
        self.assertIn("planning", str_repr)
        self.assertIn("strong", str_repr)
        self.assertIn("strands", str_repr)
        
        # __repr__ should be same as __str__
        self.assertEqual(str_repr, repr(self.universal_agent))

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
        with patch.object(self.factory, 'create_universal_agent') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            agent = self.universal_agent.assume_role(
                "planning", 
                LLMType.DEFAULT, 
                context=context
            )
            
            # Should work without errors
            self.assertEqual(agent, mock_agent)

    def test_tool_decorator_integration(self):
        """Test integration with @tool decorator."""
        # Clear global registry for clean test
        ToolRegistry.get_global_registry().clear()
        
        # Define a tool using decorator
        @tool(name="test_decorated_tool", description="A test tool", role="testing")
        def decorated_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        # Tool should be registered
        global_registry = ToolRegistry.get_global_registry()
        self.assertIn("test_decorated_tool", global_registry.list_tools())
        
        # Tool should be associated with role
        role_tools = global_registry.get_tools_for_role("testing")
        self.assertIn("test_decorated_tool", role_tools)


if __name__ == '__main__':
    unittest.main()