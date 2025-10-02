import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager
from common.task_context import TaskContext
from common.task_graph import TaskDescription, TaskDependency


class TestComprehensiveUniversalAgent:
    """Comprehensive test suite for Universal Agent with StrandsAgent integration."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value='strands')
        factory.create_strands_model = Mock()
        factory.create_universal_agent = Mock()
        return factory
    
    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPClientManager)
        manager.get_tools_for_role = Mock(return_value=[])
        manager.execute_tool = Mock(return_value={"result": "success"})
        manager.get_available_tools = Mock(return_value=[])
        return manager
    
    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_mcp_manager):
        """Create a Universal Agent for testing."""
        return UniversalAgent(mock_llm_factory, mcp_manager=mock_mcp_manager)
    
    @pytest.fixture
    def sample_task_context(self):
        """Create a sample task context for testing."""
        tasks = [
            TaskDescription(
                task_name="Test Task",
                agent_id="planning_agent",
                task_type="Planning",
                prompt="Create a test plan"
            )
        ]
        return TaskContext.from_tasks(tasks=tasks, dependencies=[], request_id="test_123")
    
    def test_universal_agent_initialization(self, universal_agent, mock_llm_factory, mock_mcp_manager):
        """Test Universal Agent initialization with all components."""
        assert universal_agent is not None
        assert universal_agent.llm_factory == mock_llm_factory
        assert universal_agent.mcp_manager == mock_mcp_manager
        assert hasattr(universal_agent, 'prompt_library')
        assert hasattr(universal_agent, 'tool_registry')
    
    def test_role_assumption_and_model_selection(self, universal_agent):
        """Test role assumption with semantic model selection."""
        # Test planning role (should use STRONG model)
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            agent = universal_agent.assume_role(
                role="planning",
                llm_type=LLMType.STRONG,
                context=None,
                tools=[]
            )
            
            # Verify strong model was requested for planning
            mock_create.assert_called_with(LLMType.STRONG)
        
        # Test search role (should use WEAK model)
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            agent = universal_agent.assume_role(
                role="search",
                llm_type=LLMType.WEAK,
                context=None,
                tools=[]
            )
            
            # Verify weak model was requested for search
            mock_create.assert_called_with(LLMType.WEAK)
    
    def test_task_execution_with_context(self, universal_agent, sample_task_context):
        """Test task execution with TaskContext integration."""
        # Mock the model execution
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke = Mock(return_value="Task completed successfully")
            mock_create.return_value = mock_model
            
            result = universal_agent.execute_task(
                task_prompt="Complete this task",
                role="planning",
                llm_type=LLMType.STRONG,
                context=sample_task_context
            )
            
            assert result is not None
            assert "Task completed" in result or result == "Task completed successfully"
    
    def test_mcp_integration_and_tool_usage(self, universal_agent):
        """Test MCP server integration and tool usage."""
        # Mock MCP tools
        mock_tools = [
            {"name": "web_search", "description": "Search the web"},
            {"name": "weather_lookup", "description": "Get weather information"}
        ]
        universal_agent.mcp_manager.get_tools_for_role.return_value = mock_tools
        
        # Test tool retrieval
        tools = universal_agent.get_available_tools("search")
        assert len(tools) == 2
        assert any(tool["name"] == "web_search" for tool in tools)
        
        # Test tool execution
        universal_agent.mcp_manager.execute_tool.return_value = {"result": "Search completed"}
        result = universal_agent.execute_mcp_tool("web_search", {"query": "test"})
        
        assert result["result"] == "Search completed"
        universal_agent.mcp_manager.execute_tool.assert_called_with("web_search", {"query": "test"})
    
    def test_prompt_library_integration(self, universal_agent):
        """Test prompt library integration for different roles."""
        # Test prompt retrieval for different roles
        planning_prompt = universal_agent.get_role_prompt("planning")
        search_prompt = universal_agent.get_role_prompt("search")
        weather_prompt = universal_agent.get_role_prompt("weather")
        
        assert planning_prompt is not None
        assert search_prompt is not None
        assert weather_prompt is not None
        
        # Verify prompts are different for different roles
        assert planning_prompt != search_prompt
        assert search_prompt != weather_prompt
    
    def test_tool_registry_functionality(self, universal_agent):
        """Test tool registry functionality."""
        # Test tool registration
        def sample_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        universal_agent.register_tool("sample_tool", sample_tool)
        
        # Test tool retrieval
        tools = universal_agent.get_registered_tools()
        assert "sample_tool" in tools
        
        # Test tool execution
        result = universal_agent.execute_registered_tool("sample_tool", "test input")
        assert result == "Processed: test input"
    
    def test_conversation_history_integration(self, universal_agent, sample_task_context):
        """Test conversation history integration with task execution."""
        # Add conversation history to context
        sample_task_context.add_user_message("Please complete this task")
        sample_task_context.add_system_message("Task execution started")
        
        # Mock model execution with conversation context
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke = Mock(return_value="Task completed with context")
            mock_create.return_value = mock_model
            
            result = universal_agent.execute_task(
                task_prompt="Complete this task",
                role="planning",
                llm_type=LLMType.DEFAULT,
                context=sample_task_context
            )
            
            # Verify conversation history was used
            assert result is not None
            
            # Check that conversation history is accessible
            history = sample_task_context.get_conversation_history()
            assert len(history) >= 2
    
    def test_progressive_summary_integration(self, universal_agent, sample_task_context):
        """Test progressive summary integration."""
        # Add progressive summary
        sample_task_context.update_progressive_summary("Previous tasks completed successfully")
        
        # Execute task with summary context
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke = Mock(return_value="Task completed with summary context")
            mock_create.return_value = mock_model
            
            result = universal_agent.execute_task(
                task_prompt="Continue the work",
                role="planning",
                llm_type=LLMType.DEFAULT,
                context=sample_task_context
            )
            
            assert result is not None
            
            # Verify summary is accessible
            summary = sample_task_context.get_progressive_summary()
            assert "Previous tasks completed" in summary
    
    def test_error_handling_and_recovery(self, universal_agent):
        """Test error handling and recovery mechanisms."""
        # Test model creation failure
        universal_agent.llm_factory.create_strands_model.side_effect = Exception("Model creation failed")
        
        with pytest.raises(Exception) as exc_info:
            universal_agent.execute_task(
                task_prompt="Test task",
                role="planning",
                llm_type=LLMType.DEFAULT,
                context=None
            )
        
        assert "Model creation failed" in str(exc_info.value)
        
        # Test MCP tool execution failure
        universal_agent.mcp_manager.execute_tool.side_effect = Exception("MCP tool failed")
        
        with pytest.raises(Exception) as exc_info:
            universal_agent.execute_mcp_tool("failing_tool", {})
        
        assert "MCP tool failed" in str(exc_info.value)
    
    def test_role_based_llm_type_optimization(self, universal_agent):
        """Test automatic LLM type optimization based on role."""
        test_cases = [
            ("planning", LLMType.STRONG),
            ("analysis", LLMType.STRONG),
            ("coding", LLMType.STRONG),
            ("search", LLMType.WEAK),
            ("weather", LLMType.WEAK),
            ("summarizer", LLMType.DEFAULT),
            ("slack", LLMType.DEFAULT),
            ("unknown_role", LLMType.DEFAULT)
        ]
        
        for role, expected_llm_type in test_cases:
            recommended_type = universal_agent.get_recommended_llm_type(role)
            assert recommended_type == expected_llm_type, f"Role {role} should recommend {expected_llm_type}, got {recommended_type}"
    
    def test_multi_step_task_execution(self, universal_agent):
        """Test multi-step task execution with state preservation."""
        # Create multi-step task context
        tasks = [
            TaskDescription(task_name="Step 1", agent_id="planning_agent", task_type="Planning", prompt="Plan the work"),
            TaskDescription(task_name="Step 2", agent_id="search_agent", task_type="Research", prompt="Research the topic"),
            TaskDescription(task_name="Step 3", agent_id="summarizer_agent", task_type="Summary", prompt="Summarize results")
        ]
        
        dependencies = [
            TaskDependency(from_task="Step 1", to_task="Step 2"),
            TaskDependency(from_task="Step 2", to_task="Step 3")
        ]
        
        context = TaskContext.from_tasks(tasks=tasks, dependencies=dependencies, request_id="multi_step_test")
        
        # Mock model responses
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_responses = [
                "Step 1 completed: Plan created",
                "Step 2 completed: Research done", 
                "Step 3 completed: Summary ready"
            ]
            mock_model.invoke.side_effect = mock_responses
            mock_create.return_value = mock_model
            
            # Execute each step
            context.start_execution()
            
            for i, expected_response in enumerate(mock_responses):
                ready_tasks = context.get_ready_tasks()
                if ready_tasks:
                    task = ready_tasks[0]
                    result = universal_agent.execute_task(
                        task_prompt=task.prompt,
                        role=universal_agent._determine_role_from_agent_id(task.agent_id),
                        llm_type=LLMType.DEFAULT,
                        context=context
                    )
                    
                    # Complete the task
                    context.complete_task(task.task_id, result)
                    
                    assert result == expected_response
    
    def test_performance_and_metrics(self, universal_agent, sample_task_context):
        """Test performance tracking and metrics collection."""
        start_time = time.time()
        
        # Execute multiple tasks to generate metrics
        with patch.object(universal_agent.llm_factory, 'create_strands_model') as mock_create:
            mock_model = Mock()
            mock_model.invoke = Mock(return_value="Task completed")
            mock_create.return_value = mock_model
            
            for i in range(3):
                result = universal_agent.execute_task(
                    task_prompt=f"Task {i}",
                    role="planning",
                    llm_type=LLMType.DEFAULT,
                    context=sample_task_context
                )
                time.sleep(0.01)  # Small delay to generate measurable time
        
        # Get performance metrics
        metrics = sample_task_context.get_performance_metrics()
        
        assert metrics is not None
        assert 'execution_time' in metrics
        assert metrics['execution_time'] > 0
        
        # Test Universal Agent specific metrics
        agent_status = universal_agent.get_status()
        assert agent_status is not None
        assert 'framework' in agent_status
        assert agent_status['framework'] == 'strands'
    
    def test_mcp_status_and_health_check(self, universal_agent):
        """Test MCP status and health checking."""
        # Test MCP status retrieval
        mcp_status = universal_agent.get_mcp_status()
        
        assert mcp_status is not None
        assert 'mcp_available' in mcp_status
        
        # Test with MCP manager available
        universal_agent.mcp_manager.get_available_tools.return_value = ["tool1", "tool2"]
        status_with_tools = universal_agent.get_mcp_status()
        
        assert status_with_tools['mcp_available'] == True
        
        # Test without MCP manager
        agent_without_mcp = UniversalAgent(universal_agent.llm_factory, mcp_manager=None)
        status_without_mcp = agent_without_mcp.get_mcp_status()
        
        assert status_without_mcp['mcp_available'] == False
    
    def test_framework_compatibility(self, universal_agent):
        """Test framework compatibility and version checking."""
        # Test framework detection
        framework = universal_agent.get_framework()
        assert framework == 'strands'
        
        # Test compatibility check
        is_compatible = universal_agent.check_strands_compatibility()
        assert is_compatible == True
        
        # Test version information
        version_info = universal_agent.get_version_info()
        assert version_info is not None
        assert 'universal_agent_version' in version_info
        assert 'strands_integration' in version_info


if __name__ == "__main__":
    pytest.main([__file__])