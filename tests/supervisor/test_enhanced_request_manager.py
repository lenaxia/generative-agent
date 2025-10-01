import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

from supervisor.enhanced_request_manager import EnhancedRequestManager
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from common.message_bus import MessageBus, MessageType
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskGraph, TaskDescription, TaskNode, TaskStatus
from common.request_model import RequestMetadata


class TestEnhancedRequestManager:
    """Test suite for EnhancedRequestManager."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.create_universal_agent = Mock()
        factory.prompt_library = Mock()
        factory.prompt_library.get_prompt = Mock(return_value="Test prompt for {role}")
        return factory
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create a mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus
    
    @pytest.fixture
    def mock_universal_agent(self):
        """Create a mock universal agent."""
        agent = Mock(spec=UniversalAgent)
        agent.assume_role = Mock()
        agent.execute_task = Mock(return_value="Mock task result")
        return agent
    
    @pytest.fixture
    def enhanced_request_manager(self, mock_llm_factory, mock_message_bus, mock_universal_agent):
        """Create an enhanced request manager for testing."""
        with patch('supervisor.enhanced_request_manager.UniversalAgent', return_value=mock_universal_agent):
            manager = EnhancedRequestManager(mock_llm_factory, mock_message_bus)
        return manager
    
    def test_initialization(self, mock_llm_factory, mock_message_bus):
        """Test that EnhancedRequestManager initializes correctly."""
        with patch('supervisor.enhanced_request_manager.UniversalAgent') as mock_ua_class:
            manager = EnhancedRequestManager(mock_llm_factory, mock_message_bus)
            
            assert manager.llm_factory == mock_llm_factory
            assert manager.message_bus == mock_message_bus
            assert manager.request_contexts == {}
            mock_ua_class.assert_called_once_with(mock_llm_factory)
            mock_message_bus.subscribe.assert_called_once()
    
    def test_agent_id_to_role_mapping(self, enhanced_request_manager):
        """Test mapping of agent IDs to roles."""
        test_cases = [
            ("planning_agent", "planning"),
            ("search_agent", "search"),
            ("weather_agent", "weather"),
            ("summarizer_agent", "summarizer"),
            ("slack_agent", "slack"),
            ("unknown_agent", "default")
        ]
        
        for agent_id, expected_role in test_cases:
            role = enhanced_request_manager._determine_role_from_agent_id(agent_id)
            assert role == expected_role
    
    def test_llm_type_for_role_mapping(self, enhanced_request_manager):
        """Test mapping of roles to appropriate LLM types."""
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
            llm_type = enhanced_request_manager._determine_llm_type_for_role(role)
            assert llm_type == expected_llm_type
    
    def test_handle_request_creates_task_context(self, enhanced_request_manager, mock_universal_agent):
        """Test that handle_request creates a TaskContext instead of TaskGraph."""
        request = RequestMetadata(
            prompt="Test request",
            source_id="test_source",
            target_id="test_target"
        )
        
        # Mock the planning agent execution
        mock_universal_agent.execute_task.return_value = "Mock planning result"
        
        with patch('supervisor.enhanced_request_manager.TaskContext') as mock_tc_class:
            mock_task_context = Mock()
            mock_tc_class.from_tasks.return_value = mock_task_context
            mock_task_context.get_ready_tasks.return_value = []
            
            request_id = enhanced_request_manager.handle_request(request)
            
            assert request_id.startswith('req_')
            assert request_id in enhanced_request_manager.request_contexts
            mock_tc_class.from_tasks.assert_called_once()
    
    def test_delegate_task_uses_universal_agent(self, enhanced_request_manager, mock_universal_agent):
        """Test that delegate_task uses Universal Agent with role-based execution."""
        # Create a mock task context and task
        task_context = Mock(spec=TaskContext)
        task = Mock(spec=TaskNode)
        task.agent_id = "planning_agent"
        task.task_id = "test_task_1"
        task.prompt = "Test task prompt"
        task.status = TaskStatus.PENDING
        
        # Mock task context methods
        task_context.prepare_task_execution.return_value = {
            "task_id": "test_task_1",
            "prompt": "Test task prompt",
            "context": "Test context"
        }
        task_context.complete_task.return_value = []
        
        # Execute delegation
        enhanced_request_manager.delegate_task(task_context, task)
        
        # Verify Universal Agent was called with correct parameters
        mock_universal_agent.execute_task.assert_called_once()
        call_args = mock_universal_agent.execute_task.call_args
        
        # Check that the task was executed with the right role
        assert "role" in call_args.kwargs
        assert call_args.kwargs["role"] == "planning"
        assert "llm_type" in call_args.kwargs
        assert call_args.kwargs["llm_type"] == LLMType.STRONG
    
    def test_task_execution_with_context_integration(self, enhanced_request_manager, mock_universal_agent):
        """Test task execution integrates properly with TaskContext."""
        # Setup
        task_context = Mock(spec=TaskContext)
        task = Mock(spec=TaskNode)
        task.agent_id = "search_agent"
        task.task_id = "search_task_1"
        task.prompt = "Search for information"
        task.status = TaskStatus.PENDING
        
        # Mock task context preparation
        execution_config = {
            "task_id": "search_task_1",
            "prompt": "Search for information",
            "context": "Previous task results",
            "conversation_history": []
        }
        task_context.prepare_task_execution.return_value = execution_config
        task_context.complete_task.return_value = []  # No next tasks
        
        # Mock agent execution
        mock_universal_agent.execute_task.return_value = "Search results found"
        
        # Execute
        enhanced_request_manager.delegate_task(task_context, task)
        
        # Verify task context integration
        task_context.prepare_task_execution.assert_called_once_with("search_task_1")
        task_context.complete_task.assert_called_once_with("search_task_1", "Search results found")
        
        # Verify agent was called with search role and weak model
        call_args = mock_universal_agent.execute_task.call_args
        assert call_args.kwargs["role"] == "search"
        assert call_args.kwargs["llm_type"] == LLMType.WEAK
    
    def test_error_handling_preserved(self, enhanced_request_manager, mock_universal_agent):
        """Test that error handling and retry logic is preserved."""
        task_context = Mock(spec=TaskContext)
        task = Mock(spec=TaskNode)
        task.agent_id = "planning_agent"
        task.task_id = "failing_task"
        task.prompt = "This will fail"
        task.status = TaskStatus.PENDING
        
        # Mock task context
        task_context.prepare_task_execution.return_value = {
            "task_id": "failing_task",
            "prompt": "This will fail"
        }
        
        # Mock agent to raise an exception
        mock_universal_agent.execute_task.side_effect = Exception("Task execution failed")
        
        # Execute and expect error handling
        with patch.object(enhanced_request_manager, 'handle_task_error') as mock_error_handler:
            enhanced_request_manager.delegate_task(task_context, task)
            mock_error_handler.assert_called_once()
    
    def test_pause_resume_functionality(self, enhanced_request_manager):
        """Test pause and resume functionality using TaskContext checkpoints."""
        # Create a request with TaskContext
        request = RequestMetadata(
            prompt="Long running task",
            source_id="test_source",
            target_id="test_target"
        )
        
        with patch('supervisor.enhanced_request_manager.TaskContext') as mock_tc_class:
            mock_task_context = Mock()
            mock_tc_class.from_tasks.return_value = mock_task_context
            mock_task_context.get_ready_tasks.return_value = []
            mock_task_context.pause_execution.return_value = {"checkpoint": "data"}
            
            request_id = enhanced_request_manager.handle_request(request)
            
            # Test pause
            checkpoint = enhanced_request_manager.pause_request(request_id)
            assert checkpoint == {"checkpoint": "data"}
            mock_task_context.pause_execution.assert_called_once()
            
            # Test resume
            enhanced_request_manager.resume_request(request_id, checkpoint)
            mock_task_context.resume_execution.assert_called_once_with(checkpoint)
    
    def test_performance_metrics_integration(self, enhanced_request_manager):
        """Test that performance metrics are properly integrated."""
        request = RequestMetadata(
            prompt="Test metrics",
            source_id="test_source",
            target_id="test_target"
        )
        
        with patch('supervisor.enhanced_request_manager.TaskContext') as mock_tc_class:
            mock_task_context = Mock()
            mock_tc_class.from_tasks.return_value = mock_task_context
            mock_task_context.get_ready_tasks.return_value = []
            mock_task_context.get_performance_metrics.return_value = {
                "total_tasks": 5,
                "completed_tasks": 3,
                "execution_time": 120.5
            }
            
            request_id = enhanced_request_manager.handle_request(request)
            metrics = enhanced_request_manager.get_request_metrics(request_id)
            
            assert metrics["total_tasks"] == 5
            assert metrics["completed_tasks"] == 3
            assert metrics["execution_time"] == 120.5
    
    def test_message_bus_integration_preserved(self, enhanced_request_manager, mock_message_bus):
        """Test that message bus integration is preserved."""
        # Verify subscription was set up
        mock_message_bus.subscribe.assert_called_once_with(
            enhanced_request_manager, 
            MessageType.INCOMING_REQUEST, 
            enhanced_request_manager.handle_request
        )
        
        # Test task assignment publishing
        task_context = Mock(spec=TaskContext)
        task = Mock(spec=TaskNode)
        task.agent_id = "test_agent"
        task.task_id = "test_task"
        task.status = TaskStatus.PENDING
        
        task_context.prepare_task_execution.return_value = {"task_id": "test_task"}
        task_context.complete_task.return_value = []
        
        enhanced_request_manager.delegate_task(task_context, task)
        
        # Verify message was published (should be called for task assignment)
        assert mock_message_bus.publish.called


if __name__ == "__main__":
    pytest.main([__file__])