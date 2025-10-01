import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

from supervisor.request_manager import RequestManager
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from common.message_bus import MessageBus, MessageType
from common.task_context import TaskContext, ExecutionState
from common.request_model import RequestMetadata


class TestStrandsRequestManager:
    """Test suite for the new StrandsAgent-based RequestManager."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value="strands")
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
        agent.execute_task = Mock(return_value="Mock task result")
        return agent
    
    @pytest.fixture
    def request_manager(self, mock_llm_factory, mock_message_bus, mock_universal_agent):
        """Create a RequestManager for testing."""
        with patch('supervisor.request_manager.UniversalAgent', return_value=mock_universal_agent):
            manager = RequestManager(mock_llm_factory, mock_message_bus)
        return manager
    
    def test_initialization(self, request_manager, mock_llm_factory, mock_message_bus):
        """Test that RequestManager initializes correctly."""
        assert request_manager.llm_factory == mock_llm_factory
        assert request_manager.message_bus == mock_message_bus
        assert request_manager.universal_agent is not None
        assert request_manager.request_contexts == {}
        assert request_manager.max_retries == 3
        assert request_manager.retry_delay == 1.0
        
        # Verify subscription to incoming requests
        mock_message_bus.subscribe.assert_called_once_with(
            request_manager, 
            MessageType.INCOMING_REQUEST, 
            request_manager.handle_request
        )
    
    def test_agent_id_to_role_mapping(self, request_manager):
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
            role = request_manager._determine_role_from_agent_id(agent_id)
            assert role == expected_role
    
    def test_llm_type_for_role_mapping(self, request_manager):
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
            llm_type = request_manager._determine_llm_type_for_role(role)
            assert llm_type == expected_llm_type
    
    def test_handle_request_creates_task_context(self, request_manager, mock_universal_agent):
        """Test that handle_request creates a TaskContext."""
        request = RequestMetadata(
            prompt="Test request",
            source_id="test_source",
            target_id="test_target"
        )
        
        # Mock the planning agent execution
        mock_universal_agent.execute_task.return_value = "Mock planning result"
        
        with patch('supervisor.request_manager.TaskContext') as mock_tc_class:
            mock_task_context = Mock()
            mock_tc_class.from_tasks.return_value = mock_task_context
            mock_task_context.get_ready_tasks.return_value = []
            mock_task_context.start_execution = Mock()
            
            request_id = request_manager.handle_request(request)
            
            assert request_id.startswith('req_')
            assert request_id in request_manager.request_contexts
            mock_tc_class.from_tasks.assert_called_once()
            mock_task_context.start_execution.assert_called_once()
    
    def test_universal_agent_status(self, request_manager):
        """Test that Universal Agent status is correctly reported."""
        status = request_manager.get_universal_agent_status()
        
        assert status["universal_agent_enabled"] == True
        assert status["has_llm_factory"] == True
        assert status["has_universal_agent"] == True
        assert status["active_contexts"] == 0
        assert status["framework"] == "strands"
    
    def test_pause_resume_functionality(self, request_manager):
        """Test pause and resume functionality."""
        # Create a mock task context
        mock_task_context = Mock(spec=TaskContext)
        mock_task_context.pause_execution.return_value = {"checkpoint": "data"}
        mock_task_context.resume_execution = Mock()
        mock_task_context.get_ready_tasks.return_value = []
        
        request_id = "test_req_123"
        request_manager.request_contexts[request_id] = mock_task_context
        
        # Test pause
        checkpoint = request_manager.pause_request(request_id)
        assert checkpoint == {"checkpoint": "data"}
        mock_task_context.pause_execution.assert_called_once()
        
        # Test resume
        success = request_manager.resume_request(request_id, checkpoint)
        assert success == True
        mock_task_context.resume_execution.assert_called_once_with(checkpoint)
    
    def test_list_active_requests(self, request_manager):
        """Test listing active requests."""
        # Create mock contexts
        running_context = Mock(spec=TaskContext)
        running_context.execution_state = ExecutionState.RUNNING
        
        paused_context = Mock(spec=TaskContext)
        paused_context.execution_state = ExecutionState.PAUSED
        
        completed_context = Mock(spec=TaskContext)
        completed_context.execution_state = ExecutionState.COMPLETED
        
        request_manager.request_contexts = {
            "req_1": running_context,
            "req_2": paused_context,
            "req_3": completed_context
        }
        
        active_requests = request_manager.list_active_requests()
        assert set(active_requests) == {"req_1", "req_2"}
    
    def test_cleanup_completed_requests(self, request_manager):
        """Test cleanup of old completed requests."""
        # Create mock contexts
        old_completed_context = Mock(spec=TaskContext)
        old_completed_context.execution_state = ExecutionState.COMPLETED
        old_completed_context.end_time = time.time() - 7200  # 2 hours ago
        
        recent_completed_context = Mock(spec=TaskContext)
        recent_completed_context.execution_state = ExecutionState.COMPLETED
        recent_completed_context.end_time = time.time() - 1800  # 30 minutes ago
        
        running_context = Mock(spec=TaskContext)
        running_context.execution_state = ExecutionState.RUNNING
        
        request_manager.request_contexts = {
            "req_old": old_completed_context,
            "req_recent": recent_completed_context,
            "req_running": running_context
        }
        request_manager.request_map = {
            "req_old": Mock(),
            "req_recent": Mock(),
            "req_running": Mock()
        }
        
        # Cleanup with 1 hour threshold
        request_manager.cleanup_completed_requests(max_age_seconds=3600)
        
        # Only old completed request should be removed
        assert "req_old" not in request_manager.request_contexts
        assert "req_old" not in request_manager.request_map
        assert "req_recent" in request_manager.request_contexts
        assert "req_running" in request_manager.request_contexts


if __name__ == "__main__":
    pytest.main([__file__])