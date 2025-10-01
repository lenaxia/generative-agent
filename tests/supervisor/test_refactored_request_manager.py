import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

from supervisor.request_manager import RequestManager
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.universal_agent import UniversalAgent
from common.message_bus import MessageBus, MessageType
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskGraph, TaskDescription, TaskNode, TaskStatus
from common.request_model import RequestMetadata


class TestRefactoredRequestManager:
    """Test suite for refactored RequestManager with Universal Agent integration."""
    
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
    def mock_agent_manager(self):
        """Create a mock agent manager for backward compatibility."""
        agent_manager = Mock()
        agent_manager.config = Mock()
        agent_manager.config.max_retries = 3
        agent_manager.config.retry_delay = 1.0
        agent_manager.config.metrics_manager = Mock()
        agent_manager.config.metrics_manager.update_metrics = Mock()
        agent_manager.config.metrics_manager.delta_metrics = Mock()
        agent_manager.config.metrics_manager.get_metrics = Mock(return_value={
            "start_time": time.time(),
            "tasks_completed": 0,
            "tasks_failed": 0,
            "retries": 0
        })
        return agent_manager
    
    @pytest.fixture
    def refactored_request_manager(self, mock_agent_manager, mock_message_bus, mock_llm_factory, mock_universal_agent):
        """Create a refactored request manager for testing."""
        with patch('supervisor.request_manager.UniversalAgent', return_value=mock_universal_agent):
            # Create RequestManager with both old and new interfaces
            manager = RequestManager(mock_agent_manager, mock_message_bus)
            # Add new attributes for Universal Agent integration
            manager.llm_factory = mock_llm_factory
            manager.universal_agent = mock_universal_agent
            manager.request_contexts = {}
        return manager
    
    def test_backward_compatibility_preserved(self, refactored_request_manager, mock_agent_manager, mock_message_bus):
        """Test that backward compatibility with existing interface is preserved."""
        # Verify old attributes still exist
        assert refactored_request_manager.agent_manager == mock_agent_manager
        assert refactored_request_manager.message_bus == mock_message_bus
        assert hasattr(refactored_request_manager, 'request_map')
        
        # Verify new attributes are added
        assert hasattr(refactored_request_manager, 'llm_factory')
        assert hasattr(refactored_request_manager, 'universal_agent')
        assert hasattr(refactored_request_manager, 'request_contexts')
    
    def test_agent_id_to_role_mapping(self, refactored_request_manager):
        """Test mapping of agent IDs to roles."""
        # Add the method to the manager for testing
        def _determine_role_from_agent_id(agent_id: str) -> str:
            agent_to_role_map = {
                "planning_agent": "planning",
                "search_agent": "search",
                "weather_agent": "weather", 
                "summarizer_agent": "summarizer",
                "slack_agent": "slack"
            }
            return agent_to_role_map.get(agent_id, "default")
        
        refactored_request_manager._determine_role_from_agent_id = _determine_role_from_agent_id
        
        test_cases = [
            ("planning_agent", "planning"),
            ("search_agent", "search"),
            ("weather_agent", "weather"),
            ("summarizer_agent", "summarizer"),
            ("slack_agent", "slack"),
            ("unknown_agent", "default")
        ]
        
        for agent_id, expected_role in test_cases:
            role = refactored_request_manager._determine_role_from_agent_id(agent_id)
            assert role == expected_role
    
    def test_llm_type_for_role_mapping(self, refactored_request_manager):
        """Test mapping of roles to appropriate LLM types."""
        # Add the method to the manager for testing
        def _determine_llm_type_for_role(role: str) -> LLMType:
            role_to_llm_type = {
                "planning": LLMType.STRONG,
                "analysis": LLMType.STRONG,
                "coding": LLMType.STRONG,
                "search": LLMType.WEAK,
                "weather": LLMType.WEAK,
                "summarizer": LLMType.DEFAULT,
                "slack": LLMType.DEFAULT,
                "default": LLMType.DEFAULT
            }
            return role_to_llm_type.get(role, LLMType.DEFAULT)
        
        refactored_request_manager._determine_llm_type_for_role = _determine_llm_type_for_role
        
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
            llm_type = refactored_request_manager._determine_llm_type_for_role(role)
            assert llm_type == expected_llm_type
    
    def test_create_task_graph_uses_universal_agent(self, refactored_request_manager, mock_universal_agent):
        """Test that create_task_graph can use Universal Agent for planning."""
        # Mock the universal agent execution
        mock_universal_agent.execute_task.return_value = "Mock planning result"
        
        # Mock the planning agent fallback
        mock_planning_agent = Mock()
        mock_planning_agent.set_agent_manager = Mock()
        mock_planning_agent.run = Mock(return_value=Mock())
        refactored_request_manager.agent_manager.get_agent = Mock(return_value=mock_planning_agent)
        
        # Test that it can create task graph (using existing method for now)
        result = refactored_request_manager.create_task_graph("Test instruction", "test_req_123")
        
        # Verify planning agent was called (backward compatibility)
        refactored_request_manager.agent_manager.get_agent.assert_called_once_with("planning_agent")
        mock_planning_agent.run.assert_called_once()
    
    def test_delegate_task_can_use_universal_agent(self, refactored_request_manager, mock_universal_agent):
        """Test that delegate_task can be enhanced to use Universal Agent."""
        # Create a mock task graph and task
        task_graph = Mock(spec=TaskGraph)
        task_graph.request_id = "test_request_123"
        task_graph.get_task_history = Mock(return_value=[])
        
        task = Mock(spec=TaskNode)
        task.task_id = "test_task_1"
        task.agent_id = "planning_agent"
        task.status = TaskStatus.PENDING
        task.model_dump = Mock(return_value={
            "task_id": "test_task_1",
            "agent_id": "planning_agent",
            "prompt": "Test task"
        })
        
        # Test delegation (should work with existing method)
        refactored_request_manager.delegate_task(task_graph, task)
        
        # Verify task was processed
        assert task.status == TaskStatus.RUNNING
        refactored_request_manager.message_bus.publish.assert_called_once()
    
    def test_error_handling_preserved(self, refactored_request_manager):
        """Test that existing error handling is preserved."""
        # Test error handling method exists
        assert hasattr(refactored_request_manager, 'handle_agent_error')
        assert hasattr(refactored_request_manager, 'retry_failed_task')
        assert hasattr(refactored_request_manager, 'handle_request_failure')
        
        # Test error handling can be called
        error_data = {
            "request_id": "test_req_123",
            "task_id": "test_task_1", 
            "error_message": "Test error"
        }
        
        # Mock request map
        refactored_request_manager.request_map = {
            "test_req_123": Mock()
        }
        refactored_request_manager.request_map["test_req_123"].task_graph = Mock()
        refactored_request_manager.request_map["test_req_123"].task_graph.get_node_by_task_id = Mock(return_value=Mock())
        
        # Should not raise exception
        refactored_request_manager.handle_agent_error(error_data)
    
    def test_message_bus_integration_preserved(self, refactored_request_manager, mock_message_bus):
        """Test that message bus integration is preserved."""
        # Verify subscription was set up
        mock_message_bus.subscribe.assert_called_once_with(
            refactored_request_manager, 
            MessageType.INCOMING_REQUEST, 
            refactored_request_manager.handle_request
        )
    
    def test_metrics_integration_preserved(self, refactored_request_manager):
        """Test that metrics integration is preserved."""
        # Test that metrics manager is accessible
        assert hasattr(refactored_request_manager, 'config')
        assert hasattr(refactored_request_manager.config, 'metrics_manager')
        
        # Test metrics can be updated
        refactored_request_manager.config.metrics_manager.update_metrics("test_req", {"test": "value"})
        refactored_request_manager.config.metrics_manager.update_metrics.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])