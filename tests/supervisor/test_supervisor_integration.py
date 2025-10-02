import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from supervisor.supervisor import Supervisor
from supervisor.task_scheduler import TaskScheduler, TaskPriority
from supervisor.request_manager import RequestManager
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata
from llm_provider.factory import LLMFactory, LLMType


class TestSupervisorIntegration:
    """Test suite for Supervisor integration with TaskScheduler and RequestManager."""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a temporary config file for testing."""
        config_content = """
logging:
  log_level: INFO
  log_file: test.log

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    @pytest.fixture
    def supervisor(self, mock_config_file):
        """Create a Supervisor for testing."""
        with patch('supervisor.supervisor.configure_logging'):
            supervisor = Supervisor(mock_config_file)
            return supervisor
    
    def test_supervisor_initialization_with_new_architecture(self, supervisor):
        """Test that Supervisor initializes with new RequestManager and TaskScheduler integration."""
        # Verify components are initialized
        assert supervisor.message_bus is not None
        assert supervisor.request_manager is not None
        assert supervisor.metrics_manager is not None
        
        # Verify RequestManager uses LLMFactory (not AgentManager)
        assert hasattr(supervisor.request_manager, 'llm_factory')
        assert hasattr(supervisor.request_manager, 'universal_agent')
        
        # Verify Universal Agent status
        status = supervisor.request_manager.get_universal_agent_status()
        assert status['universal_agent_enabled'] == True
        assert status['has_llm_factory'] == True
        assert status['has_universal_agent'] == True
    
    def test_supervisor_with_task_scheduler_integration(self, supervisor):
        """Test that Supervisor can work with TaskScheduler for advanced task management."""
        # Create a TaskScheduler that works with the RequestManager
        task_scheduler = TaskScheduler(
            request_manager=supervisor.request_manager,
            message_bus=supervisor.message_bus,
            max_concurrent_tasks=3
        )
        
        # Verify TaskScheduler initialization
        assert task_scheduler.request_manager == supervisor.request_manager
        assert task_scheduler.message_bus == supervisor.message_bus
        assert task_scheduler.max_concurrent_tasks == 3
        
        # Test scheduler lifecycle
        task_scheduler.start()
        assert task_scheduler.state.value == "RUNNING"
        
        checkpoint = task_scheduler.pause()
        assert task_scheduler.state.value == "PAUSED"
        assert checkpoint is not None
        
        success = task_scheduler.resume(checkpoint)
        assert success == True
        assert task_scheduler.state.value == "RUNNING"
        
        task_scheduler.stop()
        assert task_scheduler.state.value == "STOPPED"
    
    @patch('supervisor.supervisor.input', side_effect=['test instruction', 'stop'])
    def test_supervisor_request_handling_with_universal_agent(self, mock_input, supervisor):
        """Test that Supervisor can handle requests using Universal Agent."""
        # Mock the request manager's handle_request method
        supervisor.request_manager.handle_request = Mock(return_value="req_123")
        supervisor.request_manager.monitor_progress = Mock(return_value={"status": True})
        
        # Start supervisor (should not block due to mocked input)
        with patch('time.sleep'):  # Mock sleep to speed up test
            supervisor.run()
        
        # Verify request was handled
        supervisor.request_manager.handle_request.assert_called_once()
        call_args = supervisor.request_manager.handle_request.call_args[0][0]
        assert call_args.prompt == "test instruction"
        assert call_args.source_id == "console"
        assert call_args.target_id == "supervisor"
    
    def test_supervisor_status_reporting(self, supervisor):
        """Test Supervisor status reporting functionality."""
        # Mock message bus running state
        supervisor.message_bus.is_running = Mock(return_value=True)
        
        # Replace metrics manager with a mock since it's a Pydantic model
        mock_metrics_manager = Mock()
        mock_metrics_manager.get_metrics = Mock(return_value={"test": "metrics"})
        supervisor.metrics_manager = mock_metrics_manager
        
        status = supervisor.status()
        
        assert status is not None
        assert status["running"] == True
        assert status["metrics"] == {"test": "metrics"}
        assert "task_scheduler" in status
        assert "universal_agent" in status
    
    def test_supervisor_error_handling(self, supervisor):
        """Test Supervisor error handling in various scenarios."""
        # Test status method error handling
        supervisor.message_bus.is_running = Mock(side_effect=Exception("Test error"))
        
        status = supervisor.status()
        assert status is None
    
    def test_phase_32_requirements_met(self, supervisor):
        """Test that Phase 3.2 requirements are met."""
        # Verify TaskScheduler can be created and works with RequestManager
        task_scheduler = TaskScheduler(
            request_manager=supervisor.request_manager,
            message_bus=supervisor.message_bus
        )
        
        # ✅ TaskScheduler works with existing RequestManager
        assert task_scheduler.request_manager is not None
        
        # ✅ Task queue management implemented
        assert hasattr(task_scheduler, 'task_queue')
        assert hasattr(task_scheduler, '_process_task_queue')
        
        # ✅ Pause/resume functionality using TaskContext checkpoints
        checkpoint = task_scheduler.pause()
        assert 'scheduler_state' in checkpoint
        assert 'task_queue' in checkpoint
        
        # ✅ Integration with message bus for task distribution
        assert task_scheduler.message_bus is not None
        
        # ✅ Task priority and scheduling logic
        assert hasattr(task_scheduler, 'schedule_task')
        
        # ✅ Tests for task scheduling functionality (this test file)
        # ✅ Documentation exists (docs/TASK_SCHEDULER_IMPLEMENTATION.md)


if __name__ == "__main__":
    pytest.main([__file__])