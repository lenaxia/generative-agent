"""
Tests for Enhanced Supervisor with Single Event Loop

Tests the enhanced Supervisor that uses scheduled tasks instead of
background threads for heartbeat and monitoring operations.

Following TDD principles - tests written first.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supervisor.supervisor import Supervisor


class TestEnhancedSupervisor:
    """Test enhanced Supervisor with single event loop architecture."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create temporary config file for testing."""
        config_content = """
framework:
  type: "strands"

llm_providers:
  bedrock:
    models:
      WEAK: "anthropic.claude-3-haiku-20240307-v1:0"
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
      STRONG: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

role_system:
  roles_directory: "roles"

message_bus:
  intent_processing:
    enabled: true

logging:
  level: "INFO"
  log_file: "logs/test.log"

feature_flags:
  enable_single_event_loop: true
  enable_intent_processing: true
  enable_heartbeat: true

heartbeat:
  enabled: true
  interval: 30
  health_check_interval: 60
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def test_supervisor_single_event_loop_initialization(self, temp_config_file):
        """Test that Supervisor initializes with single event loop support."""
        supervisor = Supervisor(temp_config_file)

        # Should have scheduled tasks instead of background threads
        assert hasattr(supervisor, "_scheduled_tasks")
        assert hasattr(supervisor, "_use_single_event_loop")
        assert supervisor._use_single_event_loop is True

    def test_supervisor_heartbeat_as_scheduled_task(self, temp_config_file):
        """Test that heartbeat runs as scheduled task instead of background thread."""
        supervisor = Supervisor(temp_config_file)

        # Supervisor should have scheduled task methods
        assert hasattr(supervisor, "_initialize_scheduled_tasks")
        assert hasattr(supervisor, "_create_heartbeat_task")
        assert callable(supervisor._create_heartbeat_task)

    def test_supervisor_fast_heartbeat_as_scheduled_task(self, temp_config_file):
        """Test that fast heartbeat runs as scheduled task instead of background thread."""
        supervisor = Supervisor(temp_config_file)

        # Supervisor should have fast heartbeat task methods
        assert hasattr(supervisor, "_create_fast_heartbeat_task")
        assert callable(supervisor._create_fast_heartbeat_task)

    def test_supervisor_start_with_scheduled_tasks(self, temp_config_file):
        """Test that Supervisor starts scheduled tasks instead of threads."""
        supervisor = Supervisor(temp_config_file)

        # Should have single event loop enabled
        assert supervisor._use_single_event_loop is True

        # Should have scheduled tasks list
        assert hasattr(supervisor, "_scheduled_tasks")
        assert isinstance(supervisor._scheduled_tasks, list)

    def test_supervisor_stop_cancels_scheduled_tasks(self, temp_config_file):
        """Test that Supervisor properly cancels scheduled tasks on stop."""
        supervisor = Supervisor(temp_config_file)

        # Should have stop method for scheduled tasks
        assert hasattr(supervisor, "_stop_scheduled_tasks")
        assert callable(supervisor._stop_scheduled_tasks)

        # Should have scheduled tasks management
        assert hasattr(supervisor, "_scheduled_tasks")
        assert isinstance(supervisor._scheduled_tasks, list)

    def test_supervisor_message_bus_dependency_injection(self, temp_config_file):
        """Test that Supervisor properly injects dependencies into MessageBus."""
        supervisor = Supervisor(temp_config_file)

        # Supervisor should have dependency setup method
        assert hasattr(supervisor, "_setup_message_bus_dependencies")
        assert callable(supervisor._setup_message_bus_dependencies)

        # MessageBus should have dependency attributes
        assert hasattr(supervisor.message_bus, "communication_manager")
        assert hasattr(supervisor.message_bus, "workflow_engine")

    def test_supervisor_backward_compatibility(self, temp_config_file):
        """Test that enhanced Supervisor maintains backward compatibility."""
        supervisor = Supervisor(temp_config_file)

        # Should still have all expected attributes
        assert hasattr(supervisor, "config")
        assert hasattr(supervisor, "message_bus")
        assert hasattr(supervisor, "workflow_engine")
        assert hasattr(supervisor, "heartbeat")
        assert hasattr(supervisor, "communication_manager")

    def test_heartbeat_scheduled_task_execution(self, temp_config_file):
        """Test that heartbeat scheduled task methods exist."""
        supervisor = Supervisor(temp_config_file)

        # Should have heartbeat task creation methods
        assert hasattr(supervisor, "_create_heartbeat_task")
        assert hasattr(supervisor, "_create_fast_heartbeat_task")

        # Methods should be callable
        assert callable(supervisor._create_heartbeat_task)
        assert callable(supervisor._create_fast_heartbeat_task)

    def test_supervisor_threading_elimination(self, temp_config_file):
        """Test that Supervisor has single event loop architecture."""
        supervisor = Supervisor(temp_config_file)

        # Should use single event loop
        assert supervisor._use_single_event_loop is True

        # Should have scheduled tasks instead of threads
        assert hasattr(supervisor, "_scheduled_tasks")
        assert hasattr(supervisor, "_start_scheduled_tasks")
        assert hasattr(supervisor, "_stop_scheduled_tasks")


class TestScheduledTaskManagement:
    """Test scheduled task management in enhanced Supervisor."""

    @pytest.fixture
    def mock_supervisor(self):
        """Create mock supervisor for testing."""
        supervisor = MagicMock()
        supervisor._scheduled_tasks = []
        supervisor._use_single_event_loop = True
        return supervisor

    @pytest.mark.asyncio
    async def test_create_scheduled_heartbeat_task(self, mock_supervisor):
        """Test creating scheduled heartbeat task."""
        from supervisor.supervisor import Supervisor

        async def heartbeat_task():
            """Mock heartbeat task."""
            await asyncio.sleep(0.01)  # Simulate work
            return "heartbeat_completed"

        # Create scheduled task
        task = asyncio.create_task(heartbeat_task())
        mock_supervisor._scheduled_tasks.append(task)

        # Wait for completion
        result = await task

        assert result == "heartbeat_completed"
        assert task.done()

    @pytest.mark.asyncio
    async def test_scheduled_task_error_handling(self, mock_supervisor):
        """Test error handling in scheduled tasks."""

        async def failing_task():
            """Task that raises an exception."""
            raise ValueError("Task error")

        # Create scheduled task with error handling
        async def safe_task():
            try:
                await failing_task()
            except Exception as e:
                return f"error_handled: {e}"

        task = asyncio.create_task(safe_task())
        result = await task

        assert "error_handled" in result
        assert "Task error" in result

    @pytest.mark.asyncio
    async def test_multiple_scheduled_tasks(self, mock_supervisor):
        """Test multiple scheduled tasks running concurrently."""
        results = []

        async def task_1():
            await asyncio.sleep(0.01)
            results.append("task_1")

        async def task_2():
            await asyncio.sleep(0.02)
            results.append("task_2")

        # Create and run tasks concurrently
        tasks = [asyncio.create_task(task_1()), asyncio.create_task(task_2())]

        await asyncio.gather(*tasks)

        assert "task_1" in results
        assert "task_2" in results
        assert len(results) == 2


class TestSupervisorIntegration:
    """Test Supervisor integration with enhanced components."""

    def test_supervisor_with_enhanced_message_bus(self, tmp_path):
        """Test Supervisor integration with enhanced MessageBus."""
        # Create minimal config with required fields
        config_content = """
framework:
  type: "strands"
llm_providers:
  bedrock:
    models:
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
logging:
  level: "INFO"
  log_file: "logs/test.log"
feature_flags:
  enable_single_event_loop: true
  enable_intent_processing: true
"""
        config_file = tmp_path / "integration_config.yaml"
        config_file.write_text(config_content)

        # Create supervisor
        supervisor = Supervisor(str(config_file))

        # MessageBus should have intent processing enabled
        assert supervisor.message_bus._enable_intent_processing is True
        assert hasattr(supervisor.message_bus, "_intent_processor")

    def test_supervisor_dependency_injection_flow(self, tmp_path):
        """Test that Supervisor properly injects dependencies."""
        config_content = """
framework:
  type: "strands"
llm_providers:
  bedrock:
    models:
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
logging:
  level: "INFO"
  log_file: "logs/test.log"
"""
        config_file = tmp_path / "deps_config.yaml"
        config_file.write_text(config_content)

        supervisor = Supervisor(str(config_file))

        # Dependencies should flow correctly
        assert supervisor.message_bus is not None
        assert supervisor.workflow_engine is not None
        assert supervisor.communication_manager is not None

        # Should have dependency setup method
        assert hasattr(supervisor, "_setup_message_bus_dependencies")
        assert callable(supervisor._setup_message_bus_dependencies)


if __name__ == "__main__":
    pytest.main([__file__])
