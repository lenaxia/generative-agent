import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional
from enum import Enum

from supervisor.task_scheduler import TaskScheduler, TaskPriority, SchedulerState
from supervisor.request_manager import RequestManager
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskGraph, TaskNode, TaskStatus
from common.message_bus import MessageBus, MessageType
from llm_provider.factory import LLMFactory, LLMType


class TestTaskScheduler:
    """Test suite for TaskScheduler with pause/resume functionality."""
    
    @pytest.fixture
    def mock_request_manager(self):
        """Create a mock request manager."""
        manager = Mock(spec=RequestManager)
        manager.delegate_task = Mock()
        manager.pause_request = Mock(return_value={"checkpoint": "data"})
        manager.resume_request = Mock(return_value=True)
        manager.get_request_context = Mock()
        return manager
    
    @pytest.fixture
    def mock_message_bus(self):
        """Create a mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = Mock()
        bus.publish = Mock()
        return bus
    
    @pytest.fixture
    def task_scheduler(self, mock_request_manager, mock_message_bus):
        """Create a TaskScheduler for testing."""
        return TaskScheduler(
            request_manager=mock_request_manager,
            message_bus=mock_message_bus,
            max_concurrent_tasks=3,
            checkpoint_interval=60
        )
    
    def test_initialization(self, task_scheduler, mock_request_manager, mock_message_bus):
        """Test TaskScheduler initialization."""
        assert task_scheduler.request_manager == mock_request_manager
        assert task_scheduler.message_bus == mock_message_bus
        assert task_scheduler.max_concurrent_tasks == 3
        assert task_scheduler.checkpoint_interval == 60
        assert task_scheduler.state == SchedulerState.IDLE
        assert task_scheduler.task_queue == []
        assert task_scheduler.running_tasks == {}
        
        # Verify message bus subscriptions
        expected_calls = [
            (task_scheduler, MessageType.TASK_RESPONSE, task_scheduler.handle_task_completion),
            (task_scheduler, MessageType.AGENT_ERROR, task_scheduler.handle_task_error)
        ]
        assert mock_message_bus.subscribe.call_count == 2
    
    def test_task_priority_enum(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.LOW.value == 1
        assert TaskPriority.NORMAL.value == 2
        assert TaskPriority.HIGH.value == 3
        assert TaskPriority.CRITICAL.value == 4
    
    def test_scheduler_state_enum(self):
        """Test SchedulerState enum values."""
        assert SchedulerState.IDLE == "IDLE"
        assert SchedulerState.RUNNING == "RUNNING"
        assert SchedulerState.PAUSED == "PAUSED"
        assert SchedulerState.STOPPED == "STOPPED"
    
    def test_schedule_task_basic(self, task_scheduler):
        """Test basic task scheduling."""
        task_context = Mock(spec=TaskContext)
        task = Mock(spec=TaskNode)
        task.task_id = "test_task_1"
        task.priority = TaskPriority.NORMAL
        
        task_scheduler.schedule_task(task_context, task)
        
        assert len(task_scheduler.task_queue) == 1
        queued_item = task_scheduler.task_queue[0]
        assert queued_item.task == task
        assert queued_item.context == task_context
        assert queued_item.priority == TaskPriority.NORMAL
        assert queued_item.scheduled_time is not None
    
    def test_schedule_task_with_priority(self, task_scheduler):
        """Test task scheduling with different priorities."""
        # Schedule tasks with different priorities
        high_task = Mock(spec=TaskNode)
        high_task.task_id = "high_task"
        
        low_task = Mock(spec=TaskNode)
        low_task.task_id = "low_task"
        
        normal_task = Mock(spec=TaskNode)
        normal_task.task_id = "normal_task"
        
        task_scheduler.schedule_task(Mock(), low_task, TaskPriority.LOW)
        task_scheduler.schedule_task(Mock(), high_task, TaskPriority.HIGH)
        task_scheduler.schedule_task(Mock(), normal_task, TaskPriority.NORMAL)
        
        # Verify that high priority task is first when we pop from queue
        first_task = task_scheduler.task_queue[0]  # heapq keeps min at index 0, but our __lt__ makes higher priority "smaller"
        assert first_task.priority == TaskPriority.HIGH
        
        # Verify we have all three tasks
        assert len(task_scheduler.task_queue) == 3
        
        # Verify all priorities are represented
        all_priorities = {item.priority for item in task_scheduler.task_queue}
        assert all_priorities == {TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW}
    
    def test_start_scheduler(self, task_scheduler):
        """Test starting the scheduler."""
        task_scheduler.start()
        assert task_scheduler.state == SchedulerState.RUNNING
        assert task_scheduler.start_time is not None
    
    def test_pause_scheduler(self, task_scheduler):
        """Test pausing the scheduler."""
        task_scheduler.start()
        checkpoint = task_scheduler.pause()
        
        assert task_scheduler.state == SchedulerState.PAUSED
        assert checkpoint is not None
        assert "scheduler_state" in checkpoint
        assert "task_queue" in checkpoint
        assert "running_tasks" in checkpoint
    
    def test_resume_scheduler(self, task_scheduler):
        """Test resuming the scheduler from checkpoint."""
        # Start and pause to create checkpoint
        task_scheduler.start()
        checkpoint = task_scheduler.pause()
        
        # Resume from checkpoint
        success = task_scheduler.resume(checkpoint)
        
        assert success == True
        assert task_scheduler.state == SchedulerState.RUNNING
    
    def test_process_task_queue_respects_concurrency_limit(self, task_scheduler, mock_request_manager):
        """Test that task queue processing respects concurrency limits."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = Mock(spec=TaskNode)
            task.task_id = f"task_{i}"
            task.status = TaskStatus.PENDING
            tasks.append(task)
            task_scheduler.schedule_task(Mock(), task)
        
        # Start scheduler and process queue
        task_scheduler.start()
        task_scheduler._process_task_queue()
        
        # Should only start max_concurrent_tasks (3) tasks
        assert len(task_scheduler.running_tasks) == 3
        assert mock_request_manager.delegate_task.call_count == 3
    
    def test_handle_task_completion(self, task_scheduler):
        """Test handling task completion."""
        # Add a running task
        task_id = "completed_task"
        task_scheduler.running_tasks[task_id] = {
            "task": Mock(),
            "context": Mock(),
            "start_time": time.time()
        }
        
        # Handle completion
        completion_data = {
            "task_id": task_id,
            "result": "Task completed successfully",
            "status": "completed"
        }
        
        task_scheduler.handle_task_completion(completion_data)
        
        # Task should be removed from running tasks
        assert task_id not in task_scheduler.running_tasks
    
    def test_handle_task_error(self, task_scheduler):
        """Test handling task errors."""
        # Add a running task
        task_id = "failed_task"
        task_scheduler.running_tasks[task_id] = {
            "task": Mock(),
            "context": Mock(),
            "start_time": time.time()
        }
        
        # Handle error
        error_data = {
            "task_id": task_id,
            "error_message": "Task failed",
            "request_id": "req_123"
        }
        
        task_scheduler.handle_task_error(error_data)
        
        # Task should be removed from running tasks
        assert task_id not in task_scheduler.running_tasks
    
    def test_get_scheduler_metrics(self, task_scheduler):
        """Test scheduler metrics reporting."""
        # Add some tasks to queue and running
        mock_task1 = Mock(spec=TaskNode)
        mock_task1.task_id = "mock_task_1"
        mock_task2 = Mock(spec=TaskNode)
        mock_task2.task_id = "mock_task_2"
        
        task_scheduler.schedule_task(Mock(), mock_task1)
        task_scheduler.schedule_task(Mock(), mock_task2)
        task_scheduler.running_tasks["task_1"] = {
            "start_time": time.time(),
            "priority": TaskPriority.NORMAL
        }
        
        metrics = task_scheduler.get_metrics()
        
        assert metrics["state"] == SchedulerState.IDLE
        assert metrics["queued_tasks"] == 2
        assert metrics["running_tasks"] == 1
        assert metrics["max_concurrent_tasks"] == 3
        assert "uptime" in metrics
    
    def test_stop_scheduler(self, task_scheduler):
        """Test stopping the scheduler."""
        task_scheduler.start()
        task_scheduler.stop()
        
        assert task_scheduler.state == SchedulerState.STOPPED
        assert task_scheduler.end_time is not None


if __name__ == "__main__":
    pytest.main([__file__])