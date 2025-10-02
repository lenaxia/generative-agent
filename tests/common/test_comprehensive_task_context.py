import pytest
import time
import json
from unittest.mock import Mock, MagicMock
from typing import Dict, List

from common.task_context import TaskContext, ExecutionState, ConversationHistory, ProgressiveSummary
from common.task_graph import TaskGraph, TaskNode, TaskStatus, TaskDescription, TaskDependency


class TestComprehensiveTaskContext:
    """Comprehensive test suite for TaskContext enhancements in Universal Agent architecture."""
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDescription(
                task_name="Plan Project",
                agent_id="planning_agent",
                task_type="Planning",
                prompt="Create a project plan"
            ),
            TaskDescription(
                task_name="Research Topic",
                agent_id="search_agent", 
                task_type="Research",
                prompt="Research the topic"
            ),
            TaskDescription(
                task_name="Summarize Results",
                agent_id="summarizer_agent",
                task_type="Summarization", 
                prompt="Summarize the research results"
            )
        ]
    
    @pytest.fixture
    def sample_dependencies(self):
        """Create sample dependencies for testing."""
        return [
            TaskDependency(from_task="Plan Project", to_task="Research Topic"),
            TaskDependency(from_task="Research Topic", to_task="Summarize Results")
        ]
    
    @pytest.fixture
    def task_context(self, sample_tasks, sample_dependencies):
        """Create a TaskContext for testing."""
        return TaskContext.from_tasks(
            tasks=sample_tasks,
            dependencies=sample_dependencies,
            request_id="test_request_123"
        )
    
    def test_task_context_initialization(self, task_context):
        """Test TaskContext initialization with enhanced features."""
        assert task_context is not None
        assert task_context.context_id is not None
        assert task_context.execution_state == ExecutionState.PENDING
        assert task_context.task_graph is not None
        assert hasattr(task_context, 'conversation_history')
        assert hasattr(task_context, 'progressive_summary')
        assert hasattr(task_context, 'checkpoints')
        assert hasattr(task_context, 'metadata')
    
    def test_conversation_history_management(self, task_context):
        """Test conversation history management functionality."""
        # Add system message
        task_context.add_system_message("System initialized")
        
        # Add user message
        task_context.add_user_message("Please complete the task")
        
        # Add assistant message
        task_context.add_assistant_message("Task completed successfully")
        
        # Verify conversation history
        history = task_context.get_conversation_history()
        assert len(history) == 3
        assert history[0]['role'] == 'system'
        assert history[1]['role'] == 'user'
        assert history[2]['role'] == 'assistant'
        
        # Test conversation context retrieval
        context = task_context.get_conversation_context()
        assert 'messages' in context
        assert len(context['messages']) == 3
    
    def test_progressive_summary_functionality(self, task_context):
        """Test progressive summary management."""
        # Add initial summary
        task_context.update_progressive_summary("Initial task planning completed")
        
        # Add more summaries
        task_context.update_progressive_summary("Research phase started")
        task_context.update_progressive_summary("Research completed, moving to summarization")
        
        # Get current summary
        summary = task_context.get_progressive_summary()
        assert summary is not None
        assert "Research completed" in summary
        
        # Test summary history
        summary_history = task_context.get_summary_history()
        assert len(summary_history) >= 3
    
    def test_checkpoint_creation_and_restoration(self, task_context):
        """Test comprehensive checkpoint functionality."""
        # Modify task context state
        task_context.start_execution()
        task_context.add_user_message("Test message")
        task_context.update_progressive_summary("Test summary")
        task_context.set_metadata("test_key", "test_value")
        
        # Create checkpoint
        checkpoint = task_context.create_checkpoint()
        
        # Verify checkpoint structure
        assert checkpoint is not None
        assert 'context_id' in checkpoint
        assert 'execution_state' in checkpoint
        assert 'task_graph_state' in checkpoint
        assert 'conversation_history' in checkpoint
        assert 'progressive_summary' in checkpoint
        assert 'metadata' in checkpoint
        assert 'timestamp' in checkpoint
        
        # Test checkpoint serialization
        checkpoint_json = json.dumps(checkpoint, default=str)
        assert checkpoint_json is not None
        
        # Test restoration from checkpoint
        new_context = TaskContext.from_checkpoint(checkpoint)
        assert new_context.context_id == task_context.context_id
        assert new_context.execution_state == task_context.execution_state
        assert new_context.get_metadata("test_key") == "test_value"
    
    def test_pause_and_resume_functionality(self, task_context):
        """Test pause/resume functionality using checkpoints."""
        # Start execution
        task_context.start_execution()
        assert task_context.execution_state == ExecutionState.RUNNING
        
        # Add some state
        task_context.add_user_message("Working on task")
        task_context.update_progressive_summary("Task in progress")
        
        # Pause execution
        pause_checkpoint = task_context.pause_execution()
        assert task_context.execution_state == ExecutionState.PAUSED
        assert pause_checkpoint is not None
        
        # Resume execution
        task_context.resume_execution(pause_checkpoint)
        assert task_context.execution_state == ExecutionState.RUNNING
        
        # Verify state is preserved
        history = task_context.get_conversation_history()
        assert len(history) > 0
        assert "Working on task" in str(history)
    
    def test_task_execution_preparation(self, task_context):
        """Test task execution preparation functionality."""
        # Get ready tasks
        ready_tasks = task_context.get_ready_tasks()
        assert len(ready_tasks) > 0
        
        # Prepare task execution
        first_task = ready_tasks[0]
        execution_config = task_context.prepare_task_execution(first_task.task_id)
        
        assert execution_config is not None
        assert 'prompt' in execution_config
        assert 'context' in execution_config
        assert 'conversation_history' in execution_config
        assert 'progressive_summary' in execution_config
    
    def test_task_completion_and_progression(self, task_context):
        """Test task completion and workflow progression."""
        # Start execution
        task_context.start_execution()
        
        # Get and complete first task
        ready_tasks = task_context.get_ready_tasks()
        first_task = ready_tasks[0]
        
        # Complete the task
        result = "Task completed successfully"
        next_tasks = task_context.complete_task(first_task.task_id, result)
        
        # Verify task completion
        completed_task = task_context.task_graph.get_node(first_task.task_id)
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == result
        
        # Verify next tasks are available
        if next_tasks:
            assert len(next_tasks) > 0
            for task in next_tasks:
                assert task.status == TaskStatus.PENDING
    
    def test_performance_metrics_tracking(self, task_context):
        """Test performance metrics tracking."""
        # Start execution and simulate some work
        task_context.start_execution()
        time.sleep(0.1)  # Simulate work
        
        # Get performance metrics
        metrics = task_context.get_performance_metrics()
        
        assert metrics is not None
        assert 'execution_time' in metrics
        assert 'tasks_completed' in metrics
        assert 'tasks_pending' in metrics
        assert 'tasks_failed' in metrics
        assert metrics['execution_time'] > 0
    
    def test_metadata_management(self, task_context):
        """Test metadata management functionality."""
        # Set various metadata
        task_context.set_metadata("user_id", "user123")
        task_context.set_metadata("priority", "high")
        task_context.set_metadata("tags", ["urgent", "important"])
        
        # Get metadata
        assert task_context.get_metadata("user_id") == "user123"
        assert task_context.get_metadata("priority") == "high"
        assert task_context.get_metadata("tags") == ["urgent", "important"]
        assert task_context.get_metadata("nonexistent") is None
        
        # Get all metadata
        all_metadata = task_context.get_all_metadata()
        assert "user_id" in all_metadata
        assert "priority" in all_metadata
        assert "tags" in all_metadata
    
    def test_error_handling_and_recovery(self, task_context):
        """Test error handling and recovery mechanisms."""
        # Start execution
        task_context.start_execution()
        
        # Simulate task failure
        ready_tasks = task_context.get_ready_tasks()
        first_task = ready_tasks[0]
        
        error_message = "Task failed due to network error"
        task_context.fail_task(first_task.task_id, error_message)
        
        # Verify task failure
        failed_task = task_context.task_graph.get_node(first_task.task_id)
        assert failed_task.status == TaskStatus.FAILED
        assert error_message in failed_task.stop_reason
        
        # Test recovery
        task_context.retry_task(first_task.task_id)
        retried_task = task_context.task_graph.get_node(first_task.task_id)
        assert retried_task.status == TaskStatus.PENDING
    
    def test_context_serialization_and_persistence(self, task_context):
        """Test context serialization for persistence."""
        # Add comprehensive state
        task_context.start_execution()
        task_context.add_user_message("Test message")
        task_context.update_progressive_summary("Test summary")
        task_context.set_metadata("test", "value")
        
        # Serialize context
        serialized = task_context.to_dict()
        
        assert serialized is not None
        assert 'context_id' in serialized
        assert 'execution_state' in serialized
        assert 'task_graph' in serialized
        assert 'conversation_history' in serialized
        assert 'progressive_summary' in serialized
        assert 'metadata' in serialized
        
        # Test JSON serialization
        json_str = json.dumps(serialized, default=str)
        assert json_str is not None
        
        # Test deserialization
        deserialized_data = json.loads(json_str)
        restored_context = TaskContext.from_dict(deserialized_data)
        
        assert restored_context.context_id == task_context.context_id
        assert restored_context.execution_state == task_context.execution_state
    
    def test_integration_with_strands_agent(self, task_context):
        """Test integration points with StrandsAgent execution model."""
        # Test context preparation for StrandsAgent
        context_for_agent = task_context.prepare_for_strands_agent()
        
        assert context_for_agent is not None
        assert 'conversation_history' in context_for_agent
        assert 'current_task' in context_for_agent
        assert 'progressive_summary' in context_for_agent
        assert 'metadata' in context_for_agent
        
        # Test result integration from StrandsAgent
        agent_result = {
            "response": "Task completed",
            "reasoning": "Applied logical steps",
            "confidence": 0.95
        }
        
        task_context.integrate_strands_result(agent_result)
        
        # Verify integration
        history = task_context.get_conversation_history()
        assert len(history) > 0
        
        summary = task_context.get_progressive_summary()
        assert "Task completed" in summary or "confidence" in str(task_context.get_all_metadata())


if __name__ == "__main__":
    pytest.main([__file__])