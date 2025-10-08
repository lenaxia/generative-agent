"""Unit tests for WorkflowEngine core functionality.

Tests the WorkflowEngine class which consolidates RequestManager + TaskScheduler
functionality into a unified workflow management system with DAG execution
and state persistence.
"""

import time
from unittest.mock import Mock, call, patch

import pytest

from common.message_bus import MessageBus, MessageType
from common.task_context import TaskContext
from common.task_graph import TaskDependency, TaskDescription, TaskNode
from llm_provider.factory import LLMFactory
from llm_provider.universal_agent import UniversalAgent
from supervisor.workflow_engine import (
    QueuedTask,
    TaskPriority,
    WorkflowEngine,
    WorkflowState,
)


class TestWorkflowEngineUnit:
    """Unit tests for WorkflowEngine core functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        return factory

    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus."""
        bus = Mock(spec=MessageBus)
        return bus

    @pytest.fixture
    def mock_universal_agent(self):
        """Create mock Universal Agent."""
        agent = Mock(spec=UniversalAgent)
        agent.execute_task.return_value = {
            "result": "success",
            "output": "Task completed",
        }
        return agent

    @pytest.fixture
    def workflow_engine(self, mock_llm_factory, mock_message_bus):
        """Create WorkflowEngine instance for testing."""
        with patch("supervisor.workflow_engine.UniversalAgent") as mock_ua_class:
            mock_ua_instance = Mock(spec=UniversalAgent)
            mock_ua_class.return_value = mock_ua_instance

            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                max_retries=3,
                retry_delay=1.0,
                max_concurrent_tasks=5,
                checkpoint_interval=300,
            )

            # Replace the universal agent with our mock
            engine.universal_agent = mock_ua_instance
            return engine

    @pytest.fixture
    def sample_task_context(self):
        """Create sample TaskContext for testing."""
        # Create sample tasks
        task1 = TaskDescription(
            task_name="Test Task 1",
            agent_id="planning_agent",
            task_type="Planning",
            prompt="Plan the first task",
        )

        task2 = TaskDescription(
            task_name="Test Task 2",
            agent_id="search_agent",
            task_type="Search",
            prompt="Search for information",
        )

        tasks = [task1, task2]
        dependencies = [TaskDependency(source="Test Task 1", target="Test Task 2")]

        return TaskContext.from_tasks(
            tasks=tasks, dependencies=dependencies, request_id="test_request_123"
        )

    def test_workflow_engine_initialization(self, mock_llm_factory, mock_message_bus):
        """Test WorkflowEngine initializes correctly with required dependencies."""
        with patch("supervisor.workflow_engine.UniversalAgent") as mock_ua_class:
            mock_ua_instance = Mock(spec=UniversalAgent)
            mock_ua_class.return_value = mock_ua_instance

            engine = WorkflowEngine(
                llm_factory=mock_llm_factory,
                message_bus=mock_message_bus,
                max_retries=3,
                retry_delay=1.0,
                max_concurrent_tasks=5,
                checkpoint_interval=300,
            )

            # Verify initialization
            assert engine.llm_factory == mock_llm_factory
            assert engine.message_bus == mock_message_bus
            assert engine.max_retries == 3
            assert engine.retry_delay == 1.0
            assert engine.max_concurrent_tasks == 5
            assert engine.checkpoint_interval == 300
            assert engine.state == WorkflowState.IDLE
            assert engine.start_time is None
            assert engine.end_time is None
            assert len(engine.active_workflows) == 0
            assert len(engine.task_queue) == 0
            assert len(engine.running_tasks) == 0

            # Verify Universal Agent was created
            mock_ua_class.assert_called_once_with(
                mock_llm_factory,
                role_registry=engine.role_registry,
                mcp_manager=engine.mcp_manager,
            )

            # Verify message bus subscriptions
            expected_calls = [
                call(engine, MessageType.INCOMING_REQUEST, engine.handle_request),
                call(engine, MessageType.TASK_RESPONSE, engine.handle_task_completion),
                call(engine, MessageType.AGENT_ERROR, engine.handle_task_error_event),
            ]
            mock_message_bus.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_start_workflow_success(self, workflow_engine):
        """Test successful workflow creation and startup."""
        instruction = "Create a test workflow"

        # Mock the task planning
        with patch.object(workflow_engine, "_create_task_plan") as mock_create_plan:
            mock_task_context = Mock(spec=TaskContext)
            mock_task_context.context_id = "test_context_123"
            mock_create_plan.return_value = mock_task_context

            with patch.object(
                workflow_engine, "_execute_dag_parallel"
            ) as mock_execute_dag:
                # Execute
                workflow_id = workflow_engine.start_workflow(instruction)

                # Verify workflow creation
                assert workflow_id is not None
                assert workflow_id.startswith("wf_")
                assert workflow_id in workflow_engine.active_workflows
                assert workflow_engine.state == WorkflowState.RUNNING
                assert workflow_engine.start_time is not None

                # Verify task context was created and started
                mock_create_plan.assert_called_once_with(instruction, workflow_id)
                mock_task_context.start_execution.assert_called_once()

                # Verify DAG execution was initiated
                mock_execute_dag.assert_called_once_with(mock_task_context)

    def test_pause_workflow_success(self, workflow_engine, sample_task_context):
        """Test workflow can be paused successfully."""
        # Setup active workflow
        workflow_id = "test_workflow_123"
        workflow_engine.active_workflows[workflow_id] = sample_task_context
        workflow_engine.state = WorkflowState.RUNNING
        workflow_engine.start_time = time.time() - 100

        # Add some queued and running tasks
        mock_task = Mock(spec=TaskNode)
        mock_task.task_id = "task_1"
        queued_task = QueuedTask(
            priority=TaskPriority.NORMAL,
            scheduled_time=time.time(),
            task=mock_task,
            context=sample_task_context,
        )
        workflow_engine.task_queue.append(queued_task)

        workflow_engine.running_tasks["running_task_1"] = {
            "start_time": time.time() - 50,
            "context": sample_task_context,
        }

        # Execute pause
        checkpoint = workflow_engine.pause_workflow()

        # Verify pause state
        assert workflow_engine.state == WorkflowState.PAUSED
        assert checkpoint is not None
        assert "workflow_engine_state" in checkpoint
        assert "task_queue" in checkpoint
        assert "running_tasks" in checkpoint
        assert "active_workflows" in checkpoint
        assert "timestamp" in checkpoint

        # Verify checkpoint content
        engine_state = checkpoint["workflow_engine_state"]
        assert engine_state["state"] == WorkflowState.PAUSED.value
        assert engine_state["max_concurrent_tasks"] == 5
        assert engine_state["checkpoint_interval"] == 300

        assert len(checkpoint["task_queue"]) == 1
        assert checkpoint["task_queue"][0]["task_id"] == "task_1"
        assert checkpoint["task_queue"][0]["priority"] == TaskPriority.NORMAL.value

        assert len(checkpoint["running_tasks"]) == 1
        assert "running_task_1" in checkpoint["running_tasks"]

    def test_resume_workflow_success(self, workflow_engine):
        """Test workflow can be resumed from checkpoint."""
        # Create checkpoint data
        checkpoint_time = time.time() - 200
        checkpoint = {
            "workflow_engine_state": {
                "state": WorkflowState.PAUSED.value,
                "start_time": checkpoint_time,
                "last_checkpoint_time": checkpoint_time + 100,
                "max_concurrent_tasks": 5,
                "checkpoint_interval": 300,
            },
            "task_queue": [],
            "running_tasks": {},
            "active_workflows": 1,
            "timestamp": checkpoint_time + 150,
        }

        # Set initial paused state
        workflow_engine.state = WorkflowState.PAUSED

        with patch.object(workflow_engine, "_process_task_queue") as mock_process_queue:
            # Execute resume
            result = workflow_engine.resume_workflow(checkpoint=checkpoint)

            # Verify resume success
            assert result is True
            assert workflow_engine.state == WorkflowState.RUNNING
            assert workflow_engine.start_time == checkpoint_time
            assert workflow_engine.last_checkpoint_time == checkpoint_time + 100

            # Verify task queue processing was initiated
            mock_process_queue.assert_called_once()

    def test_get_workflow_metrics_success(self, workflow_engine, sample_task_context):
        """Test workflow metrics are returned correctly."""
        # Setup workflow state
        workflow_engine.state = WorkflowState.RUNNING
        workflow_engine.start_time = time.time() - 300
        workflow_engine.active_workflows["wf_1"] = sample_task_context
        workflow_engine.active_workflows["wf_2"] = sample_task_context

        # Add queued tasks
        mock_task = Mock(spec=TaskNode)
        mock_task.task_id = "queued_task_1"
        queued_task = QueuedTask(
            priority=TaskPriority.HIGH,
            scheduled_time=time.time(),
            task=mock_task,
            context=sample_task_context,
        )
        workflow_engine.task_queue.append(queued_task)

        # Add running tasks
        workflow_engine.running_tasks["running_task_1"] = {
            "start_time": time.time() - 50,
            "context": sample_task_context,
            "priority": TaskPriority.NORMAL,
        }
        workflow_engine.running_tasks["running_task_2"] = {
            "start_time": time.time() - 30,
            "context": sample_task_context,
            "priority": TaskPriority.HIGH,
        }

        # Execute
        metrics = workflow_engine.get_workflow_metrics()

        # Verify metrics structure and content
        assert isinstance(metrics, dict)
        assert "state" in metrics
        assert "uptime" in metrics
        assert "queued_tasks" in metrics
        assert "running_tasks" in metrics
        assert "max_concurrent_tasks" in metrics
        assert "active_workflows" in metrics
        assert "queue_priorities" in metrics
        assert "running_task_priorities" in metrics
        assert "universal_agent_status" in metrics

        # Verify specific metrics
        assert metrics["state"] == WorkflowState.RUNNING
        assert metrics["uptime"] >= 300
        assert metrics["active_workflows"] == 2
        assert metrics["queued_tasks"] == 1
        assert metrics["running_tasks"] == 2
        assert metrics["max_concurrent_tasks"] == 5

        # Verify priority information
        assert len(metrics["queue_priorities"]) == 1
        assert metrics["queue_priorities"][0] == "HIGH"
        assert len(metrics["running_task_priorities"]) == 2
        assert "NORMAL" in metrics["running_task_priorities"]
        assert "HIGH" in metrics["running_task_priorities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
