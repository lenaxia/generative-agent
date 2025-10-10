"""Integration tests for unified result storage architecture.

Tests that fast-reply and complex workflows both store results in TaskContext
and can be retrieved consistently by Slack bot and other consumers.
"""

import time
from unittest.mock import Mock, patch

import pytest

from common.message_bus import MessageBus
from common.request_model import RequestMetadata
from common.task_context import ExecutionState
from common.task_graph import TaskStatus
from llm_provider.factory import LLMFactory, LLMType
from supervisor.workflow_engine import WorkflowEngine


class TestUnifiedResultStorage:
    """Test unified result storage for fast-reply and complex workflows."""

    @pytest.fixture
    def workflow_engine(self):
        """Create a WorkflowEngine instance for testing."""
        # Mock LLMFactory
        llm_factory = Mock(spec=LLMFactory)
        llm_factory.warm_models.return_value = None
        llm_factory.warm_agent_pool.return_value = None

        # Mock message bus
        message_bus = Mock(spec=MessageBus)

        # Create workflow engine with mocked dependencies
        engine = WorkflowEngine(
            llm_factory=llm_factory,
            message_bus=message_bus,
            fast_path_config={"enabled": True, "confidence_threshold": 0.7},
        )

        return engine

    def test_fast_reply_creates_task_context(self, workflow_engine):
        """Test that fast-reply workflows create TaskContext with completed TaskNode."""
        # Mock the universal agent execution
        mock_result = "In Seattle right now, it's a mostly sunny afternoon with a comfortable temperature of 61°F."
        workflow_engine.universal_agent = Mock()
        workflow_engine.universal_agent.execute_task.return_value = mock_result

        # Mock role registry
        workflow_engine.role_registry = Mock()
        workflow_engine.role_registry.get_role_execution_type.return_value = "hybrid"

        # Mock duration logger
        with patch("supervisor.workflow_engine.get_duration_logger") as mock_logger:
            mock_duration_logger = Mock()
            mock_logger.return_value = mock_duration_logger

            # Create request
            request = RequestMetadata(
                prompt="what is the weather in seattle",
                source_id="test_user",
                target_id="supervisor",
            )

            # Mock routing result
            routing_result = {
                "route": "weather",
                "confidence": 0.95,
                "parameters": {"location": "seattle"},
            }

            # Execute fast-reply
            request_id = workflow_engine._handle_fast_reply(request, routing_result)

            # Verify request_id format
            assert request_id.startswith("fr_")

            # Verify TaskContext was created and stored
            assert request_id in workflow_engine.active_workflows
            task_context = workflow_engine.active_workflows[request_id]

            # Verify TaskContext properties
            assert task_context.context_id == request_id
            assert task_context.execution_state == ExecutionState.COMPLETED
            assert task_context.end_time is not None

            # Verify TaskNode was created with result
            task_node = task_context.task_graph.nodes[request_id]
            assert task_node.task_id == request_id
            assert task_node.task_name == "fast_reply_weather"
            assert task_node.status == TaskStatus.COMPLETED
            assert task_node.result == mock_result
            assert task_node.role == "weather"
            assert task_node.llm_type == "WEAK"

            # Verify task context contains metadata
            assert task_node.task_context["confidence"] == 0.95
            assert task_node.task_context["parameters"] == {"location": "seattle"}
            assert task_node.task_context["execution_type"] == "hybrid"

    def test_fast_reply_result_retrieval_like_slack_bot(self, workflow_engine):
        """Test that fast-reply results can be retrieved like Slack bot does."""
        # Mock the universal agent execution
        mock_result = "Current weather in Seattle: 61°F, mostly sunny with light northwest breeze."
        workflow_engine.universal_agent = Mock()
        workflow_engine.universal_agent.execute_task.return_value = mock_result

        # Mock role registry
        workflow_engine.role_registry = Mock()
        workflow_engine.role_registry.get_role_execution_type.return_value = "hybrid"

        # Mock duration logger
        with patch("supervisor.workflow_engine.get_duration_logger") as mock_logger:
            mock_duration_logger = Mock()
            mock_logger.return_value = mock_duration_logger

            # Create and execute fast-reply
            request = RequestMetadata(
                prompt="weather in seattle",
                source_id="test_user",
                target_id="supervisor",
            )
            routing_result = {
                "route": "weather",
                "confidence": 0.9,
                "parameters": {"location": "seattle"},
            }
            request_id = workflow_engine._handle_fast_reply(request, routing_result)

            # Simulate Slack bot result retrieval
            task_context = workflow_engine.active_workflows.get(request_id)
            assert task_context is not None

            # Test _get_result_from_completed_nodes logic (from Slack bot)
            completed_results = []
            for node in task_context.task_graph.nodes.values():
                if node.status.value == "COMPLETED" and node.result:
                    completed_results.append(node.result)

            # Verify result is found
            assert len(completed_results) == 1
            assert completed_results[0] == mock_result

            # Verify final result format (like Slack bot would create)
            final_result = completed_results[-1]
            formatted_result = f"🤖 {final_result}"
            assert formatted_result == f"🤖 {mock_result}"

    def test_no_fast_reply_results_dict_used(self, workflow_engine):
        """Test that fast_reply_results dict is no longer used."""
        # Verify fast_reply_results attribute doesn't exist
        assert not hasattr(workflow_engine, "fast_reply_results")

        # Verify _store_fast_reply_result method doesn't exist
        assert not hasattr(workflow_engine, "_store_fast_reply_result")

    def test_complex_workflow_still_works(self, workflow_engine):
        """Test that complex workflows still work with unified storage."""
        # Mock complex workflow creation
        with patch.object(workflow_engine, "_create_task_plan") as mock_create_plan:
            with patch.object(workflow_engine, "_execute_dag_parallel") as mock_execute:
                with patch(
                    "supervisor.workflow_engine.get_duration_logger"
                ) as mock_logger:
                    mock_duration_logger = Mock()
                    mock_logger.return_value = mock_duration_logger

                    # Mock task context
                    mock_task_context = Mock()
                    mock_task_context.start_execution.return_value = None
                    mock_create_plan.return_value = mock_task_context

                    # Create request
                    request = RequestMetadata(
                        prompt="complex multi-step task",
                        source_id="test_user",
                        target_id="supervisor",
                    )

                    # Execute complex workflow
                    request_id = workflow_engine._handle_complex_workflow(request)

                    # Verify workflow was created
                    assert request_id.startswith("wf_")
                    assert request_id in workflow_engine.active_workflows
                    assert (
                        workflow_engine.active_workflows[request_id]
                        == mock_task_context
                    )

    def test_unified_retrieval_interface(self, workflow_engine):
        """Test that both fast-reply and complex workflows use same retrieval interface."""
        # Mock universal agent
        workflow_engine.universal_agent = Mock()
        workflow_engine.universal_agent.execute_task.return_value = "Fast reply result"

        # Mock role registry
        workflow_engine.role_registry = Mock()
        workflow_engine.role_registry.get_role_execution_type.return_value = "hybrid"

        with patch("supervisor.workflow_engine.get_duration_logger") as mock_logger:
            mock_duration_logger = Mock()
            mock_logger.return_value = mock_duration_logger

            # Create fast-reply workflow
            request = RequestMetadata(
                prompt="test request", source_id="test_user", target_id="supervisor"
            )
            routing_result = {"route": "test", "confidence": 0.8, "parameters": {}}
            fast_reply_id = workflow_engine._handle_fast_reply(request, routing_result)

            # Verify both use active_workflows for storage
            assert fast_reply_id in workflow_engine.active_workflows

            # Verify retrieval interface is consistent
            fast_reply_context = workflow_engine.active_workflows[fast_reply_id]
            assert hasattr(fast_reply_context, "task_graph")
            assert hasattr(fast_reply_context, "execution_state")
            assert hasattr(fast_reply_context, "context_id")

            # Verify TaskNode structure is consistent
            task_node = fast_reply_context.task_graph.nodes[fast_reply_id]
            assert hasattr(task_node, "result")
            assert hasattr(task_node, "status")
            assert hasattr(task_node, "task_id")

    def test_performance_maintained(self, workflow_engine):
        """Test that fast-reply performance is maintained with unified storage."""
        # Mock universal agent for fast execution
        workflow_engine.universal_agent = Mock()
        workflow_engine.universal_agent.execute_task.return_value = "Quick result"

        # Mock role registry
        workflow_engine.role_registry = Mock()
        workflow_engine.role_registry.get_role_execution_type.return_value = "hybrid"

        with patch("supervisor.workflow_engine.get_duration_logger") as mock_logger:
            mock_duration_logger = Mock()
            mock_logger.return_value = mock_duration_logger

            # Measure execution time
            start_time = time.time()

            request = RequestMetadata(
                prompt="fast test", source_id="test_user", target_id="supervisor"
            )
            routing_result = {"route": "test", "confidence": 0.9, "parameters": {}}
            request_id = workflow_engine._handle_fast_reply(request, routing_result)

            execution_time = time.time() - start_time

            # Verify fast execution (should be well under 1 second for mocked execution)
            assert execution_time < 0.1  # 100ms threshold for mocked execution

            # Verify result is immediately available
            task_context = workflow_engine.active_workflows[request_id]
            assert task_context.execution_state == ExecutionState.COMPLETED

            task_node = task_context.task_graph.nodes[request_id]
            assert task_node.status == TaskStatus.COMPLETED
            assert task_node.result == "Quick result"
