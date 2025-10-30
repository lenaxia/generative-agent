"""Integration tests for planning role TaskGraph execution.

Tests the complete flow from planning role to TaskGraph execution
following the post-lifecycle fix implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from common.task_graph import TaskStatus
from roles.core_planning import execute_task_graph


class TestPlanningRoleExecution:
    """Test end-to-end planning role execution."""

    def test_simple_workflow_execution(self):
        """Test execution of a simple two-task workflow."""
        # Create mock context with WorkflowEngine
        mock_context = Mock()
        mock_workflow_engine = Mock()
        mock_context.workflow_engine = mock_workflow_engine
        mock_context.context_id = "integration_test_123"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "test_channel"

        # Create mock completed task nodes
        mock_weather_task = Mock()
        mock_weather_task.status = TaskStatus.COMPLETED
        mock_weather_task.result = "Current weather: Sunny, 72°F"
        mock_weather_task.task_name = "Get Weather"

        mock_timer_task = Mock()
        mock_timer_task.status = TaskStatus.COMPLETED
        mock_timer_task.result = "Timer set for 30 minutes"
        mock_timer_task.task_name = "Set Timer"

        # Mock TaskContext that completes successfully
        mock_task_context = Mock()
        mock_task_context.is_completed.return_value = True
        mock_task_context.task_graph.nodes = {
            "weather_task": mock_weather_task,
            "timer_task": mock_timer_task,
        }

        # Mock workflow engine execution
        def mock_dag_execution(task_context):
            # Simulate successful execution
            task_context.is_completed.return_value = True

        mock_workflow_engine._execute_dag_parallel = Mock(
            side_effect=mock_dag_execution
        )

        # TaskGraph JSON from planning role
        task_graph_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "weather_task",
                        "name": "Get Weather",
                        "description": "Get current weather conditions",
                        "role": "weather",
                        "parameters": {"location": "current"},
                    },
                    {
                        "id": "timer_task",
                        "name": "Set Timer",
                        "description": "Set a 30-minute timer",
                        "role": "timer",
                        "parameters": {"duration": "30m", "label": "Weather check"},
                    },
                ],
                "dependencies": [
                    {
                        "source_task_id": "weather_task",
                        "target_task_id": "timer_task",
                        "type": "sequential",
                    }
                ],
            }
        )

        # Mock the TaskGraph creation dependencies
        with (
            patch("common.task_context.TaskContext") as mock_task_context_class,
            patch("common.task_graph.TaskGraph") as mock_task_graph_class,
            patch("common.task_graph.TaskNode") as mock_task_node_class,
        ):
            mock_task_context_class.return_value = mock_task_context
            mock_task_graph_class.return_value = Mock()
            mock_task_node_class.return_value = Mock()

            # Execute the TaskGraph
            result = execute_task_graph(task_graph_json, mock_context, {})

            # Verify WorkflowIntent was created
            from common.intents import WorkflowIntent

            assert isinstance(result, WorkflowIntent)
            assert result.workflow_type == "task_graph_execution"
            assert result.tasks is not None and len(result.tasks) == 2
            assert result.dependencies is not None and len(result.dependencies) == 1

    def test_complex_multi_role_workflow(self):
        """Test execution of a complex workflow with multiple roles."""
        # Create mock context
        mock_context = Mock()
        mock_workflow_engine = Mock()
        mock_context.workflow_engine = mock_workflow_engine
        mock_context.context_id = "complex_workflow_456"

        # Create mock completed task nodes for complex workflow
        mock_search_task = Mock()
        mock_search_task.status = TaskStatus.COMPLETED
        mock_search_task.result = "Found 5 articles about Thailand travel"
        mock_search_task.task_name = "Search Thailand Info"

        mock_weather_task = Mock()
        mock_weather_task.status = TaskStatus.COMPLETED
        mock_weather_task.result = "Chicago weather: Cloudy, 45°F"
        mock_weather_task.task_name = "Check Chicago Weather"

        mock_conversation_task = Mock()
        mock_conversation_task.status = TaskStatus.COMPLETED
        mock_conversation_task.result = "Essay on Thomas Paine completed: 500 words on his contributions to American independence"
        mock_conversation_task.task_name = "Write Thomas Paine Essay"

        # Mock TaskContext with all completed tasks
        mock_task_context = Mock()
        mock_task_context.is_completed.return_value = True
        mock_task_context.task_graph.nodes = {
            "search_task": mock_search_task,
            "weather_task": mock_weather_task,
            "essay_task": mock_conversation_task,
        }

        mock_workflow_engine._execute_dag_parallel = Mock()

        # Complex TaskGraph JSON (the kind that was failing before)
        complex_task_graph = json.dumps(
            {
                "tasks": [
                    {
                        "id": "search_task",
                        "name": "Search Thailand Info",
                        "description": "Search for Thailand travel information",
                        "role": "search",
                        "parameters": {"query": "Thailand travel guide"},
                    },
                    {
                        "id": "weather_task",
                        "name": "Check Chicago Weather",
                        "description": "Get current weather in Chicago",
                        "role": "weather",
                        "parameters": {"location": "Chicago"},
                    },
                    {
                        "id": "essay_task",
                        "name": "Write Thomas Paine Essay",
                        "description": "Write an essay about Thomas Paine",
                        "role": "conversation",
                        "parameters": {"topic": "Thomas Paine", "length": "500 words"},
                    },
                ],
                "dependencies": [],  # All tasks can run in parallel
            }
        )

        # Mock dependencies
        with (
            patch("common.task_context.TaskContext") as mock_task_context_class,
            patch("common.task_graph.TaskGraph"),
            patch("common.task_graph.TaskNode"),
        ):
            mock_task_context_class.return_value = mock_task_context

            # Execute the complex workflow
            result = execute_task_graph(complex_task_graph, mock_context, {})

            # Verify WorkflowIntent was created
            from common.intents import WorkflowIntent

            assert isinstance(result, WorkflowIntent)
            assert result.workflow_type == "task_graph_execution"
            assert result.tasks is not None and len(result.tasks) == 3
            assert result.dependencies is not None and len(result.dependencies) == 0

    def test_workflow_execution_with_dependencies(self):
        """Test workflow execution with task dependencies."""
        mock_context = Mock()
        mock_workflow_engine = Mock()
        mock_context.workflow_engine = mock_workflow_engine
        mock_context.context_id = "dependency_test_789"

        # Mock sequential task execution
        mock_task_1 = Mock()
        mock_task_1.status = TaskStatus.COMPLETED
        mock_task_1.result = "Step 1 completed"
        mock_task_1.task_name = "First Task"

        mock_task_2 = Mock()
        mock_task_2.status = TaskStatus.COMPLETED
        mock_task_2.result = "Step 2 completed using results from Step 1"
        mock_task_2.task_name = "Second Task"

        mock_task_context = Mock()
        mock_task_context.is_completed.return_value = True
        mock_task_context.task_graph.nodes = {
            "task_1": mock_task_1,
            "task_2": mock_task_2,
        }

        mock_workflow_engine._execute_dag_parallel = Mock()

        # TaskGraph with dependencies
        dependent_workflow = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "First Task",
                        "description": "Complete the first step",
                        "role": "search",
                        "parameters": {"query": "initial data"},
                    },
                    {
                        "id": "task_2",
                        "name": "Second Task",
                        "description": "Complete the second step using first step results",
                        "role": "conversation",
                        "parameters": {"context": "use_previous_results"},
                    },
                ],
                "dependencies": [
                    {
                        "source_task_id": "task_1",
                        "target_task_id": "task_2",
                        "type": "sequential",
                    }
                ],
            }
        )

        # Mock dependencies
        with (
            patch("common.task_context.TaskContext") as mock_task_context_class,
            patch("common.task_graph.TaskGraph"),
            patch("common.task_graph.TaskNode"),
        ):
            mock_task_context_class.return_value = mock_task_context

            result = execute_task_graph(dependent_workflow, mock_context, {})

            # Verify WorkflowIntent was created
            from common.intents import WorkflowIntent

            assert isinstance(result, WorkflowIntent)
            assert result.workflow_type == "task_graph_execution"
            assert result.tasks is not None and len(result.tasks) == 2
            assert result.dependencies is not None and len(result.dependencies) == 1


# Integration test marker
pytestmark = pytest.mark.integration
