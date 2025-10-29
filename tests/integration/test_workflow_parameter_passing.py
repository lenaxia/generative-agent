"""
Integration tests for parameter passing through workflow execution.

Tests that parameters defined in planning role's task graph are correctly
passed through to role execution, addressing issues found in Document 43.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from common.intent_processor import IntentProcessor
from common.intents import WorkflowIntent
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from supervisor.workflow_engine import WorkflowEngine


class TestWorkflowParameterPassing:
    """Test parameter passing through workflow execution pipeline."""

    @pytest.fixture
    def setup_workflow_system(self):
        """Set up workflow system with mocked components."""
        message_bus = MessageBus()
        message_bus.start()

        llm_factory = Mock(spec=LLMFactory)

        # WorkflowEngine creates its own role_registry
        workflow_engine = WorkflowEngine(
            llm_factory=llm_factory,
            message_bus=message_bus,
        )

        universal_agent = workflow_engine.universal_agent
        role_registry = workflow_engine.role_registry

        intent_processor = IntentProcessor(
            workflow_engine=workflow_engine, message_bus=message_bus
        )

        universal_agent.intent_processor = intent_processor

        return {
            "message_bus": message_bus,
            "workflow_engine": workflow_engine,
            "universal_agent": universal_agent,
            "intent_processor": intent_processor,
            "role_registry": role_registry,
        }

    def test_parameters_preserved_through_taskgraph_creation(
        self, setup_workflow_system
    ):
        """Test that parameters are preserved when TaskGraph creates TaskNodes."""
        workflow_engine = setup_workflow_system["workflow_engine"]

        # Create WorkflowIntent with parameters
        intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Test Task",
                    "description": "Test task with parameters",
                    "role": "weather",
                    "parameters": {"location": "Chicago", "timeframe": "current"},
                }
            ],
            dependencies=[],
            request_id="test_request",
            user_id="test_user",
            channel_id="test_channel",
            original_instruction="Test instruction",
        )

        # Convert intent to task nodes
        task_nodes = workflow_engine._convert_intent_to_task_nodes(intent)

        # Verify parameters are in task_context field
        assert len(task_nodes) == 1
        task_node = task_nodes[0]
        assert task_node.task_context == {
            "location": "Chicago",
            "timeframe": "current",
        }
        assert task_node.task_name == "Test Task"
        assert task_node.agent_id == "weather"

    def test_parameters_preserved_after_taskgraph_init(self, setup_workflow_system):
        """Test that TaskGraph.__init__ preserves task_context when creating TaskNodes."""
        from common.task_graph import TaskGraph, TaskNode, TaskStatus

        # Create TaskNode with parameters
        task_node = TaskNode(
            task_id="task_1",
            task_name="Weather Check",
            request_id="test_request",
            agent_id="weather",
            task_type="workflow_generated",
            prompt="Check weather",
            status=TaskStatus.PENDING,
            inbound_edges=[],
            outbound_edges=[],
            result=None,
            stop_reason=None,
            include_full_history=False,
            start_time=None,
            duration=None,
            retry_count=0,
            role="weather",
            llm_type="DEFAULT",
            required_tools=[],
            task_context={"location": "Seattle", "format": "brief"},
        )

        # TaskGraph expects TaskDescription, but will accept TaskNode and transfer fields
        # This tests that task_context is preserved during the conversion
        task_graph = TaskGraph(
            tasks=[task_node], dependencies=[], request_id="test_request"
        )

        # Verify parameters are preserved in the recreated TaskNode
        assert len(task_graph.nodes) == 1
        recreated_node = list(task_graph.nodes.values())[0]
        assert recreated_node.task_context == {"location": "Seattle", "format": "brief"}
        assert recreated_node.task_name == "Weather Check"

    def test_parameters_passed_to_universal_agent(self, setup_workflow_system):
        """Test that parameters are passed to universal agent during task execution."""
        workflow_engine = setup_workflow_system["workflow_engine"]
        universal_agent = setup_workflow_system["universal_agent"]

        # Mock universal agent execute_task to capture parameters
        captured_params = {}

        def mock_execute_task(
            instruction, role, llm_type, context, extracted_parameters
        ):
            captured_params["extracted_parameters"] = extracted_parameters
            return "Task completed"

        universal_agent.execute_task = mock_execute_task

        # Create WorkflowIntent with parameters
        intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Weather Task",
                    "description": "Get weather",
                    "role": "weather",
                    "parameters": {"location": "Portland", "timeframe": "tomorrow"},
                }
            ],
            dependencies=[],
            request_id="test_request",
            user_id="test_user",
            channel_id="test_channel",
            original_instruction="Test",
        )

        # Execute workflow
        workflow_engine.execute_workflow_intent(intent)

        # Verify parameters were passed to universal agent
        assert captured_params["extracted_parameters"] == {
            "location": "Portland",
            "timeframe": "tomorrow",
        }

    def test_end_to_end_parameter_flow(self, setup_workflow_system):
        """Test complete parameter flow from WorkflowIntent to role execution."""
        workflow_engine = setup_workflow_system["workflow_engine"]
        universal_agent = setup_workflow_system["universal_agent"]

        # Track parameter flow through the system
        parameter_flow = []

        # Mock _convert_intent_to_task_nodes to track parameters
        original_convert = workflow_engine._convert_intent_to_task_nodes

        def track_convert(intent):
            nodes = original_convert(intent)
            for node in nodes:
                parameter_flow.append(
                    {
                        "stage": "convert_to_nodes",
                        "task_id": node.task_id,
                        "parameters": node.task_context,
                    }
                )
            return nodes

        workflow_engine._convert_intent_to_task_nodes = track_convert

        # Mock universal agent to track parameters received
        def track_execute(instruction, role, llm_type, context, extracted_parameters):
            parameter_flow.append(
                {
                    "stage": "universal_agent_execute",
                    "role": role,
                    "parameters": extracted_parameters,
                }
            )
            return "Completed"

        universal_agent.execute_task = track_execute

        # Create WorkflowIntent with multiple tasks with different parameters
        intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Search Task",
                    "description": "Search for info",
                    "role": "search",
                    "parameters": {"query": "test query", "type": "web"},
                },
                {
                    "id": "task_2",
                    "name": "Weather Task",
                    "description": "Get weather",
                    "role": "weather",
                    "parameters": {"location": "Boston"},
                },
            ],
            dependencies=[
                {
                    "source_task_id": "task_1",
                    "target_task_id": "task_2",
                    "type": "sequential",
                }
            ],
            request_id="test_request",
            user_id="test_user",
            channel_id="test_channel",
            original_instruction="Test",
        )

        # Execute workflow
        workflow_engine.execute_workflow_intent(intent)

        # Verify parameters flowed through correctly
        assert len(parameter_flow) >= 3  # At least convert + 2 executions

        # Check convert stage preserved parameters
        convert_entries = [
            e for e in parameter_flow if e["stage"] == "convert_to_nodes"
        ]
        assert len(convert_entries) == 2
        assert convert_entries[0]["parameters"] == {
            "query": "test query",
            "type": "web",
        }
        assert convert_entries[1]["parameters"] == {"location": "Boston"}

        # Check execution stage received parameters
        execute_entries = [
            e for e in parameter_flow if e["stage"] == "universal_agent_execute"
        ]
        assert len(execute_entries) == 2

        # Find search and weather executions
        search_exec = next(e for e in execute_entries if e["role"] == "search")
        weather_exec = next(e for e in execute_entries if e["role"] == "weather")

        assert search_exec["parameters"] == {"query": "test query", "type": "web"}
        assert weather_exec["parameters"] == {"location": "Boston"}

    def test_empty_parameters_handled_correctly(self, setup_workflow_system):
        """Test that tasks without parameters work correctly."""
        workflow_engine = setup_workflow_system["workflow_engine"]
        universal_agent = setup_workflow_system["universal_agent"]

        captured_params = {}

        def mock_execute(instruction, role, llm_type, context, extracted_parameters):
            captured_params["params"] = extracted_parameters
            return "Done"

        universal_agent.execute_task = mock_execute

        # Create task without parameters
        intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Simple Task",
                    "description": "Task without parameters",
                    "role": "conversation",
                }
            ],
            dependencies=[],
            request_id="test_request",
            user_id="test_user",
            channel_id="test_channel",
            original_instruction="Test",
        )

        # Execute workflow
        workflow_engine.execute_workflow_intent(intent)

        # Verify empty dict is passed (not None)
        assert captured_params["params"] == {}
