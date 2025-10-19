"""Global pytest configuration and fixtures for the test suite."""

from unittest.mock import Mock, patch

import pytest

from common.task_graph import TaskGraph, TaskNode, TaskStatus
from llm_provider.role_registry import RoleRegistry


@pytest.fixture
def mock_planning_phase():
    """Global fixture to mock the planning phase and avoid real LLM calls.

    This fixture mocks create_task_plan to return a simple single-task plan
    instead of making real Bedrock API calls for planning.
    """

    def mock_plan_side_effect(instruction, llm_factory, request_id):
        """Create a mock task plan without real LLM calls."""
        # Create a simple single-task plan
        task = TaskNode(
            task_id="mock_task_1",
            task_name="Execute Request",
            request_id=request_id,
            agent_id="default",
            task_type="execution",
            prompt=instruction,
            status=TaskStatus.PENDING,
        )
        task_graph = TaskGraph(tasks=[task], dependencies=[], request_id=request_id)

        return {
            "task_graph": task_graph,
            "tasks": [task],
            "dependencies": [],
            "request_id": request_id,
        }

        # Legacy planning tools removed - no longer using task decomposition
        # Complex requests now route through router to appropriate roles
        yield mock_create_plan


@pytest.fixture
def mock_llm_execution():
    """Global fixture to mock LLM task execution.

    This fixture mocks UniversalAgent.execute_task to return fast responses
    instead of making real Bedrock API calls.
    """

    def mock_execute_side_effect(
        instruction, role, llm_type=None, context=None, extracted_parameters=None
    ):
        """Return appropriate mock responses based on the role."""
        if role == "router":
            return _get_router_response(instruction)
        else:
            return _get_role_response(role)

    with patch(
        "llm_provider.universal_agent.UniversalAgent.execute_task"
    ) as mock_execute:
        mock_execute.side_effect = mock_execute_side_effect
        yield mock_execute


def _get_router_response(instruction: str) -> str:
    """Get router response based on instruction content."""
    user_request = _extract_user_request(instruction.lower())
    return _route_user_request(user_request)


def _extract_user_request(instruction_lower: str) -> str:
    """Extract user request from routing prompt."""
    # Handle both old and new formats
    if "request:" in instruction_lower:
        # New format: "Request: "Turn off the living room lights""
        user_request_start = instruction_lower.find("request:") + len("request:")
        # Look for end markers in order of preference
        end_markers = ["available roles", "analyze the request", "options:", "\n\n"]
        user_request_end = len(instruction_lower)
        for marker in end_markers:
            marker_pos = instruction_lower.find(marker, user_request_start)
            if marker_pos != -1:
                user_request_end = marker_pos
                break
        extracted = (
            instruction_lower[user_request_start:user_request_end]
            .strip()
            .strip('"')
            .strip()
        )
        return extracted
    elif "user request:" in instruction_lower:
        # Old format: "User Request: "Turn off the living room lights""
        user_request_start = instruction_lower.find("user request:") + len(
            "user request:"
        )
        user_request_end = instruction_lower.find("options:", user_request_start)
        if user_request_end == -1:
            user_request_end = len(instruction_lower)
        return instruction_lower[user_request_start:user_request_end].strip().strip('"')
    else:
        return instruction_lower


def _route_user_request(user_request: str) -> str:
    """Route user request to appropriate service."""
    if "timer" in user_request and "set" in user_request:
        return '{"route": "timer", "confidence": 0.95}'
    elif (
        "meeting" in user_request or "schedule" in user_request
    ) and "tomorrow" in user_request:
        return '{"route": "calendar", "confidence": 0.85}'
    elif "weather" in user_request and "seattle" in user_request:
        return '{"route": "weather", "confidence": 0.9}'
    elif "lights" in user_request and (
        "turn off" in user_request or "turn on" in user_request
    ):
        return '{"route": "smart_home", "confidence": 0.88}'
    elif "plan" in user_request and _is_planning_request(user_request):
        return '{"route": "PLANNING", "confidence": 0.95}'
    elif "maybe" in user_request or "related" in user_request:
        return '{"route": "weather", "confidence": 0.3}'  # Low confidence for fallback tests
    else:
        return '{"route": "PLANNING", "confidence": 0.8}'


def _is_planning_request(user_request: str) -> bool:
    """Check if request is for planning."""
    return any(
        keyword in user_request for keyword in ["comprehensive", "multiple", "phases"]
    )


def _get_role_response(role: str) -> str:
    """Get response for non-router roles."""
    role_responses = {
        "planning": "Task plan created successfully",
        "search": "Search results found",
        "analysis": "Analysis completed",
        "coding": "Code generated successfully",
        "weather": "Weather data retrieved",
        "summarizer": "Document summarized",
        "research_analyst": "Research completed",
        "calendar": "Task completed for role: calendar",
        "timer": "Task completed for role: timer",
        "smart_home": "Task completed for role: smart_home",
    }

    return role_responses.get(role, f"Task completed for role: {role}")


@pytest.fixture(scope="session")
def shared_role_registry():
    """Shared RoleRegistry fixture for performance optimization.

    This fixture creates a single RoleRegistry instance that's reused across
    all tests in the session, avoiding expensive role loading on every test setup.
    Uses session scope to maximize performance benefits.
    """
    return RoleRegistry("roles")


@pytest.fixture(autouse=True)
def mock_slack_api():
    """Global fixture to mock Slack API calls across all tests.

    This prevents any test from making real network calls to Slack API,
    avoiding network errors and improving test performance.
    """
    with (
        patch("slack_sdk.WebClient") as mock_client_class,
        patch("slack_bolt.App") as mock_bolt_app_class,
        patch("slack_sdk.socket_mode.SocketModeClient") as mock_socket_client_class,
    ):
        # Mock WebClient
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {
            "ok": True,
            "channel": "C1234567890",
            "ts": "1234567890.123456",
            "message": {"ts": "1234567890.123456", "text": "Test message"},
        }
        mock_client.conversations_list.return_value = {
            "ok": True,
            "channels": [{"id": "C1234567890", "name": "general"}],
        }
        mock_client_class.return_value = mock_client

        # Mock Bolt App to prevent "⚡️ Bolt app is running!" messages
        mock_bolt_app = Mock()
        mock_bolt_app.start.return_value = None
        mock_bolt_app.stop.return_value = None
        mock_bolt_app_class.return_value = mock_bolt_app

        # Mock SocketModeClient to prevent connection attempts
        mock_socket_client = Mock()
        mock_socket_client.connect.return_value = None
        mock_socket_client.disconnect.return_value = None
        mock_socket_client_class.return_value = mock_socket_client

        yield {
            "client": mock_client,
            "bolt_app": mock_bolt_app,
            "socket_client": mock_socket_client,
        }


@pytest.fixture
def fast_integration_test(mock_planning_phase, mock_llm_execution):
    """Combined fixture for fast integration tests.

    This fixture combines planning and execution mocking to make integration
    tests run quickly without any real LLM calls.
    """
    return {"planning_mock": mock_planning_phase, "execution_mock": mock_llm_execution}
