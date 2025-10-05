"""
Global pytest configuration and fixtures for the test suite.
"""
import pytest
from unittest.mock import Mock, patch
from common.task_graph import TaskGraph, TaskNode, TaskStatus


@pytest.fixture
def mock_planning_phase():
    """
    Global fixture to mock the planning phase and avoid real LLM calls.
    
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
            status=TaskStatus.PENDING
        )
        task_graph = TaskGraph(tasks=[task], dependencies=[], request_id=request_id)
        
        return {
            "task_graph": task_graph,
            "tasks": [task],
            "dependencies": [],
            "request_id": request_id
        }
    
    with patch('llm_provider.planning_tools.create_task_plan') as mock_create_plan:
        mock_create_plan.side_effect = mock_plan_side_effect
        yield mock_create_plan


@pytest.fixture
def mock_llm_execution():
    """
    Global fixture to mock LLM task execution.
    
    This fixture mocks UniversalAgent.execute_task to return fast responses
    instead of making real Bedrock API calls.
    """
    def mock_execute_side_effect(instruction, role, llm_type, context=None):
        """Return appropriate mock responses based on the role."""
        # Handle router role specially - return JSON routing decisions
        if role == "router":
            # Parse the instruction to determine what route to return
            # Look for the actual user request within the routing prompt
            instruction_lower = instruction.lower()
            
            # Extract the user request from the routing prompt
            if 'user request:' in instruction_lower:
                # Find the user request part
                user_request_start = instruction_lower.find('user request:') + len('user request:')
                user_request_end = instruction_lower.find('options:', user_request_start)
                if user_request_end == -1:
                    user_request_end = len(instruction_lower)
                user_request = instruction_lower[user_request_start:user_request_end].strip().strip('"')
            else:
                user_request = instruction_lower
            
            # Route based on the actual user request content
            if "timer" in user_request and "set" in user_request:
                return '{"route": "timer", "confidence": 0.95}'
            elif ("meeting" in user_request or "schedule" in user_request) and "tomorrow" in user_request:
                return '{"route": "calendar", "confidence": 0.85}'
            elif "weather" in user_request and "seattle" in user_request:
                return '{"route": "weather", "confidence": 0.9}'
            elif "lights" in user_request and ("turn off" in user_request or "turn on" in user_request):
                return '{"route": "smart_home", "confidence": 0.88}'
            elif "plan" in user_request and ("comprehensive" in user_request or "multiple" in user_request or "phases" in user_request):
                return '{"route": "PLANNING", "confidence": 0.95}'
            elif "maybe" in user_request or "related" in user_request:
                return '{"route": "weather", "confidence": 0.3}'  # Low confidence for fallback tests
            else:
                return '{"route": "PLANNING", "confidence": 0.8}'
        
        # Handle other roles with appropriate responses
        role_responses = {
            'planning': 'Task plan created successfully',
            'search': 'Search results found',
            'analysis': 'Analysis completed',
            'coding': 'Code generated successfully',
            'weather': 'Weather data retrieved',
            'summarizer': 'Document summarized',
            'research_analyst': 'Research completed',
            'calendar': 'Task completed for role: calendar',
            'timer': 'Task completed for role: timer',
            'smart_home': 'Task completed for role: smart_home'
        }
        
        # Return role-specific response or generic response
        return role_responses.get(role, f"Task completed for role: {role}")
    
    with patch('llm_provider.universal_agent.UniversalAgent.execute_task') as mock_execute:
        mock_execute.side_effect = mock_execute_side_effect
        yield mock_execute


@pytest.fixture
def fast_integration_test(mock_planning_phase, mock_llm_execution):
    """
    Combined fixture for fast integration tests.
    
    This fixture combines planning and execution mocking to make integration
    tests run quickly without any real LLM calls.
    """
    return {
        'planning_mock': mock_planning_phase,
        'execution_mock': mock_llm_execution
    }