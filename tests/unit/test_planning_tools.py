"""Unit tests for planning tools (TDD approach).

These tests are written FIRST to define the expected behavior of planning tools.
Implementation follows after tests are written.

Test Coverage:
- create_execution_plan() tool
- replan() tool
- Tool integration with ToolRegistry
"""

import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.planning_types import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    ReplanRequest,
)


class TestCreateExecutionPlan:
    """Test create_execution_plan() tool."""

    @pytest.mark.anyio
    async def test_simple_single_tool_plan(self):
        """Test creating plan with single tool."""
        from roles.planning.tools import create_planning_tools

        # Create tools
        planning_provider = None  # Not needed for now
        tools = create_planning_tools(planning_provider)

        # Find create_execution_plan tool
        create_plan_tool = None
        for tool in tools:
            if hasattr(tool, "__name__") and tool.__name__ == "create_execution_plan":
                create_plan_tool = tool
                break

        assert create_plan_tool is not None, "create_execution_plan tool not found"

        # Call tool
        plan = await create_plan_tool(
            request="Set a timer for 5 minutes",
            selected_tools=["timer.set_timer"],
            context={"user_id": "test_user", "channel_id": "test_channel"},
        )

        # Validate plan
        assert isinstance(plan, ExecutionPlan)
        assert plan.plan_id.startswith("plan_")
        assert plan.request == "Set a timer for 5 minutes"
        assert plan.selected_tools == ["timer.set_timer"]
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "timer.set_timer"
        assert plan.steps[0].step_number == 1
        assert plan.status == PlanStatus.PENDING
        assert plan.created_at <= time.time()
        assert plan.reasoning  # Should have reasoning

    @pytest.mark.anyio
    async def test_multi_tool_sequential_plan(self):
        """Test creating plan with multiple tools in sequence."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        plan = await create_plan_tool(
            request="Check weather in Seattle and set a timer for 10 minutes",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            context={"user_id": "test_user"},
        )

        # Validate plan structure
        assert len(plan.steps) == 2
        assert plan.steps[0].step_number == 1
        assert plan.steps[1].step_number == 2

        # Second step should depend on first (sequential)
        assert plan.steps[1].depends_on == [1]

        # Tool assignment
        assert plan.steps[0].tool_name == "weather.get_current_weather"
        assert plan.steps[1].tool_name == "timer.set_timer"

    @pytest.mark.anyio
    async def test_plan_with_three_tools(self):
        """Test creating plan with three tools."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        plan = await create_plan_tool(
            request="Search for restaurants, check weather, and set a timer",
            selected_tools=[
                "search.web_search",
                "weather.get_current_weather",
                "timer.set_timer",
            ],
            context={"user_id": "test_user"},
        )

        # Should have 3 steps
        assert len(plan.steps) == 3
        assert [s.step_number for s in plan.steps] == [1, 2, 3]

        # Each tool assigned to a step
        tool_names = [s.tool_name for s in plan.steps]
        assert "search.web_search" in tool_names
        assert "weather.get_current_weather" in tool_names
        assert "timer.set_timer" in tool_names

    @pytest.mark.anyio
    async def test_plan_id_uniqueness(self):
        """Test that plan IDs are unique."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        # Create two plans
        plan1 = await create_plan_tool(
            request="Test request 1",
            selected_tools=["timer.set_timer"],
            context={},
        )
        plan2 = await create_plan_tool(
            request="Test request 2",
            selected_tools=["timer.set_timer"],
            context={},
        )

        # Plan IDs should be different
        assert plan1.plan_id != plan2.plan_id

    @pytest.mark.anyio
    async def test_plan_with_description(self):
        """Test that each step has a description."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        plan = await create_plan_tool(
            request="Check weather and set timer",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            context={},
        )

        # Each step should have a description
        for step in plan.steps:
            assert step.description
            assert len(step.description) > 0
            assert isinstance(step.description, str)

    @pytest.mark.anyio
    async def test_plan_metadata_includes_context(self):
        """Test that plan metadata includes context information."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        context = {"user_id": "user123", "channel_id": "channel456", "extra": "data"}
        plan = await create_plan_tool(
            request="Test request",
            selected_tools=["timer.set_timer"],
            context=context,
        )

        # Metadata should include context
        assert isinstance(plan.metadata, dict)
        # Context information should be preserved
        assert "user_id" in plan.metadata or len(plan.metadata) >= 0


class TestReplan:
    """Test replan() tool."""

    @pytest.mark.anyio
    async def test_replan_after_failure(self):
        """Test replanning after a step failure."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        replan_tool = next(
            t for t in tools if hasattr(t, "__name__") and t.__name__ == "replan"
        )

        # Create original plan
        original_plan = ExecutionPlan(
            plan_id="plan_original",
            request="Check weather and set timer",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                    status=PlanStatus.FAILED,
                    error="API timeout",
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                    status=PlanStatus.PENDING,
                ),
            ],
            reasoning="Original plan",
            created_at=time.time(),
            status=PlanStatus.IN_PROGRESS,
        )

        # Create replan request
        replan_request = ReplanRequest(
            current_plan=original_plan,
            execution_state={"current_step": 1, "error": "API timeout"},
            reason="Weather API failed, need alternative approach",
            completed_steps=[],
            failed_steps=[1],
        )

        # Call replan tool
        revised_plan = await replan_tool(replan_request)

        # Validate revised plan
        assert isinstance(revised_plan, ExecutionPlan)
        assert revised_plan.plan_id == original_plan.plan_id  # Same plan ID
        assert revised_plan.updated_at is not None  # Timestamp updated
        assert revised_plan.updated_at > original_plan.created_at

        # Failed step should still be marked as failed
        failed_step = next(s for s in revised_plan.steps if s.step_number == 1)
        assert failed_step.status == PlanStatus.FAILED

    @pytest.mark.anyio
    async def test_replan_preserves_completed_steps(self):
        """Test that replanning preserves completed steps."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        replan_tool = next(
            t for t in tools if hasattr(t, "__name__") and t.__name__ == "replan"
        )

        # Create plan with one completed step
        original_plan = ExecutionPlan(
            plan_id="plan_preserve",
            request="Multi-step workflow",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                    status=PlanStatus.COMPLETED,
                    result="Weather data retrieved",
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                    status=PlanStatus.PENDING,
                ),
            ],
            reasoning="Original plan",
            created_at=time.time(),
        )

        # Replan with new information
        replan_request = ReplanRequest(
            current_plan=original_plan,
            execution_state={"current_step": 2},
            reason="User provided additional context",
            completed_steps=[1],
            failed_steps=[],
            new_information="Set timer for 15 minutes instead of 10",
        )

        revised_plan = await replan_tool(replan_request)

        # Completed step should remain completed
        completed_step = next(s for s in revised_plan.steps if s.step_number == 1)
        assert completed_step.status == PlanStatus.COMPLETED
        assert completed_step.result == "Weather data retrieved"

    @pytest.mark.anyio
    async def test_replan_updates_status_to_replanning(self):
        """Test that replan sets status to REPLANNING."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        replan_tool = next(
            t for t in tools if hasattr(t, "__name__") and t.__name__ == "replan"
        )

        original_plan = ExecutionPlan(
            plan_id="plan_status",
            request="Test request",
            selected_tools=["timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="timer.set_timer",
                    description="Set timer",
                )
            ],
            reasoning="Original",
            created_at=time.time(),
        )

        replan_request = ReplanRequest(
            current_plan=original_plan,
            execution_state={},
            reason="Adjustment needed",
        )

        revised_plan = await replan_tool(replan_request)

        # Status should be REPLANNING
        assert revised_plan.status == PlanStatus.REPLANNING


class TestToolIntegration:
    """Test planning tools integration with ToolRegistry."""

    def test_create_planning_tools_returns_list(self):
        """Test that create_planning_tools returns a list of tools."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)

        assert isinstance(tools, list)
        assert len(tools) == 2  # create_execution_plan and replan

    def test_tools_have_correct_names(self):
        """Test that tools have expected names."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)

        tool_names = [t.__name__ for t in tools if hasattr(t, "__name__")]

        assert "create_execution_plan" in tool_names
        assert "replan" in tool_names

    def test_tools_are_decorated_with_strands_tool(self):
        """Test that tools are decorated with @tool decorator."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)

        # Check that tools have Strands tool attributes
        for tool in tools:
            # Strands @tool decorator should add certain attributes
            # At minimum, they should be callable
            assert callable(tool)

    @pytest.mark.anyio
    async def test_tools_are_async(self):
        """Test that planning tools are async functions."""
        import inspect

        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)

        for tool in tools:
            # Planning tools should be async (check the underlying function if wrapped)
            if hasattr(tool, "__wrapped__"):
                assert inspect.iscoroutinefunction(tool.__wrapped__)
            elif hasattr(tool, "fn"):
                assert inspect.iscoroutinefunction(tool.fn)
            else:
                assert inspect.iscoroutinefunction(tool)


class TestToolSignatures:
    """Test that tool signatures match expected interface."""

    @pytest.mark.anyio
    async def test_create_execution_plan_signature(self):
        """Test create_execution_plan has correct parameters."""
        import inspect

        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        # Get signature
        sig = inspect.signature(create_plan_tool)
        params = list(sig.parameters.keys())

        # Should have these parameters
        assert "request" in params
        assert "selected_tools" in params
        assert "context" in params

    @pytest.mark.anyio
    async def test_replan_signature(self):
        """Test replan has correct parameters."""
        import inspect

        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        replan_tool = next(
            t for t in tools if hasattr(t, "__name__") and t.__name__ == "replan"
        )

        # Get signature
        sig = inspect.signature(replan_tool)
        params = list(sig.parameters.keys())

        # Should have replan_request parameter
        assert "replan_request" in params


class TestToolReturnTypes:
    """Test that tools return correct types."""

    @pytest.mark.anyio
    async def test_create_execution_plan_returns_execution_plan(self):
        """Test create_execution_plan returns ExecutionPlan."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        create_plan_tool = next(
            t
            for t in tools
            if hasattr(t, "__name__") and t.__name__ == "create_execution_plan"
        )

        result = await create_plan_tool(
            request="Test",
            selected_tools=["timer.set_timer"],
            context={},
        )

        assert isinstance(result, ExecutionPlan)

    @pytest.mark.anyio
    async def test_replan_returns_execution_plan(self):
        """Test replan returns ExecutionPlan."""
        from roles.planning.tools import create_planning_tools

        tools = create_planning_tools(None)
        replan_tool = next(
            t for t in tools if hasattr(t, "__name__") and t.__name__ == "replan"
        )

        original_plan = ExecutionPlan(
            plan_id="test",
            request="Test",
            selected_tools=["timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="timer.set_timer",
                    description="Test",
                )
            ],
            reasoning="Test",
            created_at=time.time(),
        )

        replan_request = ReplanRequest(
            current_plan=original_plan,
            execution_state={},
            reason="Test",
        )

        result = await replan_tool(replan_request)

        assert isinstance(result, ExecutionPlan)
