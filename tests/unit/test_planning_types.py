"""Unit tests for planning types.

Tests the type-safe planning data structures in common/planning_types.py.
"""

import time

import pytest
from pydantic import ValidationError

from common.planning_types import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    ReplanRequest,
)


class TestPlanStatus:
    """Test PlanStatus enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert PlanStatus.PENDING == "pending"
        assert PlanStatus.IN_PROGRESS == "in_progress"
        assert PlanStatus.COMPLETED == "completed"
        assert PlanStatus.FAILED == "failed"
        assert PlanStatus.REPLANNING == "replanning"

    def test_enum_comparison(self):
        """Test enum comparisons."""
        assert PlanStatus.PENDING == PlanStatus.PENDING
        assert PlanStatus.PENDING != PlanStatus.IN_PROGRESS


class TestExecutionStep:
    """Test ExecutionStep model."""

    def test_valid_step(self):
        """Test creating a valid execution step."""
        step = ExecutionStep(
            step_number=1,
            tool_name="weather.get_current_weather",
            description="Get current weather for Seattle",
            parameters={"location": "Seattle"},
            depends_on=[],
            status=PlanStatus.PENDING,
        )

        assert step.step_number == 1
        assert step.tool_name == "weather.get_current_weather"
        assert step.description == "Get current weather for Seattle"
        assert step.parameters == {"location": "Seattle"}
        assert step.depends_on == []
        assert step.status == PlanStatus.PENDING
        assert step.result is None
        assert step.error is None

    def test_minimal_step(self):
        """Test creating a step with minimal required fields."""
        step = ExecutionStep(
            step_number=1,
            tool_name="timer.set_timer",
            description="Set timer",
        )

        assert step.step_number == 1
        assert step.parameters == {}
        assert step.depends_on == []
        assert step.status == PlanStatus.PENDING

    def test_invalid_tool_name_no_dot(self):
        """Test tool name without domain.tool format fails."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=1,
                tool_name="get_weather",  # Missing domain
                description="Get weather",
            )

        error = exc_info.value.errors()[0]
        assert "domain.tool" in error["msg"].lower()

    def test_invalid_tool_name_multiple_dots(self):
        """Test tool name with multiple dots fails."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=1,
                tool_name="weather.api.get_current_weather",  # Too many dots
                description="Get weather",
            )

        error = exc_info.value.errors()[0]
        assert "exactly one dot" in error["msg"].lower()

    def test_invalid_tool_name_empty_parts(self):
        """Test tool name with empty domain or tool fails."""
        with pytest.raises(ValidationError):
            ExecutionStep(
                step_number=1,
                tool_name=".get_weather",  # Empty domain
                description="Get weather",
            )

        with pytest.raises(ValidationError):
            ExecutionStep(
                step_number=1,
                tool_name="weather.",  # Empty tool
                description="Get weather",
            )

    def test_invalid_step_number_zero(self):
        """Test step_number must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=0,
                tool_name="weather.get_current_weather",
                description="Get weather",
            )

        error = exc_info.value.errors()[0]
        assert error["type"] == "greater_than_equal"

    def test_invalid_step_number_negative(self):
        """Test step_number cannot be negative."""
        with pytest.raises(ValidationError):
            ExecutionStep(
                step_number=-1,
                tool_name="weather.get_current_weather",
                description="Get weather",
            )

    def test_self_dependency(self):
        """Test step cannot depend on itself."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=1,
                tool_name="weather.get_current_weather",
                description="Get weather",
                depends_on=[1],  # Self-dependency
            )

        error = exc_info.value.errors()[0]
        assert "cannot depend on itself" in error["msg"].lower()

    def test_forward_dependency(self):
        """Test step cannot depend on later steps."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=1,
                tool_name="weather.get_current_weather",
                description="Get weather",
                depends_on=[2],  # Forward dependency
            )

        error = exc_info.value.errors()[0]
        assert "forward dependencies not allowed" in error["msg"].lower()

    def test_invalid_dependency_negative(self):
        """Test dependencies must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionStep(
                step_number=2,
                tool_name="weather.get_current_weather",
                description="Get weather",
                depends_on=[0],  # Invalid dependency
            )

        error = exc_info.value.errors()[0]
        assert "positive integers" in error["msg"].lower()

    def test_sequential_dependency(self):
        """Test valid sequential dependency."""
        step = ExecutionStep(
            step_number=2,
            tool_name="timer.set_timer",
            description="Set timer",
            depends_on=[1],  # Depends on step 1
        )

        assert step.depends_on == [1]

    def test_multiple_dependencies(self):
        """Test step can depend on multiple earlier steps."""
        step = ExecutionStep(
            step_number=3,
            tool_name="notification.send",
            description="Send notification",
            depends_on=[1, 2],  # Depends on steps 1 and 2
        )

        assert step.depends_on == [1, 2]


class TestExecutionPlan:
    """Test ExecutionPlan model."""

    def test_valid_plan(self):
        """Test creating a valid execution plan."""
        plan = ExecutionPlan(
            plan_id="plan_abc123",
            request="Check weather in Seattle and set a timer for 10 minutes",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                    parameters={"location": "Seattle"},
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                    parameters={"duration": "10m"},
                    depends_on=[1],
                ),
            ],
            reasoning="Sequential execution: first get weather, then set timer",
            created_at=time.time(),
            status=PlanStatus.PENDING,
        )

        assert plan.plan_id == "plan_abc123"
        assert len(plan.steps) == 2
        assert plan.status == PlanStatus.PENDING
        assert plan.updated_at is None
        assert plan.metadata == {}

    def test_invalid_empty_tools(self):
        """Test plan must have at least one tool."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=[],  # Empty tools list
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather",
                    )
                ],
                reasoning="Test",
                created_at=time.time(),
            )

        error = exc_info.value.errors()[0]
        assert error["type"] == "too_short"

    def test_invalid_empty_steps(self):
        """Test plan must have at least one step."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["weather.get_current_weather"],
                steps=[],  # Empty steps list
                reasoning="Test",
                created_at=time.time(),
            )

        error = exc_info.value.errors()[0]
        # Pydantic's constraint error for min_length
        assert (
            "at least 1 item" in error["msg"].lower()
            or "at least one step" in error["msg"].lower()
        )

    def test_invalid_tool_format(self):
        """Test selected tools must follow domain.tool format."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["get_weather"],  # Invalid format
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather",
                    )
                ],
                reasoning="Test",
                created_at=time.time(),
            )

        error = exc_info.value.errors()[0]
        assert "domain.tool" in error["msg"].lower()

    def test_duplicate_step_numbers(self):
        """Test steps cannot have duplicate step numbers."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["weather.get_current_weather"],
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather 1",
                    ),
                    ExecutionStep(
                        step_number=1,  # Duplicate
                        tool_name="weather.get_current_weather",
                        description="Get weather 2",
                    ),
                ],
                reasoning="Test",
                created_at=time.time(),
            )

        error = exc_info.value.errors()[0]
        assert "duplicate" in error["msg"].lower()

    def test_step_number_gaps(self):
        """Test steps must be sequential (1, 2, 3, ...)."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["weather.get_current_weather"],
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather",
                    ),
                    ExecutionStep(
                        step_number=3,  # Gap: missing step 2
                        tool_name="timer.set_timer",
                        description="Set timer",
                    ),
                ],
                reasoning="Test",
                created_at=time.time(),
            )

        error = exc_info.value.errors()[0]
        assert "sequential" in error["msg"].lower()

    def test_nonexistent_dependency(self):
        """Test step dependencies must reference existing steps.

        Note: Creating a step with dependency on non-existent step can trigger
        either "forward dependencies not allowed" (if dep > step_number) or
        "non-existent step" (checked at ExecutionPlan level). Both are valid errors.
        """
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["weather.get_current_weather", "timer.set_timer"],
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather",
                    ),
                    ExecutionStep(
                        step_number=2,
                        tool_name="timer.set_timer",
                        description="Set timer",
                        depends_on=[99],  # Non-existent step (also forward dep)
                    ),
                ],
                reasoning="Test",
                created_at=time.time(),
            )

        # ValidationError is raised - either for forward dependency or non-existent step
        assert exc_info.value.errors()  # Just verify validation failed

    def test_future_timestamp(self):
        """Test timestamp cannot be in the future."""
        future_time = time.time() + 3600  # 1 hour in the future

        with pytest.raises(ValidationError) as exc_info:
            ExecutionPlan(
                plan_id="plan_abc123",
                request="Do something",
                selected_tools=["weather.get_current_weather"],
                steps=[
                    ExecutionStep(
                        step_number=1,
                        tool_name="weather.get_current_weather",
                        description="Get weather",
                    )
                ],
                reasoning="Test",
                created_at=future_time,
            )

        error = exc_info.value.errors()[0]
        assert "future" in error["msg"].lower()

    def test_complex_dependency_graph(self):
        """Test plan with complex dependency graph."""
        plan = ExecutionPlan(
            plan_id="plan_complex",
            request="Complex workflow",
            selected_tools=[
                "weather.get_current_weather",
                "timer.set_timer",
                "notification.send",
            ],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                ),
                ExecutionStep(
                    step_number=3,
                    tool_name="notification.send",
                    description="Send notification",
                    depends_on=[1, 2],  # Depends on both 1 and 2
                ),
            ],
            reasoning="Parallel execution with merge",
            created_at=time.time(),
        )

        assert len(plan.steps) == 3
        assert plan.steps[2].depends_on == [1, 2]


class TestReplanRequest:
    """Test ReplanRequest model."""

    def test_valid_replan_request(self):
        """Test creating a valid replan request."""
        original_plan = ExecutionPlan(
            plan_id="plan_abc123",
            request="Do something",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                ),
            ],
            reasoning="Original plan",
            created_at=time.time(),
        )

        replan_req = ReplanRequest(
            current_plan=original_plan,
            execution_state={"current_step": 2},
            reason="Step 1 failed due to API error",
            completed_steps=[],
            failed_steps=[1],
            new_information="Weather API is down, need alternative",
        )

        assert replan_req.current_plan.plan_id == "plan_abc123"
        assert replan_req.reason == "Step 1 failed due to API error"
        assert replan_req.completed_steps == []
        assert replan_req.failed_steps == [1]

    def test_minimal_replan_request(self):
        """Test replan request with minimal fields."""
        original_plan = ExecutionPlan(
            plan_id="plan_abc123",
            request="Do something",
            selected_tools=["weather.get_current_weather"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                )
            ],
            reasoning="Original plan",
            created_at=time.time(),
        )

        replan_req = ReplanRequest(
            current_plan=original_plan,
            execution_state={},
            reason="Need to adjust approach",
        )

        assert replan_req.completed_steps == []
        assert replan_req.failed_steps == []
        assert replan_req.new_information is None

    def test_invalid_step_not_in_plan(self):
        """Test completed/failed steps must exist in plan."""
        original_plan = ExecutionPlan(
            plan_id="plan_abc123",
            request="Do something",
            selected_tools=["weather.get_current_weather"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                )
            ],
            reasoning="Original plan",
            created_at=time.time(),
        )

        with pytest.raises(ValidationError) as exc_info:
            ReplanRequest(
                current_plan=original_plan,
                execution_state={},
                reason="Test",
                completed_steps=[99],  # Non-existent step
            )

        error = exc_info.value.errors()[0]
        assert "not found" in error["msg"].lower()

    def test_overlap_completed_and_failed(self):
        """Test step cannot be both completed and failed."""
        original_plan = ExecutionPlan(
            plan_id="plan_abc123",
            request="Do something",
            selected_tools=["weather.get_current_weather", "timer.set_timer"],
            steps=[
                ExecutionStep(
                    step_number=1,
                    tool_name="weather.get_current_weather",
                    description="Get weather",
                ),
                ExecutionStep(
                    step_number=2,
                    tool_name="timer.set_timer",
                    description="Set timer",
                ),
            ],
            reasoning="Original plan",
            created_at=time.time(),
        )

        with pytest.raises(ValidationError) as exc_info:
            ReplanRequest(
                current_plan=original_plan,
                execution_state={},
                reason="Test",
                completed_steps=[1],
                failed_steps=[1],  # Same step in both lists
            )

        error = exc_info.value.errors()[0]
        assert "both completed and failed" in error["msg"].lower()
