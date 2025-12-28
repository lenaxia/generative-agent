"""Type-safe planning data structures.

This module provides Pydantic models and enums for execution planning in Phase 4
meta-planning workflows. All planning operations use these types for validation
and type safety.

Architecture: Type-Safe Planning with Pydantic Validation
Created: 2025-12-27
"""

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PlanStatus(str, Enum):
    """Execution plan status.

    Status progression:
        PENDING → IN_PROGRESS → COMPLETED
                ↓
           REPLANNING → IN_PROGRESS
                ↓
             FAILED
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


class ExecutionStep(BaseModel):
    """Single step in execution plan.

    Each step represents one tool invocation with its parameters and dependencies.
    Steps can have dependencies on other steps, creating a execution graph.

    Attributes:
        step_number: Unique step identifier (1-indexed)
        tool_name: Tool to execute in "domain.tool" format
        description: Human-readable description of what this step does
        parameters: Tool parameters extracted from user request
        depends_on: List of step_numbers this step depends on
        status: Current execution status
        result: Result from tool execution (populated after execution)
        error: Error message if step failed
    """

    step_number: int = Field(..., ge=1, description="Step sequence number (1-indexed)")
    tool_name: str = Field(..., min_length=1, description="Tool to execute")
    description: str = Field(
        ..., min_length=1, description="Human-readable step description"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this step depends on"
    )
    status: PlanStatus = Field(
        default=PlanStatus.PENDING, description="Step execution status"
    )
    result: Any | None = Field(
        default=None, description="Step execution result (populated after execution)"
    )
    error: str | None = Field(default=None, description="Error message if step failed")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool_name follows domain.tool pattern.

        Args:
            v: Tool name to validate

        Returns:
            str: Validated tool name

        Raises:
            ValueError: If tool name doesn't follow domain.tool pattern

        Examples:
            Valid: "weather.get_current_weather", "timer.set_timer"
            Invalid: "get_weather", "set timer"
        """
        if "." not in v:
            raise ValueError(
                f"Tool name must follow 'domain.tool' format, got: '{v}'. "
                f"Examples: 'weather.get_current_weather', 'timer.set_timer'"
            )
        parts = v.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Tool name must have exactly one dot (domain.tool), got: '{v}'"
            )
        domain, tool = parts
        if not domain or not tool:
            raise ValueError(f"Both domain and tool must be non-empty, got: '{v}'")
        return v

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, v: list[int], info) -> list[int]:
        """Ensure dependencies are valid step numbers.

        Args:
            v: List of step numbers
            info: Validation context

        Returns:
            list[int]: Validated dependencies

        Raises:
            ValueError: If dependencies contain invalid step numbers
        """
        if not v:
            return v

        # Get current step number from context
        step_number = info.data.get("step_number")
        if step_number is not None:
            # Ensure no self-dependencies
            if step_number in v:
                raise ValueError(f"Step {step_number} cannot depend on itself")
            # Ensure no forward dependencies (can only depend on earlier steps)
            for dep in v:
                if dep >= step_number:
                    raise ValueError(
                        f"Step {step_number} cannot depend on step {dep} "
                        f"(forward dependencies not allowed)"
                    )

        # Ensure all dependencies are positive
        for dep in v:
            if dep < 1:
                raise ValueError(
                    f"Step dependencies must be positive integers, got: {dep}"
                )

        return v

    model_config = {
        "extra": "forbid",  # Don't allow extra fields
        "str_strip_whitespace": True,  # Strip whitespace from strings
        "validate_assignment": True,  # Validate on attribute assignment
    }


class ExecutionPlan(BaseModel):
    """Structured execution plan for agent.

    An execution plan defines the steps an agent should take to fulfill a request.
    It's created after tool selection and guides the agent during execution.

    The plan is not set in stone - agents can call replan() if they encounter
    issues or discover better approaches during execution.

    Attributes:
        plan_id: Unique plan identifier
        request: Original user request
        selected_tools: Tools selected by meta-planner
        steps: Ordered list of execution steps
        reasoning: Why this plan was created (LLM's reasoning)
        created_at: Timestamp of plan creation
        updated_at: Timestamp of last update (e.g., after replanning)
        status: Overall plan status
        metadata: Additional metadata about the plan
    """

    plan_id: str = Field(..., min_length=1, description="Unique plan identifier")
    request: str = Field(..., min_length=1, description="Original user request")
    selected_tools: list[str] = Field(
        ..., min_items=1, description="Tools selected by meta-planner"
    )
    steps: list[ExecutionStep] = Field(
        ..., min_items=1, description="Ordered execution steps"
    )
    reasoning: str = Field(..., min_length=1, description="Why this plan was created")
    created_at: float = Field(..., description="Timestamp of plan creation")
    updated_at: float | None = Field(
        default=None, description="Timestamp of last update"
    )
    status: PlanStatus = Field(
        default=PlanStatus.PENDING, description="Overall plan status"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional plan metadata"
    )

    @field_validator("selected_tools")
    @classmethod
    def validate_selected_tools(cls, v: list[str]) -> list[str]:
        """Ensure all selected tools follow domain.tool pattern.

        Args:
            v: List of tool names

        Returns:
            list[str]: Validated tool names

        Raises:
            ValueError: If any tool name is invalid
        """
        for tool_name in v:
            if "." not in tool_name:
                raise ValueError(
                    f"Tool name must follow 'domain.tool' format, got: '{tool_name}'"
                )
        return v

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: list[ExecutionStep]) -> list[ExecutionStep]:
        """Ensure steps are properly ordered and have unique step_numbers.

        Args:
            v: List of execution steps

        Returns:
            list[ExecutionStep]: Validated steps

        Raises:
            ValueError: If steps have duplicate numbers or gaps
        """
        if not v:
            raise ValueError("Plan must have at least one step")

        # Check for duplicate step numbers
        step_numbers = [step.step_number for step in v]
        if len(step_numbers) != len(set(step_numbers)):
            raise ValueError(f"Duplicate step numbers found: {step_numbers}")

        # Check for gaps in step numbers (should be 1, 2, 3, ...)
        expected = list(range(1, len(v) + 1))
        sorted_numbers = sorted(step_numbers)
        if sorted_numbers != expected:
            raise ValueError(
                f"Step numbers must be sequential starting from 1. "
                f"Expected: {expected}, Got: {sorted_numbers}"
            )

        # Validate all dependencies reference existing steps
        all_step_numbers = set(step_numbers)
        for step in v:
            for dep in step.depends_on:
                if dep not in all_step_numbers:
                    raise ValueError(
                        f"Step {step.step_number} depends on non-existent step {dep}"
                    )

        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, v: float | None) -> float | None:
        """Ensure timestamp is reasonable (not in the future).

        Args:
            v: Timestamp to validate

        Returns:
            Optional[float]: Validated timestamp

        Raises:
            ValueError: If timestamp is in the future
        """
        if v is None:
            return v

        current_time = time.time()
        if v > current_time + 60:  # Allow 60s clock skew
            raise ValueError(
                f"Timestamp cannot be in the future. Got: {v}, Current: {current_time}"
            )
        return v

    model_config = {
        "extra": "forbid",  # Don't allow extra fields
        "str_strip_whitespace": True,  # Strip whitespace from strings
        "validate_assignment": True,  # Validate on attribute assignment
    }


class ReplanRequest(BaseModel):
    """Request to revise execution plan.

    Used when an agent needs to adjust its plan during execution due to:
    - Step failures requiring alternative approach
    - New information suggesting better path
    - User providing additional context mid-execution

    Attributes:
        current_plan: Current execution plan
        execution_state: Current execution state (which steps completed, results, etc.)
        reason: Why replanning is needed
        completed_steps: Step numbers that have been completed
        failed_steps: Step numbers that have failed
        new_information: Additional context for replanning
    """

    current_plan: ExecutionPlan = Field(..., description="Current execution plan")
    execution_state: dict[str, Any] = Field(..., description="Current execution state")
    reason: str = Field(..., min_length=1, description="Why replanning is needed")
    completed_steps: list[int] = Field(
        default_factory=list, description="Steps already completed"
    )
    failed_steps: list[int] = Field(
        default_factory=list, description="Steps that have failed"
    )
    new_information: str | None = Field(
        default=None, description="Additional context for replanning"
    )

    @field_validator("completed_steps", "failed_steps")
    @classmethod
    def validate_step_lists(cls, v: list[int], info) -> list[int]:
        """Ensure step lists contain valid step numbers from current plan.

        Args:
            v: List of step numbers
            info: Validation context

        Returns:
            list[int]: Validated step numbers

        Raises:
            ValueError: If step numbers are invalid
        """
        if not v:
            return v

        # Get current plan from context
        current_plan = info.data.get("current_plan")
        if current_plan is not None:
            plan_step_numbers = {step.step_number for step in current_plan.steps}
            for step_num in v:
                if step_num not in plan_step_numbers:
                    raise ValueError(
                        f"Step {step_num} not found in current plan. "
                        f"Valid steps: {sorted(plan_step_numbers)}"
                    )

        return v

    @model_validator(mode="after")
    def validate_no_overlap(self) -> "ReplanRequest":
        """Ensure completed_steps and failed_steps don't overlap.

        Returns:
            ReplanRequest: Validated replan request

        Raises:
            ValueError: If there's overlap between completed and failed steps
        """
        overlap = set(self.completed_steps) & set(self.failed_steps)
        if overlap:
            raise ValueError(
                f"Steps cannot be both completed and failed: {sorted(overlap)}"
            )
        return self

    model_config = {
        "extra": "forbid",  # Don't allow extra fields
        "str_strip_whitespace": True,  # Strip whitespace from strings
        "validate_assignment": True,  # Validate on attribute assignment
    }
