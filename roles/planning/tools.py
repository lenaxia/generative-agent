"""Planning Domain Tools

Provides structured execution planning for meta-planning agents.

These tools create and manage execution plans that guide agents during autonomous
execution in Phase 4 meta-planning workflows.

Architecture: Type-Safe Planning with LLM-Guided Plan Generation
Created: 2025-12-27
"""

import logging
import time
import uuid
from typing import Any

from strands import tool

from common.planning_types import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    ReplanRequest,
)

logger = logging.getLogger(__name__)


def create_planning_tools(planning_provider: Any) -> list:
    """Create planning domain tools.

    Args:
        planning_provider: Planning provider instance (reserved for future use)

    Returns:
        List of tool functions for planning domain

    Tools:
        - create_execution_plan: Create structured plan from selected tools
        - replan: Revise execution plan based on new information or failures
    """

    @tool
    async def create_execution_plan(
        request: str,
        selected_tools: list[str],
        context: dict[str, Any],
    ) -> ExecutionPlan:
        """Create structured execution plan based on selected tools.

        This tool is called by meta-planning AFTER tool selection to create
        a detailed step-by-step execution plan. The plan keeps the agent on track
        during execution.

        Args:
            request: Original user request
            selected_tools: Tools selected by meta-planner (e.g., ["weather.get_current_weather", "timer.set_timer"])
            context: Additional context about the request (user_id, channel_id, etc.)

        Returns:
            ExecutionPlan: Structured plan with ordered steps

        Example:
            Request: "Check weather in Seattle and set a timer for 10 minutes"
            Selected Tools: ["weather.get_current_weather", "timer.set_timer"]

            Returns ExecutionPlan with steps:
            1. weather.get_current_weather(location="Seattle")
            2. timer.set_timer(duration="10m")

        Note:
            This is a simplified implementation. Future versions will use LLM
            to generate more intelligent plans with parameter extraction and
            dependency analysis.
        """
        logger.info(f"Creating execution plan for request: {request[:100]}...")
        logger.info(f"Selected tools: {selected_tools}")

        # Generate unique plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Create steps from selected tools
        # Simple sequential plan: each step depends on the previous one
        steps: list[ExecutionStep] = []
        for idx, tool_name in enumerate(selected_tools, start=1):
            # Extract tool domain and name
            domain, tool = tool_name.split(".")

            # Generate step description
            description = f"Execute {tool} from {domain} domain"

            # Create step
            step = ExecutionStep(
                step_number=idx,
                tool_name=tool_name,
                description=description,
                parameters={},  # TODO: LLM will extract from request in future
                depends_on=[idx - 1] if idx > 1 else [],  # Sequential dependency
                status=PlanStatus.PENDING,
            )
            steps.append(step)

        # Create execution plan
        plan = ExecutionPlan(
            plan_id=plan_id,
            request=request,
            selected_tools=selected_tools,
            steps=steps,
            reasoning=f"Sequential execution of {len(steps)} selected tools",
            created_at=time.time(),
            status=PlanStatus.PENDING,
            metadata={
                **context,  # Include context in metadata
                "plan_type": "sequential",
                "num_steps": len(steps),
            },
        )

        logger.info(f"Created execution plan {plan_id} with {len(steps)} steps")
        logger.debug(f"Plan: {plan.model_dump_json(indent=2)}")

        return plan

    @tool
    async def replan(replan_request: ReplanRequest) -> ExecutionPlan:
        """Revise execution plan based on new information or failures.

        Called by the agent when:
        - A step fails and needs alternative approach
        - New information suggests better path
        - User provides additional context mid-execution

        Args:
            replan_request: Request containing current plan, state, and reason for replanning

        Returns:
            ExecutionPlan: Revised plan with updated steps

        Example:
            Current plan step failed: weather API returned error
            Agent calls replan() with failure context
            Returns new plan with fallback approach or adjusted steps

        Note:
            This is a simplified implementation. Future versions will use LLM
            to generate intelligent replans based on failure context and
            available alternatives.
        """
        current_plan = replan_request.current_plan
        reason = replan_request.reason
        completed_steps = replan_request.completed_steps
        failed_steps = replan_request.failed_steps

        logger.info(f"Replanning for plan {current_plan.plan_id}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Completed steps: {completed_steps}")
        logger.info(f"Failed steps: {failed_steps}")

        # Update plan status
        current_plan.status = PlanStatus.REPLANNING
        current_plan.updated_at = time.time()

        # Mark completed steps
        for step in current_plan.steps:
            if step.step_number in completed_steps:
                step.status = PlanStatus.COMPLETED
                logger.debug(f"Step {step.step_number} marked as completed")

        # Mark failed steps
        for step in current_plan.steps:
            if step.step_number in failed_steps:
                step.status = PlanStatus.FAILED
                logger.debug(f"Step {step.step_number} marked as failed")

        # Update metadata with replan information
        current_plan.metadata["replan_history"] = current_plan.metadata.get(
            "replan_history", []
        ) + [
            {
                "timestamp": time.time(),
                "reason": reason,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
            }
        ]

        logger.info(f"Created revised plan for {current_plan.plan_id}")
        logger.debug(f"Revised plan: {current_plan.model_dump_json(indent=2)}")

        # TODO: Future enhancement - Use LLM to generate alternative steps
        # for failed operations or optimize remaining steps based on new info

        return current_plan

    tools = [create_execution_plan, replan]
    logger.info(f"Created {len(tools)} planning tools")

    return tools
