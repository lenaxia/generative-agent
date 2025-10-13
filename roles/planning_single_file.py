"""Planning role - LLM-friendly single file implementation.

This role consolidates all planning functionality into a single file following
the new LLM-safe architecture patterns from Documents 25, 26, and 27.

Migrated from: roles/planning/ (definition.yaml + tools.py)
Total reduction: ~174 lines → ~280 lines (expanded for LLM-safe patterns)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "planning",
    "version": "2.0.0",
    "description": "Specialized role for task planning and decomposition using LLM-safe architecture",
    "llm_type": "STRONG",  # Planning requires complex reasoning and analysis
    "fast_reply": False,  # Planning is not a fast-reply role
    "when_to_use": "Break down complex tasks, create detailed plans with dependencies, identify resources and constraints, coordinate multi-step workflows",
}


# 2. ROLE-SPECIFIC INTENTS (owned by planning role)
@dataclass
class PlanningIntent(Intent):
    """Planning-specific intent - owned by planning role."""

    action: str  # "create_plan", "analyze_dependencies", "validate_plan"
    instruction: Optional[str] = None
    plan_data: Optional[dict[str, Any]] = None
    complexity: Optional[str] = None  # "simple", "medium", "complex"

    def validate(self) -> bool:
        """Validate planning intent parameters."""
        valid_actions = ["create_plan", "analyze_dependencies", "validate_plan"]
        return bool(self.action and self.action in valid_actions)


@dataclass
class TaskPlanIntent(Intent):
    """Task plan processing intent - owned by planning role."""

    plan: dict[str, Any]
    operation: str  # "create", "validate", "execute"
    metadata: dict[str, Any]

    def validate(self) -> bool:
        """Validate task plan intent parameters."""
        valid_operations = ["create", "validate", "execute"]
        return bool(
            self.plan
            and isinstance(self.plan, dict)
            and self.operation in valid_operations
            and isinstance(self.metadata, dict)
        )


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_planning_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for planning request events."""
    try:
        # Parse event data
        instruction, complexity = _parse_planning_event_data(event_data)

        # Create intents
        return [
            PlanningIntent(
                action="create_plan", instruction=instruction, complexity=complexity
            ),
            AuditIntent(
                action="planning_request",
                details={
                    "instruction": instruction,
                    "complexity": complexity,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Planning handler error: {e}")
        return [
            NotificationIntent(
                message=f"Planning processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


def handle_plan_validation(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for plan validation events."""
    try:
        # Parse plan data from event
        plan_data = _parse_plan_validation_event(event_data)

        return [
            TaskPlanIntent(
                plan=plan_data,
                operation="validate",
                metadata={"timestamp": time.time(), "source": "validation_event"},
            ),
            AuditIntent(
                action="plan_validation",
                details={"plan_tasks": len(plan_data.get("tasks", []))},
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Plan validation error: {e}")
        return [
            NotificationIntent(
                message=f"Plan validation error: {e}",
                channel=context.get_safe_channel(),
                priority="medium",
                notification_type="warning",
            )
        ]


# 4. TOOLS (migrated from tools.py with @tool decorators)
@tool
def create_task_plan(instruction: str) -> dict[str, Any]:
    """Create a task plan by breaking down the instruction into manageable steps."""
    logger.info(f"Creating task plan for: {instruction[:50]}...")

    try:
        # Enhanced planning logic
        plan = {
            "instruction": instruction,
            "tasks": [
                {
                    "task_name": "analyze_request",
                    "agent_id": "analysis_agent",
                    "task_type": "analysis",
                    "prompt": f"Analyze the following request: {instruction}",
                    "priority": 1,
                    "estimated_duration": "5 minutes",
                },
                {
                    "task_name": "execute_main_task",
                    "agent_id": "execution_agent",
                    "task_type": "execution",
                    "prompt": f"Execute the main task: {instruction}",
                    "priority": 2,
                    "estimated_duration": "15 minutes",
                },
                {
                    "task_name": "validate_results",
                    "agent_id": "validation_agent",
                    "task_type": "validation",
                    "prompt": f"Validate the results of: {instruction}",
                    "priority": 3,
                    "estimated_duration": "5 minutes",
                },
            ],
            "dependencies": [
                {
                    "source": "analyze_request",
                    "target": "execute_main_task",
                    "type": "sequential",
                },
                {
                    "source": "execute_main_task",
                    "target": "validate_results",
                    "type": "sequential",
                },
            ],
            "metadata": {
                "created_at": time.time(),
                "complexity": _assess_complexity(instruction),
                "total_estimated_duration": "25 minutes",
            },
        }

        logger.info(f"Created task plan with {len(plan['tasks'])} tasks")
        return plan

    except Exception as e:
        logger.error(f"Failed to create task plan: {e}")
        return {"error": str(e), "tasks": [], "dependencies": []}


@tool
def analyze_task_dependencies(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze dependencies between tasks."""
    logger.info(f"Analyzing dependencies for {len(tasks)} tasks")

    try:
        dependencies = []

        # Enhanced dependency analysis
        for i in range(1, len(tasks)):
            # Sequential dependency
            dependencies.append(
                {
                    "source": tasks[i - 1]["task_name"],
                    "target": tasks[i]["task_name"],
                    "dependency_type": "sequential",
                    "condition": "completion",
                    "estimated_delay": "0 minutes",
                }
            )

            # Check for parallel opportunities
            if _can_run_parallel(tasks[i - 1], tasks[i]):
                dependencies[-1]["dependency_type"] = "optional"
                dependencies[-1]["condition"] = "can_start_parallel"

        logger.info(f"Analyzed {len(dependencies)} dependencies")
        return dependencies

    except Exception as e:
        logger.error(f"Failed to analyze dependencies: {e}")
        return []


@tool
def validate_task_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a task plan for completeness and correctness."""
    logger.info("Validating task plan")

    try:
        validation_result = {"valid": True, "errors": [], "warnings": [], "score": 1.0}

        # Check if plan has required fields
        required_plan_fields = ["tasks", "dependencies", "instruction"]
        for field in required_plan_fields:
            if field not in plan:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Plan missing required field: {field}"
                )
                validation_result["score"] -= 0.3

        # Validate each task
        tasks = plan.get("tasks", [])
        if not tasks:
            validation_result["valid"] = False
            validation_result["errors"].append("Plan has no tasks")
            validation_result["score"] = 0.0
        else:
            for i, task in enumerate(tasks):
                required_task_fields = ["task_name", "agent_id", "task_type", "prompt"]
                for field in required_task_fields:
                    if field not in task:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Task {i} missing required field: {field}"
                        )
                        validation_result["score"] -= 0.1

        # Validate dependencies
        dependencies = plan.get("dependencies", [])
        task_names = {task["task_name"] for task in tasks}
        for dep in dependencies:
            if dep.get("source") not in task_names:
                validation_result["warnings"].append(
                    f"Dependency source '{dep.get('source')}' not found in tasks"
                )
            if dep.get("target") not in task_names:
                validation_result["warnings"].append(
                    f"Dependency target '{dep.get('target')}' not found in tasks"
                )

        # Ensure score doesn't go below 0
        validation_result["score"] = max(0.0, validation_result["score"])

        logger.info(
            f"Plan validation: {'valid' if validation_result['valid'] else 'invalid'} (score: {validation_result['score']:.2f})"
        )
        return validation_result

    except Exception as e:
        logger.error(f"Failed to validate task plan: {e}")
        return {"valid": False, "errors": [str(e)], "warnings": [], "score": 0.0}


# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_planning_event_data(event_data: Any) -> tuple[str, str]:
    """LLM-SAFE: Parse planning event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("instruction", "unknown instruction"),
                event_data.get("complexity", "medium"),
            )
        elif isinstance(event_data, str):
            return event_data, _assess_complexity(event_data)
        else:
            return str(event_data), "medium"
    except Exception as e:
        return f"parse_error: {e}", "error"


def _parse_plan_validation_event(event_data: Any) -> dict[str, Any]:
    """LLM-SAFE: Parse plan validation event data."""
    try:
        if isinstance(event_data, dict) and "plan" in event_data:
            return event_data["plan"]
        elif isinstance(event_data, dict):
            return event_data
        else:
            return {"tasks": [], "dependencies": [], "instruction": str(event_data)}
    except Exception as e:
        return {"error": str(e), "tasks": [], "dependencies": []}


def _assess_complexity(instruction: str) -> str:
    """Assess the complexity of an instruction."""
    try:
        instruction_lower = instruction.lower()
        word_count = len(instruction.split())

        # Simple heuristics for complexity assessment
        complex_keywords = [
            "analyze",
            "plan",
            "strategy",
            "multiple",
            "complex",
            "integrate",
            "coordinate",
        ]
        simple_keywords = ["get", "show", "tell", "what", "when", "where"]

        complex_count = sum(1 for word in complex_keywords if word in instruction_lower)
        simple_count = sum(1 for word in simple_keywords if word in instruction_lower)

        if complex_count > 2 or word_count > 20:
            return "complex"
        elif simple_count > 1 and word_count < 10:
            return "simple"
        else:
            return "medium"

    except Exception:
        return "medium"


def _can_run_parallel(task1: dict[str, Any], task2: dict[str, Any]) -> bool:
    """Determine if two tasks can run in parallel."""
    try:
        # Simple heuristic: different task types can potentially run in parallel
        return task1.get("task_type") != task2.get("task_type")
    except Exception:
        return False


# 6. INTENT HANDLER REGISTRATION
async def process_planning_intent(intent: PlanningIntent):
    """Process planning-specific intents - called by IntentProcessor."""
    logger.info(f"Processing planning intent: {intent.action}")

    # In full implementation, this would:
    # - Create detailed task plans
    # - Analyze task dependencies
    # - Validate plan completeness
    # For now, just log the intent processing


async def process_task_plan_intent(intent: TaskPlanIntent):
    """Process task plan intents - called by IntentProcessor."""
    logger.info(f"Processing task plan intent: {intent.operation}")

    # In full implementation, this would:
    # - Execute plan operations
    # - Validate plan structures
    # - Coordinate task execution
    # For now, just log the intent processing


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "PLANNING_REQUEST": handle_planning_request,
            "PLAN_VALIDATION": handle_plan_validation,
        },
        "tools": [create_task_plan, analyze_task_dependencies, validate_task_plan],
        "intents": {
            PlanningIntent: process_planning_intent,
            TaskPlanIntent: process_task_plan_intent,
        },
    }


# 8. CONSTANTS AND CONFIGURATION
PLANNING_COMPLEXITY_LEVELS = ["simple", "medium", "complex"]
MAX_PLAN_TASKS = 20
MAX_PLAN_DEPENDENCIES = 50
PLANNING_TIMEOUT = 300  # 5 minutes

# Planning action mappings for LLM understanding
PLANNING_ACTIONS = {
    "plan": "create_plan",
    "create": "create_plan",
    "analyze": "analyze_dependencies",
    "validate": "validate_plan",
    "check": "validate_plan",
}


def normalize_planning_action(action: str) -> str:
    """Normalize planning action to standard form."""
    return PLANNING_ACTIONS.get(action.lower(), action.lower())


# 9. ENHANCED ERROR HANDLING
def create_planning_error_intent(
    error: Exception, context: LLMSafeEventContext
) -> list[Intent]:
    """Create error intents for planning operations."""
    return [
        NotificationIntent(
            message=f"Planning error: {error}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="high",
            notification_type="error",
        ),
        AuditIntent(
            action="planning_error",
            details={"error": str(error), "context": context.to_dict()},
            user_id=context.user_id,
            severity="error",
        ),
    ]


# 10. PLANNING UTILITIES
def estimate_plan_duration(tasks: list[dict[str, Any]]) -> str:
    """Estimate total duration for a plan."""
    try:
        total_minutes = 0
        for task in tasks:
            duration_str = task.get("estimated_duration", "10 minutes")
            # Extract number from duration string
            import re

            match = re.search(r"(\d+)", duration_str)
            if match:
                total_minutes += int(match.group(1))
            else:
                total_minutes += 10  # Default

        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if minutes == 0:
                return f"{hours} hours"
            else:
                return f"{hours} hours {minutes} minutes"

    except Exception as e:
        logger.error(f"Error estimating duration: {e}")
        return "unknown duration"


def validate_plan_structure(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate plan structure comprehensively."""
    try:
        issues = {"errors": [], "warnings": [], "suggestions": []}

        # Check basic structure
        if not isinstance(plan.get("tasks"), list):
            issues["errors"].append("Plan must have 'tasks' as a list")

        if not isinstance(plan.get("dependencies"), list):
            issues["warnings"].append("Plan should have 'dependencies' as a list")

        # Check task structure
        tasks = plan.get("tasks", [])
        if len(tasks) == 0:
            issues["errors"].append("Plan must have at least one task")
        elif len(tasks) > MAX_PLAN_TASKS:
            issues["warnings"].append(
                f"Plan has {len(tasks)} tasks, consider breaking into smaller plans"
            )

        # Check dependencies
        dependencies = plan.get("dependencies", [])
        if len(dependencies) > MAX_PLAN_DEPENDENCIES:
            issues["warnings"].append(
                f"Plan has {len(dependencies)} dependencies, may be overly complex"
            )

        return {
            "valid": len(issues["errors"]) == 0,
            "issues": issues,
            "task_count": len(tasks),
            "dependency_count": len(dependencies),
        }

    except Exception as e:
        return {
            "valid": False,
            "issues": {"errors": [str(e)], "warnings": [], "suggestions": []},
            "task_count": 0,
            "dependency_count": 0,
        }
