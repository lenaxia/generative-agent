"""Planning role - Enhanced LLM-driven TaskGraph generation following Document 34.

This role generates executable TaskGraphs using available system roles with:
- LLM-driven task breakdown using STRONG model
- Role-aware planning with actual system role discovery
- Structured JSON output with BNF grammar validation
- Pre-processing for role loading and post-processing for validation

Architecture: Single Event Loop + Intent-Based + Lifecycle Functions
Created: 2025-10-19 (Document 34 Implementation)
Updated: 2025-10-28 (Document 35 Phase 1 - Intent-Based Processing)
"""

import json
import logging
import re
from typing import Union

from common.intents import WorkflowIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (Enhanced with lifecycle and BNF grammar prompt)
ROLE_CONFIG = {
    "name": "planning",
    "version": "5.0.0",  # Updated for Document 35 intent-based processing
    "description": "Generate WorkflowExecutionIntents using available system roles",
    "llm_type": "STRONG",
    "fast_reply": False,
    "when_to_use": "Create multi-step workflows, break down complex tasks, coordinate multiple roles",
    "tools": {
        "automatic": False,  # No tools needed
        "shared": [],
        "include_builtin": False,
    },
    "lifecycle": {
        "pre_processing": {"enabled": True, "functions": ["load_available_roles"]},
        "post_processing": {
            "enabled": True,
            "functions": ["validate_task_graph", "execute_task_graph"],
        },
    },
    "prompts": {
        "system": """CRITICAL: RESPOND WITH ONLY VALID JSON - NO EXPLANATIONS, NO ADDITIONAL TEXT

AVAILABLE ROLES:
{{available_roles}}

TASK: Create TaskGraph JSON that breaks down the user's request into executable tasks.

STRICT OUTPUT REQUIREMENTS:
- ONLY valid JSON following the TaskGraph BNF grammar
- NO explanatory text before or after JSON
- NO markdown formatting or code blocks
- Use only the roles listed above

BNF GRAMMAR:
<TaskGraph> ::= {
  "tasks": [<Task>+],
  "dependencies": [<Dependency>*]
}

<Task> ::= {
  "id": <string>,
  "name": <string>,
  "description": <string>,
  "role": <role_name>,
  "parameters": <object>
}

<Dependency> ::= {
  "source_task_id": <string>,
  "target_task_id": <string>,
  "type": "sequential" | "parallel"
}

EXAMPLE (RESPOND EXACTLY LIKE THIS - JSON ONLY):
{
  "tasks": [
    {
      "id": "task_1",
      "name": "Get Weather",
      "description": "Check current weather conditions",
      "role": "weather",
      "parameters": {"location": "current"}
    }
  ],
  "dependencies": []
}

RESPOND WITH ONLY JSON - NO OTHER TEXT."""
    },
}


# 2. PRE-PROCESSING: ROLE DISCOVERY
def load_available_roles(instruction: str, context, parameters: dict) -> dict:
    """Load available roles and their metadata for planning."""
    try:
        from llm_provider.role_registry import RoleRegistry

        # Get all available roles
        role_registry = RoleRegistry.get_global_registry()
        all_roles = role_registry.roles

        logger.info(f"Loading roles for planning - found {len(all_roles)} total roles")
        logger.info(f"All role names: {list(all_roles.keys())}")

        # Filter out planning and router roles (can't plan planning or route routing)
        available_roles = [
            name for name in all_roles.keys() if name not in ["planning", "router"]
        ]

        logger.info(f"Filtered roles for planning: {available_roles}")

        # Format roles for prompt injection
        formatted_roles = []
        for role_name in available_roles:
            role_def = all_roles[role_name]
            role_config = getattr(role_def, "config", {})
            description = role_config.get("description", "No description available")
            when_to_use = role_config.get("when_to_use", "")

            formatted_roles.append(f"**{role_name}**: {description}")
            if when_to_use:
                formatted_roles.append(f"  - When to use: {when_to_use}")

        roles_text = "\n".join(formatted_roles)
        logger.info(f"Formatted roles for planning prompt: {roles_text[:200]}...")

        return {"available_roles": roles_text}

    except Exception as e:
        logger.error(f"Failed to load available roles: {e}")
        # Fallback to basic roles
        return {
            "available_roles": "**weather**: Weather information\n**timer**: Timer management\n**search**: Web search\n**conversation**: General conversation"
        }


# 3. POST-PROCESSING: TASKGRAPH VALIDATION
def validate_task_graph(llm_result: str, context, pre_data: dict) -> str:
    """Validate TaskGraph JSON structure and role references."""
    try:
        import json

        # Extract JSON from mixed content if needed
        clean_json = llm_result
        if not llm_result.strip().startswith("{"):
            # Try to extract JSON using regex
            json_match = re.search(r"\{.*\}", llm_result, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                logger.info("Extracted JSON from mixed content")
            else:
                return "No valid JSON found in LLM response"

        # Parse and validate JSON structure
        try:
            task_graph = json.loads(clean_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in TaskGraph: {e}")
            return f"Invalid JSON in TaskGraph: {e}"

        # Validate TaskGraph structure
        structure_errors = _validate_task_graph_structure(task_graph)
        if structure_errors:
            logger.error(f"TaskGraph structure validation failed: {structure_errors}")
            return f"Invalid TaskGraph structure: {'; '.join(structure_errors)}"

        # Validate role references
        available_roles = []
        if "available_roles" in pre_data:
            available_roles = _extract_available_role_names(pre_data["available_roles"])
        else:
            # Load roles directly if not in pre_data
            logger.warning("No roles found in pre_data, loading directly from registry")
            try:
                from llm_provider.role_registry import RoleRegistry

                role_registry = RoleRegistry.get_global_registry()
                available_roles = [
                    name
                    for name in role_registry.roles.keys()
                    if name not in ["planning", "router"]
                ]
                logger.info(f"Loaded roles directly from registry: {available_roles}")
            except Exception as e:
                logger.error(f"Failed to load roles from registry: {e}")
                available_roles = [
                    "weather",
                    "timer",
                    "conversation",
                    "search",
                ]  # Basic fallback

        logger.info(f"Final available roles for validation: {available_roles}")
        logger.info(
            f"Task roles to validate: {[task.get('role') for task in task_graph.get('tasks', [])]}"
        )

        role_errors = _validate_role_references(task_graph, available_roles)
        if role_errors:
            logger.error(f"Role validation failed: {role_errors}")
            return f"Invalid role references: {'; '.join(role_errors)}"

        # Success - return the clean JSON for execute_task_graph to process
        return clean_json

    except Exception as e:
        logger.error(f"TaskGraph validation failed: {e}")
        return f"Validation error occurred. Please try again. Error: {e}"


def _validate_task_graph_structure(task_graph: dict) -> list[str]:
    """Validate TaskGraph has required structure."""
    errors = []

    # Check required top-level keys
    if "tasks" not in task_graph:
        errors.append("Missing 'tasks' array")
    if "dependencies" not in task_graph:
        errors.append("Missing 'dependencies' array")

    # Validate tasks
    tasks = task_graph.get("tasks", [])
    if not isinstance(tasks, list):
        errors.append("'tasks' must be an array")
    elif len(tasks) == 0:
        errors.append("At least one task is required")
    else:
        for i, task in enumerate(tasks):
            task_errors = _validate_task_structure(task, i)
            errors.extend(task_errors)

    # Validate dependencies
    dependencies = task_graph.get("dependencies", [])
    if not isinstance(dependencies, list):
        errors.append("'dependencies' must be an array")
    else:
        # Only get task_ids if tasks is actually a list
        if isinstance(tasks, list):
            task_ids = {task.get("id") for task in tasks}
            for i, dep in enumerate(dependencies):
                dep_errors = _validate_dependency_structure(dep, i, task_ids)
                errors.extend(dep_errors)

    return errors


def _validate_task_structure(task: dict, index: int) -> list[str]:
    """Validate individual task structure."""
    errors = []
    required_fields = ["id", "name", "description", "role"]

    for field in required_fields:
        if field not in task:
            errors.append(f"Task {index}: missing '{field}'")
        elif not isinstance(task[field], str) or not task[field].strip():
            errors.append(f"Task {index}: '{field}' must be non-empty string")

    return errors


def _validate_dependency_structure(dep: dict, index: int, task_ids: set) -> list[str]:
    """Validate individual dependency structure."""
    errors = []

    if "source_task_id" not in dep:
        errors.append(f"Dependency {index}: missing 'source_task_id'")
    elif dep["source_task_id"] not in task_ids:
        errors.append(f"Dependency {index}: invalid 'source_task_id'")

    if "target_task_id" not in dep:
        errors.append(f"Dependency {index}: missing 'target_task_id'")
    elif dep["target_task_id"] not in task_ids:
        errors.append(f"Dependency {index}: invalid 'target_task_id'")

    if "type" not in dep:
        errors.append(f"Dependency {index}: missing 'type'")
    elif dep["type"] not in ["sequential", "parallel"]:
        errors.append(f"Dependency {index}: 'type' must be 'sequential' or 'parallel'")

    return errors


def _validate_role_references(
    task_graph: dict, available_roles: list[str]
) -> list[str]:
    """Validate all task roles are available."""
    errors = []

    for task in task_graph.get("tasks", []):
        role = task.get("role")
        if role and role not in available_roles:
            errors.append(f"Task '{task.get('id')}' uses unavailable role '{role}'")

    return errors


def _extract_available_role_names(roles_text: str) -> list[str]:
    """Extract role names from formatted roles text."""
    matches = re.findall(r"\*\*([^*]+)\*\*:", roles_text)
    return matches


# 4. POST-PROCESSING: TASKGRAPH EXECUTION (Document 35 - Intent-Based)
def execute_task_graph(llm_result: str, context, pre_data: dict) -> WorkflowIntent:
    """Document 35: Create WorkflowIntent with task graph (LLM-SAFE, pure function).

    This function creates WorkflowIntent with explicit task graph following Documents 25 & 26
    LLM-safe architecture. All legacy execution logic has been removed in favor of
    intent-based processing.

    Refactored to use consolidated WorkflowIntent instead of separate WorkflowExecutionIntent.
    """
    try:
        import json

        # Check if input is already an error message from validate_task_graph
        if llm_result.startswith(("Invalid", "No valid JSON", "Validation error")):
            # For error cases, raise an exception since we can't return a string
            raise ValueError(
                f"Cannot create WorkflowIntent from error message: {llm_result}"
            )

        # Parse validated TaskGraph JSON (extract from mixed content if needed)
        try:
            task_graph_data = json.loads(llm_result)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content using regex (like validate_task_graph does)
            json_match = re.search(r"\{.*\}", llm_result, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                task_graph_data = json.loads(clean_json)
            else:
                raise

        # LLM-SAFE: Create WorkflowIntent with task graph (declarative - what should happen)
        workflow_intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=task_graph_data["tasks"],
            dependencies=task_graph_data["dependencies"],
            request_id=getattr(context, "context_id", "planning_exec"),
            user_id=getattr(context, "user_id", "unknown"),
            channel_id=getattr(context, "channel_id", "console"),
            original_instruction=getattr(
                context, "original_prompt", "Multi-step workflow"
            ),
        )

        logger.info(
            f"Created WorkflowIntent with task graph for request {workflow_intent.request_id} with {len(workflow_intent.tasks)} tasks"
        )

        # LLM-SAFE: Return intent only, no execution (pure function)
        return workflow_intent

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse TaskGraph JSON: {e}")
        raise ValueError(f"Invalid JSON in TaskGraph: {e}")
    except Exception as e:
        logger.error(f"TaskGraph intent creation failed: {e}")
        raise ValueError(f"TaskGraph intent creation error: {e}")


# 5. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Register the planning role."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers needed
        "tools": [],  # No tools needed
        "intents": {},  # No intents needed
    }
