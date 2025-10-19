"""Planning role - Enhanced LLM-driven TaskGraph generation following Document 34.

This role generates executable TaskGraphs using available system roles with:
- LLM-driven task breakdown using STRONG model
- Role-aware planning with actual system role discovery
- Structured JSON output with BNF grammar validation
- Pre-processing for role loading and post-processing for validation

Architecture: Single Event Loop + Intent-Based + Lifecycle Functions
Created: 2025-10-19 (Document 34 Implementation)
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (Enhanced with lifecycle and BNF grammar prompt)
ROLE_CONFIG = {
    "name": "planning",
    "version": "3.0.0",
    "description": "Generate executable TaskGraphs using available system roles",
    "llm_type": "STRONG",
    "fast_reply": False,
    "when_to_use": "Create multi-step workflows, break down complex tasks, coordinate multiple roles",
    "tools": {
        "automatic": False,  # No tools needed
        "shared": [],
        "include_builtin": False,
    },
    "prompts": {
        "system": """You are a task planning specialist that creates executable workflows using available system roles.

AVAILABLE ROLES:
{{available_roles}}

Your task is to analyze the user's request and create a TaskGraph that breaks it down into executable tasks using the available roles.

OUTPUT REQUIREMENTS:
- Generate valid JSON following the TaskGraph BNF grammar
- Use only the roles listed above
- Include proper task dependencies
- Provide clear task descriptions

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

EXAMPLE OUTPUT:
{
  "tasks": [
    {
      "id": "task_1",
      "name": "Get Weather",
      "description": "Check current weather conditions",
      "role": "weather",
      "parameters": {"location": "current"}
    },
    {
      "id": "task_2",
      "name": "Set Reminder",
      "description": "Set reminder based on weather",
      "role": "timer",
      "parameters": {"duration": "1h", "label": "Check weather again"}
    }
  ],
  "dependencies": [
    {
      "source_task_id": "task_1",
      "target_task_id": "task_2",
      "type": "sequential"
    }
  ]
}

Generate a TaskGraph for the user's request using only the available roles."""
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

        # Filter out planning and router roles
        filtered_roles = {
            name: role_def
            for name, role_def in all_roles.items()
            if name not in ["planning", "router"]
        }

        # Format role information for prompt injection
        role_info = []
        for name, role_def in filtered_roles.items():
            config = role_def.config
            role_info.append(
                {
                    "name": name,
                    "description": config.get("description", ""),
                    "when_to_use": config.get("when_to_use", ""),
                    "parameters": config.get("parameters", {}),
                }
            )

        return {"available_roles": _format_roles_for_prompt(role_info)}

    except Exception as e:
        logger.error(f"Failed to load available roles: {e}")
        return {
            "available_roles": "Error loading roles. Using basic roles: weather, timer, conversation"
        }


def _format_roles_for_prompt(role_info: list) -> str:
    """Format role information for prompt injection."""
    formatted = []
    for role in role_info:
        role_text = f"**{role['name']}**: {role['description']}\n"
        if role["when_to_use"]:
            role_text += f"  When to use: {role['when_to_use']}\n"
        if role["parameters"]:
            role_text += f"  Parameters: {role['parameters']}\n"
        formatted.append(role_text)

    return "\n".join(formatted)


# 3. POST-PROCESSING: TASKGRAPH VALIDATION
def validate_task_graph(llm_result: str, context, pre_data: dict) -> str:
    """Validate LLM output is valid JSON and TaskGraph structure."""
    try:
        # Parse JSON
        try:
            task_graph = json.loads(llm_result)
        except json.JSONDecodeError as e:
            return f"Invalid JSON generated. Please try again. Error: {e}"

        # Validate TaskGraph structure
        validation_errors = _validate_task_graph_structure(task_graph)
        if validation_errors:
            return f"Invalid TaskGraph structure: {'; '.join(validation_errors)}"

        # Validate role references
        available_roles = _extract_available_role_names(
            pre_data.get("available_roles", "")
        )
        role_errors = _validate_role_references(task_graph, available_roles)
        if role_errors:
            return f"Invalid role references: {'; '.join(role_errors)}"

        # Success - return formatted result
        task_count = len(task_graph.get("tasks", []))
        dependency_count = len(task_graph.get("dependencies", []))

        return f"TaskGraph created successfully with {task_count} tasks and {dependency_count} dependencies.\n\n{llm_result}"

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


# 4. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Register the planning role."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers needed
        "tools": [],  # No tools needed
        "intents": {},  # No intents needed
    }
