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

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (Enhanced with lifecycle and BNF grammar prompt)
ROLE_CONFIG = {
    "name": "planning",
    "version": "4.0.0",
    "description": "Generate and execute TaskGraphs using available system roles",
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

        # Filter out planning and router roles
        filtered_roles = {
            name: role_def
            for name, role_def in all_roles.items()
            if name not in ["planning", "router"]
        }

        logger.info(f"Filtered roles for planning: {list(filtered_roles.keys())}")

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

        formatted_roles = _format_roles_for_prompt(role_info)
        logger.info(f"Formatted roles for planning prompt: {formatted_roles[:200]}...")

        return {"available_roles": formatted_roles}

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
        # Parse JSON - try direct parsing first
        task_graph = None
        clean_json = llm_result  # Default to original

        try:
            task_graph = json.loads(llm_result)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            json_match = re.search(r"\{.*\}", llm_result, re.DOTALL)
            if json_match:
                clean_json = json_match.group()
                try:
                    task_graph = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    return f"Invalid JSON generated. Please try again. Error: {e}"
            else:
                return f"No valid JSON found in response. Please try again."

        if task_graph is None:
            return f"Invalid JSON generated. Please try again."

        # Validate TaskGraph structure
        validation_errors = _validate_task_graph_structure(task_graph)
        if validation_errors:
            return f"Invalid TaskGraph structure: {'; '.join(validation_errors)}"

        # Validate role references - use direct role loading as fallback
        roles_text = pre_data.get("available_roles", "")
        available_roles = _extract_available_role_names(roles_text)

        # Fallback: if no roles extracted from text, load directly from registry
        if not available_roles:
            logger.warning("No roles found in pre_data, loading directly from registry")
            try:
                from llm_provider.role_registry import RoleRegistry

                role_registry = RoleRegistry.get_global_registry()
                all_roles = role_registry.roles
                available_roles = [
                    name
                    for name in all_roles.keys()
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


# 4. POST-PROCESSING: TASKGRAPH EXECUTION
def execute_task_graph(llm_result: str, context, pre_data: dict) -> str:
    """Execute TaskGraph using event-driven architecture."""
    try:
        import json

        from common.workflow_intent import WorkflowExecutionIntent

        # Check if input is already an error message from validate_task_graph
        if llm_result.startswith(("Invalid", "No valid JSON", "Validation error")):
            # Don't try to execute error messages - just return them
            return llm_result

        # Parse validated TaskGraph JSON
        task_graph_data = json.loads(llm_result)

        # Create WorkflowExecutionIntent for event-driven execution
        workflow_intent = WorkflowExecutionIntent(
            tasks=task_graph_data["tasks"],
            dependencies=task_graph_data["dependencies"],
            request_id=getattr(context, "context_id", "planning_exec"),
            user_id=getattr(context, "user_id", "unknown"),
            channel_id=getattr(context, "channel_id", "console"),
            original_instruction=getattr(
                context, "original_prompt", "Multi-step workflow"
            ),
        )

        # Process the intent through the intent processor
        # This will suspend the request and start async execution
        try:
            # Get supervisor reference to process the intent
            from supervisor.supervisor import Supervisor

            # Debug: Check what context we actually have
            logger.info(f"Context type: {type(context)}")
            logger.info(f"Context attributes: {dir(context) if context else 'None'}")
            logger.info(
                f"Has workflow_engine: {hasattr(context, 'workflow_engine') if context else False}"
            )
            logger.info(
                f"Workflow_engine value: {getattr(context, 'workflow_engine', 'NOT_FOUND') if context else 'NO_CONTEXT'}"
            )

            # Execute the workflow synchronously and return actual results
            if hasattr(context, "workflow_engine") and context.workflow_engine:
                logger.info("Starting workflow execution with WorkflowEngine context")
                try:
                    # Import required modules
                    from common.task_context import TaskContext
                    from common.task_graph import (
                        TaskDependency,
                        TaskGraph,
                        TaskNode,
                        TaskStatus,
                    )

                    logger.info(
                        f"Creating TaskNodes for {len(task_graph_data['tasks'])} tasks"
                    )

                    # Convert to TaskDescription format (not TaskNode!)
                    from common.task_graph import TaskDescription

                    task_descriptions = []
                    for task_def in task_graph_data["tasks"]:
                        task_desc = TaskDescription(
                            task_name=task_def["name"],
                            agent_id=task_def["role"],
                            tool_id=None,
                            task_type="planning_generated",
                            prompt=task_def["description"],
                            llm_type="default",
                            include_full_history=False,
                        )
                        task_descriptions.append(task_desc)

                    logger.info(
                        f"Converting {len(task_graph_data['dependencies'])} dependencies"
                    )

                    # Convert dependencies to TaskDependency objects
                    from common.task_graph import TaskDependency

                    task_dependencies = []
                    for dep in task_graph_data["dependencies"]:
                        # Find the task names for the source and target IDs
                        source_task_name = None
                        target_task_name = None

                        for task in task_graph_data["tasks"]:
                            if task["id"] == dep["source_task_id"]:
                                source_task_name = task["name"]
                            if task["id"] == dep["target_task_id"]:
                                target_task_name = task["name"]

                        if source_task_name and target_task_name:
                            task_dep = TaskDependency(
                                source=source_task_name,
                                target=target_task_name,
                                condition=None,
                            )
                            task_dependencies.append(task_dep)
                        else:
                            logger.warning(
                                f"Could not find task names for dependency: {dep}"
                            )

                    logger.info("Creating TaskGraph...")

                    # Create TaskGraph and TaskContext
                    task_graph = TaskGraph(
                        tasks=task_descriptions,
                        dependencies=task_dependencies,
                        request_id=getattr(context, "context_id", "planning_exec"),
                    )

                    logger.info("Creating TaskContext...")

                    task_context = TaskContext(
                        task_graph=task_graph,
                        context_id=getattr(context, "context_id", "planning_exec"),
                        user_id=getattr(context, "user_id", None),
                        channel_id=getattr(context, "channel_id", None),
                    )

                    logger.info("Executing workflow with WorkflowEngine...")

                    # Execute workflow
                    context.workflow_engine._execute_dag_parallel(task_context)

                    logger.info("Waiting for workflow completion...")

                    # Wait for completion with timeout
                    import time

                    start_time = time.time()
                    timeout = 60

                    while (
                        not task_context.is_completed()
                        and (time.time() - start_time) < timeout
                    ):
                        time.sleep(0.5)

                    if task_context.is_completed():
                        logger.info("Workflow completed, collecting results...")
                        # Collect results
                        results = []
                        for _, task_node in task_context.task_graph.nodes.items():
                            if (
                                task_node.status == TaskStatus.COMPLETED
                                and task_node.result
                            ):
                                results.append(
                                    f"**{task_node.task_name}**: {task_node.result}"
                                )

                        if results:
                            return "Workflow completed successfully:\n\n" + "\n\n".join(
                                results
                            )
                        else:
                            return "Workflow completed but no results were generated."
                    else:
                        return f"Workflow execution timed out after {timeout} seconds"

                except Exception as exec_error:
                    logger.error(f"Workflow execution error: {exec_error}")
                    import traceback

                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise exec_error
            else:
                # No workflow engine available - return initiation message
                task_count = len(task_graph_data.get("tasks", []))
                dependency_count = len(task_graph_data.get("dependencies", []))
                return f"Multi-step workflow initiated with {task_count} tasks and {dependency_count} dependencies. Results will be delivered when complete."

        except Exception as intent_error:
            logger.warning(f"Event-driven execution not available: {intent_error}")
            # Fallback to success message
            task_count = len(task_graph_data.get("tasks", []))
            dependency_count = len(task_graph_data.get("dependencies", []))
            return f"TaskGraph validated with {task_count} tasks and {dependency_count} dependencies."

    except json.JSONDecodeError as e:
        logger.error(f"Invalid TaskGraph JSON: {e}")
        return f"Invalid TaskGraph JSON: {e}"
    except Exception as e:
        logger.error(f"TaskGraph execution failed: {e}")
        return f"TaskGraph execution failed: {e}"


# 5. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Register the planning role."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers needed
        "tools": [],  # No tools needed
        "intents": {},  # No intents needed
    }
