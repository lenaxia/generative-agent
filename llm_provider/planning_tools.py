"""
Planning tools for StrandsAgent with Dynamic Role Support.

These tools create task graphs using dynamically loaded roles instead of hardcoded agents.
The planning LLM selects appropriate roles based on available role definitions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator
from common.task_graph import TaskGraph, TaskDescription, TaskDependency
from llm_provider.role_registry import RoleRegistry
from llm_provider.factory import LLMFactory, LLMType
import json
import logging

logger = logging.getLogger(__name__)


class PlanningOutput(BaseModel):
    """Output schema for planning tools."""
    tasks: List[TaskDescription] = []
    dependencies: Optional[List[TaskDependency]] = []

    @field_validator('tasks')
    def check_tasks(cls, tasks):
        for task in tasks:
            if not task.task_name or not task.agent_id or not task.task_type or not task.prompt:
                raise ValueError("All tasks must have agent_id, task_type, and prompt.")
        return tasks

    @field_validator('dependencies')
    def check_dependencies(cls, dependencies):
        if dependencies is None or len(dependencies) == 0:
            return dependencies  # Return as is if dependencies is None or empty

        for dependency in dependencies:
            if not dependency.source or not dependency.target:
                raise ValueError("All dependencies must have source and target set.")
        return dependencies


def create_task_plan(instruction: str, llm_factory: LLMFactory = None, request_id: str = "default") -> Dict[str, Any]:
    """
    Create a task plan using dynamic role selection via LLM.
    
    This tool uses an LLM to intelligently break down tasks and select
    appropriate roles from the dynamic role registry.
    
    Args:
        instruction: The user instruction to create a plan for
        llm_factory: LLM factory for role selection (injected by system)
        request_id: Request ID for tracking
        
    Returns:
        Dict containing the task graph with tasks and dependencies
    """
    logger.info(f"Creating dynamic task plan for instruction: {instruction}")
    
    # Get available roles from registry
    role_registry = RoleRegistry.get_global_registry()
    available_roles = role_registry.get_role_summaries()
    
    if not available_roles:
        logger.warning("No roles available, creating generic task")
        # Fallback to generic task if no roles available
        tasks = [
            TaskDescription(
                task_id="task_1",
                task_name=f"Execute: {instruction[:50]}...",
                agent_id="default",
                task_type="execution",
                prompt=instruction,
                status="pending"
            )
        ]
        dependencies = []
    else:
        # Use LLM to create intelligent task plan with role selection
        tasks, dependencies = _create_llm_task_plan(instruction, available_roles, llm_factory)
    
    # Create task graph
    task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id=request_id)
    
    logger.info(f"Created dynamic task plan with {len(tasks)} tasks and {len(dependencies)} dependencies")
    
    return {
        "task_graph": task_graph,
        "tasks": tasks,
        "dependencies": dependencies,
        "request_id": request_id
    }


def _create_llm_task_plan(instruction: str, available_roles: List[Dict], llm_factory: LLMFactory) -> tuple:
    """Use LLM to create intelligent task plan with role selection."""
    
    # Format roles for LLM prompt
    roles_text = "\n".join([
        f"- {role['name']}: {role['description']}\n  When to use: {role['when_to_use']}"
        for role in available_roles
    ])
    
    # Create planning prompt with EBNF grammar for strict JSON output
    planning_prompt = f"""
You are a planning agent responsible for breaking down complex tasks into subtasks and selecting appropriate roles.

Task: {instruction}

Available roles:
{roles_text}

Create a task plan by:
1. Breaking down the task into logical subtasks (if needed)
2. Selecting the most appropriate role for each subtask from the available roles
3. If NO appropriate role exists for a subtask, use "None" as the agent_id - this will trigger dynamic role generation
4. Assign appropriate LLM type based on task complexity: "WEAK" for simple tasks, "DEFAULT" for moderate tasks, "STRONG" for complex reasoning tasks
5. Creating dependencies between tasks if needed

For simple tasks, you may create just one task. For complex tasks, break them down.

IMPORTANT:
- Only use roles from the available roles list above
- If no suitable role exists for a task, use agent_id: "None"
- Always include llm_type field for each task

OUTPUT MUST CONFORM TO THIS EBNF GRAMMAR FOR JSON:

json        = element ;
element     = ws , value , ws ;
value       = object | array | string | number | "true" | "false" | "null" ;
object      = "{{"  , ws , "}}" | "{{"  , members , "}}" ;
members     = member , {{ "," , member }} ;
member      = ws , string , ws , ":" , element ;
array       = "[" , ws , "]" | "[" , elements , "]" ;
elements    = element , {{ "," , element }} ;
string      = "\\"" , characters , "\\"" ;
ws          = {{ " " | "\\n" | "\\r" | "\\t" }} ;

Your response MUST be valid JSON conforming to this grammar. No markdown, no code blocks, no additional text.

Required JSON structure:
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "task_name": "Descriptive name for the task",
            "agent_id": "selected_role_name_or_None",
            "task_type": "execution",
            "prompt": "Specific instruction for this subtask",
            "llm_type": "WEAK|DEFAULT|STRONG",
            "status": "pending"
        }}
    ],
    "dependencies": [
        {{"source": "Descriptive name for the task", "target": "Another task name"}}
    ]
}}
"""
    
    try:
        # Use Universal Agent for planning instead of direct StrandsAgent
        if llm_factory:
            # Create Universal Agent for planning
            from llm_provider.universal_agent import UniversalAgent
            from llm_provider.role_registry import RoleRegistry
            
            # Create temporary Universal Agent for planning
            role_registry = RoleRegistry.get_global_registry()
            universal_agent = UniversalAgent(llm_factory, role_registry)
            
            # Use Universal Agent with planning role
            response = universal_agent.execute_task(
                instruction=planning_prompt,
                role="planning",
                llm_type=LLMType.STRONG
            )
            
            # Parse JSON response directly (EBNF grammar should ensure pure JSON)
            plan_data = json.loads(response.strip())
            
            # Create task objects
            tasks = [TaskDescription(**task) for task in plan_data.get("tasks", [])]
            
            # Create dependency objects, filtering out invalid fields
            dependencies = []
            for dep in plan_data.get("dependencies", []):
                # Only include valid TaskDependency fields (source, target, condition)
                valid_dep = {
                    "source": dep.get("source"),
                    "target": dep.get("target")
                }
                if "condition" in dep:
                    valid_dep["condition"] = dep["condition"]
                dependencies.append(TaskDependency(**valid_dep))
            
            return tasks, dependencies
            
        else:
            raise ValueError("No LLM factory available for task planning")
            
    except Exception as e:
        logger.error(f"LLM task planning failed: {e}")
        raise e  # Re-raise the error instead of using fallback


# Removed fallback functions - no more fallbacks, proper error handling instead


def analyze_task_dependencies(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze and create dependencies between tasks.
    
    Args:
        tasks: List of task dictionaries
        
    Returns:
        List of dependency dictionaries
    """
    logger.info(f"Analyzing dependencies for {len(tasks)} tasks")
    
    dependencies = []
    
    # Simple dependency analysis - sequential execution
    for i in range(1, len(tasks)):
        dependency = {
            "source": tasks[i-1]["task_id"],
            "target": tasks[i]["task_id"],
            "dependency_type": "sequential"
        }
        dependencies.append(dependency)
    
    logger.info(f"Created {len(dependencies)} dependencies")
    return dependencies


def validate_task_plan(tasks: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a task plan for correctness.
    
    Args:
        tasks: List of task dictionaries
        dependencies: List of dependency dictionaries
        
    Returns:
        Validation result with status and any errors
    """
    logger.info("Validating task plan")
    
    errors = []
    warnings = []
    
    # Check that all tasks have required fields
    for task in tasks:
        required_fields = ["task_id", "task_name", "agent_id", "task_type", "prompt"]
        for field in required_fields:
            if field not in task or not task[field]:
                errors.append(f"Task {task.get('task_id', 'unknown')} missing required field: {field}")
    
    # Check that all dependencies reference valid tasks
    task_ids = {task["task_id"] for task in tasks}
    for dep in dependencies:
        if dep["source"] not in task_ids:
            errors.append(f"Dependency references unknown source task: {dep['source']}")
        if dep["target"] not in task_ids:
            errors.append(f"Dependency references unknown target task: {dep['target']}")
    
    # Check for circular dependencies (simple check)
    if len(dependencies) > len(tasks):
        warnings.append("More dependencies than tasks - possible circular dependency")
    
    is_valid = len(errors) == 0
    
    result = {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "task_count": len(tasks),
        "dependency_count": len(dependencies)
    }
    
    logger.info(f"Task plan validation: {'PASSED' if is_valid else 'FAILED'} with {len(errors)} errors")
    return result


def optimize_task_plan(tasks: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optimize a task plan for better execution.
    
    Args:
        tasks: List of task dictionaries
        dependencies: List of dependency dictionaries
        
    Returns:
        Optimized task plan
    """
    logger.info("Optimizing task plan")
    
    # Simple optimization - remove redundant dependencies
    optimized_dependencies = []
    seen_pairs = set()
    
    for dep in dependencies:
        pair = (dep["source"], dep["target"])
        if pair not in seen_pairs:
            optimized_dependencies.append(dep)
            seen_pairs.add(pair)
    
    # Could add more optimizations like:
    # - Parallel task identification
    # - Task merging opportunities
    # - Resource optimization
    
    result = {
        "tasks": tasks,
        "dependencies": optimized_dependencies,
        "optimizations_applied": [
            f"Removed {len(dependencies) - len(optimized_dependencies)} redundant dependencies"
        ]
    }
    
    logger.info(f"Task plan optimized: {len(dependencies)} -> {len(optimized_dependencies)} dependencies")
    return result


# Tool registry for planning tools
PLANNING_TOOLS = {
    "create_task_plan": create_task_plan,
    "analyze_task_dependencies": analyze_task_dependencies,
    "validate_task_plan": validate_task_plan,
    "optimize_task_plan": optimize_task_plan
}


def get_planning_tools() -> Dict[str, Any]:
    """Get all available planning tools."""
    return PLANNING_TOOLS


def get_planning_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all planning tools."""
    return {
        "create_task_plan": "Create a task plan from user instruction",
        "analyze_task_dependencies": "Analyze and create dependencies between tasks",
        "validate_task_plan": "Validate a task plan for correctness",
        "optimize_task_plan": "Optimize a task plan for better execution"
    }