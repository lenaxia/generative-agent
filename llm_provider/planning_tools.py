"""
Planning tools for StrandsAgent - converted from PlanningAgent.

These tools replace the LangChain-based PlanningAgent with @tool decorated functions
that can be used by the Universal Agent for task planning and decomposition.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator
from common.task_graph import TaskGraph, TaskDescription, TaskDependency
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


def create_task_plan(instruction: str, available_agents: List[str] = None, request_id: str = "default") -> Dict[str, Any]:
    """
    Create a task plan from user instruction - converted from PlanningAgent.
    
    This tool breaks down complex tasks into a sequence of smaller subtasks
    and creates a task graph to accomplish the given instruction.
    
    Args:
        instruction: The user instruction to create a plan for
        available_agents: List of available agent IDs and descriptions
        request_id: Request ID for tracking
        
    Returns:
        Dict containing the task graph with tasks and dependencies
    """
    logger.info(f"Creating task plan for instruction: {instruction}")
    
    # Default available agents if none provided
    if available_agents is None:
        available_agents = [
            "search_agent (Search the web for information)",
            "weather_agent (Get weather information for locations)",
            "summarizer_agent (Summarize text content)",
            "slack_agent (Send messages to Slack channels)"
        ]
    
    # Create a simple task plan - this would normally use the LLM
    # For now, create a basic single-task plan
    tasks = [
        TaskDescription(
            task_id="task_1",
            task_name=f"Execute: {instruction[:50]}...",
            agent_id="search_agent",  # Default to search agent for general tasks
            task_type="execution",
            prompt=instruction,
            status="pending"
        )
    ]
    
    # No dependencies for single task
    dependencies = []
    
    # Create task graph
    task_graph = TaskGraph(tasks=tasks, dependencies=dependencies, request_id=request_id)
    
    logger.info(f"Created task plan with {len(tasks)} tasks and {len(dependencies)} dependencies")
    
    return {
        "task_graph": task_graph,
        "tasks": tasks,
        "dependencies": dependencies,
        "request_id": request_id
    }


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