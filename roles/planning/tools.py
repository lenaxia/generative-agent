"""
Custom tools for the planning role.
"""

from typing import List, Optional, Dict, Any
import json
import logging
from strands import tool

logger = logging.getLogger(__name__)


@tool
def create_task_plan(instruction: str) -> Dict[str, Any]:
    """
    Create a task plan by breaking down the instruction into manageable steps.
    
    Args:
        instruction: The user instruction to create a plan for
        
    Returns:
        Dict containing the task plan with tasks and dependencies
    """
    try:
        # For now, create a simple plan structure
        # This can be enhanced later with LLM-based planning
        plan = {
            "instruction": instruction,
            "tasks": [
                {
                    "task_name": f"analyze_request",
                    "agent_id": "analysis_agent",
                    "task_type": "analysis",
                    "prompt": f"Analyze the following request: {instruction}"
                },
                {
                    "task_name": f"execute_main_task", 
                    "agent_id": "execution_agent",
                    "task_type": "execution",
                    "prompt": f"Execute the main task: {instruction}"
                }
            ],
            "dependencies": []
        }
        
        logger.info(f"Created task plan for: {instruction}")
        return plan
        
    except Exception as e:
        logger.error(f"Failed to create task plan: {e}")
        return {"error": str(e), "tasks": [], "dependencies": []}


@tool
def analyze_task_dependencies(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze dependencies between tasks.
    
    Args:
        tasks: List of task dictionaries
        
    Returns:
        List of dependency relationships
    """
    try:
        dependencies = []
        
        # Simple dependency analysis - each task depends on the previous one
        for i in range(1, len(tasks)):
            dependencies.append({
                "source": tasks[i-1]["task_name"],
                "target": tasks[i]["task_name"],
                "dependency_type": "sequential"
            })
        
        logger.info(f"Analyzed {len(dependencies)} dependencies")
        return dependencies
        
    except Exception as e:
        logger.error(f"Failed to analyze dependencies: {e}")
        return []


@tool
def validate_task_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a task plan for completeness and correctness.
    
    Args:
        plan: Task plan dictionary
        
    Returns:
        Validation result with status and any errors
    """
    try:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if plan has required fields
        if "tasks" not in plan:
            validation_result["valid"] = False
            validation_result["errors"].append("Plan missing 'tasks' field")
        
        if "dependencies" not in plan:
            validation_result["warnings"].append("Plan missing 'dependencies' field")
        
        # Validate each task
        tasks = plan.get("tasks", [])
        for i, task in enumerate(tasks):
            required_fields = ["task_name", "agent_id", "task_type", "prompt"]
            for field in required_fields:
                if field not in task:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Task {i} missing required field: {field}")
        
        logger.info(f"Validated task plan: {'valid' if validation_result['valid'] else 'invalid'}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate task plan: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": []
        }