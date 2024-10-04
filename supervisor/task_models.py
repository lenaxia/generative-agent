from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "inProgress"
    COMPLETED = "completed"
    FAILED = "failed"

class Tool(BaseModel):
    name: str
    description: str
    args_schema: dict

class Agent(BaseModel):
    id: str
    name: str
    description: str
    tools: List[Tool]

class Task(BaseModel):
    id: str
    agent_id: str
    task_type: str
    prompt_template: str
    prompt_args: dict
    tools: List[str]
    dependencies: Optional[List[str]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None

