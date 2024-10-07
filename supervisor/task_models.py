from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class Tool(BaseModel):
    name: str
    description: str
    args_schema: dict

class Agent(BaseModel):
    id: str
    name: str
    description: str
    tools: List[Tool]

