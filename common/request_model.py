"""Request data models for the StrandsAgent Universal Agent System.

Defines data structures for request handling, metadata management,
and workflow coordination across the system components.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from common.task_graph import TaskGraph


class RequestMetadata(BaseModel):
    prompt: str
    metadata: Optional[dict] = None
    source_id: str = Field(..., description="The source agent")
    target_id: str = Field(..., description="The target agent")
    callback_details: Optional[dict] = None
    response_requested: Optional[bool] = Field(
        default=False, description="Whether the request should return a response"
    )


class Request:
    def __init__(self, metadata: RequestMetadata, task_graph: TaskGraph):
        self.metadata = metadata
        self.task_graph = task_graph
