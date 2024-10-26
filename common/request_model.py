from typing import Optional, Dict
from typing_extensions import Unpack
from pydantic import BaseModel, ConfigDict, Field

from common.task_graph import TaskGraph

class RequestMetadata(BaseModel):
    prompt: str
    metadata: Optional[Dict] = None
    source_id: str = Field(..., description="The source agent")
    target_id: str = Field(..., description="The target agent")
    callback_details: Optional[Dict] = None
    response_requested: Optional[bool] = Field(default=False, description="Whether the request should return a response")
    
class Request:
    def __init__(self, metadata: RequestMetadata, task_graph: TaskGraph):
        self.metadata = metadata
        self.task_graph = task_graph
