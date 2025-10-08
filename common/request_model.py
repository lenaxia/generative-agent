"""Request data models for the StrandsAgent Universal Agent System.

Defines data structures for request handling, metadata management,
and workflow coordination across the system components.
"""

from typing import Optional

from pydantic import BaseModel, Field

from common.task_graph import TaskGraph


class RequestMetadata(BaseModel):
    """Metadata container for request handling and routing.

    Contains essential information for request processing including
    source and target identification, callback configuration, and
    response handling preferences.
    """

    prompt: str
    metadata: Optional[dict] = None
    source_id: str = Field(..., description="The source agent")
    target_id: str = Field(..., description="The target agent")
    callback_details: Optional[dict] = None
    response_requested: Optional[bool] = Field(
        default=False, description="Whether the request should return a response"
    )


class Request:
    """Request container combining metadata and task execution graph.

    Encapsulates a complete request with its associated metadata and
    task graph for workflow execution within the StrandsAgent system.
    """

    def __init__(self, metadata: RequestMetadata, task_graph: TaskGraph):
        """Initialize a Request with metadata and task graph.

        Args:
            metadata: Request metadata containing routing and callback information.
            task_graph: Task execution graph defining the workflow to be executed.
        """
        self.metadata = metadata
        self.task_graph = task_graph
