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
    source_id: str = Field(..., description="The source agent")
    target_id: str = Field(..., description="The target agent")

    # Common routing fields - present in all requests
    user_id: str | None = Field(None, description="User identifier across all channels")
    channel_id: str | None = Field(
        None,
        description="Channel identifier (e.g., 'slack:C123', 'home_assistant:entity')",
    )

    # Channel-specific metadata - plain dict for flexibility
    # Should contain channel-specific routing data like:
    # - Slack: slack_thread_ts, slack_initial_ts, slack_initial_content
    # - Home Assistant: ha_entity_id, ha_initial_state
    # - Sonos: sonos_device, sonos_initial_announcement
    metadata: dict | None = Field(
        None,
        description="Channel-specific routing and context data (thread IDs, message timestamps, etc.)",
    )

    callback_details: dict | None = None
    response_requested: bool | None = Field(
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
