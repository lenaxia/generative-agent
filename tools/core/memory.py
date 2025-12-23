"""Core Memory Tools

Provides unified memory system tools for the agent system.
All tools are query-only (read from memory).

These are core infrastructure tools - DO NOT MODIFY.
For custom tools, use tools/custom/.

Originally extracted from: roles/shared_tools/memory_tools.py
Moved from: roles/memory/tools.py
"""

import logging
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def create_memory_tools(memory_provider: Any) -> list:
    """Create memory domain tools.

    Args:
        memory_provider: UniversalMemoryProvider instance

    Returns:
        List of tool functions for memory domain
    """
    # Store provider reference for tools to use
    global _memory_provider
    _memory_provider = memory_provider

    tools = [
        search_memory,
        get_recent_memories,
    ]

    logger.info(f"Created {len(tools)} memory tools")
    return tools


# Global provider reference (set by create_memory_tools)
_memory_provider = None


def _get_memory_provider():
    """Get the memory provider instance."""
    if _memory_provider is None:
        # Fallback: try to get from context or raise error
        raise RuntimeError("Memory provider not initialized. Call create_memory_tools first.")
    return _memory_provider


def _get_current_user_id() -> str:
    """Get current user ID from context."""
    # In dynamic agent architecture, user_id should be in task context
    # For now, return default
    # TODO: In Phase 3, integrate with TaskContext
    return "default_user"


# QUERY TOOLS (read-only)


@tool
def search_memory(
    query: str | None = None,
    memory_types: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search unified memory across all types.

    Query tool - searches memory, no side effects.

    This tool searches the unified memory system and returns matching memories.
    It can search by keyword, filter by memory type, filter by tags, and limit results.

    Args:
        query: Optional text search query for keyword matching
        memory_types: Optional filter by types (conversation, event, plan, preference, fact)
        tags: Optional filter by tags (any match)
        limit: Maximum number of results to return (default: 10)

    Returns:
        Dictionary with:
        - success: Boolean indicating if search succeeded
        - memories: List of matching memories with content, type, source, etc.
        - count: Number of memories returned
        - error: Error message if search failed
    """
    logger.info(f"Searching memory: query={query}, types={memory_types}, limit={limit}")

    try:
        user_id = _get_current_user_id()
        provider = _get_memory_provider()

        memories = provider.search_memories(
            user_id=user_id,
            query=query,
            memory_types=memory_types,
            tags=tags,
            limit=limit,
        )

        result = {
            "success": True,
            "memories": [
                {
                    "content": m.content,
                    "type": m.memory_type,
                    "source": m.source_role,
                    "timestamp": m.timestamp,
                    "importance": m.importance,
                    "tags": m.tags or [],
                    "metadata": m.metadata or {},
                }
                for m in memories
            ],
            "count": len(memories),
        }

        logger.info(f"Found {len(memories)} memories")
        return result

    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return {"success": False, "error": str(e), "count": 0}


@tool
def get_recent_memories(
    memory_types: list[str] | None = None, limit: int = 10
) -> dict[str, Any]:
    """Get recent memories, optionally filtered by type.

    Query tool - retrieves recent memories, no side effects.

    This tool retrieves the most recent memories for quick access.
    Useful for loading context at the start of a conversation or task.

    Args:
        memory_types: Optional filter by types (conversation, event, plan, preference, fact)
        limit: Maximum number of results to return (default: 10)

    Returns:
        Dictionary with:
        - success: Boolean indicating if retrieval succeeded
        - memories: List of recent memories with content, type, source, etc.
        - error: Error message if retrieval failed
    """
    logger.info(f"Getting recent memories: types={memory_types}, limit={limit}")

    try:
        user_id = _get_current_user_id()
        provider = _get_memory_provider()

        memories = provider.get_recent_memories(
            user_id=user_id, memory_types=memory_types, limit=limit
        )

        result = {
            "success": True,
            "memories": [
                {
                    "content": m.content,
                    "type": m.memory_type,
                    "source": m.source_role,
                    "timestamp": m.timestamp,
                }
                for m in memories
            ],
        }

        logger.info(f"Retrieved {len(memories)} recent memories")
        return result

    except Exception as e:
        logger.error(f"Error getting recent memories: {e}")
        return {"success": False, "error": str(e)}
