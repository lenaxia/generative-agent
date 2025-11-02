"""Shared memory tools for unified memory access across all roles.

This module provides tools that any role can use to search and retrieve
memories from the unified memory system. These tools enable:
- Cross-role memory search
- Recent memory retrieval
- Filtered searches by type, tags, etc.
"""

import logging
import threading
from typing import Any, Optional

from strands import tool

from common.providers.universal_memory_provider import UniversalMemoryProvider

logger = logging.getLogger(__name__)

# Thread-local storage for context
_context_storage = threading.local()

# Global memory provider instance
_memory_provider: UniversalMemoryProvider | None = None


def _get_memory_provider() -> UniversalMemoryProvider:
    """Get or create the global memory provider instance.

    Returns:
        UniversalMemoryProvider instance
    """
    global _memory_provider
    if _memory_provider is None:
        _memory_provider = UniversalMemoryProvider()
    return _memory_provider


def _get_current_user_id() -> str:
    """Get current user ID from context.

    Returns:
        User ID from context, or "unknown" if not available
    """
    try:
        context = getattr(_context_storage, "context", None)
        if context and hasattr(context, "user_id"):
            return context.user_id
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to get user ID from context: {e}")
        return "unknown"


@tool
def search_memory(
    query: str | None = None,
    memory_types: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search unified memory across all types.

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

        return {
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
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return {"success": False, "error": str(e)}


@tool
def get_recent_memories(
    memory_types: list[str] | None = None, limit: int = 10
) -> dict[str, Any]:
    """Get recent memories, optionally filtered by type.

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
    try:
        user_id = _get_current_user_id()
        provider = _get_memory_provider()

        memories = provider.get_recent_memories(
            user_id=user_id, memory_types=memory_types, limit=limit
        )

        return {
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
    except Exception as e:
        logger.error(f"Error getting recent memories: {e}")
        return {"success": False, "error": str(e)}
