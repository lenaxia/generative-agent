"""Shared lifecycle helper functions for multi-turn roles.

This module provides common pre-processing and post-processing functions
used by conversation, calendar, and planning roles to eliminate code duplication
while maintaining LLM-friendly explicit imports.

Design Principles:
- Explicit over implicit
- Simple, flat function calls
- Comprehensive documentation
- No inheritance or complex patterns
- Maintains single-file role readability
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_dual_layer_context(
    context,
    memory_types: list[str] | None = None,
    realtime_limit: int = 10,
    assessed_limit: int = 5,
    importance_threshold: float = 0.7,
) -> dict[str, Any]:
    """Load dual-layer context (realtime log + assessed memories).

    This is the standard pre-processing function for multi-turn roles.
    It loads two layers of memory:
    1. Layer 1: Realtime log (recent messages, 24h TTL)
    2. Layer 2: Assessed memories (important memories, graduated TTL)

    Used by: conversation, calendar, planning roles

    Args:
        context: Event context with user_id attribute
        memory_types: Memory types to filter (default: all types)
        realtime_limit: Max realtime messages to load (default: 10)
        assessed_limit: Max assessed memories to load (default: 5)
        importance_threshold: Minimum importance for assessed memories (default: 0.7)

    Returns:
        Dict with:
        - realtime_context: Formatted realtime messages
        - assessed_memories: Formatted assessed memories
        - user_id: User identifier
    """
    try:
        from common.providers.universal_memory_provider import UniversalMemoryProvider
        from common.realtime_log import get_recent_messages

        user_id = getattr(context, "user_id", "unknown")

        # Layer 1: Realtime log (last N messages)
        realtime_messages = get_recent_messages(user_id, limit=realtime_limit)

        # Layer 2: Assessed memories (last N, importance >= threshold)
        memory_provider = UniversalMemoryProvider()
        assessed_memories = memory_provider.get_recent_memories(
            user_id=user_id, memory_types=memory_types, limit=assessed_limit
        )

        # Filter for important memories only
        important_memories = [
            m for m in assessed_memories if m.importance >= importance_threshold
        ]

        return {
            "realtime_context": _format_realtime_messages(realtime_messages),
            "assessed_memories": _format_assessed_memories(important_memories),
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Failed to load dual-layer context: {e}")
        return {
            "realtime_context": "No recent messages.",
            "assessed_memories": "No important memories.",
            "user_id": getattr(context, "user_id", "unknown"),
        }


def save_to_realtime_log(
    llm_result: str | Any, context, pre_data: dict, role_name: str
) -> str | Any:
    """Save interaction to realtime log.

    This is the standard post-processing function for multi-turn roles.
    It saves the user's message and assistant's response to the realtime log
    for later analysis and memory assessment.

    Used by: conversation, calendar, planning roles

    Args:
        llm_result: LLM response (string or other type)
        context: Event context with user_id and original_prompt
        pre_data: Pre-processing data with _instruction
        role_name: Name of the role (for logging)

    Returns:
        Original llm_result unchanged
    """
    try:
        from common.realtime_log import add_message

        user_id = getattr(context, "user_id", "unknown")

        # Get user message from context or pre_data
        user_message = getattr(context, "original_prompt", None) or pre_data.get(
            "_instruction", "unknown"
        )

        # Save to universal realtime log (24h TTL)
        add_message(
            user_id=user_id,
            user_message=user_message,
            assistant_response=str(llm_result),  # Convert to string for storage
            role=role_name,
            metadata=None,
        )

        logger.debug(f"Saved {role_name} interaction to realtime log for {user_id}")

        return llm_result

    except Exception as e:
        logger.error(f"Failed to save to realtime log: {e}")
        return llm_result


def _format_realtime_messages(messages: list[dict[str, Any]]) -> str:
    """Format realtime messages for prompt.

    Args:
        messages: List of message dicts with user/assistant fields

    Returns:
        Formatted string for LLM prompt
    """
    if not messages:
        return "No recent messages."

    formatted = []
    for msg in messages:
        formatted.append(f"User: {msg['user']}")
        formatted.append(f"Assistant: {msg['assistant']}")
    return "\n".join(formatted)


def _format_assessed_memories(memories: list) -> str:
    """Format assessed memories for prompt.

    Args:
        memories: List of UniversalMemory objects

    Returns:
        Formatted string for LLM prompt
    """
    if not memories:
        return "No important memories."

    formatted = []
    for mem in memories:
        summary = mem.summary or mem.content
        tags = ", ".join(mem.tags or [])
        formatted.append(f"- {summary} (tags: {tags})")
    return "\n".join(formatted)
