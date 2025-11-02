"""Universal realtime log for tracking recent interactions across all roles.

This module provides a universal realtime log that all roles can use to track
recent user interactions. Messages are stored in a Redis sorted set with
per-message TTL, enabling efficient retrieval and automatic cleanup.

Key features:
- Universal: All roles write to the same log
- Per-message TTL: Each message expires individually after 24 hours
- Sorted by timestamp: Automatic chronological ordering
- Analysis tracking: Mark messages as analyzed
- Efficient cleanup: Automatic removal of old messages
"""

import json
import logging
import time
import uuid
from typing import Any, Optional

from roles.shared_tools.redis_tools import _get_redis_client

logger = logging.getLogger(__name__)


def redis_zadd(key: str, score: float, member: str) -> dict[str, Any]:
    """Add member to sorted set with score.

    Args:
        key: Redis key
        score: Score for sorting (timestamp)
        member: Member to add (JSON string)

    Returns:
        Dict with success status
    """
    try:
        client = _get_redis_client()
        client.zadd(key, {member: score})
        return {"success": True}
    except Exception as e:
        logger.error(f"Redis ZADD failed: {e}")
        return {"success": False, "error": str(e)}


def redis_zrevrange(key: str, start: int, stop: int) -> dict[str, Any]:
    """Get members from sorted set in reverse order (newest first).

    Args:
        key: Redis key
        start: Start index (0-based)
        stop: Stop index (inclusive)

    Returns:
        Dict with success status and values
    """
    try:
        client = _get_redis_client()
        values = client.zrevrange(key, start, stop)
        return {
            "success": True,
            "values": [v.decode() if isinstance(v, bytes) else v for v in values],
        }
    except Exception as e:
        logger.error(f"Redis ZREVRANGE failed: {e}")
        return {"success": False, "error": str(e), "values": []}


def redis_zremrangebyscore(
    key: str, min_score: str, max_score: float
) -> dict[str, Any]:
    """Remove members from sorted set by score range.

    Args:
        key: Redis key
        min_score: Minimum score (use "-inf" for negative infinity)
        max_score: Maximum score

    Returns:
        Dict with success status
    """
    try:
        client = _get_redis_client()
        count = client.zremrangebyscore(key, min_score, max_score)
        return {"success": True, "removed_count": count}
    except Exception as e:
        logger.error(f"Redis ZREMRANGEBYSCORE failed: {e}")
        return {"success": False, "error": str(e)}


def add_message(
    user_id: str,
    user_message: str,
    assistant_response: str,
    role: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Add a message to the universal realtime log.

    Args:
        user_id: User identifier
        user_message: User's message
        assistant_response: Assistant's response
        role: Role that generated the response
        metadata: Optional additional metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        message = {
            "id": str(uuid.uuid4()),
            "user": user_message,
            "assistant": assistant_response,
            "role": role,
            "timestamp": time.time(),
            "analyzed": False,
            "metadata": metadata or {},
        }

        key = f"realtime:{user_id}"
        score = message["timestamp"]
        member = json.dumps(message)

        result = redis_zadd(key, score, member)

        if result.get("success"):
            # Cleanup old messages (> 24 hours)
            cleanup_old_messages(user_id, max_age_hours=24)
            return True

        return False

    except Exception as e:
        logger.error(f"Failed to add message to realtime log: {e}")
        return False


def get_recent_messages(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get recent messages from realtime log.

    Args:
        user_id: User identifier
        limit: Maximum number of messages to return

    Returns:
        List of message dictionaries, newest first
    """
    try:
        key = f"realtime:{user_id}"
        result = redis_zrevrange(key, 0, limit - 1)

        if not result.get("success"):
            return []

        messages = []
        for value in result.get("values", []):
            try:
                message = json.loads(value)
                messages.append(message)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse message: {e}")
                continue

        return messages

    except Exception as e:
        logger.error(f"Failed to get recent messages: {e}")
        return []


def get_unanalyzed_messages(user_id: str) -> list[dict[str, Any]]:
    """Get messages that haven't been analyzed yet.

    Args:
        user_id: User identifier

    Returns:
        List of unanalyzed message dictionaries
    """
    try:
        key = f"realtime:{user_id}"
        # Get all messages
        result = redis_zrevrange(key, 0, -1)

        if not result.get("success"):
            return []

        unanalyzed = []
        for value in result.get("values", []):
            try:
                message = json.loads(value)
                if not message.get("analyzed", False):
                    unanalyzed.append(message)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse message: {e}")
                continue

        return unanalyzed

    except Exception as e:
        logger.error(f"Failed to get unanalyzed messages: {e}")
        return []


def mark_as_analyzed(user_id: str, message_ids: list[str]) -> bool:
    """Mark messages as analyzed.

    Args:
        user_id: User identifier
        message_ids: List of message IDs to mark as analyzed

    Returns:
        True if successful, False otherwise
    """
    try:
        key = f"realtime:{user_id}"
        # Get all messages
        result = redis_zrevrange(key, 0, -1)

        if not result.get("success"):
            return False

        # Update analyzed flag for matching messages
        updated = False
        for value in result.get("values", []):
            try:
                message = json.loads(value)
                if message["id"] in message_ids:
                    # Update analyzed flag
                    message["analyzed"] = True
                    # Use timestamp from message, or current time if missing
                    timestamp = message.get("timestamp", time.time())
                    # Remove old entry and add updated one
                    client = _get_redis_client()
                    client.zrem(key, value)
                    redis_zadd(key, timestamp, json.dumps(message))
                    updated = True
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to update message: {e}")
                continue

        return updated

    except Exception as e:
        logger.error(f"Failed to mark messages as analyzed: {e}")
        return False


def cleanup_old_messages(user_id: str, max_age_hours: int = 24) -> bool:
    """Remove messages older than specified age.

    Args:
        user_id: User identifier
        max_age_hours: Maximum age in hours

    Returns:
        True if successful, False otherwise
    """
    try:
        key = f"realtime:{user_id}"
        cutoff = time.time() - (max_age_hours * 60 * 60)

        result = redis_zremrangebyscore(key, "-inf", cutoff)

        if result.get("success"):
            removed = result.get("removed_count", 0)
            if removed > 0:
                logger.debug(f"Cleaned up {removed} old messages for {user_id}")
            return True

        return False

    except Exception as e:
        logger.error(f"Failed to cleanup old messages: {e}")
        return False


def get_last_message_time(user_id: str) -> float | None:
    """Get timestamp of the most recent message.

    Args:
        user_id: User identifier

    Returns:
        Timestamp of last message, or None if no messages
    """
    try:
        messages = get_recent_messages(user_id, limit=1)
        if messages:
            return messages[0]["timestamp"]
        return None
    except Exception as e:
        logger.error(f"Failed to get last message time: {e}")
        return None
