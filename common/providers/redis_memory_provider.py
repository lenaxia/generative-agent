"""
Redis-based memory provider implementation.

This module provides a Redis-based implementation of the MemoryProvider interface
using the existing Redis tools for storage and retrieval operations.
"""

import json
import logging
import time
from datetime import datetime
from typing import List

from common.interfaces.context_interfaces import MemoryEntry, MemoryProvider
from roles.shared_tools.redis_tools import redis_get_keys, redis_read, redis_write

logger = logging.getLogger(__name__)


class RedisMemoryProvider(MemoryProvider):
    """Redis-based memory storage using existing Redis tools.

    This provider implements the MemoryProvider interface using Redis as the
    backend storage system. It uses importance-based TTL for automatic cleanup
    and provides efficient storage and retrieval of user memories.
    """

    def __init__(self):
        """Initialize Redis memory provider."""
        self.key_prefix = "memory"

    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store memory in Redis with importance-based TTL.

        Args:
            memory: MemoryEntry to store

        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            memory_key = f"{self.key_prefix}:{memory.user_id}:{int(memory.timestamp.timestamp())}"
            memory_data = {
                "content": memory.content,
                "location": memory.location,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": memory.metadata or {},
            }

            # TTL based on importance (30-60 days)
            base_ttl = 2592000  # 30 days in seconds
            ttl = int(base_ttl * min(memory.importance * 2, 1.0))

            result = redis_write(memory_key, memory_data, ttl=ttl)
            return result.get("success", False)

        except Exception as e:
            logger.error(f"Redis memory storage failed: {e}")
            return False

    async def get_recent_memories(
        self, user_id: str, limit: int = 3
    ) -> list[MemoryEntry]:
        """Get recent memories from Redis.

        Args:
            user_id: User identifier
            limit: Maximum number of memories to return

        Returns:
            List[MemoryEntry]: Recent memories for the user
        """
        try:
            keys_result = redis_get_keys(f"{self.key_prefix}:{user_id}:*")
            if not keys_result.get("success"):
                return []

            recent_keys = sorted(keys_result["keys"])[-limit:]
            memories = []

            for key in recent_keys:
                memory_result = redis_read(key)
                if memory_result.get("success"):
                    data = memory_result["value"]
                    memories.append(
                        MemoryEntry(
                            user_id=user_id,
                            content=data["content"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            location=data.get("location"),
                            importance=data.get("importance", 0.5),
                            metadata=data.get("metadata", {}),
                        )
                    )

            return memories

        except Exception as e:
            logger.error(f"Redis memory retrieval failed: {e}")
            return []

    async def search_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[MemoryEntry]:
        """Search memories using keyword matching.

        Args:
            user_id: User identifier
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List[MemoryEntry]: Relevant memories matching the query
        """
        try:
            keys_result = redis_get_keys(f"{self.key_prefix}:{user_id}:*")
            if not keys_result.get("success"):
                return []

            query_words = set(query.lower().split())
            scored_memories = []

            for key in keys_result["keys"]:
                memory_result = redis_read(key)
                if memory_result.get("success"):
                    data = memory_result["value"]
                    content_words = set(data["content"].lower().split())

                    overlap = len(query_words.intersection(content_words))
                    if overlap > 0:
                        relevance = overlap / len(query_words.union(content_words))
                        scored_memories.append(
                            (
                                relevance,
                                MemoryEntry(
                                    user_id=user_id,
                                    content=data["content"],
                                    timestamp=datetime.fromisoformat(data["timestamp"]),
                                    location=data.get("location"),
                                    importance=data.get("importance", 0.5),
                                    metadata=data.get("metadata", {}),
                                ),
                            )
                        )

            scored_memories.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in scored_memories[:limit]]

        except Exception as e:
            logger.error(f"Redis memory search failed: {e}")
            return []
