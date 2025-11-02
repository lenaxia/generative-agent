"""Universal Memory Provider for unified memory across all roles.

This module provides a unified memory system that all roles can use for
storing and retrieving memories. It supports:
- Multiple memory types (conversation, event, plan, preference, fact)
- Rich metadata and tagging
- Cross-role memory linking
- Importance-based TTL
- Flexible search with multiple filters
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from roles.shared_tools.redis_tools import redis_get_keys, redis_read, redis_write

logger = logging.getLogger(__name__)


@dataclass
class UniversalMemory:
    """Unified memory structure across all roles.

    This data class represents a single memory entry that can be created
    by any role and searched by any role. It provides a consistent schema
    for all memory types.

    Attributes:
        id: Unique identifier for this memory
        user_id: User who owns this memory
        memory_type: Type of memory (conversation, event, plan, preference, fact)
        content: The actual memory content
        source_role: Role that created this memory
        timestamp: When this memory was created (Unix timestamp)
        importance: Importance score 0.0-1.0 (affects TTL)
        metadata: Role-specific additional data
        tags: Tags for categorization and search
        related_memories: IDs of related memories for cross-referencing
    """

    id: str
    user_id: str
    memory_type: str
    content: str
    source_role: str
    timestamp: float
    importance: float = 0.5
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None
    related_memories: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary for storage.

        Returns:
            Dictionary representation with None values converted to empty collections
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "source_role": self.source_role,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata or {},
            "tags": self.tags or [],
            "related_memories": self.related_memories or [],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniversalMemory":
        """Create memory from dictionary.

        Args:
            data: Dictionary containing memory data

        Returns:
            UniversalMemory instance
        """
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            memory_type=data["memory_type"],
            content=data["content"],
            source_role=data["source_role"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            related_memories=data.get("related_memories", []),
        )


class UniversalMemoryProvider:
    """Unified memory provider for all roles.

    This provider implements a unified memory system that all roles can use.
    It provides:
    - Consistent memory storage across roles
    - Flexible search with multiple filters
    - Importance-based TTL for automatic cleanup
    - Cross-role memory linking
    - Secondary indices for fast queries
    """

    def __init__(self):
        """Initialize the universal memory provider."""
        self.key_prefix = "memory"

    def write_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        source_role: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        related_memories: list[str] | None = None,
    ) -> str | None:
        """Write memory synchronously, return memory_id.

        Args:
            user_id: User who owns this memory
            memory_type: Type of memory (conversation, event, plan, preference, fact)
            content: The memory content
            source_role: Role creating this memory
            importance: Importance score 0.0-1.0 (affects TTL)
            metadata: Optional role-specific metadata
            tags: Optional tags for categorization
            related_memories: Optional IDs of related memories

        Returns:
            Memory ID if successful, None if failed
        """
        try:
            memory_id = str(uuid.uuid4())

            memory = UniversalMemory(
                id=memory_id,
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                source_role=source_role,
                timestamp=time.time(),
                importance=importance,
                metadata=metadata,
                tags=tags,
                related_memories=related_memories,
            )

            # Write to Redis with importance-based TTL
            memory_key = f"{self.key_prefix}:{user_id}:{memory_id}"
            ttl = self._calculate_ttl(importance)

            result = redis_write(memory_key, memory.to_dict(), ttl=ttl)

            if result.get("success"):
                # Update indices for fast queries
                self._update_indices(memory)
                logger.debug(f"Wrote memory {memory_id} for user {user_id}")
                return memory_id

            logger.error(f"Failed to write memory: {result.get('error')}")
            return None

        except Exception as e:
            logger.error(f"Error writing memory: {e}")
            return None

    def search_memories(
        self,
        user_id: str,
        query: str | None = None,
        memory_types: list[str] | None = None,
        source_roles: list[str] | None = None,
        tags: list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[UniversalMemory]:
        """Unified search across all memory types.

        Args:
            user_id: User to search memories for
            query: Optional text query for keyword matching
            memory_types: Optional filter by memory types
            source_roles: Optional filter by source roles
            tags: Optional filter by tags (any match)
            start_time: Optional filter by start timestamp
            end_time: Optional filter by end timestamp
            min_importance: Minimum importance threshold
            limit: Maximum number of results

        Returns:
            List of matching memories, sorted by relevance
        """
        try:
            # Get all memory keys for user
            pattern = f"{self.key_prefix}:{user_id}:*"
            keys_result = redis_get_keys(pattern)

            if not keys_result.get("success"):
                logger.warning(f"Failed to get keys for user {user_id}")
                return []

            memories = []
            for key in keys_result["keys"]:
                memory_result = redis_read(key)
                if not memory_result.get("success"):
                    continue

                try:
                    memory = UniversalMemory.from_dict(memory_result["value"])

                    # Apply filters
                    if memory_types and memory.memory_type not in memory_types:
                        continue
                    if source_roles and memory.source_role not in source_roles:
                        continue
                    if min_importance and memory.importance < min_importance:
                        continue
                    if start_time and memory.timestamp < start_time:
                        continue
                    if end_time and memory.timestamp > end_time:
                        continue
                    if tags and not any(tag in (memory.tags or []) for tag in tags):
                        continue

                    # Keyword matching if query provided
                    if query:
                        query_lower = query.lower()
                        content_lower = memory.content.lower()
                        if query_lower not in content_lower:
                            # Check tags too
                            if not any(
                                query_lower in tag.lower()
                                for tag in (memory.tags or [])
                            ):
                                continue

                    memories.append(memory)

                except Exception as e:
                    logger.warning(f"Failed to parse memory from {key}: {e}")
                    continue

            # Sort by importance * recency (more recent and more important first)
            current_time = time.time()
            memories.sort(
                key=lambda m: m.importance * (1.0 / (current_time - m.timestamp + 1)),
                reverse=True,
            )

            return memories[:limit]

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def get_recent_memories(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[UniversalMemory]:
        """Get recent memories, optionally filtered by type.

        Args:
            user_id: User to get memories for
            memory_types: Optional filter by memory types
            limit: Maximum number of results

        Returns:
            List of recent memories
        """
        return self.search_memories(
            user_id=user_id, memory_types=memory_types, limit=limit
        )

    def link_memories(self, memory_id1: str, memory_id2: str, user_id: str) -> bool:
        """Create bidirectional link between memories.

        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            user_id: User who owns these memories

        Returns:
            True if linking succeeded, False otherwise
        """
        try:
            # Read both memories
            mem1_key = f"{self.key_prefix}:{user_id}:{memory_id1}"
            mem2_key = f"{self.key_prefix}:{user_id}:{memory_id2}"

            mem1_result = redis_read(mem1_key)
            mem2_result = redis_read(mem2_key)

            if not (mem1_result.get("success") and mem2_result.get("success")):
                logger.warning(f"Failed to read memories for linking")
                return False

            mem1_data = mem1_result["value"]
            mem2_data = mem2_result["value"]

            # Add cross-references
            if "related_memories" not in mem1_data:
                mem1_data["related_memories"] = []
            if "related_memories" not in mem2_data:
                mem2_data["related_memories"] = []

            if memory_id2 not in mem1_data["related_memories"]:
                mem1_data["related_memories"].append(memory_id2)
            if memory_id1 not in mem2_data["related_memories"]:
                mem2_data["related_memories"].append(memory_id1)

            # Write back
            redis_write(mem1_key, mem1_data)
            redis_write(mem2_key, mem2_data)

            logger.debug(f"Linked memories {memory_id1} and {memory_id2}")
            return True

        except Exception as e:
            logger.error(f"Error linking memories: {e}")
            return False

    def _calculate_ttl(self, importance: float) -> int:
        """Calculate TTL based on importance (30-90 days).

        Args:
            importance: Importance score 0.0-1.0

        Returns:
            TTL in seconds
        """
        base_ttl = 30 * 24 * 60 * 60  # 30 days in seconds
        max_ttl = 90 * 24 * 60 * 60  # 90 days in seconds
        return int(base_ttl + (max_ttl - base_ttl) * importance)

    def _update_indices(self, memory: UniversalMemory) -> None:
        """Update secondary indices for fast queries.

        This creates indices for:
        - Memory type
        - Tags

        Args:
            memory: Memory to index
        """
        try:
            ttl = self._calculate_ttl(memory.importance)

            # Type index
            type_key = f"index:type:{memory.user_id}:{memory.memory_type}"
            type_data = {"memory_ids": [memory.id], "updated_at": time.time()}
            redis_write(type_key, type_data, ttl=ttl)

            # Tag indices
            for tag in memory.tags or []:
                tag_key = f"index:tag:{memory.user_id}:{tag}"
                tag_data = {"memory_ids": [memory.id], "updated_at": time.time()}
                redis_write(tag_key, tag_data, ttl=ttl)

        except Exception as e:
            logger.warning(f"Failed to update indices for memory {memory.id}: {e}")
            # Non-critical, don't raise
