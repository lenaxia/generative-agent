"""
Context interfaces for the Universal Agent System.

This module defines abstract interfaces for context providers that can be
implemented by different backends (Redis, MQTT, etc.) to provide context
information to the agent system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """Standard memory entry format for storing user interactions."""

    user_id: str
    content: str
    timestamp: datetime
    location: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] | None = None


@dataclass
class LocationData:
    """Standard location data format for user location tracking."""

    user_id: str
    current_location: str
    timestamp: datetime
    previous_location: str | None = None
    confidence: float = 1.0


@dataclass
class ContextData:
    """Generic context data container for any context type."""

    user_id: str
    context_type: str
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


class MemoryProvider(ABC):
    """Interface for memory storage and retrieval backends."""

    @abstractmethod
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry.

        Args:
            memory: MemoryEntry to store

        Returns:
            bool: True if storage was successful, False otherwise
        """

    @abstractmethod
    async def get_recent_memories(
        self, user_id: str, limit: int = 3
    ) -> list[MemoryEntry]:
        """Get recent memories for user.

        Args:
            user_id: User identifier
            limit: Maximum number of memories to return

        Returns:
            List[MemoryEntry]: Recent memories for the user
        """

    @abstractmethod
    async def search_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[MemoryEntry]:
        """Search memories by content relevance.

        Args:
            user_id: User identifier
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List[MemoryEntry]: Relevant memories matching the query
        """


class LocationProvider(ABC):
    """Interface for location tracking backends."""

    @abstractmethod
    async def get_current_location(self, user_id: str) -> str | None:
        """Get user's current location.

        Args:
            user_id: User identifier

        Returns:
            Optional[str]: Current location or None if not available
        """

    @abstractmethod
    async def update_location(
        self, user_id: str, location: str, confidence: float = 1.0
    ) -> bool:
        """Update user location.

        Args:
            user_id: User identifier
            location: New location string
            confidence: Confidence level (0.0-1.0)

        Returns:
            bool: True if update was successful, False otherwise
        """


class ContextProvider(ABC):
    """Generic interface for context providers."""

    @abstractmethod
    async def get_context(
        self, user_id: str, context_type: str
    ) -> ContextData | None:
        """Get context data for user.

        Args:
            user_id: User identifier
            context_type: Type of context to retrieve

        Returns:
            Optional[ContextData]: Context data or None if not available
        """


class EnvironmentProvider(ABC):
    """Interface for environment data providers (weather, time, etc.)."""

    @abstractmethod
    async def get_environment_data(
        self, location: str | None = None
    ) -> dict[str, Any]:
        """Get environment data for location.

        Args:
            location: Optional location string, uses default if None

        Returns:
            Dict[str, Any]: Environment data (weather, time, etc.)
        """
