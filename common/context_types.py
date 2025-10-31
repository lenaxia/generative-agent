"""
Context types and collector for the Universal Agent System.

This module provides enum-based context types and a ContextCollector class
for surgical context gathering with interface-driven design.
"""

import logging
from enum import Enum
from typing import Any

from common.interfaces.context_interfaces import LocationProvider, MemoryProvider
from roles.shared_tools.redis_tools import redis_get_keys, redis_read

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Enumeration of available context types for surgical context gathering."""

    LOCATION = "location"
    RECENT_MEMORY = "recent_memory"
    PRESENCE = "presence"
    SCHEDULE = "schedule"


class ContextCollector:
    """Enum-based context collector with interface-driven design.

    This class provides surgical context gathering based on specific context types
    determined by the router role. It uses provider interfaces for extensibility
    and implements graceful degradation when providers fail.
    """

    def __init__(
        self,
        memory_provider: MemoryProvider,
        location_provider: LocationProvider,
    ):
        """Initialize ContextCollector with provider interfaces.

        Args:
            memory_provider: Provider for memory storage and retrieval
            location_provider: Provider for location tracking
        """
        self.memory_provider = memory_provider
        self.location_provider = location_provider

    async def initialize(self):
        """Initialize providers if they have initialization methods."""
        try:
            if hasattr(self.memory_provider, "initialize"):
                await self.memory_provider.initialize()
            if hasattr(self.location_provider, "initialize"):
                await self.location_provider.initialize()
            logger.debug("Context collector initialized successfully")
        except Exception as e:
            logger.warning(f"Context collector initialization failed: {e}")
            # Don't raise - system should work without context

    async def gather_context(
        self, user_id: str, context_types: list[str]
    ) -> dict[str, Any]:
        """Gather specific contexts with error handling and graceful degradation.

        Args:
            user_id: User identifier for context gathering
            context_types: List of context type strings to gather

        Returns:
            Dict containing successfully gathered context data
        """
        if not context_types or not user_id:
            return {}

        context = {}

        for context_type in context_types:
            try:
                if context_type == ContextType.LOCATION.value:
                    location = await self.location_provider.get_current_location(
                        user_id
                    )
                    if location:
                        context["location"] = location

                elif context_type == ContextType.RECENT_MEMORY.value:
                    memories = await self.memory_provider.get_recent_memories(
                        user_id, limit=3
                    )
                    if memories:
                        context["recent_memory"] = [m.content for m in memories]

                elif context_type == ContextType.PRESENCE.value:
                    others_home = await self._get_others_home(user_id)
                    if others_home:
                        context["presence"] = others_home

                elif context_type == ContextType.SCHEDULE.value:
                    # Get schedule context from calendar role
                    schedule_events = await self._get_user_schedule(user_id)
                    if schedule_events:
                        context["schedule"] = schedule_events

            except Exception as e:
                logger.warning(
                    f"Failed to gather {context_type} context for {user_id}: {e}"
                )
                # Error handling: continue without this context type
                continue

        return context

    async def _get_user_schedule(self, user_id: str) -> list[dict[str, Any]]:
        """Get user's schedule events from calendar role.

        Args:
            user_id: User identifier

        Returns:
            List of schedule events for the user
        """
        try:
            # Use Redis to get cached schedule data to avoid circular imports
            schedule_key = f"schedule:{user_id}"
            schedule_result = redis_read(schedule_key)

            if schedule_result.get("success") and schedule_result.get("value"):
                schedule_data = schedule_result["value"]
                if isinstance(schedule_data, list):
                    return schedule_data
                elif isinstance(schedule_data, dict) and "events" in schedule_data:
                    return schedule_data["events"]

            return []

        except Exception as e:
            logger.warning(f"Failed to get schedule for {user_id}: {e}")
            return []

    async def _get_others_home(self, user_id: str) -> list[str]:
        """Get other users currently home using Redis location data.

        Args:
            user_id: Current user ID to exclude from results

        Returns:
            List of user IDs who are currently home (excluding current user)
        """
        try:
            # Get all location keys from Redis
            keys_result = redis_get_keys("location:*")
            if not keys_result.get("success"):
                return []

            others_home = []
            for key in keys_result["keys"]:
                other_user = key.split(":")[-1]
                if other_user != user_id:
                    location_result = redis_read(key)
                    if (
                        location_result.get("success")
                        and location_result["value"] == "home"
                    ):
                        others_home.append(other_user)

            return others_home

        except Exception as e:
            logger.warning(f"Failed to get presence for {user_id}: {e}")
            return []
