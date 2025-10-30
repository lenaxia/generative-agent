"""
Enhanced Event Context for LLM-Safe Event Processing

This module provides simplified, LLM-friendly event context objects for pure function
event handlers, eliminating complex dependencies and threading issues.

Created: 2025-10-12
Part of: Threading Architecture Improvements (Documents 25, 26, 27)
"""

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMSafeEventContext:
    """
    LLM-SAFE: Simplified event context for pure function event handlers.

    This context provides all necessary information for event handlers without
    complex dependencies or threading concerns. It's designed to be:
    1. Immutable-friendly (defensive copying)
    2. Serializable for debugging
    3. Simple for LLM agents to understand and use
    4. Thread-safe by design (no shared mutable state)
    """

    user_id: str | None = None
    channel_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metadata(self, key: str) -> Any:
        """
        Get metadata value by key.

        Args:
            key: Metadata key to retrieve

        Returns:
            Metadata value or None if key doesn't exist
        """
        return self.metadata.get(key)

    def get_all_metadata(self) -> dict[str, Any]:
        """
        Get all metadata as a copy to prevent accidental mutation.

        Returns:
            Copy of all metadata
        """
        return deepcopy(self.metadata)

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def merge_metadata(self, new_metadata: dict[str, Any]) -> None:
        """
        Merge new metadata into existing metadata.

        Args:
            new_metadata: Dictionary of metadata to merge
        """
        self.metadata.update(new_metadata)

    def get_safe_channel(self) -> str:
        """
        Get channel ID with safe fallback.

        Returns:
            Channel ID or "general" if not set
        """
        return self.channel_id or "general"

    def get_safe_user(self) -> str:
        """
        Get user ID with safe fallback.

        Returns:
            User ID or "system" if not set
        """
        return self.user_id or "system"

    def is_valid(self) -> bool:
        """
        Validate context has minimum required information.

        Returns:
            True if context is valid for processing
        """
        # For LLM-safe processing, even minimal context is valid
        return isinstance(self.timestamp, (int, float)) and self.timestamp > 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Returns:
            Dictionary representation of context
        """
        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": deepcopy(self.metadata),
        }

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"LLMSafeEventContext(user={self.user_id}, "
            f"channel={self.channel_id}, source={self.source}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


def create_context_from_event_data(
    event_data: Any,
    source: str = "unknown",
    user_id: str | None = None,
    channel_id: str | None = None,
) -> LLMSafeEventContext:
    """
    Create LLM-safe event context from various event data formats.

    This function handles the common patterns of event data and creates
    a consistent context object that pure function handlers can use.

    Args:
        event_data: Event data in various formats (dict, list, string, etc.)
        source: Source of the event (role name, component, etc.)
        user_id: Optional user ID override
        channel_id: Optional channel ID override

    Returns:
        LLMSafeEventContext object
    """
    metadata = {}
    extracted_user_id = user_id
    extracted_channel_id = channel_id

    try:
        if isinstance(event_data, dict):
            # Extract standard fields from dictionary
            extracted_user_id = extracted_user_id or event_data.get("user_id")
            extracted_channel_id = extracted_channel_id or event_data.get("channel_id")

            # Everything else goes into metadata
            metadata = {
                key: value
                for key, value in event_data.items()
                if key not in ["user_id", "channel_id"]
            }

        elif isinstance(event_data, list) and len(event_data) >= 2:
            # Handle common timer pattern: ['timer_id', 'original_request']
            metadata["timer_id"] = str(event_data[0])
            metadata["original_request"] = str(event_data[1])

            # Additional list items go into metadata
            if len(event_data) > 2:
                metadata["additional_data"] = event_data[2:]

        elif isinstance(event_data, list) and len(event_data) == 1:
            # Single item list
            metadata["data"] = event_data[0]

        elif isinstance(event_data, str):
            # String data
            metadata["raw_data"] = event_data

        else:
            # Any other type
            metadata["raw_data"] = str(event_data)
            metadata["original_type"] = type(event_data).__name__

    except Exception as e:
        logger.warning(f"Error parsing event data: {e}")
        metadata["parse_error"] = str(e)
        metadata["raw_data"] = str(event_data)

    return LLMSafeEventContext(
        user_id=extracted_user_id,
        channel_id=extracted_channel_id,
        source=source,
        metadata=metadata,
    )


def create_minimal_context(source: str = "system") -> LLMSafeEventContext:
    """
    Create minimal event context for system events.

    Args:
        source: Source of the event

    Returns:
        Minimal LLMSafeEventContext
    """
    return LLMSafeEventContext(source=source)


def create_user_context(
    user_id: str, channel_id: str, source: str = "user", **metadata
) -> LLMSafeEventContext:
    """
    Create user-specific event context.

    Args:
        user_id: User identifier
        channel_id: Channel identifier
        source: Source of the event
        **metadata: Additional metadata

    Returns:
        User-specific LLMSafeEventContext
    """
    return LLMSafeEventContext(
        user_id=user_id, channel_id=channel_id, source=source, metadata=metadata
    )


# Backward compatibility helper
def convert_legacy_context(legacy_context) -> LLMSafeEventContext:
    """
    Convert legacy EventHandlerContext to LLMSafeEventContext.

    This helper function provides backward compatibility during the migration
    from complex EventHandlerContext to simplified LLMSafeEventContext.

    Args:
        legacy_context: Legacy EventHandlerContext object

    Returns:
        Equivalent LLMSafeEventContext
    """
    try:
        # Extract information from legacy context
        user_id = getattr(legacy_context, "get_user_id", lambda: None)()
        channel_id = getattr(legacy_context, "get_channel", lambda: None)()

        # Create metadata from execution context if available
        metadata = {}
        if hasattr(legacy_context, "execution_context"):
            metadata = deepcopy(legacy_context.execution_context)

        return LLMSafeEventContext(
            user_id=user_id, channel_id=channel_id, source="legacy", metadata=metadata
        )

    except Exception as e:
        logger.warning(f"Error converting legacy context: {e}")
        return create_minimal_context("legacy_conversion_error")


# Utility functions for common context operations
def extract_timer_info(context: LLMSafeEventContext) -> dict[str, Any]:
    """
    Extract timer-specific information from context.

    Args:
        context: Event context

    Returns:
        Dictionary with timer information
    """
    return {
        "timer_id": context.get_metadata("timer_id"),
        "original_request": context.get_metadata("original_request"),
        "user_id": context.user_id,
        "channel_id": context.channel_id,
    }


def is_user_event(context: LLMSafeEventContext) -> bool:
    """
    Check if event originated from a user.

    Args:
        context: Event context

    Returns:
        True if event has user information
    """
    return context.user_id is not None and context.channel_id is not None


def is_system_event(context: LLMSafeEventContext) -> bool:
    """
    Check if event originated from system.

    Args:
        context: Event context

    Returns:
        True if event is system-generated
    """
    return context.source in ["system", "heartbeat", "scheduler", "monitor"]
