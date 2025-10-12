"""
Tests for Enhanced Event Context for LLM Safety

Tests the enhanced event context that provides simplified, LLM-friendly
context objects for pure function event handlers.

Following TDD principles - tests written first.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from common.enhanced_event_context import LLMSafeEventContext


class TestLLMSafeEventContext:
    """Test LLM-safe event context functionality."""

    def test_create_simple_context(self):
        """Test creating simple event context."""
        context = LLMSafeEventContext(
            user_id="user123",
            channel_id="channel456",
            timestamp=1234567890.0,
            source="timer",
        )

        assert context.user_id == "user123"
        assert context.channel_id == "channel456"
        assert context.timestamp == 1234567890.0
        assert context.source == "timer"
        assert context.metadata == {}

    def test_create_context_with_metadata(self):
        """Test creating context with metadata."""
        metadata = {"original_request": "Set timer for 5 minutes", "priority": "high"}

        context = LLMSafeEventContext(
            user_id="user123", channel_id="channel456", metadata=metadata
        )

        assert context.metadata == metadata
        assert context.get_metadata("original_request") == "Set timer for 5 minutes"
        assert context.get_metadata("priority") == "high"
        assert context.get_metadata("nonexistent") is None

    def test_create_context_with_defaults(self):
        """Test creating context with default values."""
        context = LLMSafeEventContext()

        assert context.user_id is None
        assert context.channel_id is None
        assert context.source == "unknown"
        assert isinstance(context.timestamp, float)
        assert context.timestamp > 0
        assert context.metadata == {}

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = LLMSafeEventContext(
            user_id="user123",
            channel_id="channel456",
            source="timer",
            metadata={"key": "value"},
        )

        result = context.to_dict()

        assert result["user_id"] == "user123"
        assert result["channel_id"] == "channel456"
        assert result["source"] == "timer"
        assert result["metadata"] == {"key": "value"}
        assert "timestamp" in result

    def test_context_is_valid(self):
        """Test context validation."""
        # Valid context
        valid_context = LLMSafeEventContext(user_id="user123", channel_id="channel456")
        assert valid_context.is_valid() is True

        # Context with minimal info is still valid
        minimal_context = LLMSafeEventContext()
        assert minimal_context.is_valid() is True

    def test_context_get_safe_channel(self):
        """Test getting safe channel ID."""
        # With channel ID
        context_with_channel = LLMSafeEventContext(channel_id="channel123")
        assert context_with_channel.get_safe_channel() == "channel123"

        # Without channel ID
        context_without_channel = LLMSafeEventContext()
        assert context_without_channel.get_safe_channel() == "general"

    def test_context_get_safe_user(self):
        """Test getting safe user ID."""
        # With user ID
        context_with_user = LLMSafeEventContext(user_id="user123")
        assert context_with_user.get_safe_user() == "user123"

        # Without user ID
        context_without_user = LLMSafeEventContext()
        assert context_without_user.get_safe_user() == "system"

    def test_context_add_metadata(self):
        """Test adding metadata to context."""
        context = LLMSafeEventContext()

        context.add_metadata("key1", "value1")
        context.add_metadata("key2", {"nested": "value"})

        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2") == {"nested": "value"}

    def test_context_merge_metadata(self):
        """Test merging metadata into context."""
        context = LLMSafeEventContext(metadata={"existing": "value"})

        new_metadata = {"new_key": "new_value", "another": "data"}
        context.merge_metadata(new_metadata)

        assert context.get_metadata("existing") == "value"
        assert context.get_metadata("new_key") == "new_value"
        assert context.get_metadata("another") == "data"

    def test_context_immutability_protection(self):
        """Test that context provides protection against accidental mutation."""
        context = LLMSafeEventContext(user_id="user123", metadata={"key": "value"})

        # Getting metadata should return copy to prevent mutation
        metadata = context.get_all_metadata()
        metadata["new_key"] = "new_value"

        # Original context should not be affected
        assert "new_key" not in context.metadata


class TestEventContextFactory:
    """Test event context factory functions."""

    def test_create_context_from_event_data(self):
        """Test creating context from event data."""
        from common.enhanced_event_context import create_context_from_event_data

        event_data = {
            "user_id": "user123",
            "channel_id": "channel456",
            "original_request": "Set timer",
        }

        context = create_context_from_event_data(event_data, source="timer")

        assert context.user_id == "user123"
        assert context.channel_id == "channel456"
        assert context.source == "timer"
        assert context.get_metadata("original_request") == "Set timer"

    def test_create_context_from_minimal_data(self):
        """Test creating context from minimal event data."""
        from common.enhanced_event_context import create_context_from_event_data

        # Minimal data
        event_data = {}
        context = create_context_from_event_data(event_data, source="test")

        assert context.user_id is None
        assert context.channel_id is None
        assert context.source == "test"
        assert isinstance(context.timestamp, float)

    def test_create_context_from_list_data(self):
        """Test creating context from list event data (common timer case)."""
        from common.enhanced_event_context import create_context_from_event_data

        # List data like ['timer_123', 'original_request']
        event_data = ["timer_123", "Set timer for 5 minutes"]
        context = create_context_from_event_data(event_data, source="timer")

        assert context.source == "timer"
        assert context.get_metadata("timer_id") == "timer_123"
        assert context.get_metadata("original_request") == "Set timer for 5 minutes"

    def test_create_context_from_string_data(self):
        """Test creating context from string event data."""
        from common.enhanced_event_context import create_context_from_event_data

        event_data = "simple_string_data"
        context = create_context_from_event_data(event_data, source="test")

        assert context.source == "test"
        assert context.get_metadata("raw_data") == "simple_string_data"


class TestEventContextIntegration:
    """Test event context integration with existing system."""

    def test_context_backward_compatibility(self):
        """Test that enhanced context maintains backward compatibility."""
        # The enhanced context should work with existing event handlers
        context = LLMSafeEventContext(user_id="user123", channel_id="channel456")

        # Should have attributes that existing handlers expect
        assert hasattr(context, "user_id")
        assert hasattr(context, "channel_id")
        assert hasattr(context, "timestamp")

    def test_context_serialization(self):
        """Test context can be serialized for debugging."""
        context = LLMSafeEventContext(
            user_id="user123", channel_id="channel456", metadata={"key": "value"}
        )

        serialized = context.to_dict()

        # Should be JSON-serializable
        import json

        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)

        # Should contain all expected fields
        deserialized = json.loads(json_str)
        assert deserialized["user_id"] == "user123"
        assert deserialized["channel_id"] == "channel456"
        assert deserialized["metadata"]["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__])
