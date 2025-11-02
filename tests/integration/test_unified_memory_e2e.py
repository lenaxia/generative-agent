"""End-to-end integration tests for unified memory system.

These tests verify the unified memory system components work together correctly.
"""

import time
from unittest.mock import MagicMock

import pytest

from common.intent_processor import IntentProcessor
from common.intents import MemoryWriteIntent
from common.providers.universal_memory_provider import UniversalMemoryProvider


class TestMemorySystemIntegration:
    """Test unified memory system integration."""

    def test_memory_ttl_calculation(self):
        """Test TTL is calculated correctly based on importance."""
        provider = UniversalMemoryProvider()

        # Low importance
        low_ttl = provider._calculate_ttl(0.0)
        assert low_ttl == 30 * 24 * 60 * 60  # 30 days

        # High importance
        high_ttl = provider._calculate_ttl(1.0)
        assert high_ttl == 90 * 24 * 60 * 60  # 90 days

        # Medium importance
        med_ttl = provider._calculate_ttl(0.5)
        assert 30 * 24 * 60 * 60 < med_ttl < 90 * 24 * 60 * 60

    def test_role_integration_config(self):
        """Test roles are configured with memory_tools."""
        from roles.core_calendar import register_role as register_calendar
        from roles.core_conversation import register_role as register_conversation
        from roles.core_planning import register_role as register_planning

        # Check conversation role
        conv_reg = register_conversation()
        assert "memory_tools" in conv_reg["config"]["tools"]["shared"]

        # Check calendar role
        cal_reg = register_calendar()
        assert "memory_tools" in cal_reg["config"]["tools"]["shared"]

        # Check planning role
        plan_reg = register_planning()
        assert "memory_tools" in plan_reg["config"]["tools"]["shared"]

    @pytest.mark.asyncio
    async def test_memory_write_intent_processing(self):
        """Test MemoryWriteIntent is processed correctly."""
        from unittest.mock import patch

        with patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.write_memory.return_value = "mem-123"
            mock_provider_class.return_value = mock_provider

            processor = IntentProcessor()

            intent = MemoryWriteIntent(
                user_id="test_user",
                memory_type="conversation",
                content="Test memory",
                source_role="conversation",
                importance=0.7,
            )

            result = await processor.process_intents([intent])

            assert result["processed"] == 1
            assert result["failed"] == 0
            mock_provider.write_memory.assert_called_once()

    def test_memory_data_model_serialization(self):
        """Test UniversalMemory serialization works correctly."""
        from common.providers.universal_memory_provider import UniversalMemory

        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Test content",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.7,
            metadata={"key": "value"},
            tags=["test"],
            related_memories=["mem1"],
        )

        # Serialize
        data = memory.to_dict()
        assert data["id"] == "test-id"
        assert data["memory_type"] == "conversation"

        # Deserialize
        restored = UniversalMemory.from_dict(data)
        assert restored.id == memory.id
        assert restored.content == memory.content
        assert restored.importance == memory.importance

    def test_memory_validation(self):
        """Test MemoryWriteIntent validation."""
        # Valid intent
        valid_intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Test",
            source_role="conversation",
        )
        assert valid_intent.validate() is True

        # Invalid type
        invalid_type = MemoryWriteIntent(
            user_id="user123",
            memory_type="invalid",
            content="Test",
            source_role="conversation",
        )
        assert invalid_type.validate() is False

        # Invalid importance
        invalid_importance = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Test",
            source_role="conversation",
            importance=1.5,
        )
        assert invalid_importance.validate() is False

    def test_lifecycle_functions_exist(self):
        """Test that lifecycle functions are properly defined."""
        from roles.core_calendar import load_calendar_context, save_calendar_event
        from roles.core_conversation import (
            load_conversation_context,
            save_message_to_log,
        )
        from roles.core_planning import load_planning_context, save_planning_result

        # Verify functions are callable
        assert callable(load_conversation_context)
        assert callable(save_message_to_log)
        assert callable(load_calendar_context)
        assert callable(save_calendar_event)
        assert callable(load_planning_context)
        assert callable(save_planning_result)

    def test_memory_tools_exist(self):
        """Test that memory tools are properly defined."""
        from roles.shared_tools.memory_tools import get_recent_memories, search_memory

        # Verify tools are callable
        assert callable(search_memory)
        assert callable(get_recent_memories)

        # Verify they have the @tool decorator
        assert hasattr(search_memory, "__wrapped__")
        assert hasattr(get_recent_memories, "__wrapped__")
