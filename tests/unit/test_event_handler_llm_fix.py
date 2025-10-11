"""Test EventHandlerLLM Agent integration fix.

Tests that EventHandlerLLM correctly uses Strands Agent with no tools
instead of trying to call model.invoke() directly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.event_handler_llm import EventHandlerLLM


class TestEventHandlerLLMFix:
    """Test EventHandlerLLM Agent integration fix."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = MagicMock()
        mock_model = MagicMock()
        factory.create_strands_model.return_value = mock_model
        return factory, mock_model

    @pytest.fixture
    def event_context(self):
        """Create test event context."""
        return {
            "execution_context": {
                "user_id": "U52L1U8M6",
                "channel": "slack:C52L1UK5E",
                "original_request": "set timer 10s",
            },
            "original_request": "set timer 10s",
            "timer_id": "timer_12345",
        }

    @patch("common.event_handler_llm.Agent")
    @pytest.mark.asyncio
    async def test_invoke_uses_agent_correctly(
        self, mock_agent_class, mock_llm_factory, event_context
    ):
        """Test that invoke method creates and uses Agent correctly."""
        factory, mock_model = mock_llm_factory

        # Mock Agent instance and response
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.message = "This is a test response"
        mock_agent.invoke_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Test invoke
        result = await llm.invoke("Test prompt", model_type="WEAK")

        # Verify factory was called correctly
        factory.create_strands_model.assert_called_once()

        # Verify Agent was created with model and no tools
        mock_agent_class.assert_called_once_with(model=mock_model, tools=[])

        # Verify Agent.invoke_async was called with prompt
        mock_agent.invoke_async.assert_called_once_with(
            'Test prompt\nContext: {\n  "user_id": "U52L1U8M6",\n  "channel": "slack:C52L1UK5E",\n  "original_request": "set timer 10s"\n}\n'
        )

        # Verify response
        assert result == "This is a test response"

    @patch("common.event_handler_llm.Agent")
    @pytest.mark.asyncio
    async def test_invoke_with_different_model_types(
        self, mock_agent_class, mock_llm_factory, event_context
    ):
        """Test that invoke works with different model types."""
        factory, mock_model = mock_llm_factory

        # Mock Agent instance and response
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.message = "Strong model response"
        mock_agent.invoke_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Test with STRONG model type
        result = await llm.invoke("Complex prompt", model_type="STRONG")

        # Verify LLMType.STRONG was used
        from llm_provider.factory import LLMType

        factory.create_strands_model.assert_called_with(LLMType.STRONG)

        assert result == "Strong model response"

    @pytest.mark.asyncio
    async def test_invoke_fallback_when_strands_not_available(
        self, mock_llm_factory, event_context
    ):
        """Test fallback behavior when Strands Agent is not available."""
        factory, mock_model = mock_llm_factory

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Mock ImportError for Strands Agent
        with patch(
            "common.event_handler_llm.Agent",
            side_effect=ImportError("Strands not available"),
        ):
            result = await llm.invoke("Test prompt", model_type="WEAK")

        # Should return mock response
        assert (
            result
            == 'Mock response to: Test prompt\nContext: {\n  "user_id": "U52L1U8M6",\n  "channel": "slack:C52L1UK5E",\n  "original_request": "set timer 10s"\n}\n'
        )

    @patch("common.event_handler_llm.Agent")
    @pytest.mark.asyncio
    async def test_parse_json_uses_agent(
        self, mock_agent_class, mock_llm_factory, event_context
    ):
        """Test that parse_json method also uses Agent correctly."""
        factory, mock_model = mock_llm_factory

        # Mock Agent instance and response with JSON
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.message = '{"action": "workflow", "device": "lights"}'
        mock_agent.invoke_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Test parse_json
        result = await llm.parse_json("Parse this action", model_type="WEAK")

        # Verify Agent was used
        mock_agent_class.assert_called_once_with(model=mock_model, tools=[])
        mock_agent.invoke_async.assert_called_once()

        # Verify JSON parsing
        assert result == {"action": "workflow", "device": "lights"}

    @patch("common.event_handler_llm.Agent")
    @pytest.mark.asyncio
    async def test_quick_decision_uses_agent(
        self, mock_agent_class, mock_llm_factory, event_context
    ):
        """Test that quick_decision method uses Agent correctly."""
        factory, mock_model = mock_llm_factory

        # Mock Agent instance and response
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.message = "workflow"
        mock_agent.invoke_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Test quick_decision
        result = await llm.quick_decision(
            "Should this create a workflow?", ["workflow", "notification"]
        )

        # Verify Agent was used with WEAK model (for quick decisions)
        mock_agent_class.assert_called_once_with(model=mock_model, tools=[])

        # Verify response
        assert result == "workflow"

    def test_context_methods_work_correctly(self, mock_llm_factory, event_context):
        """Test that context access methods work correctly."""
        factory, _ = mock_llm_factory

        # Create EventHandlerLLM
        llm = EventHandlerLLM(factory, event_context)

        # Test context access methods
        assert llm.get_user_id() == "U52L1U8M6"
        assert llm.get_channel() == "slack:C52L1UK5E"
        assert llm.get_original_request() == "set timer 10s"
        assert llm.get_timer_id() == "timer_12345"

        # Test get_context with key
        assert llm.get_context("user_id") == "U52L1U8M6"
        assert llm.get_context("channel") == "slack:C52L1UK5E"

        # Test get_context with full context
        full_context = llm.get_context()
        assert full_context["user_id"] == "U52L1U8M6"
        assert full_context["channel"] == "slack:C52L1UK5E"
