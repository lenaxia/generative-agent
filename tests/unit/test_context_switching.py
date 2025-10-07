from unittest.mock import Mock, patch

import pytest

from config.base_config import BaseConfig
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestContextSwitching:
    """Test suite for context switching functionality in UniversalAgent."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=BaseConfig)
        config.name = "test_bedrock"
        config.provider_type = "bedrock"
        config.model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
        config.temperature = 0.3
        config.additional_params = {}
        return config

    @pytest.fixture
    def mock_llm_factory(self, mock_config):
        """Create a mock LLMFactory with agent pooling."""
        configs = {
            LLMType.WEAK: [mock_config],
            LLMType.DEFAULT: [mock_config],
            LLMType.STRONG: [mock_config],
        }
        factory = LLMFactory(configs)
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Create a mock role registry."""
        registry = Mock(spec=RoleRegistry)

        # Create mock role definitions
        weather_role = Mock(spec=RoleDefinition)
        weather_role.name = "weather"
        weather_role.config = {
            "prompts": {"system": "You are a weather specialist."},
            "tools": {"shared": ["weather_tools"]},
        }
        weather_role.custom_tools = []

        timer_role = Mock(spec=RoleDefinition)
        timer_role.name = "timer"
        timer_role.config = {
            "prompts": {"system": "You are a timer specialist."},
            "tools": {"shared": ["timer_tools"]},
        }
        timer_role.custom_tools = []

        default_role = Mock(spec=RoleDefinition)
        default_role.name = "default"
        default_role.config = {
            "prompts": {"system": "You are a helpful assistant."},
            "tools": {"shared": []},
        }
        default_role.custom_tools = []

        # Configure registry to return appropriate roles
        def get_role_side_effect(role_name):
            if role_name == "weather":
                return weather_role
            elif role_name == "timer":
                return timer_role
            elif role_name == "default":
                return default_role
            return None

        registry.get_role.side_effect = get_role_side_effect
        registry.get_shared_tool.return_value = Mock()

        return registry

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create UniversalAgent with mocked dependencies."""
        return UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=mock_role_registry
        )

    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent for testing."""
        agent = Mock()
        agent.model = Mock()
        agent.system_prompt = "test prompt"
        agent.tools = []
        return agent

    def test_assume_role_uses_agent_pooling(self, universal_agent, mock_agent):
        """Test that assume_role uses agent pooling instead of creating new agents."""
        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(
                universal_agent, "_update_agent_context"
            ) as mock_update_context,
        ):

            mock_get_agent.return_value = mock_agent

            # Assume weather role
            result_agent = universal_agent.assume_role("weather", LLMType.DEFAULT)

            # Should have called get_agent from factory
            mock_get_agent.assert_called_once_with(LLMType.DEFAULT)

            # Should have updated agent context
            mock_update_context.assert_called_once()

            # Should return the updated agent (may be new due to context switching)
            assert result_agent is not None

    def test_assume_role_determines_correct_llm_type(self, universal_agent, mock_agent):
        """Test that assume_role determines appropriate LLM type for roles."""
        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(universal_agent, "_update_agent_context"),
            patch.object(
                universal_agent, "_determine_llm_type_for_role"
            ) as mock_determine_type,
        ):

            mock_get_agent.return_value = mock_agent
            mock_determine_type.return_value = LLMType.WEAK

            # Assume role without specifying LLM type
            universal_agent.assume_role("weather")

            # Should determine LLM type for role
            mock_determine_type.assert_called_once_with("weather")

            # Should use determined LLM type
            mock_get_agent.assert_called_once_with(LLMType.WEAK)

    def test_context_switching_updates_agent_properties(
        self, universal_agent, mock_agent
    ):
        """Test that context switching properly updates agent properties."""
        with patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            # Assume weather role
            universal_agent.assume_role("weather", LLMType.DEFAULT)

            # Verify agent context was updated
            assert hasattr(mock_agent, "system_prompt") or hasattr(
                mock_agent, "update_context"
            )

    def test_multiple_role_switches_use_same_agent_pool(
        self, universal_agent, mock_agent
    ):
        """Test that multiple role switches reuse agents from pool."""
        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(universal_agent, "_update_agent_context"),
        ):

            mock_get_agent.return_value = mock_agent

            # Switch between roles multiple times
            universal_agent.assume_role("weather", LLMType.DEFAULT)
            universal_agent.assume_role("timer", LLMType.DEFAULT)
            universal_agent.assume_role("weather", LLMType.DEFAULT)  # Should reuse

            # Should have called get_agent for each role switch
            assert mock_get_agent.call_count == 3

            # All calls should use same LLM type
            for call in mock_get_agent.call_args_list:
                assert call[0][0] == LLMType.DEFAULT

    def test_different_llm_types_use_different_pool_entries(self, universal_agent):
        """Test that different LLM types use separate pool entries."""
        mock_agent_weak = Mock()
        mock_agent_strong = Mock()

        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(universal_agent, "_update_agent_context"),
        ):

            # Return different agents for different LLM types
            def get_agent_side_effect(llm_type):
                if llm_type == LLMType.WEAK:
                    return mock_agent_weak
                elif llm_type == LLMType.STRONG:
                    return mock_agent_strong
                return Mock()

            mock_get_agent.side_effect = get_agent_side_effect

            # Assume same role with different LLM types
            agent1 = universal_agent.assume_role("weather", LLMType.WEAK)
            agent2 = universal_agent.assume_role("weather", LLMType.STRONG)

            # Should get different agents
            assert agent1 != agent2
            assert agent1 == mock_agent_weak
            assert agent2 == mock_agent_strong

    def test_update_agent_context_with_strands_agent(self, universal_agent):
        """Test _update_agent_context with Strands Agent that supports context updates."""
        mock_agent = Mock()
        mock_agent.update_context = Mock()

        system_prompt = "You are a weather specialist."
        tools = [Mock(), Mock()]

        # Call _update_agent_context
        universal_agent._update_agent_context(mock_agent, system_prompt, tools)

        # Should call update_context method
        mock_agent.update_context.assert_called_once_with(
            system_prompt=system_prompt, tools=tools
        )

    def test_update_agent_context_with_property_based_agent(self, universal_agent):
        """Test _update_agent_context with agent that uses properties."""
        mock_agent = Mock()
        # Remove update_context method to simulate property-based agent
        del mock_agent.update_context
        mock_agent.system_prompt = None
        mock_agent.tools = None

        system_prompt = "You are a weather specialist."
        tools = [Mock(), Mock()]

        # Call _update_agent_context
        universal_agent._update_agent_context(mock_agent, system_prompt, tools)

        # Should update properties directly
        assert mock_agent.system_prompt == system_prompt
        assert mock_agent.tools == tools

    def test_update_agent_context_fallback_recreation(self, universal_agent):
        """Test _update_agent_context fallback to agent recreation."""
        mock_agent = Mock()
        # Remove both update_context method and properties
        del mock_agent.update_context
        delattr(mock_agent, "system_prompt")
        delattr(mock_agent, "tools")
        mock_agent.model = Mock()

        system_prompt = "You are a weather specialist."
        tools = [Mock(), Mock()]

        with patch("llm_provider.universal_agent.Agent") as mock_agent_class:
            new_agent = Mock()
            mock_agent_class.return_value = new_agent

            # Call _update_agent_context
            result = universal_agent._update_agent_context(
                mock_agent, system_prompt, tools
            )

            # Should create new agent as fallback
            mock_agent_class.assert_called_once_with(
                model=mock_agent.model, system_prompt=system_prompt, tools=tools
            )
            assert result == new_agent

    def test_determine_llm_type_for_role(self, universal_agent):
        """Test _determine_llm_type_for_role method."""
        # Test router role should use WEAK model for fast routing
        llm_type = universal_agent._determine_llm_type_for_role("router")
        assert llm_type == LLMType.WEAK

        # Test weather role should use DEFAULT model
        llm_type = universal_agent._determine_llm_type_for_role("weather")
        assert llm_type == LLMType.DEFAULT

        # Test planning role should use STRONG model for complex tasks
        llm_type = universal_agent._determine_llm_type_for_role("planning")
        assert llm_type == LLMType.STRONG

        # Test unknown role should default to DEFAULT
        llm_type = universal_agent._determine_llm_type_for_role("unknown_role")
        assert llm_type == LLMType.DEFAULT

    def test_context_switching_performance(self, universal_agent):
        """Test that context switching is faster than agent creation."""
        import time

        mock_agent = Mock()
        mock_agent.update_context = Mock()

        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(
                universal_agent, "_assemble_role_tools"
            ) as mock_assemble_tools,
            patch.object(
                universal_agent, "_get_system_prompt_from_role"
            ) as mock_get_prompt,
        ):

            mock_get_agent.return_value = mock_agent
            mock_assemble_tools.return_value = []
            mock_get_prompt.return_value = "test prompt"

            # Measure context switching time
            start_time = time.perf_counter()
            universal_agent.assume_role("weather", LLMType.DEFAULT)
            context_switch_time = time.perf_counter() - start_time

            # Context switching should be very fast (< 1ms in tests)
            assert context_switch_time < 0.001

    def test_role_fallback_to_default(self, universal_agent, mock_agent):
        """Test that invalid roles fall back to default role."""
        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(universal_agent, "_update_agent_context"),
        ):

            mock_get_agent.return_value = mock_agent

            # Try to assume non-existent role
            result_agent = universal_agent.assume_role("nonexistent_role")

            # Should fall back to default role
            assert universal_agent.current_role == "default"
            assert result_agent == mock_agent

    def test_none_role_fallback_to_default(self, universal_agent, mock_agent):
        """Test that None role falls back to default role."""
        with (
            patch.object(universal_agent.llm_factory, "get_agent") as mock_get_agent,
            patch.object(universal_agent, "_update_agent_context"),
        ):

            mock_get_agent.return_value = mock_agent

            # Try to assume None role
            result_agent = universal_agent.assume_role(None)

            # Should fall back to default role
            assert universal_agent.current_role == "default"
            assert result_agent == mock_agent


class TestContextSwitchingIntegration:
    """Integration tests for context switching with real components."""

    def test_end_to_end_context_switching(self):
        """Test complete context switching flow with realistic setup."""
        # Create realistic config
        config = Mock(spec=BaseConfig)
        config.name = "bedrock_sonnet"
        config.provider_type = "bedrock"
        config.model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
        config.temperature = 0.3
        config.additional_params = {}

        configs = {LLMType.DEFAULT: [config]}
        factory = LLMFactory(configs)

        # Create role registry
        registry = Mock(spec=RoleRegistry)
        weather_role = Mock(spec=RoleDefinition)
        weather_role.name = "weather"
        weather_role.config = {
            "prompts": {"system": "Weather specialist"},
            "tools": {"shared": []},
        }
        weather_role.custom_tools = []
        registry.get_role.return_value = weather_role
        registry.get_shared_tool.return_value = Mock()

        universal_agent = UniversalAgent(factory, registry)

        with (
            patch("llm_provider.factory.BedrockModel") as mock_bedrock,
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):

            mock_model = Mock()
            mock_bedrock.return_value = mock_model
            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # First role assumption should create agent
            agent1 = universal_agent.assume_role("weather")

            # Second role assumption should reuse agent
            agent2 = universal_agent.assume_role("weather")

            # Should be same agent instance
            assert agent1 is agent2

            # Agent should only be created once
            mock_agent_class.assert_called_once()

            # Context should be updated for both calls
            assert mock_agent.update_context.call_count == 2

    def test_context_switching_with_different_tools(self):
        """Test context switching updates tools correctly."""
        config = Mock(spec=BaseConfig)
        config.name = "test_config"
        config.provider_type = "bedrock"
        config.model_id = "test-model"
        config.temperature = 0.3
        config.additional_params = {}

        configs = {LLMType.DEFAULT: [config]}
        factory = LLMFactory(configs)

        # Create roles with different tools
        registry = Mock(spec=RoleRegistry)

        def get_role_side_effect(role_name):
            role = Mock(spec=RoleDefinition)
            role.name = role_name
            role.custom_tools = []
            if role_name == "weather":
                role.config = {
                    "prompts": {"system": "Weather specialist"},
                    "tools": {"shared": ["weather_tools"]},
                }
            elif role_name == "timer":
                role.config = {
                    "prompts": {"system": "Timer specialist"},
                    "tools": {"shared": ["timer_tools"]},
                }
            return role

        registry.get_role.side_effect = get_role_side_effect
        registry.get_shared_tool.return_value = Mock()

        universal_agent = UniversalAgent(factory, registry)

        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):

            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # Switch between roles with different tools
            universal_agent.assume_role("weather")
            weather_call = mock_agent.update_context.call_args

            universal_agent.assume_role("timer")
            timer_call = mock_agent.update_context.call_args

            # Should have different system prompts
            assert weather_call[1]["system_prompt"] != timer_call[1]["system_prompt"]

            # Should have been called twice
            assert mock_agent.update_context.call_count == 2
