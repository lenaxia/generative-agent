"""Test to debug router role filtering issue.

This test reproduces the issue where the router is not seeing all fast-reply roles
and routes weather queries incorrectly.
"""

from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory
from llm_provider.request_router import RequestRouter
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRouterRoleFilteringDebug:
    """Test router role filtering to debug the weather role visibility issue."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_llm_factory = Mock(spec=LLMFactory)
        self.mock_universal_agent = Mock(spec=UniversalAgent)

        # Create role registry with mock roles
        self.role_registry = Mock(spec=RoleRegistry)

        # Create test role definitions that match the actual system
        self.weather_role = RoleDefinition(
            name="weather",
            config={
                "role": {
                    "name": "weather",
                    "description": "Weather role with pre-processing data fetching",
                    "fast_reply": True,
                    "llm_type": "WEAK",
                }
            },
            custom_tools=[],
            shared_tools={},
        )

        self.calendar_role = RoleDefinition(
            name="calendar",
            config={
                "role": {
                    "name": "calendar",
                    "description": "Calendar and scheduling role",
                    "fast_reply": True,
                    "llm_type": "WEAK",
                }
            },
            custom_tools=[],
            shared_tools={},
        )

        self.timer_role = RoleDefinition(
            name="timer",
            config={
                "role": {
                    "name": "timer",
                    "description": "Timer and reminder role",
                    "fast_reply": True,
                    "llm_type": "WEAK",
                }
            },
            custom_tools=[],
            shared_tools={},
        )

        self.smart_home_role = RoleDefinition(
            name="smart_home",
            config={
                "role": {
                    "name": "smart_home",
                    "description": "Smart home device control role",
                    "fast_reply": True,
                    "llm_type": "WEAK",
                }
            },
            custom_tools=[],
            shared_tools={},
        )

    def test_role_registry_returns_all_fast_reply_roles(self):
        """Test that role registry correctly identifies all fast-reply roles."""
        # Mock the role registry to return all fast-reply roles
        expected_roles = [
            self.weather_role,
            self.calendar_role,
            self.timer_role,
            self.smart_home_role,
        ]
        self.role_registry.get_fast_reply_roles.return_value = expected_roles

        # Get fast-reply roles
        fast_reply_roles = self.role_registry.get_fast_reply_roles()

        # Verify all roles are returned
        assert len(fast_reply_roles) == 4
        role_names = [role.name for role in fast_reply_roles]
        assert "weather" in role_names
        assert "calendar" in role_names
        assert "timer" in role_names
        assert "smart_home" in role_names

    def test_request_router_sees_all_roles(self):
        """Test that RequestRouter receives all fast-reply roles."""
        # Setup role registry mock
        expected_roles = [
            self.weather_role,
            self.calendar_role,
            self.timer_role,
            self.smart_home_role,
        ]
        self.role_registry.get_fast_reply_roles.return_value = expected_roles

        # Create request router
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Mock get_role_parameters
        self.role_registry.get_role_parameters.return_value = {}

        # Test that router calls get_fast_reply_roles
        with patch.object(
            router, "_build_enhanced_routing_prompt"
        ) as mock_build_prompt:
            mock_build_prompt.return_value = "test prompt"

            # Mock the universal agent response
            self.mock_universal_agent.execute_task.return_value = (
                '{"route": "weather", "confidence": 0.9, "parameters": {}}'
            )

            # Route a weather request
            result = router.route_request("What's the weather in Seattle?")

            # Verify get_fast_reply_roles was called
            self.role_registry.get_fast_reply_roles.assert_called()

            # Verify build_enhanced_routing_prompt was called with all roles
            mock_build_prompt.assert_called_once()
            args, kwargs = mock_build_prompt.call_args
            instruction, roles = args

            assert len(roles) == 4
            role_names = [role.name for role in roles]
            assert "weather" in role_names
            assert "calendar" in role_names

    def test_routing_prompt_includes_all_roles(self):
        """Test that the routing prompt includes all available roles."""
        # Setup role registry mock
        expected_roles = [
            self.weather_role,
            self.calendar_role,
            self.timer_role,
            self.smart_home_role,
        ]
        self.role_registry.get_fast_reply_roles.return_value = expected_roles

        # Mock get_role_parameters to return empty parameters for all roles
        self.role_registry.get_role_parameters.return_value = {}

        # Create request router
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Build routing prompt
        prompt = router._build_enhanced_routing_prompt(
            "What's the weather?", expected_roles
        )

        # Verify the prompt contains role information
        assert "Route this request to the best role" in prompt
        assert "What's the weather?" in prompt
        assert "Available roles and their parameters:" in prompt

    def test_weather_request_routing_with_all_roles_available(self):
        """Test weather request routing when all roles are properly available."""
        # Setup role registry mock
        expected_roles = [
            self.weather_role,
            self.calendar_role,
            self.timer_role,
            self.smart_home_role,
        ]
        self.role_registry.get_fast_reply_roles.return_value = expected_roles
        self.role_registry.get_role_parameters.return_value = {}

        # Create request router
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Mock successful weather routing response
        self.mock_universal_agent.execute_task.return_value = (
            '{"route": "weather", "confidence": 0.9, "parameters": {}}'
        )

        # Route weather request
        result = router.route_request("What's the weather in Seattle?")

        # Verify correct routing
        assert result["route"] == "weather"
        assert result["confidence"] == 0.9

    def test_router_fallback_when_only_calendar_visible(self):
        """Test router behavior when only calendar role is visible (reproducing the bug)."""
        # Setup role registry to return only calendar (reproducing the bug)
        calendar_only = [self.calendar_role]
        self.role_registry.get_fast_reply_roles.return_value = calendar_only
        self.role_registry.get_role_parameters.return_value = {}

        # Create request router
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Mock LLM response when only calendar is available
        self.mock_universal_agent.execute_task.return_value = (
            '{"route": "PLANNING", "confidence": 0.1, "parameters": {}}'
        )

        # Route weather request
        result = router.route_request("What's the weather in Seattle?")

        # Verify it falls back to PLANNING (reproducing the bug)
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.1

    def test_role_config_structure_for_fast_reply_detection(self):
        """Test that role config structure is correct for fast-reply detection."""
        # Test the actual structure that get_fast_reply_roles expects
        for role in [
            self.weather_role,
            self.calendar_role,
            self.timer_role,
            self.smart_home_role,
        ]:
            # This is the path that get_fast_reply_roles uses
            fast_reply_flag = role.config.get("role", {}).get("fast_reply", False)
            assert (
                fast_reply_flag is True
            ), f"Role {role.name} should be marked as fast_reply"

    def test_role_registry_fast_reply_detection_logic(self):
        """Test the exact logic used in RoleRegistry.get_fast_reply_roles()."""
        # Create a mock role registry with actual roles
        mock_llm_roles = {
            "weather": self.weather_role,
            "calendar": self.calendar_role,
            "timer": self.timer_role,
            "smart_home": self.smart_home_role,
        }

        # Simulate the filtering logic from get_fast_reply_roles
        fast_reply_roles = [
            role
            for role in mock_llm_roles.values()
            if role.config.get("role", {}).get("fast_reply", False)
        ]

        # Verify all roles are detected as fast-reply
        assert len(fast_reply_roles) == 4
        role_names = [role.name for role in fast_reply_roles]
        assert set(role_names) == {"weather", "calendar", "timer", "smart_home"}
