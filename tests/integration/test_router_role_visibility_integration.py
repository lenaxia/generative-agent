"""Integration test to reproduce the router role visibility issue.

This test uses the actual role registry and request router to reproduce
the issue where the router is not seeing all fast-reply roles.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory
from llm_provider.request_router import RequestRouter
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRouterRoleVisibilityIntegration:
    """Integration test for router role visibility issue."""

    def setup_method(self):
        """Set up test fixtures with real role registry."""
        # Use the actual roles directory
        self.roles_dir = Path("roles")

        # Create role registry with actual roles
        self.role_registry = RoleRegistry(str(self.roles_dir))

        # Create mock dependencies
        self.mock_llm_factory = Mock(spec=LLMFactory)
        self.mock_universal_agent = Mock(spec=UniversalAgent)

    def test_role_registry_loads_all_fast_reply_roles(self):
        """Test that role registry correctly loads all fast-reply roles from actual files."""
        # Get fast-reply roles from actual role registry
        fast_reply_roles = self.role_registry.get_fast_reply_roles()

        # Verify we have the expected fast-reply roles
        role_names = [role.name for role in fast_reply_roles]
        print(f"Found fast-reply roles: {role_names}")

        # Based on the logs, we should have these 4 roles
        expected_roles = {"weather", "calendar", "timer", "smart_home"}
        actual_roles = set(role_names)

        assert (
            len(fast_reply_roles) >= 4
        ), f"Expected at least 4 fast-reply roles, got {len(fast_reply_roles)}"
        assert expected_roles.issubset(
            actual_roles
        ), f"Missing expected roles. Expected: {expected_roles}, Got: {actual_roles}"

    def test_weather_role_is_properly_configured(self):
        """Test that weather role is properly configured as fast-reply."""
        weather_role = self.role_registry.get_role("weather")
        assert weather_role is not None, "Weather role not found in registry"

        # Check the configuration structure
        role_config = weather_role.config.get("role", {})
        assert (
            role_config.get("fast_reply", False) is True
        ), f"Weather role not marked as fast_reply: {role_config}"

        # Verify it's detected as fast-reply
        assert self.role_registry.is_fast_reply_role(
            "weather"
        ), "Weather role not detected as fast-reply"

    def test_request_router_gets_all_roles_from_registry(self):
        """Test that RequestRouter receives all fast-reply roles from the registry."""
        # Create request router with real role registry
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Mock the universal agent to capture the prompt
        captured_prompts = []

        def capture_prompt(instruction, role, llm_type):
            captured_prompts.append(instruction)
            return '{"route": "weather", "confidence": 0.9, "parameters": {}}'

        self.mock_universal_agent.execute_task.side_effect = capture_prompt

        # Route a weather request
        result = router.route_request("What's the weather in Seattle?")

        # Verify the prompt was captured
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        print(f"Generated prompt:\n{prompt}")

        # Verify all expected roles are in the prompt
        assert "weather" in prompt, "Weather role missing from routing prompt"
        assert "calendar" in prompt, "Calendar role missing from routing prompt"
        assert "timer" in prompt, "Timer role missing from routing prompt"
        assert "smart_home" in prompt, "Smart home role missing from routing prompt"

    def test_router_prompt_generation_with_actual_roles(self):
        """Test the actual prompt generation with real role data."""
        # Get actual fast-reply roles
        fast_reply_roles = self.role_registry.get_fast_reply_roles()

        # Create request router
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Generate the actual prompt
        prompt = router._build_enhanced_routing_prompt(
            "What's the weather in Seattle?", fast_reply_roles
        )

        print(f"Generated routing prompt:\n{prompt}")

        # Verify the prompt contains all roles
        role_names = [role.name for role in fast_reply_roles]
        for role_name in role_names:
            assert role_name in prompt, f"Role {role_name} not found in prompt"

        # Verify the prompt structure
        assert "Route this request to the best role" in prompt
        assert "What's the weather in Seattle?" in prompt
        assert "Available roles and their parameters:" in prompt

    def test_role_registry_caching_behavior(self):
        """Test that role registry caching doesn't affect role visibility."""
        # First call to get_fast_reply_roles (should populate cache)
        first_call = self.role_registry.get_fast_reply_roles()
        first_names = [role.name for role in first_call]

        # Second call (should use cache)
        second_call = self.role_registry.get_fast_reply_roles()
        second_names = [role.name for role in second_call]

        # Verify both calls return the same roles
        assert set(first_names) == set(
            second_names
        ), "Caching changed the available roles"
        assert len(first_call) == len(
            second_call
        ), "Caching changed the number of roles"

    def test_router_with_mock_llm_response_showing_only_calendar(self):
        """Test router behavior when LLM response indicates only calendar is available."""
        # Create router with real role registry
        router = RequestRouter(
            llm_factory=self.mock_llm_factory,
            role_registry=self.role_registry,
            universal_agent=self.mock_universal_agent,
        )

        # Mock LLM response that mimics the bug (only seeing calendar)
        bug_response = '{"route": "PLANNING", "confidence": 0.1, "parameters": {}}'

        self.mock_universal_agent.execute_task.return_value = bug_response

        # Route weather request
        result = router.route_request("What's the weather in Seattle?")

        # This should reproduce the bug
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.1

        # But let's also verify that the role registry actually has all roles
        fast_reply_roles = self.role_registry.get_fast_reply_roles()
        role_names = [role.name for role in fast_reply_roles]
        assert "weather" in role_names, "Weather role should be available in registry"
        assert "calendar" in role_names, "Calendar role should be available in registry"

    def test_debug_role_parameter_extraction(self):
        """Debug test to see what role parameters are being extracted."""
        # Get all fast-reply roles
        fast_reply_roles = self.role_registry.get_fast_reply_roles()

        print("\nDebugging role parameter extraction:")
        for role in fast_reply_roles:
            role_name = role.name
            try:
                parameters = self.role_registry.get_role_parameters(role_name)
                print(f"Role: {role_name}")
                print(f"  Config: {role.config}")
                print(f"  Parameters: {parameters}")
                print(
                    f"  Fast-reply: {role.config.get('role', {}).get('fast_reply', False)}"
                )
                print()
            except Exception as e:
                print(f"Error getting parameters for {role_name}: {e}")
