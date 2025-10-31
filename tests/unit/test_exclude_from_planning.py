"""Tests for exclude_from_planning role configuration parameter.

This test suite validates that roles can be excluded from planning workflows
using the exclude_from_planning configuration parameter.
"""

import pytest


class TestExcludeFromPlanning:
    """Test suite for exclude_from_planning functionality."""

    def test_router_excluded_from_planning(self):
        """Test that router role is excluded from planning."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        router = registry.roles.get("router")

        assert router is not None
        config = router.config.get("role", {})
        assert config.get("exclude_from_planning") is True

    def test_planning_excluded_from_planning(self):
        """Test that planning role is excluded from planning."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        planning = registry.roles.get("planning")

        assert planning is not None
        config = planning.config.get("role", {})
        assert config.get("exclude_from_planning") is True

    def test_other_roles_not_excluded(self):
        """Test that other roles are not excluded from planning."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")

        # These roles should be available for planning
        roles_to_check = [
            "timer",
            "weather",
            "search",
            "conversation",
            "summarizer",
            "calendar",
            "smart_home",
        ]

        for role_name in roles_to_check:
            role = registry.roles.get(role_name)
            assert role is not None, f"Role {role_name} not found"

            config = role.config.get("role", {})
            excluded = config.get("exclude_from_planning", False)
            assert (
                excluded is False
            ), f"Role {role_name} should not be excluded from planning"

    def test_planning_role_filters_correctly(self):
        """Test that planning role's pre-processing filters roles correctly."""
        from unittest.mock import Mock

        from llm_provider.role_registry import RoleRegistry
        from roles.core_planning import load_available_roles

        registry = RoleRegistry("roles")

        # Create mock context
        context = Mock()
        instruction = "Test instruction"
        parameters = {}

        # Call the pre-processing function
        result = load_available_roles(instruction, context, parameters)

        # Check that available_roles text is returned
        assert "available_roles" in result
        available_roles_text = result["available_roles"]

        # Verify router and planning are NOT in the available roles
        assert "**router**" not in available_roles_text.lower()
        assert "**planning**" not in available_roles_text.lower()

        # Verify other roles ARE in the available roles
        assert "**timer**" in available_roles_text.lower()
        assert "**weather**" in available_roles_text.lower()
        assert "**summarizer**" in available_roles_text.lower()

    def test_exclude_from_planning_defaults_to_false(self):
        """Test that exclude_from_planning defaults to False when not specified."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")

        # Timer role doesn't explicitly set exclude_from_planning
        timer = registry.roles.get("timer")
        assert timer is not None

        config = timer.config.get("role", {})
        # Should default to False
        assert config.get("exclude_from_planning", False) is False

    def test_available_roles_count(self):
        """Test that the correct number of roles are available for planning."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")

        # Count roles that should be available for planning
        available_count = sum(
            1
            for role_def in registry.roles.values()
            if not role_def.config.get("role", {}).get("exclude_from_planning", False)
        )

        # Should have 7 roles available (all except router and planning)
        # calendar, conversation, search, smart_home, summarizer, timer, weather
        assert available_count == 7

    def test_excluded_roles_count(self):
        """Test that the correct number of roles are excluded from planning."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")

        # Count roles that should be excluded from planning
        excluded_count = sum(
            1
            for role_def in registry.roles.values()
            if role_def.config.get("role", {}).get("exclude_from_planning", False)
        )

        # Should have 2 roles excluded (router and planning)
        assert excluded_count == 2
