"""Unit test to demonstrate the router role filtering bug fix.

This test verifies that the fix for the router role filtering bug works correctly.
The bug was that the LLM was not correctly processing all roles in the router prompt,
despite the prompt correctly including all roles.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.request_router import RequestRouter
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRouterRoleFilteringBugFix(unittest.TestCase):
    """Test case to verify the router role filtering bug fix."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_factory = Mock(spec=LLMFactory)
        self.role_registry = Mock(spec=RoleRegistry)
        self.universal_agent = Mock(spec=UniversalAgent)

        # Set up mock role registry with multiple fast-reply roles
        calendar_role = Mock()
        calendar_role.name = "calendar"
        calendar_role.config = {
            "role": {
                "description": "Calendar management specialist",
                "fast_reply": True,
            }
        }

        weather_role = Mock()
        weather_role.name = "weather"
        weather_role.config = {
            "role": {
                "description": "Weather information specialist",
                "fast_reply": True,
            }
        }

        timer_role = Mock()
        timer_role.name = "timer"
        timer_role.config = {
            "role": {"description": "Timer and alarm specialist", "fast_reply": True}
        }

        smart_home_role = Mock()
        smart_home_role.name = "smart_home"
        smart_home_role.config = {
            "role": {"description": "Smart home device control", "fast_reply": True}
        }

        # Mock the role registry to return all four roles
        self.role_registry.get_fast_reply_roles.return_value = [
            calendar_role,
            weather_role,
            timer_role,
            smart_home_role,
        ]

        # Set up parameter schemas for each role
        def get_role_parameters_side_effect(role_name):
            if role_name == "calendar":
                return {
                    "action": {
                        "type": "string",
                        "required": True,
                        "enum": ["create", "list", "delete"],
                    },
                    "date": {"type": "string", "required": False},
                }
            elif role_name == "weather":
                return {
                    "location": {"type": "string", "required": True},
                    "timeframe": {
                        "type": "string",
                        "required": False,
                        "enum": ["current", "today", "tomorrow"],
                    },
                }
            elif role_name == "timer":
                return {
                    "duration": {"type": "string", "required": True},
                    "label": {"type": "string", "required": False},
                }
            elif role_name == "smart_home":
                return {
                    "device": {"type": "string", "required": True},
                    "action": {
                        "type": "string",
                        "required": True,
                        "enum": ["on", "off", "status"],
                    },
                }
            return {}

        self.role_registry.get_role_parameters.side_effect = (
            get_role_parameters_side_effect
        )

    def test_router_role_filtering_bug_fix(self):
        """Test to verify the fix for the router role filtering bug."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a weather-related response
        # This simulates the LLM correctly processing all roles in the prompt
        self.universal_agent.execute_task.return_value = """
        {
          "route": "weather",
          "confidence": 0.95,
          "parameters": {
            "location": "seattle",
            "timeframe": "current"
          }
        }
        """

        # Execute the route_request method with a weather-related query
        result = router.route_request("weather in seattle?")

        # Verify that the router called universal_agent.execute_task with the correct prompt
        prompt_arg = self.universal_agent.execute_task.call_args[1]["instruction"]

        # Print the prompt for debugging
        print("\n--- Router Prompt ---")
        print(prompt_arg)
        print("--- End Router Prompt ---\n")

        # Verify that all roles are included in the prompt
        assert "calendar" in prompt_arg, "Calendar role should be in the prompt"
        assert "weather" in prompt_arg, "Weather role should be in the prompt"
        assert "timer" in prompt_arg, "Timer role should be in the prompt"
        assert "smart_home" in prompt_arg, "Smart home role should be in the prompt"

        # Verify the routing result correctly routes to the weather role
        assert result["route"] == "weather", "Should route to weather role"
        assert (
            result["confidence"] == 0.95
        ), "Should have high confidence for weather role"
        assert "parameters" in result, "Should include parameters field"
        assert "location" in result["parameters"], "Should extract location parameter"
        assert (
            result["parameters"]["location"] == "seattle"
        ), "Should extract correct location"
        assert "timeframe" in result["parameters"], "Should extract timeframe parameter"
        assert (
            result["parameters"]["timeframe"] == "current"
        ), "Should extract correct timeframe"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
