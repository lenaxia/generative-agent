"""Unit test to verify the implementation of the router role filtering fix.

This test verifies that the updated router correctly processes all roles and
routes requests to the appropriate role.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.request_router_updated import RequestRouter
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRouterRoleFilteringFixImplementation(unittest.TestCase):
    """Test case to verify the implementation of the router role filtering fix."""

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

    def test_weather_request_routing(self):
        """Test that a weather-related request is correctly routed to the weather role."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a weather-related response
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

        # Verify that the router called universal_agent.execute_task with the correct parameters
        self.universal_agent.execute_task.assert_called_once()
        call_args = self.universal_agent.execute_task.call_args
        self.assertEqual(call_args[1]["role"], "router")
        self.assertEqual(call_args[1]["llm_type"], LLMType.STRONG)

        # Verify the routing result correctly routes to the weather role
        self.assertEqual(result["route"], "weather")
        self.assertEqual(result["confidence"], 0.95)
        self.assertIn("parameters", result)
        self.assertIn("location", result["parameters"])
        self.assertEqual(result["parameters"]["location"], "seattle")
        self.assertIn("timeframe", result["parameters"])
        self.assertEqual(result["parameters"]["timeframe"], "current")

        # Verify that the prompt includes all roles
        prompt = call_args[1]["instruction"]
        self.assertIn("calendar", prompt)
        self.assertIn("weather", prompt)
        self.assertIn("timer", prompt)
        self.assertIn("smart_home", prompt)

    def test_timer_request_routing(self):
        """Test that a timer-related request is correctly routed to the timer role."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a timer-related response
        self.universal_agent.execute_task.return_value = """
        {
          "route": "timer",
          "confidence": 0.9,
          "parameters": {
            "duration": "5 minutes",
            "label": "coffee"
          }
        }
        """

        # Execute the route_request method with a timer-related query
        result = router.route_request("set a timer for 5 minutes for coffee")

        # Verify that the router called universal_agent.execute_task with the correct parameters
        self.universal_agent.execute_task.assert_called_once()
        call_args = self.universal_agent.execute_task.call_args
        self.assertEqual(call_args[1]["role"], "router")
        self.assertEqual(call_args[1]["llm_type"], LLMType.STRONG)

        # Verify the routing result correctly routes to the timer role
        self.assertEqual(result["route"], "timer")
        self.assertEqual(result["confidence"], 0.9)
        self.assertIn("parameters", result)
        self.assertIn("duration", result["parameters"])
        self.assertEqual(result["parameters"]["duration"], "5 minutes")
        self.assertIn("label", result["parameters"])
        self.assertEqual(result["parameters"]["label"], "coffee")

        # Verify that the prompt includes all roles
        prompt = call_args[1]["instruction"]
        self.assertIn("calendar", prompt)
        self.assertIn("weather", prompt)
        self.assertIn("timer", prompt)
        self.assertIn("smart_home", prompt)

    def test_smart_home_request_routing(self):
        """Test that a smart home-related request is correctly routed to the smart_home role."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a smart home-related response
        self.universal_agent.execute_task.return_value = """
        {
          "route": "smart_home",
          "confidence": 0.85,
          "parameters": {
            "device": "lights",
            "action": "on"
          }
        }
        """

        # Execute the route_request method with a smart home-related query
        result = router.route_request("turn on the lights")

        # Verify that the router called universal_agent.execute_task with the correct parameters
        self.universal_agent.execute_task.assert_called_once()
        call_args = self.universal_agent.execute_task.call_args
        self.assertEqual(call_args[1]["role"], "router")
        self.assertEqual(call_args[1]["llm_type"], LLMType.STRONG)

        # Verify the routing result correctly routes to the smart_home role
        self.assertEqual(result["route"], "smart_home")
        self.assertEqual(result["confidence"], 0.85)
        self.assertIn("parameters", result)
        self.assertIn("device", result["parameters"])
        self.assertEqual(result["parameters"]["device"], "lights")
        self.assertIn("action", result["parameters"])
        self.assertEqual(result["parameters"]["action"], "on")

        # Verify that the prompt includes all roles
        prompt = call_args[1]["instruction"]
        self.assertIn("calendar", prompt)
        self.assertIn("weather", prompt)
        self.assertIn("timer", prompt)
        self.assertIn("smart_home", prompt)

    def test_calendar_request_routing(self):
        """Test that a calendar-related request is correctly routed to the calendar role."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a calendar-related response
        self.universal_agent.execute_task.return_value = """
        {
          "route": "calendar",
          "confidence": 0.88,
          "parameters": {
            "action": "create",
            "date": "tomorrow"
          }
        }
        """

        # Execute the route_request method with a calendar-related query
        result = router.route_request("schedule a meeting for tomorrow")

        # Verify that the router called universal_agent.execute_task with the correct parameters
        self.universal_agent.execute_task.assert_called_once()
        call_args = self.universal_agent.execute_task.call_args
        self.assertEqual(call_args[1]["role"], "router")
        self.assertEqual(call_args[1]["llm_type"], LLMType.STRONG)

        # Verify the routing result correctly routes to the calendar role
        self.assertEqual(result["route"], "calendar")
        self.assertEqual(result["confidence"], 0.88)
        self.assertIn("parameters", result)
        self.assertIn("action", result["parameters"])
        self.assertEqual(result["parameters"]["action"], "create")
        self.assertIn("date", result["parameters"])
        self.assertEqual(result["parameters"]["date"], "tomorrow")

        # Verify that the prompt includes all roles
        prompt = call_args[1]["instruction"]
        self.assertIn("calendar", prompt)
        self.assertIn("weather", prompt)
        self.assertIn("timer", prompt)
        self.assertIn("smart_home", prompt)

    def test_unknown_request_routing(self):
        """Test that an unknown request is correctly routed to PLANNING."""
        # Create a router with our mocked dependencies
        router = RequestRouter(
            self.llm_factory, self.role_registry, self.universal_agent
        )

        # Mock the universal agent to return a PLANNING response
        self.universal_agent.execute_task.return_value = """
        {
          "route": "PLANNING",
          "confidence": 0.3,
          "parameters": {}
        }
        """

        # Execute the route_request method with an unknown query
        result = router.route_request("write a complex essay about quantum physics")

        # Verify that the router called universal_agent.execute_task with the correct parameters
        self.universal_agent.execute_task.assert_called_once()
        call_args = self.universal_agent.execute_task.call_args
        self.assertEqual(call_args[1]["role"], "router")
        self.assertEqual(call_args[1]["llm_type"], LLMType.STRONG)

        # Verify the routing result correctly routes to PLANNING
        self.assertEqual(result["route"], "PLANNING")
        self.assertEqual(result["confidence"], 0.3)
        self.assertIn("parameters", result)
        self.assertEqual(len(result["parameters"]), 0)

        # Verify that the prompt includes all roles
        prompt = call_args[1]["instruction"]
        self.assertIn("calendar", prompt)
        self.assertIn("weather", prompt)
        self.assertIn("timer", prompt)
        self.assertIn("smart_home", prompt)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
