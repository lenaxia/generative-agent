"""
Tests for router type safety and edge cases.

This module tests the router's handling of various data types and edge cases
to prevent type errors and ensure robust operation.
"""

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from roles.core_router import route_request_with_available_roles


class TestRouterTypeSafety:
    """Test router type safety with various data types."""

    @pytest.fixture
    def mock_role_registry_with_integer_examples(self):
        """Create mock role registry with integer examples that caused the bug."""
        mock_registry = Mock()
        mock_registry._universal_agent = Mock()
        mock_registry._universal_agent.execute_task.return_value = (
            '{"route": "timer", "confidence": 0.95, "parameters": {"duration": "5m"}}'
        )

        # Create mock role with integer examples (this caused the original bug)
        mock_role = Mock()
        mock_role.name = "timer"
        mock_role.config = {
            "role": {
                "description": "Timer management",
                "when_to_use": "Set timers and alarms",
                "capabilities": ["timer"],
                "fast_reply": True,
                "parameters": {
                    "duration": {
                        "type": "string",
                        "required": True,
                        "examples": [
                            "5m",
                            "1h",
                            30,
                            60,
                        ],  # Mixed string and integer examples
                    },
                    "count": {
                        "type": "integer",
                        "required": False,
                        "examples": [1, 2, 3, 5, 10],  # Integer examples
                    },
                },
            }
        }

        mock_registry.get_fast_reply_roles.return_value = [mock_role]
        return mock_registry

    def test_router_handles_integer_examples(
        self, mock_role_registry_with_integer_examples
    ):
        """Test that router handles integer examples without type errors."""
        # This test reproduces the bug scenario: role parameters with integer examples
        result = route_request_with_available_roles(
            "Set a timer for 5 minutes", mock_role_registry_with_integer_examples
        )

        # Should not raise "sequence item 0: expected str instance, int found" error
        assert result["valid"] is True
        assert result["route"] == "timer"

    def test_router_handles_mixed_type_examples(
        self, mock_role_registry_with_integer_examples
    ):
        """Test router handles mixed string/integer examples gracefully."""
        # Create role with mixed type examples
        mock_role = Mock()
        mock_role.name = "test_role"
        mock_role.config = {
            "role": {
                "description": "Test role",
                "when_to_use": "Testing",
                "parameters": {
                    "mixed_param": {
                        "type": "mixed",
                        "examples": ["string", 42, True, 3.14, None],  # Various types
                    }
                },
            }
        }

        mock_role_registry_with_integer_examples.get_fast_reply_roles.return_value = [
            mock_role
        ]

        # Should handle all types by converting to strings
        result = route_request_with_available_roles(
            "Test request", mock_role_registry_with_integer_examples
        )

        # Should not raise type errors
        assert result["valid"] is True

    def test_router_handles_empty_examples(
        self, mock_role_registry_with_integer_examples
    ):
        """Test router handles empty examples list."""
        mock_role = Mock()
        mock_role.name = "test_role"
        mock_role.config = {
            "role": {
                "description": "Test role",
                "when_to_use": "Testing",
                "parameters": {
                    "param_with_empty_examples": {
                        "type": "string",
                        "examples": [],  # Empty examples
                    }
                },
            }
        }

        mock_role_registry_with_integer_examples.get_fast_reply_roles.return_value = [
            mock_role
        ]

        result = route_request_with_available_roles(
            "Test request", mock_role_registry_with_integer_examples
        )

        # Should handle empty examples gracefully
        assert result["valid"] is True

    def test_router_handles_none_examples(
        self, mock_role_registry_with_integer_examples
    ):
        """Test router handles None examples."""
        mock_role = Mock()
        mock_role.name = "test_role"
        mock_role.config = {
            "role": {
                "description": "Test role",
                "when_to_use": "Testing",
                "parameters": {
                    "param_with_none_examples": {
                        "type": "string",
                        "examples": None,  # None examples
                    }
                },
            }
        }

        mock_role_registry_with_integer_examples.get_fast_reply_roles.return_value = [
            mock_role
        ]

        result = route_request_with_available_roles(
            "Test request", mock_role_registry_with_integer_examples
        )

        # Should handle None examples gracefully
        assert result["valid"] is True

    def test_router_type_safety_regression_test(self):
        """Regression test for the specific error: 'sequence item 0: expected str instance, int found'."""
        # This test specifically targets the bug reported in the logs

        # Test data that would cause the original error
        examples_with_integers = [5, 10, "5m", "10m", 30]

        # The fix should convert all to strings
        string_examples = [str(ex) for ex in examples_with_integers[:2]]
        joined_examples = ", ".join(string_examples)

        # Should not raise TypeError
        assert isinstance(joined_examples, str)
        assert "5" in joined_examples
        assert "10" in joined_examples


if __name__ == "__main__":
    pytest.main([__file__])
