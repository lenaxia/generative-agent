"""Tests for the updated router_single_file.py (Current Architecture)

Tests the current router implementation that uses JSON response format
and intent-based processing without legacy functions.
"""

import os
import sys
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roles.core_router import (
    ROLE_CONFIG,
    _format_routing_summary,
    parse_routing_response,
    register_role,
    route_request_with_available_roles,
    validate_confidence_score,
    validate_routing_request,
)


class TestUpdatedRouterSingleFile:
    """Test suite for the updated router role (current architecture)."""

    def test_role_config_structure(self):
        """Test that role configuration follows the correct structure."""
        assert ROLE_CONFIG["name"] == "router"
        assert ROLE_CONFIG["version"] == "4.0.0"
        assert ROLE_CONFIG["llm_type"] == "WEAK"
        assert ROLE_CONFIG["fast_reply"] is False
        assert "description" in ROLE_CONFIG
        assert "when_to_use" in ROLE_CONFIG
        assert "prompts" in ROLE_CONFIG
        assert "system" in ROLE_CONFIG["prompts"]

    def test_role_registration_structure(self):
        """Test that role registration follows the correct structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration

        # Should have no tools (JSON response only)
        assert len(registration["tools"]) == 0

        # Should have minimal event handlers (only external events)
        assert "EXTERNAL_ROUTING_REQUEST" in registration["event_handlers"]

        # Should have minimal intents (no processing functions)
        assert len(registration["intents"]) == 1

    def test_route_request_with_available_roles_success(self):
        """Test successful routing using the current architecture."""
        request = "Set a timer for 10 minutes"

        # Create a mock role registry with proper structure
        mock_role_registry = Mock()
        mock_role_registry.get_fast_reply_roles.return_value = []
        mock_role_registry._universal_agent = Mock()
        mock_role_registry._universal_agent.execute_task.return_value = (
            '{"route": "timer", "confidence": 0.95, "parameters": {}}'
        )

        result = route_request_with_available_roles(request, mock_role_registry)
        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["confidence"] >= 0.7

    def test_parse_routing_response_valid_json(self):
        """Test parsing valid JSON routing response."""
        response_text = (
            '{"route": "timer", "confidence": 0.9, "parameters": {"duration": "10m"}}'
        )
        result = parse_routing_response(response_text)

        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["confidence"] == 0.9
        assert result["parameters"]["duration"] == "10m"

    def test_parse_routing_response_invalid_json(self):
        """Test parsing invalid JSON routing response."""
        response_text = '{"route": "timer", "confidence": invalid}'
        result = parse_routing_response(response_text)

        assert result["valid"] is False
        assert "error" in result

    def test_validate_confidence_score_valid(self):
        """Test confidence score validation with valid values."""
        result = validate_confidence_score(0.8)
        assert result["valid"] is True

    def test_validate_confidence_score_invalid_high(self):
        """Test confidence score validation with invalid high values."""
        result = validate_confidence_score(1.5)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_confidence_score_invalid_low(self):
        """Test confidence score validation with invalid low values."""
        result = validate_confidence_score(-0.1)
        assert result["valid"] is False
        assert "error" in result

    def test_validate_routing_request_valid(self):
        """Test routing request validation with valid input."""
        request = "Set a timer for 10 minutes"
        result = validate_routing_request(request)
        assert result["valid"] is True

    def test_validate_routing_request_empty(self):
        """Test routing request validation with empty input."""
        result = validate_routing_request("")
        assert result["valid"] is False
        assert "error" in result

    def test_validate_routing_request_too_short(self):
        """Test routing request validation with too short input."""
        result = validate_routing_request("hi")
        assert result["valid"] is False
        assert "error" in result

    def test_format_routing_summary(self):
        """Test routing summary formatting."""
        summary = _format_routing_summary(
            "timer", 0.95, "Request clearly asks for timer functionality"
        )
        assert "timer" in summary
        assert "95.0%" in summary
        assert "high confidence" in summary
        assert "Request clearly asks for timer functionality" in summary


class TestRouterIntegration:
    """Integration tests for router functionality."""

    def test_routing_workflow_simulation(self):
        """Test complete routing workflow simulation."""
        request = "Set a timer for 5 minutes"

        # Create a mock role registry
        mock_role_registry = Mock()
        mock_role_registry.get_fast_reply_roles.return_value = []
        mock_role_registry._universal_agent = Mock()
        mock_role_registry._universal_agent.execute_task.return_value = (
            '{"route": "timer", "confidence": 0.9, "parameters": {"duration": "5m"}}'
        )

        result = route_request_with_available_roles(request, mock_role_registry)

        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["confidence"] == 0.9
        assert result["parameters"]["duration"] == "5m"

    def test_error_recovery_workflow(self):
        """Test error recovery in routing workflow."""
        request = "Set a timer for 5 minutes"

        # Create a mock role registry
        mock_role_registry = Mock()
        mock_role_registry.get_fast_reply_roles.return_value = []
        mock_role_registry._universal_agent = Mock()
        mock_role_registry._universal_agent.execute_task.return_value = (
            "invalid json response"
        )

        result = route_request_with_available_roles(request, mock_role_registry)

        assert result["valid"] is False
        assert "error" in result
