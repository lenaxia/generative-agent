"""Tests for the updated router_single_file.py (No RequestRouter)

Tests the updated router implementation that works without RequestRouter,
where available roles are pre-injected into the system prompt.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roles.router_single_file import (
    ROLE_CONFIG,
    ROUTING_CONFIDENCE_THRESHOLDS,
    _format_routing_summary,
    _get_role_priority,
    parse_routing_response,
    register_role,
    validate_confidence_score,
    validate_routing_request,
)


class TestUpdatedRouterSingleFile:
    """Test suite for the updated router role (no RequestRouter)."""

    def test_role_config_structure(self):
        """Test that role configuration follows the correct structure."""
        assert ROLE_CONFIG["name"] == "router"
        assert ROLE_CONFIG["version"] == "3.0.0"
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

        # Should have exactly 1 tool (route_to_role only)
        assert len(registration["tools"]) == 1
        assert route_to_role in registration["tools"]

        # Should have minimal event handlers (only external events)
        assert "EXTERNAL_ROUTING_REQUEST" in registration["event_handlers"]

        # Should have minimal intents (no processing functions)
        assert len(registration["intents"]) == 1

    def test_route_to_role_success(self):
        """Test successful routing decision execution."""
        result = route_to_role(
            confidence=0.9,
            selected_role="timer",
            original_request="Set a timer for 5 minutes",
            reasoning="Request clearly asks for timer functionality",
        )

        assert result["success"] is True
        assert result["selected_role"] == "timer"
        assert result["confidence"] == 0.9
        assert "workflow_id" in result
        assert "execution_time" in result

    def test_route_to_role_low_confidence_fallback(self):
        """Test that low confidence routes to planning."""
        result = route_to_role(
            confidence=0.5,  # Below 0.7 threshold
            selected_role="timer",
            original_request="Maybe set some kind of reminder?",
            reasoning="Unclear request",
        )

        assert result["success"] is True
        assert result["selected_role"] == "planning"  # Should fallback to planning
        assert "Low confidence" in result["reasoning"]

    def test_route_to_role_invalid_confidence(self):
        """Test validation of confidence scores."""
        result = route_to_role(
            confidence=1.5,  # Invalid - above 1.0
            selected_role="timer",
            original_request="Set a timer",
            reasoning="Test",
        )

        assert result["success"] is False
        assert "Invalid confidence" in result["error"]

    def test_route_to_role_missing_parameters(self):
        """Test validation of required parameters."""
        result = route_to_role(
            confidence=0.8,
            selected_role="",  # Empty role
            original_request="Set a timer",
            reasoning="Test",
        )

        assert result["success"] is False
        assert "required" in result["error"]

    def test_validate_routing_request_valid(self):
        """Test validation of valid routing requests."""
        result = validate_routing_request("Set a timer for 5 minutes")

        assert result["valid"] is True
        assert "message" in result

    def test_validate_routing_request_empty(self):
        """Test validation of empty requests."""
        result = validate_routing_request("")

        assert result["valid"] is False
        assert "Empty" in result["error"]

    def test_validate_routing_request_too_short(self):
        """Test validation of too-short requests."""
        result = validate_routing_request("hi")

        assert result["valid"] is False
        assert "too short" in result["error"]

    def test_validate_routing_request_too_long(self):
        """Test validation of too-long requests."""
        long_request = "x" * 1001  # Over 1000 character limit
        result = validate_routing_request(long_request)

        assert result["valid"] is False
        assert "too long" in result["error"]

    def test_validate_confidence_score_valid(self):
        """Test validation of valid confidence scores."""
        result = validate_confidence_score(0.8)

        assert result["valid"] is True
        assert result["confidence_level"] == "high"

    def test_validate_confidence_score_invalid_type(self):
        """Test validation of invalid confidence score types."""
        result = validate_confidence_score("0.8")  # type: ignore

        assert result["valid"] is False
        assert "must be a number" in result["error"]

    def test_validate_confidence_score_out_of_range(self):
        """Test validation of out-of-range confidence scores."""
        result = validate_confidence_score(-0.1)

        assert result["valid"] is False
        assert "outside valid range" in result["error"]

    def test_get_role_priority(self):
        """Test role priority scoring."""
        assert _get_role_priority("timer") == 1  # Highest priority
        assert _get_role_priority("weather") == 2
        assert _get_role_priority("planning") == 5
        assert _get_role_priority("unknown_role") == 10  # Default

    def test_format_routing_summary(self):
        """Test routing summary formatting."""
        summary = _format_routing_summary("timer", 0.9, "Clear timer request")

        assert "timer" in summary
        assert "high confidence" in summary
        assert "90.0%" in summary  # Format shows decimal
        assert "Clear timer request" in summary

    def test_confidence_thresholds(self):
        """Test confidence threshold constants."""
        assert ROUTING_CONFIDENCE_THRESHOLDS["high"] == 0.8
        assert ROUTING_CONFIDENCE_THRESHOLDS["medium"] == 0.6
        assert ROUTING_CONFIDENCE_THRESHOLDS["low"] == 0.3
        assert ROUTING_CONFIDENCE_THRESHOLDS["fallback"] == 0.7


class TestRouterIntegration:
    """Integration tests for router role functionality."""

    def test_routing_workflow_simulation(self):
        """Test a routing workflow simulation."""
        # Test routing with high confidence
        routing_result = route_to_role(
            confidence=0.9,
            selected_role="timer",
            original_request="Set a timer for 10 minutes",
            reasoning="Clear timer request with specific duration",
        )

        assert routing_result["success"] is True
        assert routing_result["selected_role"] == "timer"
        assert routing_result["confidence"] == 0.9

    def test_error_recovery_workflow(self):
        """Test error recovery in routing workflow."""
        # Test routing with low confidence fallback
        routing_result = route_to_role(
            confidence=0.4,  # Low confidence
            selected_role="timer",
            original_request="Unclear request",
            reasoning="Low confidence due to ambiguity",
        )

        assert routing_result["success"] is True
        assert routing_result["selected_role"] == "planning"  # Should fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
