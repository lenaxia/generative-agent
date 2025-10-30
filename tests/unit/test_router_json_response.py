"""Tests for the JSON-response router_single_file.py (Pydantic Parsing)

Tests the router implementation that uses JSON responses with Pydantic validation
instead of tool calls for routing decisions.
"""

import os
import sys

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roles.core_router import (
    ROLE_CONFIG,
    ROUTING_CONFIDENCE_THRESHOLDS,
    RoutingResponse,
    _format_routing_summary,
    _get_role_priority,
    parse_routing_response,
    register_role,
    validate_confidence_score,
    validate_routing_request,
)


class TestJsonRouterSingleFile:
    """Test suite for the JSON-response router role."""

    def test_role_config_structure(self):
        """Test that role configuration follows the correct structure."""
        assert ROLE_CONFIG["name"] == "router"
        assert ROLE_CONFIG["version"] == "4.0.0"
        assert ROLE_CONFIG["llm_type"] == "WEAK"
        assert ROLE_CONFIG["fast_reply"] is False
        assert ROLE_CONFIG["tools"]["automatic"] is False  # No tools for JSON response
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

    def test_pydantic_model_validation(self):
        """Test Pydantic model validation for routing responses."""
        # Valid routing response
        valid_data = {"route": "weather", "confidence": 0.95, "parameters": {}}

        response = RoutingResponse(**valid_data)
        assert response.route == "weather"
        assert response.confidence == 0.95
        assert response.parameters == {}

    def test_pydantic_model_validation_errors(self):
        """Test Pydantic model validation errors."""
        # Missing required field
        with pytest.raises(Exception):  # Pydantic ValidationError
            RoutingResponse(confidence=0.95)

        # Invalid confidence range
        with pytest.raises(Exception):  # Pydantic ValidationError
            RoutingResponse(route="weather", confidence=1.5)

        # Negative confidence
        with pytest.raises(Exception):  # Pydantic ValidationError
            RoutingResponse(route="weather", confidence=-0.1)

    def test_parse_routing_response_valid_json(self):
        """Test parsing valid JSON routing response with Pydantic."""
        json_response = '{"route": "weather", "confidence": 0.95, "parameters": {}}'
        result = parse_routing_response(json_response)

        assert result["valid"] is True
        assert result["route"] == "weather"  # Keep original case
        assert result["confidence"] == 0.95
        assert result["parameters"] == {}

    def test_parse_routing_response_low_confidence_fallback(self):
        """Test that low confidence routes to planning."""
        json_response = '{"route": "weather", "confidence": 0.5, "parameters": {}}'
        result = parse_routing_response(json_response)

        assert result["valid"] is True
        assert result["route"] == "planning"  # Should fallback to planning
        assert result["confidence"] == 0.6  # Adjusted confidence

    def test_parse_routing_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        invalid_json = (
            '{"route": "weather", "confidence": 0.95'  # Missing closing brace
        )
        result = parse_routing_response(invalid_json)

        assert result["valid"] is False
        assert result["route"] == "PLANNING"  # Fallback
        assert result["confidence"] == 0.0
        assert (
            "Validation error" in result["error"]
        )  # Pydantic catches JSON errors as validation errors

    def test_parse_routing_response_missing_fields(self):
        """Test parsing JSON with missing required fields."""
        json_response = '{"confidence": 0.95}'  # Missing route
        result = parse_routing_response(json_response)

        assert result["valid"] is False
        assert result["route"] == "PLANNING"  # Fallback
        assert "Validation error" in result["error"]

    def test_parse_routing_response_invalid_confidence(self):
        """Test parsing JSON with invalid confidence value."""
        json_response = '{"route": "weather", "confidence": 1.5}'  # Out of range
        result = parse_routing_response(json_response)

        assert result["valid"] is False
        assert result["route"] == "PLANNING"  # Fallback
        assert "Validation error" in result["error"]

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

    def test_validate_confidence_score_valid(self):
        """Test validation of valid confidence scores."""
        result = validate_confidence_score(0.8)

        assert result["valid"] is True
        assert result["confidence_level"] == "high"

    def test_get_role_priority(self):
        """Test role priority scoring."""
        assert _get_role_priority("timer") == 1  # Highest priority
        assert _get_role_priority("weather") == 2
        assert _get_role_priority("planning") == 5
        assert _get_role_priority("unknown_role") == 10  # Default

    def test_format_routing_summary(self):
        """Test routing summary formatting."""
        summary = _format_routing_summary("weather", 0.95, "Clear weather request")

        assert "weather" in summary
        assert "high confidence" in summary
        assert "95.0%" in summary
        assert "Clear weather request" in summary

    def test_confidence_thresholds(self):
        """Test confidence threshold constants."""
        assert ROUTING_CONFIDENCE_THRESHOLDS["high"] == 0.8
        assert ROUTING_CONFIDENCE_THRESHOLDS["medium"] == 0.6
        assert ROUTING_CONFIDENCE_THRESHOLDS["low"] == 0.3
        assert ROUTING_CONFIDENCE_THRESHOLDS["fallback"] == 0.7


class TestJsonRouterIntegration:
    """Integration tests for JSON-based router functionality."""

    def test_json_parsing_workflow(self):
        """Test complete JSON parsing workflow."""
        # Test valid routing decision
        json_response = '{"route": "weather", "confidence": 0.95, "parameters": {"location": "seattle"}}'
        result = parse_routing_response(json_response)

        assert result["valid"] is True
        assert result["route"] == "weather"
        assert result["confidence"] == 0.95
        assert result["parameters"]["location"] == "seattle"

    def test_error_recovery_workflow(self):
        """Test error recovery in JSON parsing."""
        # Test with malformed JSON
        invalid_json = "This is not JSON at all"
        result = parse_routing_response(invalid_json)

        assert result["valid"] is False
        assert result["route"] == "PLANNING"  # Should fallback
        assert result["confidence"] == 0.0

    def test_backus_naur_form_compliance(self):
        """Test that the expected JSON format matches Backus-Naur form."""
        # Test the exact format specified in the system prompt
        bnf_compliant_json = '{"route": "timer", "confidence": 0.88, "parameters": {}}'
        result = parse_routing_response(bnf_compliant_json)

        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["confidence"] == 0.88
        assert result["parameters"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
