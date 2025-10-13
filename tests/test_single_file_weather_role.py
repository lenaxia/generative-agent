"""
Tests for single-file weather role implementation.

This module tests the consolidated weather role following the LLM-safe
architecture patterns from Documents 25, 26, and 27.

Created: 2025-10-13
Part of: Technical Debt Cleanup - Role Migration
"""

import logging
import time
from unittest.mock import patch

import pytest

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, NotificationIntent
from roles.weather_single_file import (
    ROLE_CONFIG,
    WeatherDataIntent,
    WeatherIntent,
    handle_weather_request,
    register_role,
)

logger = logging.getLogger(__name__)


class TestSingleFileWeatherRole:
    """Test single-file weather role implementation."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock LLMSafeEventContext for testing."""
        return LLMSafeEventContext(
            channel_id="C123WEATHER",
            user_id="U456WEATHER",
            timestamp=time.time(),
            source="weather_test",
        )

    def test_role_config_structure(self):
        """Test that role configuration follows expected structure."""
        assert isinstance(ROLE_CONFIG, dict), "ROLE_CONFIG should be a dictionary"
        assert "name" in ROLE_CONFIG, "Should have name"
        assert "version" in ROLE_CONFIG, "Should have version"
        assert "description" in ROLE_CONFIG, "Should have description"
        assert "llm_type" in ROLE_CONFIG, "Should have llm_type"
        assert "fast_reply" in ROLE_CONFIG, "Should have fast_reply"
        assert "when_to_use" in ROLE_CONFIG, "Should have when_to_use"

        # Verify specific values
        assert ROLE_CONFIG["name"] == "weather", "Name should be weather"
        assert ROLE_CONFIG["llm_type"] == "WEAK", "Should use WEAK LLM type"
        assert ROLE_CONFIG["fast_reply"] is True, "Should enable fast reply"

    def test_weather_intent_validation(self):
        """Test weather intent validation."""
        # Valid weather intent
        valid_intent = WeatherIntent(
            action="fetch", location="Seattle", timeframe="current", format_type="brief"
        )
        assert valid_intent.validate(), "Valid weather intent should pass validation"

        # Invalid weather intent (bad action)
        invalid_intent = WeatherIntent(action="invalid_action", location="Seattle")
        assert not invalid_intent.validate(), "Invalid action should fail validation"

        # Valid weather intent (minimal)
        minimal_intent = WeatherIntent(action="validate")
        assert minimal_intent.validate(), "Minimal valid intent should pass"

    def test_weather_data_intent_validation(self):
        """Test weather data intent validation."""
        # Valid weather data intent
        valid_intent = WeatherDataIntent(
            weather_data={"temperature": 72, "condition": "sunny"},
            location="Seattle",
            processing_type="tts_format",
        )
        assert (
            valid_intent.validate()
        ), "Valid weather data intent should pass validation"

        # Invalid weather data intent (bad processing type)
        invalid_intent = WeatherDataIntent(
            weather_data={"temperature": 72},
            location="Seattle",
            processing_type="invalid_type",
        )
        assert (
            not invalid_intent.validate()
        ), "Invalid processing type should fail validation"

    def test_handle_weather_request_returns_intents(self, mock_context):
        """Test that weather request handler returns intents."""
        # Test data
        event_data = {"location": "Seattle", "timeframe": "current"}

        # Call the pure function handler
        result = handle_weather_request(event_data, mock_context)

        # Verify it returns a list of intents
        assert isinstance(result, list), "Handler should return list of intents"
        assert len(result) >= 1, "Handler should return at least one intent"

        # Verify all items are intents
        for intent in result:
            assert hasattr(
                intent, "validate"
            ), "All items should be intents with validate method"
            assert intent.validate(), f"Intent should be valid: {intent}"

        # Verify specific intent types
        weather_intents = [i for i in result if isinstance(i, WeatherIntent)]
        audit_intents = [i for i in result if isinstance(i, AuditIntent)]

        assert len(weather_intents) >= 1, "Should have at least one weather intent"
        assert len(audit_intents) >= 1, "Should have at least one audit intent"

    def test_handle_weather_request_error_handling(self, mock_context):
        """Test weather request handler error handling."""
        # Test with invalid data that should trigger error handling
        event_data = None

        # Call handler
        result = handle_weather_request(event_data, mock_context)

        # Should return error notification intent
        assert isinstance(result, list), "Should return list even on error"
        assert len(result) >= 1, "Should return at least one intent on error"

        # Should contain notification intent
        notification_intents = [i for i in result if isinstance(i, NotificationIntent)]
        assert len(notification_intents) >= 1, "Should have error notification intent"

        # Verify error notification
        error_notification = notification_intents[0]
        assert "error" in error_notification.message.lower(), "Should mention error"
        assert error_notification.notification_type == "error", "Should be error type"

    def test_role_registration_structure(self):
        """Test that role registration follows expected structure."""
        registration = register_role()

        assert isinstance(registration, dict), "Registration should return dictionary"
        assert "config" in registration, "Should include config"
        assert "event_handlers" in registration, "Should include event handlers"
        assert "tools" in registration, "Should include tools"
        assert "intents" in registration, "Should include intents"

        # Verify config
        assert registration["config"] == ROLE_CONFIG, "Config should match ROLE_CONFIG"

        # Verify event handlers
        handlers = registration["event_handlers"]
        assert "WEATHER_REQUEST" in handlers, "Should handle weather requests"
        assert (
            "WEATHER_DATA_PROCESSING" in handlers
        ), "Should handle weather data processing"

        # Verify tools
        tools = registration["tools"]
        assert len(tools) >= 2, "Should have at least 2 tools"

        # Verify intents
        intents = registration["intents"]
        assert WeatherIntent in intents, "Should register WeatherIntent"
        assert WeatherDataIntent in intents, "Should register WeatherDataIntent"

    @patch("roles.weather_single_file.requests.get")
    def test_get_weather_tool_success(self, mock_get):
        """Test get_weather tool with successful API response."""
        # Mock successful API responses
        mock_points_response = {
            "properties": {
                "forecast": "https://api.weather.gov/gridpoints/SEW/124,67/forecast"
            }
        }
        mock_forecast_response = {
            "properties": {
                "periods": [
                    {
                        "name": "Today",
                        "temperature": 72,
                        "temperatureUnit": "F",
                        "shortForecast": "Sunny",
                        "detailedForecast": "Sunny with clear skies",
                        "windSpeed": "5 mph",
                        "windDirection": "NW",
                        "isDaytime": True,
                    }
                ]
            }
        }

        # Configure mock to return different responses for different URLs
        def mock_get_side_effect(url, **kwargs):
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            if "points" in url:
                mock_response.json.return_value = mock_points_response
            else:
                mock_response.json.return_value = mock_forecast_response
            return mock_response

        mock_get.side_effect = mock_get_side_effect

        # Test with coordinates
        from roles.weather_single_file import get_weather

        result = get_weather("47.6062,-122.3321")

        # Verify result structure
        assert (
            result["status"] == "success"
        ), f"Expected success, got {result.get('status')}"
        assert "weather" in result, "Should include weather data"
        assert "coordinates" in result, "Should include coordinates"
        assert "timestamp" in result, "Should include timestamp"

    def test_get_weather_tool_error_handling(self):
        """Test get_weather tool error handling."""
        from roles.weather_single_file import get_weather

        # Test with invalid location that should cause an error
        result = get_weather("invalid_location_12345")

        # Should return error result
        assert result["status"] == "error", "Should return error status"
        assert "error" in result, "Should include error message"
        assert "timestamp" in result, "Should include timestamp"

    def test_weather_helper_functions(self):
        """Test weather helper functions."""
        from roles.weather_single_file import (
            _is_coordinates,
            _is_zipcode,
            normalize_weather_action,
        )

        # Test coordinate detection
        assert _is_coordinates("47.6062,-122.3321") is True, "Should detect coordinates"
        assert (
            _is_coordinates("Seattle") is False
        ), "Should not detect city as coordinates"

        # Test ZIP code detection
        assert _is_zipcode("98101") is True, "Should detect ZIP code"
        assert _is_zipcode("Seattle") is False, "Should not detect city as ZIP code"
        assert (
            _is_zipcode("1234") is False
        ), "Should not detect short number as ZIP code"

        # Test action normalization
        assert (
            normalize_weather_action("get") == "fetch"
        ), "Should normalize get to fetch"
        assert (
            normalize_weather_action("check") == "fetch"
        ), "Should normalize check to fetch"
        assert (
            normalize_weather_action("validate") == "validate"
        ), "Should keep validate as validate"

    def test_weather_constants(self):
        """Test weather constants and configuration."""
        from roles.weather_single_file import (
            DEFAULT_FORECAST_DAYS,
            MAX_FORECAST_DAYS,
            WEATHER_ACTIONS,
            WEATHER_API_TIMEOUT,
        )

        # Verify constants are reasonable
        assert WEATHER_API_TIMEOUT > 0, "API timeout should be positive"
        assert DEFAULT_FORECAST_DAYS > 0, "Default forecast days should be positive"
        assert MAX_FORECAST_DAYS >= DEFAULT_FORECAST_DAYS, "Max should be >= default"

        # Verify action mappings
        assert isinstance(WEATHER_ACTIONS, dict), "Weather actions should be dict"
        assert "get" in WEATHER_ACTIONS, "Should have get action mapping"
        assert "fetch" in WEATHER_ACTIONS, "Should have fetch action mapping"

    def test_weather_error_intent_creation(self, mock_context):
        """Test weather error intent creation."""
        from roles.weather_single_file import create_weather_error_intent

        # Create test error
        test_error = ValueError("Test weather error")

        # Create error intents
        error_intents = create_weather_error_intent(test_error, mock_context)

        # Verify error intents
        assert isinstance(error_intents, list), "Should return list of intents"
        assert len(error_intents) >= 2, "Should return notification and audit intents"

        # Verify notification intent
        notification_intents = [
            i for i in error_intents if isinstance(i, NotificationIntent)
        ]
        assert len(notification_intents) >= 1, "Should have notification intent"

        notification = notification_intents[0]
        assert "error" in notification.message.lower(), "Should mention error"
        assert notification.notification_type == "error", "Should be error type"

        # Verify audit intent
        audit_intents = [i for i in error_intents if isinstance(i, AuditIntent)]
        assert len(audit_intents) >= 1, "Should have audit intent"

        audit = audit_intents[0]
        assert audit.action == "weather_error", "Should be weather error action"
        assert audit.severity == "error", "Should be error severity"

    def test_single_file_consolidation_success(self):
        """Test that single-file consolidation maintains functionality."""
        # Verify all expected components are present
        from roles.weather_single_file import (
            ROLE_CONFIG,
            WeatherDataIntent,
            WeatherIntent,
            get_weather,
            get_weather_forecast,
            handle_weather_data_processing,
            handle_weather_request,
            process_weather_data_intent,
            process_weather_intent,
            register_role,
        )

        # All components should be importable and defined
        assert ROLE_CONFIG is not None, "Role config should be defined"
        assert WeatherIntent is not None, "WeatherIntent should be defined"
        assert WeatherDataIntent is not None, "WeatherDataIntent should be defined"
        assert callable(
            handle_weather_request
        ), "Weather request handler should be callable"
        assert callable(
            handle_weather_data_processing
        ), "Weather data handler should be callable"
        assert callable(get_weather), "get_weather tool should be callable"
        assert callable(
            get_weather_forecast
        ), "get_weather_forecast tool should be callable"
        assert callable(register_role), "register_role should be callable"
        assert callable(
            process_weather_intent
        ), "process_weather_intent should be callable"
        assert callable(
            process_weather_data_intent
        ), "process_weather_data_intent should be callable"

        logger.info("✅ Single-file weather role consolidation successful")
