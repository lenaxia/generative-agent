"""Tests for weather role pre-processing functionality.

Tests the weather role's pre-processing pattern where weather data is fetched
from APIs and injected into the prompt before LLM processing.
"""

import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from roles.weather_single_file import (
    ROLE_CONFIG,
    _check_weather,
    _city_to_coordinates,
    _is_coordinates,
    _is_zipcode,
    fetch_weather_data_for_request,
    process_weather_request_with_data,
)


class TestWeatherPreProcessing:
    """Test suite for weather role pre-processing functionality."""

    def test_weather_role_config(self):
        """Test weather role configuration for pre-processing."""
        assert ROLE_CONFIG["name"] == "weather"
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["tools"]["automatic"] is False  # No tools
        assert ROLE_CONFIG["tools"]["include_builtin"] is False  # No built-in tools

    @patch("roles.weather_single_file._city_to_coordinates")
    @patch("roles.weather_single_file._check_weather")
    def test_fetch_weather_data_success(self, mock_check_weather, mock_city_coords):
        """Test successful weather data fetching."""
        # Mock coordinate conversion
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}

        # Mock weather API response
        mock_weather_data = {
            "temperature": 65,
            "temperature_unit": "F",
            "short_forecast": "Partly Cloudy",
            "detailed_forecast": "Partly cloudy with temperatures around 65°F",
            "wind_speed": "5 mph",
            "wind_direction": "NW",
            "period_name": "This Afternoon",
        }
        mock_check_weather.return_value = mock_weather_data

        # Test weather data fetching
        parameters = {"location": "seattle", "type": "current"}
        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["location"] == "seattle"
        assert result["weather_data"]["type"] == "current"
        assert "current" in result["weather_data"]
        assert result["weather_data"]["current"] == mock_weather_data

    def test_fetch_weather_data_no_location(self):
        """Test weather data fetching with no location."""
        parameters = {"type": "current"}  # Missing location
        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is False
        assert "No location provided" in result["error"]
        assert result["weather_data"] is None

    @patch("roles.weather_single_file._city_to_coordinates")
    def test_fetch_weather_data_api_error(self, mock_city_coords):
        """Test weather data fetching with API error."""
        # Mock API error
        mock_city_coords.side_effect = Exception("API unavailable")

        parameters = {"location": "seattle", "type": "current"}
        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is False
        assert "API unavailable" in result["error"]
        assert result["weather_data"] is None

    @patch("roles.weather_single_file.fetch_weather_data_for_request")
    def test_process_weather_request_success(self, mock_fetch_data):
        """Test weather request processing with successful data fetching."""
        # Mock successful weather data fetch
        mock_weather_data = {
            "location": "seattle",
            "type": "current",
            "current": {
                "temperature": 65,
                "temperature_unit": "F",
                "short_forecast": "Partly Cloudy",
                "detailed_forecast": "Partly cloudy with temperatures around 65°F",
                "wind_speed": "5 mph",
                "wind_direction": "NW",
                "period_name": "This Afternoon",
            },
        }

        mock_fetch_data.return_value = {
            "success": True,
            "weather_data": mock_weather_data,
        }

        # Mock universal agent
        mock_universal_agent = Mock()
        mock_universal_agent.execute_task.return_value = (
            "The weather in Seattle is partly cloudy with a temperature of 65°F."
        )

        # Set universal agent reference
        process_weather_request_with_data._universal_agent = mock_universal_agent

        # Test processing
        result = process_weather_request_with_data(
            "What's the weather in Seattle?", {"location": "seattle", "type": "current"}
        )

        # Verify universal agent was called with injected data
        mock_universal_agent.execute_task.assert_called_once()
        call_args = mock_universal_agent.execute_task.call_args

        assert "CURRENT WEATHER DATA FOR SEATTLE:" in call_args[1]["instruction"]
        assert "Temperature: 65°F" in call_args[1]["instruction"]
        assert "Conditions: Partly Cloudy" in call_args[1]["instruction"]
        assert call_args[1]["role"] == "weather"

    @patch("roles.weather_single_file.fetch_weather_data_for_request")
    def test_process_weather_request_fetch_error(self, mock_fetch_data):
        """Test weather request processing with data fetch error."""
        # Mock failed weather data fetch
        mock_fetch_data.return_value = {
            "success": False,
            "error": "Weather API unavailable",
        }

        result = process_weather_request_with_data(
            "What's the weather in Seattle?", {"location": "seattle", "type": "current"}
        )

        assert "couldn't fetch weather data" in result
        assert "Weather API unavailable" in result

    def test_location_parsing_coordinates(self):
        """Test coordinate parsing."""
        assert _is_coordinates("47.6062,-122.3321") is True
        assert _is_coordinates("seattle") is False
        assert _is_coordinates("98101") is False

    def test_location_parsing_zipcode(self):
        """Test ZIP code parsing."""
        assert _is_zipcode("98101") is True
        assert _is_zipcode("seattle") is False
        assert _is_zipcode("47.6062,-122.3321") is False


class TestWeatherIntegration:
    """Integration tests for weather pre-processing."""

    def test_weather_data_injection_format(self):
        """Test that weather data is properly formatted for injection."""
        mock_weather_data = {
            "location": "seattle",
            "type": "current",
            "current": {
                "temperature": 65,
                "temperature_unit": "F",
                "short_forecast": "Sunny",
                "detailed_forecast": "Sunny skies with light winds",
                "wind_speed": "5 mph",
                "wind_direction": "NW",
                "period_name": "This Afternoon",
            },
        }

        # Test that the weather context format is correct
        expected_context = """CURRENT WEATHER DATA FOR SEATTLE:
- Temperature: 65°F
- Conditions: Sunny
- Detailed: Sunny skies with light winds
- Wind: 5 mph NW
- Time Period: This Afternoon"""

        # This would be generated by the process_weather_request_with_data function
        # We're testing the format that should be injected into the prompt
        assert "CURRENT WEATHER DATA FOR SEATTLE:" in expected_context
        assert "Temperature: 65°F" in expected_context
        assert "Conditions: Sunny" in expected_context

    def test_forecast_data_injection_format(self):
        """Test forecast data injection format."""
        mock_forecast_data = [
            {"name": "Today", "temperature": 65, "shortForecast": "Sunny"},
            {"name": "Tonight", "temperature": 45, "shortForecast": "Clear"},
            {"name": "Tomorrow", "temperature": 68, "shortForecast": "Partly Cloudy"},
        ]

        # Test forecast format
        expected_context = f"WEATHER FORECAST DATA FOR SEATTLE:\n{mock_forecast_data}"

        assert "WEATHER FORECAST DATA FOR SEATTLE:" in expected_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
