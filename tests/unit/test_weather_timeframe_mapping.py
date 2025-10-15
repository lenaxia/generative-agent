"""Tests for weather role timeframe parameter handling.

Validates that the weather role correctly processes timeframe parameters
from the router and fetches appropriate weather data (current vs forecast).
"""

from unittest.mock import MagicMock, patch

import pytest

from roles.core_weather import fetch_weather_data_for_request


class TestWeatherTimeframeMapping:
    """Test weather role timeframe parameter processing."""

    @patch("roles.core_weather._city_to_coordinates")
    @patch("roles.core_weather._check_weather")
    def test_timeframe_current_fetches_current_weather(
        self, mock_check_weather, mock_city_coords
    ):
        """Test that timeframe 'current' fetches current weather data."""
        # Setup mocks
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
        mock_check_weather.return_value = {
            "temperature": 61,
            "temperature_unit": "F",
            "short_forecast": "Sunny",
            "detailed_forecast": "Sunny skies with light winds",
        }

        # Test current timeframe
        parameters = {"location": "seattle", "timeframe": "current"}

        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["type"] == "current"
        assert "current" in result["weather_data"]
        mock_check_weather.assert_called_once()

    @patch("roles.core_weather._city_to_coordinates")
    @patch("roles.core_weather._get_forecast_data")
    def test_timeframe_week_fetches_forecast_weather(
        self, mock_get_forecast, mock_city_coords
    ):
        """Test that timeframe 'week' fetches forecast weather data."""
        # Setup mocks
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
        mock_get_forecast.return_value = [
            {"name": "Today", "temperature": 61, "short_forecast": "Sunny"},
            {"name": "Tonight", "temperature": 45, "short_forecast": "Clear"},
        ]

        # Test week timeframe
        parameters = {"location": "seattle", "timeframe": "week"}

        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["type"] == "forecast"
        assert "forecast" in result["weather_data"]
        mock_get_forecast.assert_called_once_with(47.6062, -122.3321, 7)

    @patch("roles.core_weather._city_to_coordinates")
    @patch("roles.core_weather._get_forecast_data")
    def test_timeframe_this_week_fetches_forecast(
        self, mock_get_forecast, mock_city_coords
    ):
        """Test that timeframe 'this week' fetches forecast weather data."""
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
        mock_get_forecast.return_value = []

        parameters = {"location": "seattle", "timeframe": "this week"}

        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["type"] == "forecast"
        mock_get_forecast.assert_called_once()

    @patch("roles.core_weather._city_to_coordinates")
    @patch("roles.core_weather._check_weather")
    def test_timeframe_today_fetches_current(
        self, mock_check_weather, mock_city_coords
    ):
        """Test that timeframe 'today' fetches current weather data."""
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
        mock_check_weather.return_value = {"temperature": 61}

        parameters = {"location": "seattle", "timeframe": "today"}

        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["type"] == "current"
        mock_check_weather.assert_called_once()

    @patch("roles.core_weather._city_to_coordinates")
    @patch("roles.core_weather._check_weather")
    def test_timeframe_default_fallback(self, mock_check_weather, mock_city_coords):
        """Test that unknown timeframe defaults to current weather."""
        mock_city_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
        mock_check_weather.return_value = {"temperature": 61}

        parameters = {"location": "seattle", "timeframe": "unknown_timeframe"}

        result = fetch_weather_data_for_request(parameters)

        assert result["success"] is True
        assert result["weather_data"]["type"] == "current"
        mock_check_weather.assert_called_once()

    def test_timeframe_parameter_logging(self):
        """Test that timeframe parameter is properly logged."""
        parameters = {"location": "seattle", "timeframe": "week"}

        with patch(
            "roles.core_weather._city_to_coordinates", side_effect=Exception("Test")
        ):
            with patch("roles.core_weather.logger") as mock_logger:
                try:
                    fetch_weather_data_for_request(parameters)
                except:
                    pass

                # Should log the forecast type, not current
                mock_logger.info.assert_called_with(
                    "Pre-fetching weather data for seattle (type: forecast)"
                )
