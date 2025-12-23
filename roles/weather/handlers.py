"""Weather Role Handlers - Phase 3 Domain Pattern

Event handlers, intent processors, and helper functions for weather role.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from common.event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)


# INTENT DEFINITIONS
@dataclass
class WeatherIntent(Intent):
    """Weather-specific intent - owned by weather role."""

    action: str  # "fetch", "validate", "format"
    location: str | None = None
    timeframe: str | None = None
    format_type: str | None = None

    def validate(self) -> bool:
        """Validate weather intent parameters."""
        return bool(self.action and self.action in ["fetch", "validate", "format"])


@dataclass
class WeatherDataIntent(Intent):
    """Weather data processing intent - owned by weather role."""

    weather_data: dict[str, Any]
    location: str
    processing_type: str  # "tts_format", "pii_scrub", "audit_log"

    def validate(self) -> bool:
        """Validate weather data intent parameters."""
        return bool(
            self.weather_data
            and self.location
            and self.processing_type in ["tts_format", "pii_scrub", "audit_log"]
        )


# EVENT HANDLERS
def handle_weather_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for weather request events."""
    try:
        # Parse event data
        location, timeframe = _parse_weather_event_data(event_data)

        # Check if parsing failed (indicates error condition)
        if location == "parse_error" or (event_data is None and location == "unknown"):
            logger.error(f"Weather handler error: Invalid event data: {event_data}")
            return [
                NotificationIntent(
                    message=f"Weather processing error: Invalid event data",
                    channel=context.get_safe_channel(),
                    priority="high",
                    notification_type="error",
                )
            ]

        # Create intents for successful parsing
        return [
            WeatherIntent(
                action="fetch",
                location=location,
                timeframe=timeframe,
                format_type="brief",
            ),
            AuditIntent(
                action="weather_request",
                details={
                    "location": location,
                    "timeframe": timeframe,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
            ),
        ]

    except Exception as e:
        logger.error(f"Weather handler error: {e}")
        return [
            NotificationIntent(
                message=f"Weather processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


def handle_weather_data_processing(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for weather data processing events."""
    try:
        # Parse weather data from event
        weather_data, processing_type = _parse_weather_data_event(event_data)

        return [
            WeatherDataIntent(
                weather_data=weather_data,
                location=weather_data.get("location", "unknown"),
                processing_type=processing_type,
            )
        ]

    except Exception as e:
        logger.error(f"Weather data processing error: {e}")
        return [
            NotificationIntent(
                message=f"Weather data processing error: {e}",
                channel=context.get_safe_channel(),
                priority="medium",
                notification_type="warning",
            )
        ]


# HELPER FUNCTIONS
def _parse_weather_event_data(event_data: Any) -> tuple[str, str]:
    """LLM-SAFE: Parse weather event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("location", "unknown"),
                event_data.get("timeframe", "current"),
            )
        elif isinstance(event_data, list) and len(event_data) >= 2:
            return str(event_data[0]), str(event_data[1])
        else:
            return "unknown", "current"
    except Exception as e:
        return "parse_error", f"Parse error: {e}"


def _parse_weather_data_event(event_data: Any) -> tuple[dict[str, Any], str]:
    """LLM-SAFE: Parse weather data processing event."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("weather_data", {}),
                event_data.get("processing_type", "tts_format"),
            )
        else:
            return {}, "tts_format"
    except Exception as e:
        return {}, f"parse_error: {e}"


def _is_coordinates(location: str) -> bool:
    """Check if location string is coordinates."""
    return (
        "," in location
        and location.replace(",", "").replace(".", "").replace("-", "").isdigit()
    )


def _is_zipcode(location: str) -> bool:
    """Check if location string is a ZIP code."""
    return location.isdigit() and len(location) == 5


def _city_to_coordinates(city: str) -> dict[str, float]:
    """Convert a city name to latitude and longitude coordinates."""
    logger.info(f"Converting city to coordinates: {city}")

    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        response = requests.get(url, headers=headers, timeout=10)
        response_data = response.json()

        if not response_data:
            raise ValueError(f"Could not find coordinates for city '{city}'")

        lat = float(response_data[0]["lat"])
        lon = float(response_data[0]["lon"])

        coordinates = {"lat": lat, "lon": lon}
        logger.info(f"City {city} converted to coordinates: {coordinates}")
        return coordinates

    except Exception as e:
        logger.error(f"Error converting city {city} to coordinates: {e}")
        raise


def _zipcode_to_coordinates(zipcode: str, country_code: str = "US") -> dict[str, float]:
    """Convert a ZIP code to latitude and longitude coordinates."""
    logger.info(f"Converting ZIP code to coordinates: {zipcode} ({country_code})")

    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={zipcode}&format=json&country={country_code}"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        response = requests.get(url, headers=headers, timeout=10)
        response_data = response.json()

        if not response_data:
            raise ValueError(
                f"Could not find coordinates for ZIP code '{zipcode}' in {country_code}"
            )

        # Filter results to get the coordinates for the ZIP code in the specified country
        country_result = None
        for result in response_data:
            if (
                country_code.upper() == "US"
                and "United States" in result["display_name"]
            ):
                country_result = result
                break
            elif country_code.upper() in result["display_name"].upper():
                country_result = result
                break

        if not country_result:
            country_result = response_data[0]  # Fallback to first result

        lat = float(country_result["lat"])
        lon = float(country_result["lon"])

        coordinates = {"lat": lat, "lon": lon}
        logger.info(f"ZIP code {zipcode} converted to coordinates: {coordinates}")
        return coordinates

    except Exception as e:
        logger.error(f"Error converting ZIP code {zipcode} to coordinates: {e}")
        raise


def _check_weather(lat: float, lon: float) -> dict[str, Any]:
    """Check weather for specific coordinates using NOAA API."""
    logger.info(f"Checking weather for coordinates: {lat}, {lon}")

    try:
        # Fetch weather data from NOAA API
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_data = forecast_response.json()

        if (
            "properties" not in forecast_data
            or "periods" not in forecast_data["properties"]
        ):
            raise ValueError("Could not fetch weather forecast for that location")

        current_period = forecast_data["properties"]["periods"][0]

        weather_info = {
            "detailed_forecast": current_period["detailedForecast"],
            "temperature": current_period.get("temperature"),
            "temperature_unit": current_period.get("temperatureUnit"),
            "wind_speed": current_period.get("windSpeed"),
            "wind_direction": current_period.get("windDirection"),
            "short_forecast": current_period.get("shortForecast"),
            "period_name": current_period.get("name"),
            "is_daytime": current_period.get("isDaytime"),
        }

        logger.info(f"Weather data retrieved for coordinates {lat}, {lon}")
        return weather_info

    except Exception as e:
        logger.error(f"Error checking weather for coordinates {lat}, {lon}: {e}")
        raise


def _get_forecast_data(lat: float, lon: float, days: int) -> list[dict[str, Any]]:
    """Get extended forecast data from NOAA API."""
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_data = forecast_response.json()

        if (
            "properties" not in forecast_data
            or "periods" not in forecast_data["properties"]
        ):
            raise ValueError("Could not fetch weather forecast for that location")

        periods = forecast_data["properties"]["periods"][
            : days * 2
        ]  # Day and night periods

        return [
            {
                "name": period.get("name"),
                "temperature": period.get("temperature"),
                "temperature_unit": period.get("temperatureUnit"),
                "wind_speed": period.get("windSpeed"),
                "wind_direction": period.get("windDirection"),
                "short_forecast": period.get("shortForecast"),
                "detailed_forecast": period.get("detailedForecast"),
                "is_daytime": period.get("isDaytime"),
            }
            for period in periods
        ]

    except Exception as e:
        logger.error(f"Error getting forecast data: {e}")
        raise


# INTENT PROCESSORS
async def process_weather_intent(intent: WeatherIntent):
    """Process weather-specific intents - called by IntentProcessor."""
    logger.info(f"Processing weather intent: {intent.action}")
    # TODO: Implement weather intent processing logic
    # In full implementation, this would:
    # - Fetch weather data for location
    # - Validate coordinates and location formats
    # - Format weather data for different output types


async def process_weather_data_intent(intent: WeatherDataIntent):
    """Process weather data processing intents - called by IntentProcessor."""
    logger.info(f"Processing weather data intent: {intent.processing_type}")
    # TODO: Implement weather data processing logic
    # In full implementation, this would:
    # - Format weather data for TTS
    # - Scrub PII from weather responses
    # - Log weather interactions for audit
