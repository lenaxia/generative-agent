"""Weather Domain Tools

Provides weather-related tools for the dynamic agent system.
Tools are query-only (read weather data) - no action intents needed.

Extracted from: roles/core_weather.py
"""

import logging
from datetime import datetime
from typing import Any

import requests
from strands import tool

logger = logging.getLogger(__name__)


def create_weather_tools(weather_provider: Any) -> list:
    """Create weather domain tools.

    Args:
        weather_provider: Weather provider instance (currently unused, uses NOAA API directly)

    Returns:
        List of tool functions for weather domain
    """
    # Note: Current implementation uses NOAA API directly via requests
    # The weather_provider parameter is kept for future provider abstraction

    tools = [
        get_current_weather,
        get_forecast,
    ]

    logger.info(f"Created {len(tools)} weather tools")
    return tools


# QUERY TOOLS (read-only, no side effects)


@tool
async def get_current_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Query tool - reads weather data, no side effects, no intents needed.

    Args:
        location: City name, ZIP code, or coordinates (lat,lon)

    Returns:
        JSON string with current weather data
    """
    logger.info(f"Getting current weather for: {location}")

    try:
        # Convert location to coordinates
        if _is_coordinates(location):
            lat, lon = map(float, location.split(","))
            coordinates = {"lat": lat, "lon": lon}
        elif _is_zipcode(location):
            coordinates = _zipcode_to_coordinates(location, "US")
        else:
            coordinates = _city_to_coordinates(location)

        # Fetch current weather data
        weather_data = _check_weather(coordinates["lat"], coordinates["lon"])

        result = {
            "location": location,
            "coordinates": coordinates,
            "weather": weather_data,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        logger.info(f"Weather retrieved successfully for {location}")

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting weather for {location}: {e}")
        result = {
            "location": location,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        }

        import json
        return json.dumps(result, indent=2)


@tool
async def get_forecast(location: str, days: int = 7) -> str:
    """Get extended weather forecast for a location.

    Query tool - reads forecast data, no side effects, no intents needed.

    Args:
        location: City name, ZIP code, or coordinates (lat,lon)
        days: Number of days to forecast (default: 7, max: 14)

    Returns:
        JSON string with forecast data
    """
    logger.info(f"Getting {days}-day forecast for: {location}")

    try:
        # Limit days to reasonable maximum
        days = min(days, 14)

        # Convert location to coordinates
        if _is_coordinates(location):
            lat, lon = map(float, location.split(","))
        elif _is_zipcode(location):
            coords = _zipcode_to_coordinates(location, "US")
            lat, lon = coords["lat"], coords["lon"]
        else:
            coords = _city_to_coordinates(location)
            lat, lon = coords["lat"], coords["lon"]

        # Fetch forecast data
        forecast_data = _get_forecast_data(lat, lon, days)

        result = {
            "location": location,
            "coordinates": {"lat": lat, "lon": lon},
            "forecast_days": days,
            "periods": forecast_data,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        logger.info(
            f"Forecast retrieved for {location}: {len(forecast_data)} periods"
        )

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting forecast for {location}: {e}")
        result = {
            "location": location,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        }

        import json
        return json.dumps(result, indent=2)


# HELPER FUNCTIONS (not exposed as tools)


def _is_coordinates(location: str) -> bool:
    """Check if location string is coordinates (lat,lon format)."""
    return (
        "," in location
        and location.replace(",", "").replace(".", "").replace("-", "").isdigit()
    )


def _is_zipcode(location: str) -> bool:
    """Check if location string is a US ZIP code."""
    return location.isdigit() and len(location) == 5


def _city_to_coordinates(city: str) -> dict[str, float]:
    """Convert city name to coordinates using OpenStreetMap Nominatim API.

    Args:
        city: City name (e.g., "Seattle", "New York, NY")

    Returns:
        Dictionary with lat and lon keys

    Raises:
        ValueError: If city cannot be geocoded
    """
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
    """Convert ZIP code to coordinates using OpenStreetMap Nominatim API.

    Args:
        zipcode: ZIP/postal code
        country_code: Country code (default: "US")

    Returns:
        Dictionary with lat and lon keys

    Raises:
        ValueError: If ZIP code cannot be geocoded
    """
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

        # Filter for country-specific results
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
    """Fetch current weather data from NOAA API.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary with current weather data

    Raises:
        ValueError: If weather data cannot be fetched
    """
    logger.info(f"Checking weather for coordinates: {lat}, {lon}")

    try:
        # Get forecast URL from NOAA points API
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        # Get current conditions from forecast
        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_data = forecast_response.json()

        if (
            "properties" not in forecast_data
            or "periods" not in forecast_data["properties"]
        ):
            raise ValueError("Could not fetch weather forecast for that location")

        # Extract current period (first period)
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
    """Fetch extended forecast data from NOAA API.

    Args:
        lat: Latitude
        lon: Longitude
        days: Number of forecast days

    Returns:
        List of forecast period dictionaries

    Raises:
        ValueError: If forecast data cannot be fetched
    """
    logger.info(f"Getting {days}-day forecast for coordinates: {lat}, {lon}")

    try:
        # Get forecast URL from NOAA points API
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {"User-Agent": "TheKaoCloud/1.0", "Referer": "https://thekao.cloud"}

        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        # Get extended forecast
        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_data = forecast_response.json()

        if (
            "properties" not in forecast_data
            or "periods" not in forecast_data["properties"]
        ):
            raise ValueError("Could not fetch weather forecast for that location")

        # Extract periods (day and night, so days * 2)
        periods = forecast_data["properties"]["periods"][: days * 2]

        forecast_periods = [
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

        logger.info(
            f"Forecast data retrieved: {len(forecast_periods)} periods for {days} days"
        )
        return forecast_periods

    except Exception as e:
        logger.error(f"Error getting forecast data: {e}")
        raise
