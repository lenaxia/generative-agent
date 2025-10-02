"""
Weather tools for StrandsAgent - converted from WeatherAgent.

These tools replace the LangChain-based WeatherAgent with @tool decorated functions
that can be used by the Universal Agent for weather information retrieval.
"""

import requests
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from strands import tool

logger = logging.getLogger(__name__)


@tool
def get_weather(location: str) -> Dict[str, Any]:
    """
    Get weather information for a location - converted from WeatherAgent.
    
    This tool can handle city names, ZIP codes, or coordinates and returns
    current weather information using the NOAA Weather API.
    
    Args:
        location: Location as city name, ZIP code, or "lat,lon" coordinates
        
    Returns:
        Dict containing weather information and metadata
    """
    logger.info(f"Getting weather for location: {location}")
    
    try:
        # Determine if location is coordinates, ZIP code, or city name
        if ',' in location and location.replace(',', '').replace('.', '').replace('-', '').isdigit():
            # Coordinates format: "lat,lon"
            lat, lon = map(float, location.split(','))
            coordinates = {"lat": lat, "lon": lon}
        elif location.isdigit() and len(location) == 5:
            # ZIP code
            coordinates = zipcode_to_coordinates(location, "US")
        else:
            # City name
            coordinates = city_to_coordinates(location)
        
        # Get weather using coordinates
        weather_data = check_weather(coordinates["lat"], coordinates["lon"])
        
        result = {
            "location": location,
            "coordinates": coordinates,
            "weather": weather_data,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info(f"Weather retrieved successfully for {location}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting weather for {location}: {e}")
        return {
            "location": location,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


def check_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Check weather for specific coordinates using NOAA API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Dict containing detailed weather information
    """
    logger.info(f"Checking weather for coordinates: {lat}, {lon}")
    
    try:
        # Fetch weather data from NOAA API
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        points_response = requests.get(points_url, headers=headers)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_data = forecast_response.json()

        if "properties" not in forecast_data or "periods" not in forecast_data["properties"]:
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
            "is_daytime": current_period.get("isDaytime")
        }
        
        logger.info(f"Weather data retrieved for coordinates {lat}, {lon}")
        return weather_info
        
    except Exception as e:
        logger.error(f"Error checking weather for coordinates {lat}, {lon}: {e}")
        raise


def city_to_coordinates(city: str) -> Dict[str, float]:
    """
    Convert a city name to latitude and longitude coordinates.
    
    Args:
        city: City name to convert
        
    Returns:
        Dict containing lat and lon coordinates
    """
    logger.info(f"Converting city to coordinates: {city}")
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json"
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        response = requests.get(url, headers=headers)
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


def zipcode_to_coordinates(zipcode: str, country_code: str = "US") -> Dict[str, float]:
    """
    Convert a ZIP code to latitude and longitude coordinates.
    
    Args:
        zipcode: ZIP code to convert
        country_code: Two-letter country code (default: "US")
        
    Returns:
        Dict containing lat and lon coordinates
    """
    logger.info(f"Converting ZIP code to coordinates: {zipcode} ({country_code})")
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={zipcode}&format=json&country={country_code}"
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        response = requests.get(url, headers=headers)
        response_data = response.json()

        if not response_data:
            raise ValueError(f"Could not find coordinates for ZIP code '{zipcode}' in {country_code}")

        # Filter results to get the coordinates for the ZIP code in the specified country
        country_result = None
        for result in response_data:
            if country_code.upper() == "US" and "United States" in result["display_name"]:
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


def get_weather_forecast(location: str, days: int = 7) -> Dict[str, Any]:
    """
    Get extended weather forecast for a location.
    
    Args:
        location: Location as city name, ZIP code, or coordinates
        days: Number of days to forecast (default: 7)
        
    Returns:
        Dict containing extended forecast information
    """
    logger.info(f"Getting {days}-day forecast for location: {location}")
    
    try:
        # Get coordinates for location
        if ',' in location and location.replace(',', '').replace('.', '').replace('-', '').isdigit():
            lat, lon = map(float, location.split(','))
        elif location.isdigit() and len(location) == 5:
            coords = zipcode_to_coordinates(location, "US")
            lat, lon = coords["lat"], coords["lon"]
        else:
            coords = city_to_coordinates(location)
            lat, lon = coords["lat"], coords["lon"]
        
        # Get extended forecast from NOAA
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {
            'User-Agent': 'TheKaoCloud/1.0',
            'Referer': 'https://thekao.cloud'
        }

        points_response = requests.get(points_url, headers=headers)
        points_data = points_response.json()

        if "properties" not in points_data:
            raise ValueError("Could not fetch weather data for that location")

        forecast_url = points_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_data = forecast_response.json()

        if "properties" not in forecast_data or "periods" not in forecast_data["properties"]:
            raise ValueError("Could not fetch weather forecast for that location")

        periods = forecast_data["properties"]["periods"][:days * 2]  # Day and night periods
        
        forecast_result = {
            "location": location,
            "coordinates": {"lat": lat, "lon": lon},
            "forecast_days": days,
            "periods": [
                {
                    "name": period.get("name"),
                    "temperature": period.get("temperature"),
                    "temperature_unit": period.get("temperatureUnit"),
                    "wind_speed": period.get("windSpeed"),
                    "wind_direction": period.get("windDirection"),
                    "short_forecast": period.get("shortForecast"),
                    "detailed_forecast": period.get("detailedForecast"),
                    "is_daytime": period.get("isDaytime")
                }
                for period in periods
            ],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info(f"Extended forecast retrieved for {location}: {len(periods)} periods")
        return forecast_result
        
    except Exception as e:
        logger.error(f"Error getting forecast for {location}: {e}")
        return {
            "location": location,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


def validate_coordinates(lat: float, lon: float) -> Dict[str, Any]:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Dict containing validation results
    """
    logger.info(f"Validating coordinates: {lat}, {lon}")
    
    errors = []
    warnings = []
    
    # Validate latitude
    if not isinstance(lat, (int, float)):
        errors.append("Latitude must be a number")
    elif lat < -90 or lat > 90:
        errors.append("Latitude must be between -90 and 90 degrees")
    
    # Validate longitude
    if not isinstance(lon, (int, float)):
        errors.append("Longitude must be a number")
    elif lon < -180 or lon > 180:
        errors.append("Longitude must be between -180 and 180 degrees")
    
    # Check for common issues
    if abs(lat) < 0.001 and abs(lon) < 0.001:
        warnings.append("Coordinates are very close to (0,0) - verify this is correct")
    
    is_valid = len(errors) == 0
    
    result = {
        "lat": lat,
        "lon": lon,
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "hemisphere": {
            "lat": "North" if lat >= 0 else "South",
            "lon": "East" if lon >= 0 else "West"
        }
    }
    
    logger.info(f"Coordinate validation: {'PASSED' if is_valid else 'FAILED'} with {len(errors)} errors")
    return result


# Tool registry for weather tools
WEATHER_TOOLS = {
    "get_weather": get_weather,
    "check_weather": check_weather,
    "city_to_coordinates": city_to_coordinates,
    "zipcode_to_coordinates": zipcode_to_coordinates,
    "get_weather_forecast": get_weather_forecast,
    "validate_coordinates": validate_coordinates
}


def get_weather_tools() -> Dict[str, Any]:
    """Get all available weather tools."""
    return WEATHER_TOOLS


def get_weather_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all weather tools."""
    return {
        "get_weather": "Get current weather for any location (city, ZIP, or coordinates)",
        "check_weather": "Check weather for specific latitude/longitude coordinates",
        "city_to_coordinates": "Convert city name to coordinates",
        "zipcode_to_coordinates": "Convert ZIP code to coordinates",
        "get_weather_forecast": "Get extended weather forecast for a location",
        "validate_coordinates": "Validate latitude and longitude coordinates"
    }