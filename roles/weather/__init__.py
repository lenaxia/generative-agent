"""Weather Domain

Provides weather-related tools for querying current conditions and forecasts.
"""

from .tools import create_weather_tools
from .role import WeatherRole

__all__ = ["create_weather_tools", "WeatherRole"]
