"""Weather Domain

Provides weather-related tools for querying current conditions and forecasts.
"""

from .role import WeatherRole
from .tools import create_weather_tools

__all__ = ["create_weather_tools", "WeatherRole"]
