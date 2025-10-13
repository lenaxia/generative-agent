"""Weather Role Lifecycle Functions

Pre-processing and post-processing functions for the hybrid weather role.
These functions handle data fetching and result formatting for the weather role.
"""

import logging
from datetime import datetime
from typing import Any

from common.task_context import TaskContext
from roles.weather.tools import get_weather, get_weather_forecast

logger = logging.getLogger(__name__)


async def fetch_weather_data(
    instruction: str, context: TaskContext, parameters: dict
) -> dict[str, Any]:
    """Pre-processor: Fetch weather data before LLM call.

    Args:
        instruction: Original user instruction
        context: Task context
        parameters: Extracted parameters (location, timeframe)

    Returns:
        Dict containing weather data for LLM context
    """
    location = parameters.get("location")
    timeframe = parameters.get("timeframe", "current")
    format_requested = parameters.get("format", "brief")

    if not location:
        raise ValueError("Location parameter is required for weather data")

    try:
        # Fetch current weather data
        if timeframe in ["current", "now", "today"]:
            weather_result = get_weather(location)
        else:
            # For future timeframes, get forecast
            weather_result = get_weather_forecast(location, days=7)

        if weather_result.get("status") == "error":
            raise ValueError(f"Weather API error: {weather_result.get('error')}")

        return {
            "weather_current": weather_result.get("weather", {}),
            "location_resolved": weather_result.get("location", location),
            "coordinates": weather_result.get("coordinates", {}),
            "data_timestamp": datetime.now().isoformat(),
            "timeframe_requested": timeframe,
            "format_requested": format_requested,
        }

    except Exception as e:
        logger.error(f"Failed to fetch weather data for {location}: {e}")
        raise


async def validate_location(
    instruction: str, context: TaskContext, parameters: dict
) -> dict[str, Any]:
    """Pre-processor: Validate and normalize location parameter.

    Args:
        instruction: Original user instruction
        context: Task context
        parameters: Extracted parameters (location)

    Returns:
        Dict containing validated location data
    """
    location = parameters.get("location")

    if not location:
        return {"validation_error": "No location provided"}

    try:
        # Basic location validation and normalization
        location = location.strip()

        # Check if it looks like coordinates
        if "," in location and all(
            part.replace(".", "").replace("-", "").isdigit()
            for part in location.split(",")
        ):
            return {
                "location_type": "coordinates",
                "location_normalized": location,
                "validation_status": "valid",
            }

        # Check if it looks like a ZIP code
        if location.isdigit() and len(location) == 5:
            return {
                "location_type": "zip_code",
                "location_normalized": location,
                "validation_status": "valid",
            }

        # Treat as city name
        return {
            "location_type": "city_name",
            "location_normalized": location.title(),
            "validation_status": "valid",
        }

    except Exception as e:
        logger.error(f"Location validation failed for {location}: {e}")
        return {"validation_error": str(e), "validation_status": "invalid"}


async def format_for_tts(llm_result: str, context: TaskContext, pre_data: dict) -> str:
    """Post-processor: Format LLM result for text-to-speech.

    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        TTS-formatted result
    """
    try:
        # Remove markdown formatting
        tts_result = llm_result.replace("**", "").replace("*", "")

        # Add natural pauses
        tts_result = tts_result.replace(".", ". ")
        tts_result = tts_result.replace(",", ", ")

        # Replace technical terms with pronunciations
        replacements = {
            "°F": " degrees Fahrenheit",
            "°C": " degrees Celsius",
            "mph": " miles per hour",
            "km/h": " kilometers per hour",
            "%": " percent",
            "UV": "U V",
            "PM2.5": "P M 2.5",
            "AQI": "Air Quality Index",
        }

        for old, new in replacements.items():
            tts_result = tts_result.replace(old, new)

        # Ensure proper sentence structure
        tts_result = tts_result.strip()
        if not tts_result.endswith("."):
            tts_result += "."

        return tts_result

    except Exception as e:
        logger.error(f"TTS formatting failed: {e}")
        return llm_result  # Return original on failure


async def pii_scrubber(llm_result: str, context: TaskContext, pre_data: dict) -> str:
    """Post-processor: Remove sensitive data from weather responses.

    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        Scrubbed result with PII removed
    """
    try:
        # For weather data, we mainly need to be careful about
        # exact coordinates that might reveal precise locations
        scrubbed_result = llm_result

        # Remove exact coordinates if present (keep general location)
        import re

        # Pattern for precise coordinates (more than 2 decimal places)
        precise_coord_pattern = r"-?\d+\.\d{3,},-?\d+\.\d{3,}"
        scrubbed_result = re.sub(
            precise_coord_pattern, "[coordinates removed]", scrubbed_result
        )

        # Remove any API keys that might have leaked
        api_key_pattern = r"[Aa][Pp][Ii]_?[Kk][Ee][Yy][:=]\s*[A-Za-z0-9]+"
        scrubbed_result = re.sub(api_key_pattern, "[API key removed]", scrubbed_result)

        return scrubbed_result

    except Exception as e:
        logger.error(f"PII scrubbing failed: {e}")
        return llm_result  # Return original on failure


async def audit_log(llm_result: str, context: TaskContext, pre_data: dict) -> str:
    """Post-processor: Log interaction for compliance and monitoring.

    Args:
        llm_result: Result from LLM processing
        context: Task context
        pre_data: Data from pre-processing

    Returns:
        Original result (unchanged)
    """
    try:
        # Extract relevant information for audit logging
        location = None
        timeframe = None

        # Get location from pre-processing data
        for _func_name, data in pre_data.items():
            if isinstance(data, dict):
                if "location_resolved" in data:
                    location = data["location_resolved"]
                if "timeframe_requested" in data:
                    timeframe = data["timeframe_requested"]

        # Log the interaction (in production, this would go to a proper audit system)
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "context_id": getattr(context, "context_id", "unknown"),
            "role": "weather",
            "location_requested": location,
            "timeframe_requested": timeframe,
            "response_length": len(llm_result),
            "success": True,
        }

        logger.info(f"Weather interaction audit: {audit_entry}")

        return llm_result  # Return unchanged

    except Exception as e:
        logger.error(f"Audit logging failed: {e}")
        return llm_result  # Return original on failure
