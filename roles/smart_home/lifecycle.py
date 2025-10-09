"""Smart Home Role Lifecycle Functions

Pre-processing and post-processing functions for the hybrid smart home role.
Handles parameter extraction, validation, device operations, and confirmation formatting.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def parse_smart_home_parameters(
    action: str,
    device: Optional[str] = None,
    value: Optional[str] = None,
    scene: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse and normalize smart home parameters from user input.

    Args:
        action: Smart home action (lights_on, lights_off, set_brightness, etc.)
        device: Target device or room
        value: Operation value (brightness %, temperature, etc.)
        scene: Scene name for scene operations

    Returns:
        Dict containing parsed and normalized parameters
    """
    try:
        parsed_data = {
            "action_requested": action,
            "target_device": device or "",
            "operation_value": value or "",
            "scene_name": scene or "",
            "normalized_value": None,
            "device_type": None,
        }

        # Determine device type from device name or action
        if device:
            parsed_data["device_type"] = _determine_device_type(device)
        elif action in ["set_temperature"]:
            parsed_data["device_type"] = "thermostat"
        elif action in ["lights_on", "lights_off", "set_brightness"]:
            parsed_data["device_type"] = "lights"

        # Normalize value based on action
        if value and action == "set_brightness":
            normalized_brightness = _parse_brightness_value(value)
            if normalized_brightness is not None:
                parsed_data["normalized_value"] = normalized_brightness

        if value and action == "set_temperature":
            normalized_temp = _parse_temperature_value(value)
            if normalized_temp is not None:
                parsed_data["normalized_value"] = normalized_temp

        logger.info(f"Parsed smart home parameters: {parsed_data}")
        return parsed_data

    except Exception as e:
        logger.error(f"Failed to parse smart home parameters: {e}")
        return {
            "action_requested": action,
            "error": f"Parameter parsing failed: {str(e)}",
        }


async def validate_device_request(
    action: str, device: Optional[str] = None, value: Optional[str] = None, **kwargs
) -> dict[str, Any]:
    """Validate smart home device request parameters.

    Args:
        action: Smart home action to validate
        device: Device name for validation
        value: Value for validation

    Returns:
        Dict containing validation results
    """
    try:
        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Validate action
        valid_actions = [
            "lights_on",
            "lights_off",
            "set_brightness",
            "set_temperature",
            "device_status",
            "scene_activate",
        ]
        if action not in valid_actions:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Invalid action '{action}'. Must be one of: {valid_actions}"
            )

        # Validate brightness values
        if action == "set_brightness" and value:
            brightness = _parse_brightness_value(value)
            if brightness is None:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid brightness value '{value}'. Use 0-100% or descriptors like 'dim', 'bright'"
                )

        # Validate temperature values
        if action == "set_temperature" and value:
            temp = _parse_temperature_value(value)
            if temp is None:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid temperature value '{value}'. Use formats like '72°F', '22°C', or '72'"
                )
            elif temp < 50 or temp > 90:  # Reasonable thermostat range in Fahrenheit
                validation_result["warnings"].append(
                    f"Temperature {temp}°F is outside typical range (50-90°F)"
                )

        # Validate device exists (placeholder - would check actual device registry)
        if device and not _is_valid_device(device):
            validation_result["warnings"].append(
                f"Device '{device}' may not be available - please verify"
            )

        logger.info(f"Smart home validation result: {validation_result}")
        return {"device_validation": validation_result}

    except Exception as e:
        logger.error(f"Smart home validation failed: {e}")
        return {
            "device_validation": {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }
        }


async def format_device_confirmation(llm_response: str, **kwargs) -> str:
    """Format the LLM response for clear device operation confirmation.

    Args:
        llm_response: Raw response from LLM

    Returns:
        Formatted confirmation message
    """
    try:
        # Clean up the response and ensure it's user-friendly
        formatted_response = llm_response.strip()

        # Add timestamp and device status indicator
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_response += f"\n\n🏠 Device operation confirmed at {timestamp}"

        logger.info("Smart home confirmation formatted successfully")
        return formatted_response

    except Exception as e:
        logger.error(f"Failed to format device confirmation: {e}")
        return f"Smart home operation completed. (Formatting error: {str(e)})"


async def update_device_status(llm_response: str, **kwargs) -> dict[str, Any]:
    """Update device status tracking (placeholder implementation).

    Args:
        llm_response: LLM response containing device operation details

    Returns:
        Dict containing device status update result
    """
    try:
        # Placeholder implementation - would integrate with device status database
        logger.info("Device status update requested (not implemented)")
        return {
            "status_updated": False,
            "message": "Device status tracking not implemented - requires device registry integration",
        }

    except Exception as e:
        logger.error(f"Failed to update device status: {e}")
        return {"status_updated": False, "error": str(e)}


async def audit_smart_home_action(llm_response: str, **kwargs) -> dict[str, Any]:
    """Audit log smart home action for security tracking and compliance.

    Args:
        llm_response: LLM response containing device operation details

    Returns:
        Dict containing audit logging result
    """
    try:
        # Log smart home action for security audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "smart_home_operation",
            "response": llm_response[:200],  # Truncate for logging
            "user": "system",  # Would be actual user in real implementation
            "security_level": "device_control",
        }

        logger.info(f"Smart home audit entry: {audit_entry}")
        return {
            "audit_logged": True,
            "entry_id": f"smarthome_{int(datetime.now().timestamp())}",
        }

    except Exception as e:
        logger.error(f"Failed to audit smart home action: {e}")
        return {"audit_logged": False, "error": str(e)}


def _determine_device_type(device_name: str) -> str:
    """Determine device type from device name.

    Args:
        device_name: Name of the device or room

    Returns:
        Device type category
    """
    device_name_lower = device_name.lower()

    if any(word in device_name_lower for word in ["light", "lamp", "bulb"]):
        return "lights"
    elif any(
        word in device_name_lower
        for word in ["thermostat", "temperature", "heat", "cool"]
    ):
        return "thermostat"
    elif any(word in device_name_lower for word in ["switch", "outlet", "plug"]):
        return "switch"
    elif any(word in device_name_lower for word in ["fan", "ceiling"]):
        return "fan"
    else:
        return "generic"


def _parse_brightness_value(value_str: str) -> Optional[int]:
    """Parse brightness value to percentage (0-100).

    Args:
        value_str: Brightness value string

    Returns:
        Brightness percentage or None if invalid
    """
    try:
        value_str = value_str.lower().strip()

        # Handle percentage values
        if "%" in value_str:
            percent_match = re.search(r"(\d+)%", value_str)
            if percent_match:
                percent = int(percent_match.group(1))
                return max(0, min(100, percent))  # Clamp to 0-100

        # Handle numeric values (assume percentage)
        if value_str.isdigit():
            percent = int(value_str)
            return max(0, min(100, percent))

        # Handle descriptive values
        descriptive_values = {
            "off": 0,
            "dim": 25,
            "low": 25,
            "medium": 50,
            "bright": 75,
            "high": 75,
            "max": 100,
            "full": 100,
        }

        return descriptive_values.get(value_str)

    except Exception as e:
        logger.error(f"Failed to parse brightness value '{value_str}': {e}")
        return None


def _parse_temperature_value(value_str: str) -> Optional[float]:
    """Parse temperature value to Fahrenheit.

    Args:
        value_str: Temperature value string

    Returns:
        Temperature in Fahrenheit or None if invalid
    """
    try:
        value_str = value_str.strip()

        # Handle Fahrenheit
        if "°f" in value_str.lower() or "f" in value_str.lower():
            temp_match = re.search(r"(\d+(?:\.\d+)?)", value_str)
            if temp_match:
                return float(temp_match.group(1))

        # Handle Celsius (convert to Fahrenheit)
        if "°c" in value_str.lower() or "c" in value_str.lower():
            temp_match = re.search(r"(\d+(?:\.\d+)?)", value_str)
            if temp_match:
                celsius = float(temp_match.group(1))
                return (celsius * 9 / 5) + 32

        # Handle plain number (assume Fahrenheit)
        if re.match(r"^\d+(?:\.\d+)?$", value_str):
            return float(value_str)

        return None

    except Exception as e:
        logger.error(f"Failed to parse temperature value '{value_str}': {e}")
        return None


def _is_valid_device(device_name: str) -> bool:
    """Check if device name is valid (placeholder implementation).

    Args:
        device_name: Device name to validate

    Returns:
        True if device is valid (always True in placeholder)
    """
    # Placeholder implementation - would check actual device registry
    # For now, assume all devices are valid
    return True
