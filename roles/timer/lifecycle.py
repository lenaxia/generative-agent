"""Timer Role Lifecycle Functions

Pre-processing and post-processing functions for the hybrid timer role.
Handles parameter extraction, validation, timer operations, and confirmation formatting.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def parse_timer_parameters(
    action: str,
    duration: Optional[str] = None,
    time: Optional[str] = None,
    label: Optional[str] = None,
    timer_id: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse and normalize timer parameters from user input.

    Args:
        action: Timer action (set, cancel, list, alarm_set, alarm_cancel)
        duration: Timer duration string (e.g., "5m", "1h30m", "120")
        time: Alarm time string (e.g., "14:30", "2:30 PM")
        label: Optional timer/alarm label
        timer_id: Timer ID for cancel operations

    Returns:
        Dict containing parsed and normalized parameters
    """
    try:
        parsed_data = {
            "action_requested": action,
            "timer_duration": None,
            "alarm_time": None,
            "timer_label": label or "",
            "timer_id": timer_id or "",
            "duration_seconds": None,
            "alarm_datetime": None,
        }

        # Parse duration for timer operations
        if duration and action == "set":
            duration_seconds = _parse_duration_to_seconds(duration)
            if duration_seconds:
                parsed_data["timer_duration"] = duration
                parsed_data["duration_seconds"] = duration_seconds
                # Calculate expiry time
                expiry_time = datetime.now() + timedelta(seconds=duration_seconds)
                parsed_data["expiry_time"] = expiry_time.strftime("%H:%M:%S")

        # Parse time for alarm operations
        if time and action == "alarm_set":
            alarm_datetime = _parse_alarm_time(time)
            if alarm_datetime:
                parsed_data["alarm_time"] = time
                parsed_data["alarm_datetime"] = alarm_datetime

        logger.info(f"Parsed timer parameters: {parsed_data}")
        return parsed_data

    except Exception as e:
        logger.error(f"Failed to parse timer parameters: {e}")
        return {
            "action_requested": action,
            "error": f"Parameter parsing failed: {str(e)}",
        }


async def validate_timer_request(
    action: str, duration: Optional[str] = None, time: Optional[str] = None, **kwargs
) -> dict[str, Any]:
    """Validate timer request parameters.

    Args:
        action: Timer action to validate
        duration: Duration string for validation
        time: Time string for validation

    Returns:
        Dict containing validation results
    """
    try:
        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Validate action
        valid_actions = ["set", "cancel", "list", "alarm_set", "alarm_cancel"]
        if action not in valid_actions:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Invalid action '{action}'. Must be one of: {valid_actions}"
            )

        # Validate duration for timer set
        if action == "set":
            if not duration:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Duration is required for timer set operations"
                )
            elif not _parse_duration_to_seconds(duration):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid duration format '{duration}'. Use formats like '5m', '1h30m', '120'"
                )

        # Validate time for alarm set
        if action == "alarm_set":
            if not time:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Time is required for alarm set operations"
                )
            elif not _parse_alarm_time(time):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid time format '{time}'. Use formats like '14:30', '2:30 PM'"
                )

        logger.info(f"Timer validation result: {validation_result}")
        return {"validation_result": validation_result}

    except Exception as e:
        logger.error(f"Timer validation failed: {e}")
        return {
            "validation_result": {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }
        }


async def format_timer_confirmation(llm_response: str, **kwargs) -> str:
    """Format the LLM response for clear timer confirmation.

    Args:
        llm_response: Raw response from LLM

    Returns:
        Formatted confirmation message
    """
    try:
        # Clean up the response and ensure it's user-friendly
        formatted_response = llm_response.strip()

        # Add timestamp for confirmation
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_response += f"\n\n⏰ Confirmed at {timestamp}"

        logger.info("Timer confirmation formatted successfully")
        return formatted_response

    except Exception as e:
        logger.error(f"Failed to format timer confirmation: {e}")
        return f"Timer operation completed. (Formatting error: {str(e)})"


async def schedule_notification(llm_response: str, **kwargs) -> dict[str, Any]:
    """Schedule system notification for timer/alarm (placeholder implementation).

    Args:
        llm_response: LLM response containing timer details

    Returns:
        Dict containing notification scheduling result
    """
    try:
        # Placeholder implementation - would integrate with system notifications
        logger.info("Timer notification scheduling requested (not implemented)")
        return {
            "notification_scheduled": False,
            "message": "Notification scheduling not implemented - requires system integration",
        }

    except Exception as e:
        logger.error(f"Failed to schedule notification: {e}")
        return {"notification_scheduled": False, "error": str(e)}


async def audit_timer_action(llm_response: str, **kwargs) -> dict[str, Any]:
    """Audit log timer action for tracking and compliance.

    Args:
        llm_response: LLM response containing timer details

    Returns:
        Dict containing audit logging result
    """
    try:
        # Log timer action for audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "timer_operation",
            "response": llm_response[:200],  # Truncate for logging
            "user": "system",  # Would be actual user in real implementation
        }

        logger.info(f"Timer audit entry: {audit_entry}")
        return {
            "audit_logged": True,
            "entry_id": f"timer_{int(datetime.now().timestamp())}",
        }

    except Exception as e:
        logger.error(f"Failed to audit timer action: {e}")
        return {"audit_logged": False, "error": str(e)}


def _parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """Parse duration string to seconds.

    Args:
        duration_str: Duration in formats like "5m", "1h30m", "120"

    Returns:
        Duration in seconds or None if invalid
    """
    try:
        duration_str = duration_str.lower().strip()

        # Handle plain number (assume seconds)
        if duration_str.isdigit():
            return int(duration_str)

        # Parse complex duration (e.g., "1h30m", "5m", "2h")
        total_seconds = 0

        # Extract hours
        hours_match = re.search(r"(\d+)h", duration_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600

        # Extract minutes
        minutes_match = re.search(r"(\d+)m", duration_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60

        # Extract seconds
        seconds_match = re.search(r"(\d+)s", duration_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))

        return total_seconds if total_seconds > 0 else None

    except Exception as e:
        logger.error(f"Failed to parse duration '{duration_str}': {e}")
        return None


def _parse_alarm_time(time_str: str) -> Optional[datetime]:
    """Parse alarm time string to datetime.

    Args:
        time_str: Time in formats like "14:30", "2:30 PM", "2024-12-25 09:00"

    Returns:
        Parsed datetime or None if invalid
    """
    try:
        time_str = time_str.strip()

        # Try different time formats
        time_formats = [
            "%H:%M",  # 14:30
            "%I:%M %p",  # 2:30 PM
            "%Y-%m-%d %H:%M",  # 2024-12-25 09:00
            "%m/%d/%Y %H:%M",  # 12/25/2024 09:00
        ]

        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt)

                # If no date specified, assume today
                if fmt in ["%H:%M", "%I:%M %p"]:
                    today = datetime.now().date()
                    parsed_time = datetime.combine(today, parsed_time.time())

                return parsed_time

            except ValueError:
                continue

        return None

    except Exception as e:
        logger.error(f"Failed to parse alarm time '{time_str}': {e}")
        return None
