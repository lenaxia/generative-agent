"""Calendar Role Lifecycle Functions

Pre-processing and post-processing functions for the hybrid calendar role.
Handles parameter extraction, validation, calendar operations, and confirmation formatting.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def parse_calendar_parameters(
    action: str,
    title: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse and normalize calendar parameters from user input.

    Args:
        action: Calendar action (add, update, delete, get, check_availability)
        title: Event title
        start_time: Event start time
        end_time: Event end time
        event_id: Event ID for update/delete operations
        description: Event description
        location: Event location

    Returns:
        Dict containing parsed and normalized parameters
    """
    try:
        parsed_data = {
            "action_requested": action,
            "event_title": title or "",
            "start_time": start_time or "",
            "end_time": end_time or "",
            "event_id": event_id or "",
            "event_description": description or "",
            "event_location": location or "",
            "parsed_start_datetime": None,
            "parsed_end_datetime": None,
        }

        # Parse start time
        if start_time:
            parsed_start = _parse_datetime_string(start_time)
            if parsed_start:
                parsed_data["parsed_start_datetime"] = parsed_start
                parsed_data["formatted_start"] = parsed_start.strftime("%Y-%m-%d %H:%M")

        # Parse end time
        if end_time:
            parsed_end = _parse_datetime_string(end_time)
            if parsed_end:
                parsed_data["parsed_end_datetime"] = parsed_end
                parsed_data["formatted_end"] = parsed_end.strftime("%Y-%m-%d %H:%M")

        # Calculate duration if both times provided
        if parsed_data.get("parsed_start_datetime") and parsed_data.get(
            "parsed_end_datetime"
        ):
            duration = (
                parsed_data["parsed_end_datetime"]
                - parsed_data["parsed_start_datetime"]
            )
            parsed_data["event_duration"] = str(duration)

        logger.info(f"Parsed calendar parameters: {parsed_data}")
        return parsed_data

    except Exception as e:
        logger.error(f"Failed to parse calendar parameters: {e}")
        return {
            "action_requested": action,
            "error": f"Parameter parsing failed: {str(e)}",
        }


async def validate_calendar_request(
    action: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """Validate calendar request parameters.

    Args:
        action: Calendar action to validate
        start_time: Start time for validation
        end_time: End time for validation

    Returns:
        Dict containing validation results
    """
    try:
        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Validate action
        valid_actions = ["add", "update", "delete", "get", "check_availability"]
        if action not in valid_actions:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Invalid action '{action}'. Must be one of: {valid_actions}"
            )

        # Validate times for add/update operations
        if action in ["add", "update"]:
            if start_time:
                parsed_start = _parse_datetime_string(start_time)
                if not parsed_start:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Invalid start time format '{start_time}'"
                    )
                elif parsed_start < datetime.now():
                    validation_result["warnings"].append("Start time is in the past")

            if end_time:
                parsed_end = _parse_datetime_string(end_time)
                if not parsed_end:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Invalid end time format '{end_time}'"
                    )
                elif start_time:
                    parsed_start = _parse_datetime_string(start_time)
                    if parsed_start and parsed_end <= parsed_start:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            "End time must be after start time"
                        )

        logger.info(f"Calendar validation result: {validation_result}")
        return {"validation_result": validation_result}

    except Exception as e:
        logger.error(f"Calendar validation failed: {e}")
        return {
            "validation_result": {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }
        }


async def format_calendar_confirmation(llm_response: str, **kwargs) -> str:
    """Format the LLM response for clear calendar confirmation.

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
        formatted_response += f"\n\n📅 Confirmed at {timestamp}"

        logger.info("Calendar confirmation formatted successfully")
        return formatted_response

    except Exception as e:
        logger.error(f"Failed to format calendar confirmation: {e}")
        return f"Calendar operation completed. (Formatting error: {str(e)})"


async def schedule_calendar_reminder(llm_response: str, **kwargs) -> dict[str, Any]:
    """Schedule reminder notification for calendar event (placeholder implementation).

    Args:
        llm_response: LLM response containing event details

    Returns:
        Dict containing reminder scheduling result
    """
    try:
        # Placeholder implementation - would integrate with notification system
        logger.info("Calendar reminder scheduling requested (not implemented)")
        return {
            "reminder_scheduled": False,
            "message": "Reminder scheduling not implemented - requires notification system integration",
        }

    except Exception as e:
        logger.error(f"Failed to schedule calendar reminder: {e}")
        return {"reminder_scheduled": False, "error": str(e)}


async def audit_calendar_action(llm_response: str, **kwargs) -> dict[str, Any]:
    """Audit log calendar action for tracking and compliance.

    Args:
        llm_response: LLM response containing calendar details

    Returns:
        Dict containing audit logging result
    """
    try:
        # Log calendar action for audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "calendar_operation",
            "response": llm_response[:200],  # Truncate for logging
            "user": "system",  # Would be actual user in real implementation
        }

        logger.info(f"Calendar audit entry: {audit_entry}")
        return {
            "audit_logged": True,
            "entry_id": f"calendar_{int(datetime.now().timestamp())}",
        }

    except Exception as e:
        logger.error(f"Failed to audit calendar action: {e}")
        return {"audit_logged": False, "error": str(e)}


def _parse_datetime_string(datetime_str: str) -> Optional[datetime]:
    """Parse various datetime string formats.

    Args:
        datetime_str: Datetime string in various formats

    Returns:
        Parsed datetime or None if invalid
    """
    try:
        datetime_str = datetime_str.strip()

        # Handle relative time expressions
        if "tomorrow" in datetime_str.lower():
            tomorrow = datetime.now() + timedelta(days=1)
            time_part = re.search(
                r"(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)", datetime_str.lower()
            )
            if time_part:
                time_str = time_part.group(1)
                parsed_time = _parse_time_component(time_str)
                if parsed_time:
                    return datetime.combine(tomorrow.date(), parsed_time)

        if "next monday" in datetime_str.lower():
            # Find next Monday
            today = datetime.now()
            days_ahead = 7 - today.weekday()  # Monday is 0
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            next_monday = today + timedelta(days=days_ahead)

            time_part = re.search(
                r"(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)", datetime_str.lower()
            )
            if time_part:
                time_str = time_part.group(1)
                parsed_time = _parse_time_component(time_str)
                if parsed_time:
                    return datetime.combine(next_monday.date(), parsed_time)

        # Try standard datetime formats
        datetime_formats = [
            "%Y-%m-%d %H:%M",  # 2024-12-25 14:30
            "%Y-%m-%d %H:%M:%S",  # 2024-12-25 14:30:00
            "%m/%d/%Y %H:%M",  # 12/25/2024 14:30
            "%d/%m/%Y %H:%M",  # 25/12/2024 14:30
            "%Y-%m-%d %I:%M %p",  # 2024-12-25 2:30 PM
            "%H:%M",  # 14:30 (today)
            "%I:%M %p",  # 2:30 PM (today)
        ]

        for fmt in datetime_formats:
            try:
                parsed_datetime = datetime.strptime(datetime_str, fmt)

                # If no date specified, assume today
                if fmt in ["%H:%M", "%I:%M %p"]:
                    today = datetime.now().date()
                    parsed_datetime = datetime.combine(today, parsed_datetime.time())

                return parsed_datetime

            except ValueError:
                continue

        return None

    except Exception as e:
        logger.error(f"Failed to parse datetime '{datetime_str}': {e}")
        return None


def _parse_time_component(time_str: str) -> Optional[datetime.time]:
    """Parse time component from string.

    Args:
        time_str: Time string like "2pm", "14:30", "9am"

    Returns:
        Parsed time or None if invalid
    """
    try:
        time_formats = [
            "%H:%M",  # 14:30
            "%I%p",  # 2PM
            "%I:%M%p",  # 2:30PM
            "%I %p",  # 2 PM
            "%I:%M %p",  # 2:30 PM
        ]

        # Clean up the time string
        time_str = time_str.replace(" ", "").upper()

        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt).time()
                return parsed_time
            except ValueError:
                continue

        return None

    except Exception as e:
        logger.error(f"Failed to parse time component '{time_str}': {e}")
        return None
