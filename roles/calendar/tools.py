"""Calendar Domain Tools

Provides calendar and scheduling tools for the dynamic agent system.
Includes both query tools (read events) and action tools (create/modify events).

Extracted from: roles/core_calendar.py
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from strands import tool

from common.calendar_providers import get_calendar_provider

logger = logging.getLogger(__name__)


def create_calendar_tools(calendar_provider: Any) -> list:
    """Create calendar domain tools.

    Args:
        calendar_provider: Calendar provider instance (or config for CalDAV)

    Returns:
        List of tool functions for calendar domain
    """
    # Note: Current implementation uses CalDAV provider from common.calendar_providers
    # The calendar_provider parameter is kept for future provider abstraction

    tools = [
        get_schedule,
        add_calendar_event,
    ]

    logger.info(f"Created {len(tools)} calendar tools")
    return tools


# HELPER FUNCTIONS


def _is_calendar_configured() -> bool:
    """Check if calendar is properly configured."""
    url = os.getenv("CALDAV_URL", "")
    username = os.getenv("CALDAV_USERNAME", "")
    password = os.getenv("CALDAV_PASSWORD", "")

    return bool(url and username and password)


def _load_calendar_config() -> dict | None:
    """Load calendar configuration from environment.

    Returns:
        Config dict if properly configured, None otherwise
    """
    if not _is_calendar_configured():
        logger.warning(
            "Calendar not configured. Set CALDAV_URL, CALDAV_USERNAME, and CALDAV_PASSWORD "
            "environment variables to enable calendar integration."
        )
        return None

    return {
        "provider": os.getenv("CALENDAR_PROVIDER", "caldav"),
        "caldav": {
            "url": os.getenv("CALDAV_URL", ""),
            "username": os.getenv("CALDAV_USERNAME", ""),
            "password": os.getenv("CALDAV_PASSWORD", ""),
        },
    }


# QUERY TOOLS (read-only)


@tool
def get_schedule(
    user_id: str, days_ahead: int = 7, location: str = None
) -> dict[str, Any]:
    """Get user's schedule from calendar provider.

    Query tool - reads calendar events, no side effects.

    Args:
        user_id: User identifier
        days_ahead: Number of days ahead to retrieve (default: 7)
        location: Optional location context for filtering events

    Returns:
        Dict with success status, events list, and message
    """
    logger.info(f"Getting schedule for {user_id}, {days_ahead} days ahead")

    # Check if calendar is configured
    config = _load_calendar_config()
    if config is None:
        return {
            "success": False,
            "events": [],
            "message": "Calendar not configured. Please set CALDAV_URL, CALDAV_USERNAME, and CALDAV_PASSWORD environment variables.",
        }

    try:
        # Get provider
        provider = get_calendar_provider(config)

        # Get events
        start = datetime.now()
        end = start + timedelta(days=days_ahead)
        events = provider.get_events(start, end)

        # Format events for display
        formatted_events = []
        for event in events:
            formatted_events.append(
                {
                    "title": event["title"],
                    "start": (
                        event["start"].isoformat()
                        if isinstance(event["start"], datetime)
                        else str(event["start"])
                    ),
                    "end": (
                        event["end"].isoformat()
                        if isinstance(event["end"], datetime)
                        else str(event["end"])
                    ),
                    "location": event.get("location"),
                }
            )

        message = f"Retrieved {len(events)} events for {user_id}"
        if location:
            message += f" (location context: {location})"

        logger.info(f"Retrieved {len(events)} events for {user_id}")
        return {"success": True, "events": formatted_events, "message": message}

    except Exception as e:
        logger.error(f"Failed to get schedule: {e}")
        return {
            "success": False,
            "events": [],
            "message": f"Calendar integration error: {str(e)}. Check your calendar configuration.",
        }


# ACTION TOOLS (write operations)
# Note: In Phase 3, these will be updated to use intent registration


@tool
def add_calendar_event(
    title: str,
    start_time: str,
    duration: int = 60,
    location: str = None,
    user_id: str = None,
) -> dict[str, Any]:
    """Add calendar event using configured provider.

    Action tool - creates calendar event (has side effects).
    TODO Phase 3: Update to register CalendarIntent instead of executing directly.

    Args:
        title: Event title
        start_time: Event start time in ISO format (e.g., "2024-03-15T14:00:00")
        duration: Event duration in minutes (default: 60)
        location: Optional event location
        user_id: Optional user identifier

    Returns:
        Dict with success status, event_id, and message
    """
    logger.info(f"Adding calendar event: {title} at {start_time}")

    # Check if calendar is configured
    config = _load_calendar_config()
    if config is None:
        return {
            "success": False,
            "event_id": None,
            "message": "Calendar not configured. Please set CALDAV_URL, CALDAV_USERNAME, and CALDAV_PASSWORD environment variables.",
        }

    try:
        # Parse start time
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

        # Get provider
        provider = get_calendar_provider(config)

        # Add event
        event = provider.add_event(
            title=title,
            start=start,
            duration=duration,
            location=location,
            description=f"Created by {user_id}" if user_id else None,
        )

        message = f"Added event: {title} at {start_time}"
        if location:
            message += f" ({location})"

        logger.info(f"Added calendar event: {event['id']}")
        return {"success": True, "event_id": event["id"], "message": message}

    except Exception as e:
        logger.error(f"Failed to add event: {e}")
        return {
            "success": False,
            "event_id": None,
            "message": f"Calendar integration error: {str(e)}. Check your calendar configuration.",
        }
