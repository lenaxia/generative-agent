"""Calendar role - LLM-friendly single file implementation for household assistant.

This role provides context-aware calendar and scheduling management functionality
following the single-file role pattern established in the Universal Agent System.

Key features:
- Context-aware scheduling with memory and location integration
- Calendar event management (add, retrieve, query)
- Integration with household assistant context system
- LLM-safe architecture with intent-based processing
- CalDAV provider abstraction (supports iCloud, Nextcloud, etc.)

Architecture: Single Event Loop + Intent-Based + Context-Aware
Created: 2025-10-16
Updated: 2025-11-01 - Added CalDAV provider integration
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from strands import tool

from common.calendar_providers import get_calendar_provider
from common.event_context import LLMSafeEventContext
from common.intents import Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "calendar",
    "version": "1.0.0",
    "description": "Calendar and scheduling management with context awareness",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "when_to_use": "Schedule management, calendar queries, event planning",
    "memory_enabled": True,
    "location_aware": True,
    "parameters": {
        "action": {
            "type": "string",
            "required": True,
            "description": "Calendar action to perform",
            "examples": ["add_event", "get_schedule", "find_conflicts"],
            "enum": ["add_event", "get_schedule", "find_conflicts"],
        },
        "title": {
            "type": "string",
            "required": False,
            "description": "Event title for add_event operations",
            "examples": ["Team Meeting", "Doctor Appointment", "Lunch with Bob"],
        },
        "start_time": {
            "type": "string",
            "required": False,
            "description": "Event start time in ISO format",
            "examples": ["2023-10-15T14:00:00", "2023-10-16T09:30:00"],
        },
        "duration": {
            "type": "integer",
            "required": False,
            "description": "Event duration in minutes",
            "examples": [30, 60, 90, 120],
        },
        "location": {
            "type": "string",
            "required": False,
            "description": "Event location",
            "examples": ["office", "conference_room", "home"],
        },
    },
    "tools": {
        "automatic": True,  # Include custom calendar tools
        "shared": ["memory_tools"],  # Unified memory for event context
        "include_builtin": False,  # Exclude calculator, file_read, shell
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode
        },
    },
    "lifecycle": {
        "pre_processing": {"enabled": True, "functions": ["load_calendar_context"]},
        "post_processing": {"enabled": True, "functions": ["save_calendar_event"]},
    },
    "prompts": {
        "system": """You are a calendar and scheduling specialist with context awareness. You can manage calendar events, retrieve schedules, and help with event planning.

Available calendar tools:
- get_schedule(user_id, days_ahead, location): Get user's schedule with optional location context
- add_calendar_event(title, start_time, duration, location, user_id): Add new calendar event

Calendar Management:
- Parse natural language time references (today, tomorrow, next week, etc.)
- Consider location context for event planning
- Use memory context for recurring events and preferences
- Provide clear confirmations for all calendar operations

Context Awareness:
- Location: Consider user's current location for event suggestions
- Memory: Recall previous scheduling preferences and patterns
- Presence: Consider household members for shared events

When users request calendar operations:
1. Parse the time and event details from natural language
2. Use appropriate calendar tools to perform the action
3. Provide clear confirmation with event details
4. Consider context for better scheduling suggestions

Always use the calendar tools to perform calendar operations."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (owned by calendar role)
@dataclass
class CalendarIntent(Intent):
    """Calendar-specific intent for event management."""

    action: str  # "add_event", "get_schedule", "find_conflicts"
    event_data: dict[str, Any]

    def validate(self) -> bool:
        """Validate calendar intent parameters."""
        return self.action in ["add_event", "get_schedule", "find_conflicts"]


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_calendar_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """Handle calendar requests with context awareness."""
    try:
        request = event_data.get("request", "")

        return [
            CalendarIntent(
                action="get_schedule",
                event_data={"query": request, "context": context.to_dict()},
            )
        ]

    except Exception as e:
        logger.error(f"Calendar request handler error: {e}")
        return [
            NotificationIntent(
                message=f"Calendar request processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


# 4. HELPER FUNCTIONS
def _is_calendar_configured() -> bool:
    """Check if calendar is properly configured."""
    url = os.getenv("CALDAV_URL", "")
    username = os.getenv("CALDAV_USERNAME", "")
    password = os.getenv("CALDAV_PASSWORD", "")

    return bool(url and username and password)


def _load_calendar_config() -> dict | None:
    """Load calendar configuration from environment or config file.

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


# 5. TOOLS
@tool
def get_schedule(
    user_id: str, days_ahead: int = 7, location: str = None
) -> dict[str, Any]:
    """Get user's schedule from calendar provider.

    Args:
        user_id: User identifier
        days_ahead: Number of days ahead to retrieve (default: 7)
        location: Optional location context for filtering events

    Returns:
        Dict with success status, events list, and message
    """
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
                    "start": event["start"].isoformat()
                    if isinstance(event["start"], datetime)
                    else str(event["start"]),
                    "end": event["end"].isoformat()
                    if isinstance(event["end"], datetime)
                    else str(event["end"]),
                    "location": event.get("location"),
                }
            )

        message = f"Retrieved {len(events)} events for {user_id}"
        if location:
            message += f" (location context: {location})"

        return {"success": True, "events": formatted_events, "message": message}

    except Exception as e:
        logger.error(f"Failed to get schedule: {e}")
        return {
            "success": False,
            "events": [],
            "message": f"Calendar integration error: {str(e)}. Check your calendar configuration.",
        }


@tool
def add_calendar_event(
    title: str,
    start_time: str,
    duration: int = 60,
    location: str = None,
    user_id: str = None,
) -> dict[str, Any]:
    """Add calendar event using configured provider.

    Args:
        title: Event title
        start_time: Event start time in ISO format (e.g., "2024-03-15T14:00:00")
        duration: Event duration in minutes (default: 60)
        location: Optional event location
        user_id: Optional user identifier

    Returns:
        Dict with success status, event_id, and message
    """
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

        return {"success": True, "event_id": event["id"], "message": message}

    except Exception as e:
        logger.error(f"Failed to add event: {e}")
        return {
            "success": False,
            "event_id": None,
            "message": f"Calendar integration error: {str(e)}. Check your calendar configuration.",
        }


# 5. LIFECYCLE FUNCTIONS
def load_calendar_context(instruction: str, context, parameters: dict) -> dict:
    """Pre-processor: Load Tier 1 memories for calendar context."""
    try:
        from common.providers.universal_memory_provider import UniversalMemoryProvider

        user_id = getattr(context, "user_id", "unknown")

        # TIER 1: Load recent memories from unified system (last 5)
        memory_provider = UniversalMemoryProvider()
        tier1_memories = memory_provider.get_recent_memories(
            user_id=user_id, memory_types=["event", "conversation"], limit=5
        )

        return {
            "tier1_memories": tier1_memories,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Failed to load calendar context: {e}")
        return {
            "tier1_memories": [],
            "user_id": getattr(context, "user_id", "unknown"),
        }


def save_calendar_event(llm_result: str, context, pre_data: dict) -> str:
    """Post-processing: Save calendar event to unified memory."""
    try:
        # For now, just return the result
        # In future, parse llm_result to extract event details and emit MemoryWriteIntent
        return llm_result

    except Exception as e:
        logger.error(f"Failed to save calendar memory: {e}")
        return llm_result


# 6. INTENT HANDLER REGISTRATION
async def process_calendar_intent(intent: CalendarIntent):
    """Process calendar-specific intents - called by IntentProcessor."""
    logger.info(f"Processing calendar intent: {intent.action}")


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {"CALENDAR_REQUEST": handle_calendar_request},
        "tools": [get_schedule, add_calendar_event],
        "intents": {
            CalendarIntent: process_calendar_intent,
        },
    }
