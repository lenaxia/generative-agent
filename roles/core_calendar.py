"""Calendar role - LLM-friendly single file implementation for household assistant.

This role provides context-aware calendar and scheduling management functionality
following the single-file role pattern established in the Universal Agent System.

Key features:
- Context-aware scheduling with memory and location integration
- Calendar event management (add, retrieve, query)
- Integration with household assistant context system
- LLM-safe architecture with intent-based processing

Architecture: Single Event Loop + Intent-Based + Context-Aware
Created: 2025-10-16
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from strands import tool

from common.event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
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
        "shared": [],  # No shared tools needed for basic calendar
        "include_builtin": False,  # Exclude calculator, file_read, shell
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode
        },
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


# 4. TOOLS
@tool
def get_schedule(
    user_id: str, days_ahead: int = 7, location: str = None
) -> dict[str, Any]:
    """Get user's schedule with location context.

    Args:
        user_id: User identifier
        days_ahead: Number of days ahead to retrieve (default: 7)
        location: Optional location context for filtering events

    Returns:
        Dict with success status, events list, and message
    """
    # Placeholder for calendar API integration
    # Could integrate with CalDAV, Google Calendar, etc.

    message = f"Schedule retrieved for {user_id}"
    if location:
        message += f" (location: {location})"

    return {"success": True, "events": [], "message": message}


@tool
def add_calendar_event(
    title: str,
    start_time: str,
    duration: int = 60,
    location: str = None,
    user_id: str = None,
) -> dict[str, Any]:
    """Add calendar event with memory storage.

    Args:
        title: Event title
        start_time: Event start time in ISO format
        duration: Event duration in minutes (default: 60)
        location: Optional event location
        user_id: Optional user identifier

    Returns:
        Dict with success status, event_id, and message
    """
    # Placeholder for calendar API integration
    event_data = {
        "title": title,
        "start_time": start_time,
        "duration": duration,
        "location": location,
    }

    event_id = f"evt_{int(time.time())}"
    message = f"Added event: {title} at {start_time}"

    return {"success": True, "event_id": event_id, "message": message}


# 5. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {"CALENDAR_REQUEST": handle_calendar_request},
        "tools": [get_schedule, add_calendar_event],
        "intents": [CalendarIntent],
    }
