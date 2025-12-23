"""Calendar Role Handlers - Phase 3 Domain Pattern

Event handlers, intent processors, and helper functions for calendar role.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from common.event_context import LLMSafeEventContext
from common.intents import Intent, NotificationIntent

logger = logging.getLogger(__name__)


# INTENT DEFINITIONS
@dataclass
class CalendarIntent(Intent):
    """Calendar-specific intent for event management."""

    action: str  # "add_event", "get_schedule", "find_conflicts"
    event_data: dict[str, Any]

    def validate(self) -> bool:
        """Validate calendar intent parameters."""
        return self.action in ["add_event", "get_schedule", "find_conflicts"]


# EVENT HANDLERS
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


# HELPER FUNCTIONS
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


# INTENT PROCESSORS
async def process_calendar_intent(intent: CalendarIntent):
    """Process calendar-specific intents - called by IntentProcessor."""
    logger.info(f"Processing calendar intent: {intent.action}")
    # TODO: Implement calendar intent processing logic
    # This would handle actions like adding events, checking schedule, etc.
