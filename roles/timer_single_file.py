"""Timer role - LLM-friendly single file implementation with intent-based architecture.

This role consolidates all timer functionality into a single file following
the LLM-safe intent-based architecture patterns from Documents 25, 26, and 27.

Key architectural principles:
- Tools are declarative (return intents, no side effects)
- Event handlers are pure functions returning intents
- IntentProcessor handles all I/O operations
- Single event loop compliance (no threading.Timer)
- Context flows through LLMSafeEventContext

Migrated from: roles/timer/ (definition.yaml + lifecycle.py + tools.py)
Total reduction: ~1800 lines → ~300 lines (83% reduction)
"""

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from strands import tool

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent
from common.message_bus import MessageBus

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "timer",
    "version": "5.0.0",
    "description": "Timer and alarm management with intent-based LLM-safe architecture",
    "llm_type": "WEAK",
    "fast_reply": True,
    "when_to_use": "Set timers, alarms, manage time-based reminders with intent-based processing",
    "parameters": {
        "action": {
            "type": "string",
            "required": True,
            "description": "Timer action to perform",
            "examples": ["set", "cancel", "list"],
            "enum": ["set", "cancel", "list"],
        },
        "duration": {
            "type": "string",
            "required": False,
            "description": "Timer duration for set operations",
            "examples": ["5s", "2m", "1h", "30min"],
        },
        "label": {
            "type": "string",
            "required": False,
            "description": "Optional label for the timer",
            "examples": ["Meeting reminder", "Coffee break", "Workout"],
        },
        "timer_id": {
            "type": "string",
            "required": False,
            "description": "Timer ID for cancel operations",
        },
    },
    "tools": {
        "automatic": True,  # Include custom timer tools
        "shared": [],  # No shared tools needed for intent-based architecture
        "include_builtin": False,  # Exclude calculator, file_read, shell
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode
        },
    },
    "prompts": {
        "system": """You are a timer management specialist. You can set, cancel, and list timers using the available tools.

Available timer tools:
- set_timer(duration, label): Set a new timer with duration (e.g., "5s", "2m", "1h") and optional label
- cancel_timer(timer_id): Cancel an existing timer by ID
- list_timers(): List all active timers

When users request timer operations:
1. Parse the duration from natural language (5s, 2 minutes, 1 hour, etc.)
2. Use the appropriate tool to perform the action
3. Provide clear confirmation of the action taken

Always use the timer tools to perform timer operations. Do not suggest alternative approaches."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (owned by timer role)
@dataclass
class TimerCreationIntent(Intent):
    """Timer-specific intent: Create a timer with proper context routing."""

    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate timer creation intent parameters."""
        return (
            bool(self.timer_id and self.duration)
            and isinstance(self.duration_seconds, (int, float))
            and self.duration_seconds > 0
            and len(self.timer_id.strip()) > 0
            and len(self.duration.strip()) > 0
        )


@dataclass
class TimerCancellationIntent(Intent):
    """Timer-specific intent: Cancel an existing timer."""

    timer_id: str
    user_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate timer cancellation intent parameters."""
        return bool(self.timer_id and len(self.timer_id.strip()) > 0)


@dataclass
class TimerListingIntent(Intent):
    """Timer-specific intent: List active timers for a user."""

    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate timer listing intent parameters."""
        return True  # No required parameters for listing


@dataclass
class TimerExpiryIntent(Intent):
    """Timer-specific intent: Handle timer expiry notification."""

    timer_id: str
    original_duration: str
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate timer expiry intent parameters."""
        return (
            bool(self.timer_id and self.original_duration)
            and len(self.timer_id.strip()) > 0
            and len(self.original_duration.strip()) > 0
        )


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_timer_expiry(event_data: Any, context: LLMSafeEventContext) -> list[Intent]:
    """LLM-SAFE: Pure function for timer expiry events."""
    try:
        # Parse event data
        timer_id, request = _parse_timer_event_data(event_data)

        # Check if parsing failed (indicates error condition)
        if timer_id == "parse_error" or "Unparseable data:" in request:
            logger.error(f"Timer handler error: {request}")
            return [
                NotificationIntent(
                    message=f"Timer processing error: {request}",
                    channel=context.get_safe_channel(),
                    priority="high",
                    notification_type="error",
                )
            ]

        # Create intents for successful parsing
        return [
            NotificationIntent(
                message=f"⏰ Timer expired: {request}",
                channel=context.get_safe_channel(),
                user_id=context.user_id,
                priority="medium",
            ),
            AuditIntent(
                action="timer_expired",
                details={
                    "timer_id": timer_id,
                    "original_request": request,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
            ),
        ]

    except Exception as e:
        logger.error(f"Timer handler error: {e}")
        return [
            NotificationIntent(
                message=f"Timer processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


def handle_heartbeat_monitoring(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for heartbeat monitoring."""
    # For heartbeat monitoring, we typically don't need to create intents
    # unless there are expired timers to process
    return []


# 4. TOOLS (declarative, LLM-friendly, intent-based)
@tool
def set_timer(duration: str, label: str = "") -> dict[str, Any]:
    """LLM-SAFE: Declarative timer creation - returns intent, no side effects."""
    try:
        # Parse duration to seconds
        duration_seconds = _parse_duration(duration)
        if duration_seconds <= 0:
            return {"success": False, "error": f"Invalid duration: {duration}"}

        # Generate unique timer ID
        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        # CORRECT: Return intent data, no I/O operations
        return {
            "success": True,
            "timer_id": timer_id,
            "message": f"Timer set for {duration}" + (f" ({label})" if label else ""),
            "intent": {
                "type": "TimerCreationIntent",
                "timer_id": timer_id,
                "duration": duration,
                "duration_seconds": duration_seconds,
                "label": label,
                # user_id and channel_id will be injected by UniversalAgent
            },
        }
    except Exception as e:
        logger.error(f"Timer creation error: {e}")
        return {"success": False, "error": str(e)}


@tool
def cancel_timer(timer_id: str) -> dict[str, Any]:
    """LLM-SAFE: Declarative timer cancellation - returns intent, no side effects."""
    return {
        "success": True,
        "message": f"Timer {timer_id} cancelled",
        "timer_id": timer_id,
        "intent": {
            "type": "TimerCancellationIntent",
            "timer_id": timer_id,
            # user_id will be injected by UniversalAgent
        },
    }


@tool
def list_timers() -> dict[str, Any]:
    """LLM-SAFE: Declarative timer listing - returns intent, no side effects."""
    return {
        "success": True,
        "message": "Listing active timers",
        "intent": {
            "type": "TimerListingIntent",
            # user_id and channel_id will be injected by UniversalAgent
        },
    }


# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_timer_event_data(event_data: Any) -> tuple[str, str]:
    """LLM-SAFE: Parse timer event data with error handling."""
    try:
        if isinstance(event_data, list) and len(event_data) >= 2:
            return str(event_data[0]), str(event_data[1])
        elif isinstance(event_data, dict):
            return (
                event_data.get("timer_id", "unknown"),
                event_data.get("original_request", "Unknown timer"),
            )
        else:
            return "unknown", f"Unparseable data: {event_data}"
    except Exception as e:
        return "parse_error", f"Parse error: {e}"


def _parse_duration(duration_str: str) -> int:
    """LLM-SAFE: Simple duration parsing that LLMs can understand."""
    try:
        # Remove whitespace and convert to lowercase
        duration_str = duration_str.strip().lower()

        # Handle common patterns
        if duration_str.endswith("m"):
            return int(duration_str[:-1]) * 60
        elif duration_str.endswith("h"):
            return int(duration_str[:-1]) * 3600
        elif duration_str.endswith("s"):
            return int(duration_str[:-1])
        elif duration_str.endswith("min"):
            return int(duration_str[:-3]) * 60
        elif duration_str.endswith("hour"):
            return int(duration_str[:-4]) * 3600
        elif duration_str.endswith("hours"):
            return int(duration_str[:-5]) * 3600
        elif duration_str.endswith("minutes"):
            return int(duration_str[:-7]) * 60
        elif duration_str.endswith("seconds"):
            return int(duration_str[:-7])
        else:
            # Assume seconds if no unit
            return int(duration_str)
    except ValueError:
        raise ValueError(f"Invalid duration format: {duration_str}")


# 6. INTENT HANDLER REGISTRATION
async def process_timer_creation_intent(intent: TimerCreationIntent):
    """Process timer creation intents - handles actual I/O operations."""
    import asyncio

    from roles.shared_tools.redis_tools import redis_write

    try:
        # Create timer data with proper context
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": time.time() + intent.duration_seconds,
            "status": "active",
            "user_id": intent.user_id,  # ✅ From intent context
            "channel_id": intent.channel_id,  # ✅ From intent context
        }

        # Store in Redis
        redis_result = redis_write(
            f"timer:{intent.timer_id}", timer_data, ttl=intent.duration_seconds + 60
        )

        if redis_result.get("success"):
            # Schedule expiry using single event loop (not threading.Timer)
            asyncio.create_task(
                _schedule_timer_expiry_async(
                    intent.timer_id, intent.duration_seconds, timer_data
                )
            )
            logger.info(f"Timer {intent.timer_id} created and scheduled")
        else:
            logger.error(f"Failed to store timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer creation failed: {e}")


async def process_timer_cancellation_intent(intent: TimerCancellationIntent):
    """Process timer cancellation intents - handles actual I/O operations."""
    from roles.shared_tools.redis_tools import redis_delete, redis_read

    try:
        # Read timer data for validation
        timer_data = redis_read(f"timer:{intent.timer_id}")

        if not timer_data.get("success"):
            logger.warning(f"Timer {intent.timer_id} not found for cancellation")
            return

        # Validate user can cancel this timer (if user_id provided)
        stored_timer = timer_data.get("data", {})
        if intent.user_id and stored_timer.get("user_id") != intent.user_id:
            logger.warning(
                f"User {intent.user_id} cannot cancel timer {intent.timer_id} (not owner)"
            )
            return

        # Delete timer from Redis
        redis_result = redis_delete(f"timer:{intent.timer_id}")

        if redis_result.get("success"):
            logger.info(f"Timer {intent.timer_id} cancelled successfully")
        else:
            logger.error(f"Failed to cancel timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer cancellation failed: {e}")


async def process_timer_listing_intent(intent: TimerListingIntent):
    """Process timer listing intents - handles actual I/O operations."""
    from roles.shared_tools.redis_tools import redis_get_keys, redis_read

    try:
        # Get all timer keys
        keys_result = redis_get_keys("timer:*")

        if not keys_result.get("success"):
            logger.error("Failed to retrieve timer keys")
            return

        timer_keys = keys_result.get("keys", [])
        active_timers = []

        # Read each timer and filter by user if specified
        for key in timer_keys:
            timer_data = redis_read(key)
            if timer_data.get("success"):
                timer_info = timer_data.get("data", {})

                # Filter by user_id if specified
                if intent.user_id and timer_info.get("user_id") != intent.user_id:
                    continue

                # Check if timer is still active
                if timer_info.get("expires_at", 0) > time.time():
                    active_timers.append(timer_info)

        logger.info(
            f"Found {len(active_timers)} active timers for user {intent.user_id}"
        )

    except Exception as e:
        logger.error(f"Timer listing failed: {e}")


async def process_timer_expiry_intent(intent: TimerExpiryIntent):
    """Process timer expiry intents - handles notification delivery."""
    try:
        logger.info(f"Timer expiry notification for {intent.timer_id}")
        # Notification will be handled by the event handler returning NotificationIntent

    except Exception as e:
        logger.error(f"Timer expiry processing failed: {e}")


async def _schedule_timer_expiry_async(
    timer_id: str, duration_seconds: int, timer_data: dict
):
    """Schedule timer expiry using asyncio (single event loop)."""
    try:
        # Wait for timer duration
        await asyncio.sleep(duration_seconds)

        # Emit timer expiry event
        try:
            from supervisor.supervisor import get_global_supervisor

            supervisor = get_global_supervisor()
            if supervisor and supervisor.message_bus:
                supervisor.message_bus.publish(
                    "timer_role",
                    "TIMER_EXPIRED",
                    {
                        "timer_id": timer_id,
                        "original_request": f"Timer {timer_data.get('duration', 'unknown')} expired",
                        "user_id": timer_data.get("user_id"),
                        "channel_id": timer_data.get("channel_id"),
                        "label": timer_data.get("label", ""),
                        "expired_at": time.time(),
                    },
                )
                logger.info(f"Timer expiry event published for {timer_id}")
        except Exception as e:
            logger.error(f"Failed to publish timer expiry event: {e}")

        # Clean up expired timer
        from roles.shared_tools.redis_tools import redis_delete

        redis_delete(f"timer:{timer_id}")
        logger.info(f"Timer {timer_id} expired and cleaned up")

    except Exception as e:
        logger.error(f"Timer expiry scheduling failed for {timer_id}: {e}")


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "TIMER_EXPIRED": handle_timer_expiry,
            "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring,
        },
        "tools": [set_timer, cancel_timer, list_timers],
        "intents": {
            TimerCreationIntent: process_timer_creation_intent,
            TimerCancellationIntent: process_timer_cancellation_intent,
            TimerListingIntent: process_timer_listing_intent,
            TimerExpiryIntent: process_timer_expiry_intent,
        },
    }


# 8. UTILITY FUNCTIONS FOR TIMER OPERATIONS
def format_duration(seconds: int) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds == 0:
            return f"{minutes}m"
        else:
            return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {remaining_minutes}m"


def parse_time_expression(time_expr: str) -> Optional[datetime]:
    """Parse various time expressions into datetime objects."""
    try:
        # Simple time patterns (HH:MM, H:MM AM/PM)
        time_patterns = [
            r"^(\d{1,2}):(\d{2})$",  # 14:30
            r"^(\d{1,2}):(\d{2})\s*(AM|PM)$",  # 2:30 PM
        ]

        for pattern in time_patterns:
            match = re.match(pattern, time_expr.strip(), re.IGNORECASE)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))

                # Handle AM/PM
                if len(match.groups()) > 2:
                    am_pm = match.group(3).upper()
                    if am_pm == "PM" and hour != 12:
                        hour += 12
                    elif am_pm == "AM" and hour == 12:
                        hour = 0

                # Create datetime for today at specified time
                now = datetime.now()
                target_time = now.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )

                # If time has passed today, schedule for tomorrow
                if target_time <= now:
                    target_time += timedelta(days=1)

                return target_time

        return None
    except Exception as e:
        logger.error(f"Error parsing time expression '{time_expr}': {e}")
        return None


# 9. CONSTANTS AND CONFIGURATION
MAX_TIMER_DURATION = 365 * 24 * 3600  # 1 year maximum
DEFAULT_TIMER_LABEL = "Timer"
TIMER_ID_PREFIX = "timer_"

# Timer action mappings for LLM understanding
TIMER_ACTIONS = {
    "set": "create",
    "create": "create",
    "start": "create",
    "cancel": "cancel",
    "stop": "cancel",
    "delete": "cancel",
    "list": "check",
    "show": "check",
    "status": "check",
}


# 10. ENHANCED ERROR HANDLING
def create_timer_error_intent(
    error: Exception, context: LLMSafeEventContext
) -> list[Intent]:
    """Create error intents for timer operations."""
    return [
        NotificationIntent(
            message=f"Timer error: {error}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="high",
            notification_type="error",
        ),
        AuditIntent(
            action="timer_error",
            details={"error": str(error), "context": context.to_dict()},
            user_id=context.user_id,
            severity="error",
        ),
    ]
