"""Timer role - LLM-friendly single file implementation following Documents 25 & 26.

This role consolidates all timer functionality into a single file following
the LLM-safe architecture patterns from Documents 25 and 26.

Key architectural principles:
- Single event loop compliance (no asyncio.sleep())
- Intent-based processing (pure functions returning intents)
- Heartbeat-driven timer monitoring (Redis polling every 5 seconds)
- LLM-safe patterns (predictable, simple, self-contained)
- Redis sorted sets for efficient timer queuing

Architecture: Single Event Loop + Intent-Based + Heartbeat Polling
Created: 2025-10-14
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool

from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "timer",
    "version": "6.0.0",
    "description": "Timer and alarm management with LLM-safe heartbeat-driven architecture",
    "llm_type": "WEAK",
    "fast_reply": True,
    "when_to_use": "Set timers, alarms, manage time-based reminders using heartbeat polling",
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
        "shared": ["redis_tools"],  # Need Redis for timer queue
        "include_builtin": False,  # Exclude calculator, file_read, shell
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode
        },
    },
    "prompts": {
        "system": """You are a timer management specialist using heartbeat-driven architecture. You can set, cancel, and list timers using the available tools.

Available timer tools:
- set_timer(duration, label): Set a new timer with duration (e.g., "5s", "2m", "1h") and optional label
- cancel_timer(timer_id): Cancel an existing timer by ID
- list_timers(): List all active timers

Timer Architecture:
- Timers are stored in Redis sorted sets for efficient expiry queries
- Heartbeat system checks for expired timers every 5 seconds
- No persistent async tasks or threading complexity
- All operations are intent-based and LLM-safe

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
    """Timer-specific intent: Create a timer with heartbeat-driven expiry."""

    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    event_context: Optional[dict[str, Any]] = None  # Store full LLMSafeEventContext

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
    event_context: Optional[dict[str, Any]] = None  # Store full LLMSafeEventContext

    def validate(self) -> bool:
        """Validate timer expiry intent parameters."""
        return (
            bool(self.timer_id and self.original_duration)
            and len(self.timer_id.strip()) > 0
            and len(self.original_duration.strip()) > 0
        )


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_timer_expiry(event_data: Any, context) -> list[Intent]:
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
                    channel=_get_safe_channel(context),
                    priority="high",
                    notification_type="error",
                )
            ]

        # Create intents for successful parsing
        return [
            NotificationIntent(
                message=f"⏰ Timer expired: {request}",
                channel=_get_safe_channel(context),
                user_id=getattr(context, "user_id", None),
                priority="medium",
            ),
            AuditIntent(
                action="timer_expired",
                details={
                    "timer_id": timer_id,
                    "original_request": request,
                    "processed_at": time.time(),
                },
                user_id=getattr(context, "user_id", None),
            ),
        ]

    except Exception as e:
        logger.error(f"Timer handler error: {e}")
        return [
            NotificationIntent(
                message=f"Timer processing error: {e}",
                channel=_get_safe_channel(context),
                priority="high",
                notification_type="error",
            )
        ]


def handle_heartbeat_monitoring(event_data: Any, context) -> list[Intent]:
    """LLM-SAFE: Pure function for heartbeat monitoring - checks Redis for expired timers."""
    logger.info(
        f"🔥 TIMER HEARTBEAT HANDLER CALLED! event_data: {event_data}, context: {context}"
    )
    try:
        from roles.shared_tools.redis_tools import redis_read

        # Get current time for expiry check
        current_time = int(time.time())
        logger.info(f"🔥 Checking for expired timers at time: {current_time}")

        # Check Redis sorted set for expired timers
        # This is called every 5 seconds by the fast heartbeat
        expired_timer_ids = _get_expired_timers_from_redis(current_time)
        logger.info(
            f"🔥 Found {len(expired_timer_ids)} expired timers: {expired_timer_ids}"
        )

        # Create expiry intents for each expired timer
        intents = []
        for timer_id in expired_timer_ids:
            logger.info(f"🔥 Processing expired timer: {timer_id}")
            # Get timer data to create proper expiry intent
            timer_result = redis_read(f"timer:data:{timer_id}")
            if timer_result.get("success"):
                timer_data = timer_result.get("value", {})
                logger.info(f"🔥 Timer data for {timer_id}: {timer_data}")
                # Extract stored event context for full traceability
                stored_context = timer_data.get("event_context", {})

                intents.append(
                    TimerExpiryIntent(
                        timer_id=timer_id,
                        original_duration=timer_data.get("duration", "unknown"),
                        label=timer_data.get("label", ""),
                        user_id=timer_data.get("user_id"),
                        channel_id=timer_data.get("channel_id"),
                        event_context=stored_context,  # Include full context for traceability
                    )
                )

                # Log the complete context for debugging
                logger.info(
                    f"🔥 Timer {timer_id} context: user={timer_data.get('user_id')}, "
                    f"channel={timer_data.get('channel_id')}, "
                    f"source={stored_context.get('source', 'unknown')}, "
                    f"metadata_keys={list(stored_context.get('metadata', {}).keys())}"
                )
            else:
                logger.warning(
                    f"🔥 Failed to get timer data for {timer_id}: {timer_result}"
                )

        logger.info(f"🔥 Returning {len(intents)} timer expiry intents")
        return intents

    except Exception as e:
        logger.error(f"Heartbeat monitoring error: {e}")
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

        # Return intent data for processing by infrastructure
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


def _get_safe_channel(context) -> str:
    """LLM-SAFE: Get safe channel from context."""
    if hasattr(context, "get_safe_channel") and callable(context.get_safe_channel):
        try:
            return context.get_safe_channel()
        except TypeError:
            # Handle case where get_safe_channel doesn't accept arguments
            pass
    if hasattr(context, "channel_id") and context.channel_id:
        return context.channel_id
    else:
        return "console"


def _get_expired_timers_from_redis(current_time: int) -> list[str]:
    """LLM-SAFE: Get expired timer IDs from Redis sorted set."""
    try:
        from roles.shared_tools.redis_tools import _get_redis_client

        client = _get_redis_client()
        logger.info(f"🔥 Redis client: {client}")

        # First, check all active timers for debugging
        all_active_timers = client.zrange("timer:active_queue", 0, -1, withscores=True)
        logger.info(f"🔥 All active timers in Redis: {all_active_timers}")

        # Get expired timers from sorted set (score <= current_time)
        expired_timer_ids = client.zrangebyscore("timer:active_queue", 0, current_time)
        logger.info(f"🔥 Expired timer IDs from Redis: {expired_timer_ids}")

        # Remove expired timers from the active queue
        if expired_timer_ids:
            removed_count = client.zremrangebyscore(
                "timer:active_queue", 0, current_time
            )
            logger.info(f"🔥 Removed {removed_count} expired timers from Redis queue")

        decoded_ids = [
            timer_id.decode() if isinstance(timer_id, bytes) else timer_id
            for timer_id in expired_timer_ids
        ]
        logger.info(f"🔥 Returning decoded timer IDs: {decoded_ids}")
        return decoded_ids

    except Exception as e:
        logger.error(f"Redis expired timer query failed: {e}")
        return []


# 6. INTENT HANDLER REGISTRATION
async def process_timer_creation_intent(intent: TimerCreationIntent):
    """Process timer creation intents - handles actual Redis operations."""
    logger.info(f"🔥 PROCESSING TIMER CREATION INTENT: {intent}")
    from roles.shared_tools.redis_tools import _get_redis_client, redis_write

    try:
        # Calculate expiry time
        expiry_time = time.time() + intent.duration_seconds
        logger.info(
            f"🔥 Timer will expire at: {expiry_time} (current: {time.time()}, duration: {intent.duration_seconds}s)"
        )

        # Create timer data with complete context for full traceability
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": expiry_time,
            "status": "active",
            "user_id": intent.user_id,
            "channel_id": intent.channel_id,
            "event_context": intent.event_context,  # Store complete context for traceability
        }
        logger.info(f"🔥 Timer data to store: {timer_data}")

        # Store timer metadata in Redis hash
        redis_result = redis_write(
            f"timer:data:{intent.timer_id}",
            timer_data,
            ttl=intent.duration_seconds + 60,
        )
        logger.info(f"🔥 Redis write result: {redis_result}")

        if redis_result.get("success"):
            # Add timer to sorted set for efficient expiry queries
            client = _get_redis_client()
            zadd_result = client.zadd(
                "timer:active_queue", {intent.timer_id: expiry_time}
            )
            logger.info(f"🔥 Added timer to sorted set, zadd result: {zadd_result}")

            logger.info(
                f"🔥 Timer {intent.timer_id} created in Redis queue (expires at {expiry_time})"
            )
        else:
            logger.error(f"Failed to store timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer creation failed: {e}")


async def process_timer_cancellation_intent(intent: TimerCancellationIntent):
    """Process timer cancellation intents - handles actual Redis operations."""
    from roles.shared_tools.redis_tools import (
        _get_redis_client,
        redis_delete,
        redis_read,
    )

    try:
        # Read timer data for validation
        timer_data = redis_read(f"timer:data:{intent.timer_id}")

        if not timer_data.get("success"):
            logger.warning(f"Timer {intent.timer_id} not found for cancellation")
            return

        # Validate user can cancel this timer (if user_id provided)
        stored_timer = timer_data.get("value", {})
        if intent.user_id and stored_timer.get("user_id") != intent.user_id:
            logger.warning(
                f"User {intent.user_id} cannot cancel timer {intent.timer_id} (not owner)"
            )
            return

        # Remove timer from active queue
        client = _get_redis_client()
        client.zrem("timer:active_queue", intent.timer_id)

        # Delete timer metadata
        redis_result = redis_delete(f"timer:data:{intent.timer_id}")

        if redis_result.get("success"):
            logger.info(f"Timer {intent.timer_id} cancelled successfully")
        else:
            logger.error(f"Failed to cancel timer: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Timer cancellation failed: {e}")


async def process_timer_listing_intent(intent: TimerListingIntent):
    """Process timer listing intents - handles actual Redis operations."""
    from roles.shared_tools.redis_tools import _get_redis_client, redis_read

    try:
        # Get all active timers from sorted set
        client = _get_redis_client()
        current_time = time.time()

        # Get all active timers (score > current_time)
        active_timer_ids = client.zrangebyscore(
            "timer:active_queue", current_time, "+inf"
        )

        active_timers = []
        for timer_id in active_timer_ids:
            if isinstance(timer_id, bytes):
                timer_id = timer_id.decode()

            # Read timer metadata
            timer_data = redis_read(f"timer:data:{timer_id}")
            if timer_data.get("success"):
                timer_info = timer_data.get("value", {})

                # Filter by user_id if specified
                if intent.user_id and timer_info.get("user_id") != intent.user_id:
                    continue

                active_timers.append(timer_info)

        logger.info(
            f"Found {len(active_timers)} active timers for user {intent.user_id}"
        )

    except Exception as e:
        logger.error(f"Timer listing failed: {e}")


async def process_timer_expiry_intent(intent: TimerExpiryIntent):
    """Process timer expiry intents - handles notification delivery."""
    try:
        logger.info(f"Processing timer expiry notification for {intent.timer_id}")

        # Create notification message
        message = f"⏰ Timer expired: {intent.original_duration}"
        if intent.label:
            message += f" ({intent.label})"

        # Create notification intent for proper routing
        notification_intent = NotificationIntent(
            message=message,
            channel=intent.channel_id or "console",
            user_id=intent.user_id,
            priority="medium",
            notification_type="info",
        )

        # Send notification directly through the IntentProcessor's notification handler
        logger.info(
            f"Timer expiry notification ready: {notification_intent.message} -> {notification_intent.channel}"
        )

        # Get the IntentProcessor from the global registry and process the notification
        from llm_provider.role_registry import RoleRegistry

        role_registry = RoleRegistry.get_global_registry()
        if role_registry and role_registry.intent_processor:
            await role_registry.intent_processor._process_notification(
                notification_intent
            )
            logger.info(f"Timer expiry notification sent via IntentProcessor")
        else:
            logger.error("No IntentProcessor available for timer expiry notification")

    except Exception as e:
        logger.error(f"Timer expiry processing failed: {e}")


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


# 8. CONSTANTS AND CONFIGURATION
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


# 9. ERROR HANDLING UTILITIES
def create_timer_error_intent(error: Exception, context) -> list[Intent]:
    """Create error intents for timer operations."""
    return [
        NotificationIntent(
            message=f"Timer error: {error}",
            channel=_get_safe_channel(context),
            user_id=getattr(context, "user_id", None),
            priority="high",
            notification_type="error",
        ),
        AuditIntent(
            action="timer_error",
            details={"error": str(error), "context": str(context)},
            user_id=getattr(context, "user_id", None),
            severity="error",
        ),
    ]
