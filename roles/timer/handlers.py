"""Timer event handlers and intent processors.

Migrated from legacy roles/core_timer.py to Phase 3 domain structure.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)


# Intent Definitions
@dataclass
class TimerCreationIntent(Intent):
    """Timer-specific intent: Create a timer with heartbeat-driven expiry."""

    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    deferred_workflow: str = ""
    user_id: str | None = None
    channel_id: str | None = None
    event_context: dict[str, Any] | None = None

    def validate(self) -> bool:
        return (
            bool(self.timer_id and self.duration)
            and isinstance(self.duration_seconds, (int, float))
            and self.duration_seconds > 0
        )


@dataclass
class TimerCancellationIntent(Intent):
    """Timer-specific intent: Cancel an existing timer."""

    timer_id: str
    user_id: str | None = None

    def validate(self) -> bool:
        return bool(self.timer_id and len(self.timer_id.strip()) > 0)


@dataclass
class TimerListingIntent(Intent):
    """Timer-specific intent: List active timers for a user."""

    user_id: str | None = None
    channel_id: str | None = None

    def validate(self) -> bool:
        return True


@dataclass
class TimerExpiryIntent(Intent):
    """Timer-specific intent: Handle timer expiry notification."""

    timer_id: str
    original_duration: str
    label: str = ""
    deferred_workflow: str = ""
    user_id: str | None = None
    channel_id: str | None = None
    event_context: dict[str, Any] | None = None

    def validate(self) -> bool:
        return bool(self.timer_id and self.original_duration)


# Helper Functions
def _get_safe_channel(context) -> str:
    """Get safe channel from context."""
    if hasattr(context, "get_safe_channel") and callable(context.get_safe_channel):
        try:
            return context.get_safe_channel()
        except TypeError:
            pass
    if hasattr(context, "channel_id") and context.channel_id:
        return context.channel_id
    return "console"


def _get_expired_timers_from_redis(current_time: int) -> list[str]:
    """Get expired timer IDs from Redis sorted set."""
    try:
        from roles.shared_tools.redis_tools import _get_redis_client

        client = _get_redis_client()

        # Get expired timers from sorted set (score <= current_time)
        expired_timer_ids = client.zrangebyscore("timer:active_queue", 0, current_time)

        # Remove expired timers from the active queue
        if expired_timer_ids:
            removed_count = client.zremrangebyscore(
                "timer:active_queue", 0, current_time
            )
            logger.debug(f"Removed {removed_count} expired timers from Redis queue")

        # Decode timer IDs
        decoded_ids = [
            timer_id.decode() if isinstance(timer_id, bytes) else timer_id
            for timer_id in expired_timer_ids
        ]
        return decoded_ids

    except Exception as e:
        logger.error(f"Failed to get expired timers: {e}")
        return []


# Event Handlers
def handle_heartbeat_monitoring(event_data: Any, context) -> list[Intent]:
    """Event handler for FAST_HEARTBEAT_TICK - checks Redis for expired timers."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        current_time = int(time.time())
        logger.info(f"⏱️ Checking for expired timers at time: {current_time}")

        expired_timer_ids = _get_expired_timers_from_redis(current_time)
        if expired_timer_ids:
            logger.info(
                f"Found {len(expired_timer_ids)} expired timers: {expired_timer_ids}"
            )

        # Create expiry intents for each expired timer
        intents = []
        for timer_id in expired_timer_ids:
            timer_result = redis_read(f"timer:data:{timer_id}")
            if timer_result.get("success"):
                timer_data = timer_result.get("value", {})
                stored_context = timer_data.get("event_context", {})
                deferred_workflow = timer_data.get("deferred_workflow", "")

                intents.append(
                    TimerExpiryIntent(
                        timer_id=timer_id,
                        original_duration=timer_data.get("duration", "unknown"),
                        label=timer_data.get("label", ""),
                        deferred_workflow=deferred_workflow,
                        user_id=stored_context.get("user_id"),
                        channel_id=stored_context.get("channel_id"),
                        event_context=stored_context,
                    )
                )

                workflow_info = (
                    f" with deferred workflow: {deferred_workflow}"
                    if deferred_workflow
                    else ""
                )
                logger.info(
                    f"Timer {timer_id} expiring for user {stored_context.get('user_id')} "
                    f"in channel {stored_context.get('channel_id')}{workflow_info}"
                )
            else:
                logger.warning(
                    f"Failed to get timer data for {timer_id}: {timer_result}"
                )

        if intents:
            logger.info(f"Processing {len(intents)} timer expiry notifications")
        return intents

    except Exception as e:
        logger.error(f"Heartbeat monitoring error: {e}")
        return []


# Intent Processors
async def process_timer_creation_intent(intent: TimerCreationIntent):
    """Process timer creation intents - handles actual Redis operations."""
    logger.info(f"Creating timer {intent.timer_id} for {intent.duration}")
    from roles.shared_tools.redis_tools import _get_redis_client, redis_write

    try:
        expiry_time = time.time() + intent.duration_seconds

        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "deferred_workflow": intent.deferred_workflow,
            "created_at": time.time(),
            "expires_at": expiry_time,
            "status": "active",
            "event_context": intent.event_context,
        }

        # Store timer metadata in Redis hash
        redis_result = redis_write(
            f"timer:data:{intent.timer_id}",
            timer_data,
            ttl=intent.duration_seconds + 60,
        )

        if redis_result.get("success"):
            # Add timer to sorted set for efficient expiry queries
            client = _get_redis_client()
            client.zadd("timer:active_queue", {intent.timer_id: expiry_time})
            logger.info(f"Timer {intent.timer_id} created successfully")
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
        timer_data = redis_read(f"timer:data:{intent.timer_id}")
        if not timer_data.get("success"):
            logger.warning(f"Timer {intent.timer_id} not found for cancellation")
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

            timer_data = redis_read(f"timer:data:{timer_id}")
            if timer_data.get("success"):
                active_timers.append(timer_data.get("value", {}))

        logger.info(f"Found {len(active_timers)} active timers")

    except Exception as e:
        logger.error(f"Timer listing failed: {e}")


async def process_timer_expiry_intent(intent: TimerExpiryIntent):
    """Process timer expiry intents - handles notification delivery."""
    try:
        logger.info(f"Processing timer expiry notification for {intent.timer_id}")

        message = f"⏰ Timer expired: {intent.original_duration}"
        if intent.label:
            message += f" ({intent.label})"

        # Create notification intent
        notification_intent = NotificationIntent(
            message=message,
            channel=intent.channel_id or "console",
            user_id=intent.user_id,
            priority="medium",
        )

        logger.info(
            f"Timer expiry notification ready: {message} -> {notification_intent.channel}"
        )

        # Process notification through IntentProcessor
        from llm_provider.role_registry import RoleRegistry

        role_registry = RoleRegistry.get_global_registry()
        if role_registry and role_registry.intent_processor:
            await role_registry.intent_processor._process_notification(
                notification_intent
            )
            logger.info("Timer expiry notification sent via IntentProcessor")

        # TODO: Handle deferred workflow execution if specified
        if intent.deferred_workflow:
            logger.warning(
                f"Deferred workflow execution not yet implemented: {intent.deferred_workflow}"
            )

    except Exception as e:
        logger.error(f"Timer expiry processing failed: {e}")
