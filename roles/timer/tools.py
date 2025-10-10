"""Timer tools for StrandsAgent - Real implementations.

These tools provide timer and alarm functionality using the TimerManager
for Redis-based persistence and the comprehensive timer system.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from strands import tool

from roles.timer.lifecycle import get_timer_manager

logger = logging.getLogger(__name__)


@tool
def timer_set(
    duration: str,
    label: Optional[str] = None,
    custom_message: Optional[str] = None,
    user_id: str = "system",
    channel_id: str = "default",
    notification_channel: Optional[str] = None,
    notification_recipient: Optional[str] = None,
    notification_priority: str = "medium",
) -> dict[str, Any]:
    """Set a countdown timer.

    Args:
        duration: Timer duration (e.g., "5m", "30s", "1h30m", "120")
        label: Optional label for the timer
        custom_message: Custom message when timer expires
        user_id: User who created the timer
        channel_id: Channel for notifications
        notification_channel: Channel type for notifications (slack, email, etc.)
        notification_recipient: Recipient identifier for notifications
        notification_priority: Priority level (low, medium, high, critical)

    Returns:
        Dict containing timer creation result
    """
    try:
        timer_manager = get_timer_manager()

        # Parse duration to seconds
        from roles.timer.lifecycle import _parse_duration_to_seconds

        duration_seconds = _parse_duration_to_seconds(duration)

        if not duration_seconds:
            return {
                "success": False,
                "error": f"Invalid duration format '{duration}'. Use formats like '5m', '1h30m', '120'",
            }

        # Create timer using async wrapper
        import asyncio

        async def _create_timer():
            # Prepare notification metadata
            notification_metadata = {}
            if notification_channel:
                notification_metadata["notification_channel"] = notification_channel
            if notification_recipient:
                notification_metadata["notification_recipient"] = notification_recipient
            if notification_priority:
                notification_metadata["notification_priority"] = notification_priority

            return await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=duration_seconds,
                name=label or f"{duration} timer",
                label=label or "",
                custom_message=custom_message or f"Timer for {duration} has expired!",
                user_id=user_id,
                channel_id=channel_id,
                metadata=notification_metadata,
            )

        # Run async operation - handle existing event loop
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _create_timer())
                timer_id = future.result()
        except RuntimeError:
            # No running loop, create new one
            timer_id = asyncio.run(_create_timer())

        logger.info(f"Created timer {timer_id} for {duration}")

        return {
            "success": True,
            "timer_id": timer_id,
            "duration": duration,
            "duration_seconds": duration_seconds,
            "label": label,
            "message": f"Timer set for {duration}. Timer ID: {timer_id}",
        }

    except Exception as e:
        logger.error(f"Failed to set timer: {e}")
        return {"success": False, "error": str(e)}


@tool
def timer_cancel(timer_id: str) -> dict[str, Any]:
    """Cancel an active timer.

    Args:
        timer_id: Timer identifier to cancel

    Returns:
        Dict containing timer cancellation result
    """
    try:
        timer_manager = get_timer_manager()

        # Cancel timer using async wrapper
        import asyncio

        async def _cancel_timer():
            return await timer_manager.cancel_timer(timer_id)

        # Run async operation - handle existing event loop
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _cancel_timer())
                success = future.result()
        except RuntimeError:
            # No running loop, create new one
            success = asyncio.run(_cancel_timer())

        if success:
            logger.info(f"Cancelled timer {timer_id}")
            return {
                "success": True,
                "timer_id": timer_id,
                "message": f"Timer {timer_id} has been cancelled",
            }
        else:
            return {
                "success": False,
                "error": f"Timer {timer_id} not found or already completed",
            }

    except Exception as e:
        logger.error(f"Failed to cancel timer {timer_id}: {e}")
        return {"success": False, "error": str(e)}


@tool
def timer_list(
    active_only: bool = True,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
) -> dict[str, Any]:
    """List timers (active or all).

    Args:
        active_only: If True, only show active timers; if False, show all timers
        user_id: Filter by user ID (optional)
        channel_id: Filter by channel ID (optional)

    Returns:
        Dict containing list of timers
    """
    try:
        timer_manager = get_timer_manager()

        # List timers using async wrapper
        import asyncio

        async def _list_timers():
            status = "active" if active_only else None
            return await timer_manager.list_timers(
                user_id=user_id, channel_id=channel_id, status=status
            )

        # Run async operation - handle existing event loop
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _list_timers())
                timers = future.result()
        except RuntimeError:
            # No running loop, create new one
            timers = asyncio.run(_list_timers())

        # Format timer data for display
        formatted_timers = []
        for timer in timers:
            formatted_timer = {
                "timer_id": timer["id"],
                "name": timer["name"],
                "type": timer["type"],
                "status": timer["status"],
                "created_at": datetime.fromtimestamp(timer["created_at"]).isoformat(),
                "expires_at": datetime.fromtimestamp(timer["expires_at"]).isoformat(),
                "user_id": timer["user_id"],
                "channel_id": timer["channel_id"],
            }

            if timer.get("duration_seconds"):
                formatted_timer["duration_seconds"] = timer["duration_seconds"]

            if timer.get("label"):
                formatted_timer["label"] = timer["label"]

            formatted_timers.append(formatted_timer)

        logger.info(f"Listed {len(formatted_timers)} timers")

        return {
            "success": True,
            "timers": formatted_timers,
            "count": len(formatted_timers),
            "message": f"Found {len(formatted_timers)} timers",
        }

    except Exception as e:
        logger.error(f"Failed to list timers: {e}")
        return {"success": False, "error": str(e), "timers": []}


@tool
def alarm_set(
    time: str,
    label: Optional[str] = None,
    recurring: Optional[str] = None,
    custom_message: Optional[str] = None,
    user_id: str = "system",
    channel_id: str = "default",
    notification_channel: Optional[str] = None,
    notification_recipient: Optional[str] = None,
    notification_priority: str = "high",
) -> dict[str, Any]:
    """Set an alarm for a specific time.

    Args:
        time: Alarm time in format "HH:MM" or "YYYY-MM-DD HH:MM"
        label: Optional label for the alarm
        recurring: Optional recurrence pattern ("daily", "weekdays", "weekends", "weekly")
        custom_message: Custom message when alarm triggers
        user_id: User who created the alarm
        channel_id: Channel for notifications
        notification_channel: Channel type for notifications (slack, email, etc.)
        notification_recipient: Recipient identifier for notifications
        notification_priority: Priority level (low, medium, high, critical)

    Returns:
        Dict containing alarm creation result
    """
    try:
        timer_manager = get_timer_manager()

        # Parse alarm time
        from roles.timer.lifecycle import _parse_alarm_time

        alarm_datetime = _parse_alarm_time(time)

        if not alarm_datetime:
            return {
                "success": False,
                "error": f"Invalid time format '{time}'. Use formats like '14:30', '2:30 PM', '2024-12-25 09:00'",
            }

        # Check if alarm time is in the future
        if alarm_datetime <= datetime.now():
            return {
                "success": False,
                "error": f"Alarm time '{time}' must be in the future",
            }

        # Determine timer type based on recurring pattern
        timer_type = "recurring" if recurring else "alarm"
        recurring_config = None

        if recurring:
            recurring_config = {
                "pattern": recurring,
                "end_date": None,  # No end date for basic recurring alarms
                "max_occurrences": None,
            }

        # Create alarm using async wrapper
        import asyncio

        async def _create_alarm():
            # Prepare notification metadata
            notification_metadata = {}
            if notification_channel:
                notification_metadata["notification_channel"] = notification_channel
            if notification_recipient:
                notification_metadata["notification_recipient"] = notification_recipient
            if notification_priority:
                notification_metadata["notification_priority"] = notification_priority

            return await timer_manager.create_timer(
                timer_type=timer_type,
                alarm_time=alarm_datetime,
                name=label or f"Alarm at {time}",
                label=label or "",
                custom_message=custom_message or f"Alarm for {time}!",
                user_id=user_id,
                channel_id=channel_id,
                recurring=recurring_config,
                metadata=notification_metadata,
            )

        # Run async operation - handle existing event loop
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _create_alarm())
                timer_id = future.result()
        except RuntimeError:
            # No running loop, create new one
            timer_id = asyncio.run(_create_alarm())

        logger.info(f"Created alarm {timer_id} for {time}")

        return {
            "success": True,
            "timer_id": timer_id,
            "time": time,
            "alarm_datetime": alarm_datetime.isoformat(),
            "label": label,
            "recurring": recurring,
            "message": f"Alarm set for {time}. Timer ID: {timer_id}",
        }

    except Exception as e:
        logger.error(f"Failed to set alarm: {e}")
        return {"success": False, "error": str(e)}


@tool
def alarm_cancel(alarm_id: str) -> dict[str, Any]:
    """Cancel an alarm.

    Args:
        alarm_id: Alarm identifier to cancel

    Returns:
        Dict containing alarm cancellation result
    """
    try:
        # Alarm cancellation is the same as timer cancellation
        return timer_cancel(alarm_id)

    except Exception as e:
        logger.error(f"Failed to cancel alarm {alarm_id}: {e}")
        return {"success": False, "error": str(e)}


@tool
def timer_snooze(timer_id: str, snooze_minutes: int = 5) -> dict[str, Any]:
    """Snooze an active timer.

    Args:
        timer_id: Timer identifier to snooze
        snooze_minutes: Minutes to snooze (default: 5)

    Returns:
        Dict containing timer snooze result
    """
    try:
        timer_manager = get_timer_manager()

        if snooze_minutes <= 0:
            return {"success": False, "error": "Snooze minutes must be positive"}

        snooze_seconds = snooze_minutes * 60

        # Snooze timer using async wrapper
        import asyncio

        async def _snooze_timer():
            return await timer_manager.snooze_timer(timer_id, snooze_seconds)

        # Run async operation - handle existing event loop
        try:
            # Try to get existing loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _snooze_timer())
                success = future.result()
        except RuntimeError:
            # No running loop, create new one
            success = asyncio.run(_snooze_timer())

        if success:
            logger.info(f"Snoozed timer {timer_id} for {snooze_minutes} minutes")
            return {
                "success": True,
                "timer_id": timer_id,
                "snooze_minutes": snooze_minutes,
                "message": f"Timer {timer_id} snoozed for {snooze_minutes} minutes",
            }
        else:
            return {
                "success": False,
                "error": f"Timer {timer_id} not found or not active",
            }

    except Exception as e:
        logger.error(f"Failed to snooze timer {timer_id}: {e}")
        return {"success": False, "error": str(e)}
