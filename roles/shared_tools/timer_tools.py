"""Timer tools for StrandsAgent - Placeholder implementations.

These tools provide timer and alarm functionality stubs that throw NotImplementedError.
They need to be implemented with real timer/notification systems.
"""

import logging
from typing import Any, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def timer_set(duration: str, label: Optional[str] = None) -> dict[str, Any]:
    """Set a countdown timer.

    Args:
        duration: Timer duration (e.g., "5m", "30s", "1h30m", "120")
        label: Optional label for the timer

    Returns:
        Dict containing timer creation result

    Raises:
        NotImplementedError: This tool needs to be implemented with a timer system
    """
    logger.warning("timer_set called but not implemented")
    raise NotImplementedError(
        "Timer functionality not implemented. "
        "Please implement this tool with system notifications, database storage, or a timer service."
    )


@tool
def timer_cancel(timer_id: str) -> dict[str, Any]:
    """Cancel an active timer.

    Args:
        timer_id: Timer identifier to cancel

    Returns:
        Dict containing timer cancellation result

    Raises:
        NotImplementedError: This tool needs to be implemented with a timer system
    """
    logger.warning("timer_cancel called but not implemented")
    raise NotImplementedError(
        "Timer functionality not implemented. "
        "Please implement this tool with system notifications, database storage, or a timer service."
    )


@tool
def timer_list(active_only: bool = True) -> dict[str, Any]:
    """List timers (active or all).

    Args:
        active_only: If True, only show active timers; if False, show all timers

    Returns:
        Dict containing list of timers

    Raises:
        NotImplementedError: This tool needs to be implemented with a timer system
    """
    logger.warning("timer_list called but not implemented")
    raise NotImplementedError(
        "Timer functionality not implemented. "
        "Please implement this tool with system notifications, database storage, or a timer service."
    )


@tool
def alarm_set(
    time: str, label: Optional[str] = None, recurring: Optional[str] = None
) -> dict[str, Any]:
    """Set an alarm for a specific time.

    Args:
        time: Alarm time in format "HH:MM" or "YYYY-MM-DD HH:MM"
        label: Optional label for the alarm
        recurring: Optional recurrence pattern ("daily", "weekdays", "weekends", "weekly")

    Returns:
        Dict containing alarm creation result

    Raises:
        NotImplementedError: This tool needs to be implemented with an alarm system
    """
    logger.warning("alarm_set called but not implemented")
    raise NotImplementedError(
        "Alarm functionality not implemented. "
        "Please implement this tool with system notifications, cron jobs, or an alarm service."
    )


@tool
def alarm_cancel(alarm_id: str) -> dict[str, Any]:
    """Cancel an alarm.

    Args:
        alarm_id: Alarm identifier to cancel

    Returns:
        Dict containing alarm cancellation result

    Raises:
        NotImplementedError: This tool needs to be implemented with an alarm system
    """
    logger.warning("alarm_cancel called but not implemented")
    raise NotImplementedError(
        "Alarm functionality not implemented. "
        "Please implement this tool with system notifications, cron jobs, or an alarm service."
    )
