"""Timer Domain Tools

Provides timer management tools for the dynamic agent system.
Tools return intents for deferred execution (intent-based pattern).

Duration parsing is handled by LLM - tools accept duration in seconds.

Extracted from: roles/core_timer.py
"""

import logging
import uuid
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def create_timer_tools(redis_provider: Any) -> list:
    """Create timer domain tools.

    Args:
        redis_provider: Redis provider instance for timer storage

    Returns:
        List of tool functions for timer domain
    """
    # Note: Timer tools use intent-based pattern
    # They return intent data for processing by infrastructure

    tools = [
        set_timer,
        cancel_timer,
        list_timers,
    ]

    logger.info(f"Created {len(tools)} timer tools")
    return tools


# ACTION TOOLS (intent-based, declarative)


@tool
def set_timer(
    duration_seconds: int, label: str = "", deferred_workflow: str = ""
) -> dict[str, Any]:
    """Set a timer with specified duration in seconds.

    Action tool - returns intent for timer creation (declarative, no side effects).

    The LLM should convert natural language durations (like "5 minutes", "30 seconds")
    to seconds before calling this tool.

    Args:
        duration_seconds: Timer duration in seconds (LLM converts from natural language)
        label: Optional label/description for the timer
        deferred_workflow: Optional workflow/task to execute when timer expires

    Examples:
        # User says "set a timer for 10 seconds"
        set_timer(10, "coffee break")

        # User says "remind me to check weather in 5 minutes"
        set_timer(300, "check weather", "check the weather in seattle")

    Returns:
        Dict with success status, timer_id, message, and intent data
    """
    logger.info(f"Setting timer for {duration_seconds}s with label: {label}")

    try:
        if duration_seconds <= 0:
            return {
                "success": False,
                "error": f"Duration must be positive, got: {duration_seconds} seconds",
            }

        # Generate unique timer ID
        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        # Build message
        if deferred_workflow:
            message = f"Timer set for {duration_seconds}s - will execute: {deferred_workflow}"
        else:
            message = f"Timer set for {duration_seconds}s" + (
                f" ({label})" if label else ""
            )

        # Convert duration_seconds back to human-readable format for intent
        # This is required by TimerCreationIntent which expects both duration and duration_seconds
        if duration_seconds >= 3600:
            hours = duration_seconds // 3600
            duration_str = f"{hours}h"
        elif duration_seconds >= 60:
            minutes = duration_seconds // 60
            duration_str = f"{minutes}m"
        else:
            duration_str = f"{duration_seconds}s"

        # Return intent data for processing by infrastructure
        logger.info(f"Timer created: {timer_id}")
        return {
            "success": True,
            "timer_id": timer_id,
            "message": message,
            "intent": {
                "type": "TimerCreationIntent",
                "timer_id": timer_id,
                "duration": duration_str,  # Required by TimerCreationIntent
                "duration_seconds": duration_seconds,
                "label": label,
                "deferred_workflow": deferred_workflow,
                # user_id and channel_id will be injected by UniversalAgent
            },
        }
    except Exception as e:
        logger.error(f"Timer creation error: {e}")
        return {"success": False, "error": str(e)}


@tool
def cancel_timer(timer_id: str) -> dict[str, Any]:
    """Cancel an active timer.

    Action tool - returns intent for timer cancellation (declarative, no side effects).

    Args:
        timer_id: ID of the timer to cancel

    Returns:
        Dict with success status, message, and intent data
    """
    logger.info(f"Cancelling timer: {timer_id}")

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
    """List all active timers for the user.

    Query tool - returns intent for timer listing (declarative, no side effects).

    Returns:
        Dict with success status, message, and intent data
    """
    logger.info("Listing active timers")

    return {
        "success": True,
        "message": "Listing active timers",
        "intent": {
            "type": "TimerListingIntent",
            # user_id and channel_id will be injected by UniversalAgent
        },
    }
