"""Core Notification Tools

Provides notification tools for the agent system.
Tools send notifications to users via configured channels (Slack, Discord, etc.).

These are core infrastructure tools - DO NOT MODIFY.
For custom tools, use tools/custom/.

Moved from: roles/notification/tools.py
"""

import logging
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def create_notification_tools(communication_provider: Any) -> list:
    """Create notification domain tools.

    Args:
        communication_provider: Communication provider instance (Slack, Discord, etc.)

    Returns:
        List of tool functions for notification domain
    """
    # Store provider reference for tools to use
    global _communication_provider
    _communication_provider = communication_provider

    tools = [
        send_notification,
    ]

    logger.info(f"Created {len(tools)} notification tools")
    return tools


# Global provider reference (set by create_notification_tools)
_communication_provider = None


def _get_communication_provider():
    """Get the communication provider instance."""
    if _communication_provider is None:
        raise RuntimeError(
            "Communication provider not initialized. Call create_notification_tools first."
        )
    return _communication_provider


# ACTION TOOLS


@tool
def send_notification(
    message: str,
    channel: str | None = None,
    priority: str = "normal",
    notification_type: str = "info",
) -> dict[str, Any]:
    """Send a notification to the user.

    Action tool - sends notification (has side effects).

    Args:
        message: Notification message content
        channel: Optional channel/user to send to (defaults to current context)
        priority: Priority level: "low", "normal", "high" (default: "normal")
        notification_type: Type of notification: "info", "warning", "error", "success" (default: "info")

    Returns:
        Dict with success status and message ID
    """
    logger.info(f"Sending notification: type={notification_type}, priority={priority}")

    try:
        # Return intent for notification
        # The infrastructure will process this via the communication provider
        return {
            "success": True,
            "message": "Notification queued for delivery",
            "intent": {
                "type": "NotificationIntent",
                "message": message,
                "channel": channel,
                "priority": priority,
                "notification_type": notification_type,
                # user_id and additional context will be injected by infrastructure
            },
        }

    except Exception as e:
        logger.error(f"Notification error: {e}")
        return {
            "success": False,
            "error": str(e),
        }
