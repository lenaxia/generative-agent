"""
Console channel handler for the Communication Manager.

This handler provides a simple console-based notification channel
that prints messages to the console. It's useful for development
and as a fallback when other channels are unavailable.
"""

import logging
from typing import Any, Dict, Optional

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class ConsoleChannelHandler(ChannelHandler):
    """
    A simple channel handler that outputs notifications to the console.

    This handler is always available and serves as a fallback when
    other channels fail or aren't configured.
    """

    channel_type = ChannelType.CONSOLE

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the console channel handler."""
        super().__init__(config)
        self.log_level = self.config.get("log_level", logging.INFO)

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a notification by logging it to the console.

        Args:
            message: The message content
            recipient: Ignored for console handler
            message_format: Format of the message content
            metadata: Additional metadata

        Returns:
            Dict with status information
        """
        # Format the message with recipient if provided
        formatted_message = message
        if recipient:
            formatted_message = f"[To: {recipient}] {message}"

        # Add a prefix based on metadata if available
        if "timer_id" in metadata:
            formatted_message = (
                f"[Timer: {metadata.get('timer_id')}] {formatted_message}"
            )

        # Log the message
        logger.log(self.log_level, f"NOTIFICATION: {formatted_message}")

        # Also print to console for visibility
        print(f"\n⏰ NOTIFICATION: {formatted_message}\n")

        return {
            "success": True,
            "channel": self.channel_type.value,
            "timestamp": metadata.get("timestamp"),
        }
