"""
Slack channel handler for the Communication Manager.

This handler provides integration with Slack for sending notifications
with support for rich formatting and interactive buttons.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from common.communication_manager import ChannelHandler, ChannelType, MessageFormat

logger = logging.getLogger(__name__)


class SlackChannelHandler(ChannelHandler):
    """
    Channel handler for sending notifications to Slack.

    Supports:
    - Plain text and markdown messages
    - Rich formatting with blocks
    - Interactive buttons
    - Direct messages and channel posts
    """

    channel_type = ChannelType.SLACK

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the Slack channel handler."""
        super().__init__(config)

        # Extract configuration
        self.webhook_url = self.config.get("webhook_url")
        self.bot_token = self.config.get("bot_token")
        self.default_channel = self.config.get("default_channel", "#general")

        # Validate configuration
        if not (self.webhook_url or self.bot_token):
            logger.warning("Slack handler initialized without webhook_url or bot_token")
            self.enabled = False

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a notification to Slack.

        Args:
            message: The message content
            recipient: Slack channel or user ID (e.g., "#general" or "@user")
            message_format: Format of the message content
            metadata: Additional metadata including:
                - blocks: Slack blocks for rich formatting
                - attachments: Slack attachments
                - thread_ts: Thread timestamp to reply in a thread
                - buttons: List of button configs (text, value, style)

        Returns:
            Dict with status information
        """
        # Determine the channel to post to
        channel = recipient or self.default_channel

        # Check if we have buttons to add
        buttons = metadata.get("buttons", [])

        # Prepare the payload based on available credentials
        if self.webhook_url:
            return await self._send_via_webhook(
                message, channel, message_format, buttons, metadata
            )
        elif self.bot_token:
            return await self._send_via_api(
                message, channel, message_format, buttons, metadata
            )
        else:
            return {"success": False, "error": "No Slack credentials configured"}

    async def _send_via_webhook(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a message using Slack incoming webhooks."""
        # Prepare the payload
        payload = {"channel": channel, "text": message}

        # Add blocks if provided in metadata
        if "blocks" in metadata:
            payload["blocks"] = metadata["blocks"]
        # Otherwise create blocks if we have buttons
        elif buttons:
            payload["blocks"] = self._create_blocks_with_buttons(message, buttons)

        # Add attachments if provided
        if "attachments" in metadata:
            payload["attachments"] = metadata["attachments"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return {"success": True, "channel": channel}
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Slack webhook error: {response.status} - {error_text}",
                        }
        except Exception as e:
            logger.error(f"Error sending Slack webhook: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_via_api(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a message using the Slack API."""
        # Prepare the payload
        payload = {"channel": channel, "text": message}

        # Add thread_ts if provided for threading
        if "thread_ts" in metadata:
            payload["thread_ts"] = metadata["thread_ts"]

        # Add blocks if provided in metadata
        if "blocks" in metadata:
            payload["blocks"] = metadata["blocks"]
        # Otherwise create blocks if we have buttons
        elif buttons:
            payload["blocks"] = self._create_blocks_with_buttons(message, buttons)

        # Add attachments if provided
        if "attachments" in metadata:
            payload["attachments"] = metadata["attachments"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.bot_token}",
                    },
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data.get("ok"):
                            return {
                                "success": True,
                                "channel": channel,
                                "ts": response_data.get("ts"),
                                "message_id": response_data.get("ts"),
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Slack API error: {response_data.get('error')}",
                            }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Slack API error: {response.status} - {error_text}",
                        }
        except Exception as e:
            logger.error(f"Error sending Slack API request: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_blocks_with_buttons(
        self, message: str, buttons: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Create Slack blocks with text and action buttons."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": message}}]

        # Add buttons if provided
        if buttons:
            actions = []
            for button in buttons:
                action = {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": button.get("text", "Button"),
                        "emoji": True,
                    },
                    "value": button.get("value", "button_click"),
                }

                # Add style if provided
                if "style" in button:
                    action["style"] = button["style"]

                actions.append(action)

            blocks.append({"type": "actions", "elements": actions})

        return blocks
