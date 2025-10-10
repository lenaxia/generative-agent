"""
Slack channel handler for the Communication Manager.

This handler provides integration with Slack for sending notifications
with support for rich formatting and interactive buttons.
"""

import asyncio
import json
import logging
import os
import threading
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
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN") or self.config.get(
            "bot_token"
        )
        self.app_token = os.environ.get("SLACK_APP_TOKEN") or self.config.get(
            "app_token"
        )
        self.default_channel = self.config.get("default_channel", "#general")

        # WebSocket and bidirectional support
        self.slack_app = None
        self.socket_handler = None
        self.pending_questions = {}  # Track questions waiting for responses
        self.question_timeout = 300  # Default timeout for questions

        # Validate configuration - don't disable here, let _validate_requirements handle it

    def _validate_requirements(self) -> bool:
        """Validate Slack configuration."""
        # For bidirectional support, we need both bot token and app token
        if self.bot_token and self.app_token:
            return True
        elif self.webhook_url or self.bot_token:
            if not (self.bot_token and self.app_token):
                logger.info(
                    "Slack WebSocket disabled: using webhook/API-only mode (unidirectional)"
                )
            return True
        else:
            logger.error("SLACK_BOT_TOKEN/webhook_url required for Slack functionality")
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Slack channel capabilities."""
        # Only consider bidirectional if we have BOTH bot token AND app token
        has_websocket = bool(self.bot_token and self.app_token)
        return {
            "supports_rich_text": True,
            "supports_buttons": True,
            "supports_images": True,
            "bidirectional": has_websocket,
            "requires_session": has_websocket,
            "max_message_length": 4000,
        }

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

    async def _background_session_loop(self):
        """Run Slack WebSocket in background thread."""
        if not (self.bot_token and self.app_token):
            logger.warning("Slack WebSocket disabled: missing bot_token or app_token")
            return

        try:
            # Import slack_bolt here to avoid dependency issues if not installed
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError:
            logger.error(
                "slack_bolt library required for WebSocket support: pip install slack-bolt"
            )
            return

        # Create Slack app
        self.slack_app = App(token=self.bot_token)
        self.socket_handler = SocketModeHandler(self.slack_app, self.app_token)

        # Handle incoming messages
        @self.slack_app.event("message")
        def handle_message(event, say):
            if not event.get("bot_id"):  # Ignore bot messages
                # Send to main thread via queue
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(
                        {
                            "type": "incoming_message",
                            "user_id": event["user"],
                            "channel_id": event["channel"],
                            "text": event.get("text", ""),
                            "timestamp": event.get("ts"),
                        }
                    ),
                    self._get_main_event_loop(),
                )

        # Handle button interactions
        @self.slack_app.action(".*")  # Match all button actions
        def handle_button_click(ack, body):
            ack()  # Acknowledge button click

            action_id = body["actions"][0]["action_id"]
            value = body["actions"][0]["value"]
            user_id = body["user"]["id"]
            channel_id = body["channel"]["id"]

            # Check if this is a response to a pending question
            if action_id in self.pending_questions:
                question_data = self.pending_questions[action_id]
                question_data["response_future"].set_result(value)
                del self.pending_questions[action_id]

            # Also send to main thread for general processing
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    {
                        "type": "user_response",
                        "data": {
                            "action_id": action_id,
                            "value": value,
                            "user_id": user_id,
                            "channel_id": channel_id,
                        },
                    }
                ),
                self._get_main_event_loop(),
            )

        # Start WebSocket (blocks in this thread)
        logger.info("Starting Slack WebSocket connection...")
        await self.socket_handler.start_async()

    def _get_main_event_loop(self):
        """Get reference to main thread's event loop."""
        # This is a simplified approach - in production you'd want to store
        # the main loop reference during initialization
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop in current thread, create a new one
            return asyncio.new_event_loop()

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Ask question with Slack buttons and wait for response."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError(
                "Slack WebSocket not available for bidirectional communication"
            )

        # Generate unique action ID for this question
        import uuid

        action_id = f"question_{uuid.uuid4().hex[:8]}"

        # Create buttons for options
        buttons = []
        for i, option in enumerate(options or ["Yes", "No"]):
            buttons.append(
                {
                    "text": option,
                    "value": option,
                    "style": "primary" if i == 0 else "default",
                }
            )

        # Create blocks with question and buttons
        blocks = self._create_blocks_with_buttons(question, buttons)

        # Set up response tracking
        response_future = asyncio.Future()
        self.pending_questions[action_id] = {
            "question": question,
            "options": options,
            "response_future": response_future,
        }

        try:
            # Send question via API
            result = await self._send_via_api(
                question,
                self.default_channel,
                MessageFormat.RICH_TEXT,
                buttons,
                {"blocks": blocks},
            )

            if not result.get("success"):
                raise Exception(f"Failed to send question: {result.get('error')}")

            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            # Clean up pending question
            if action_id in self.pending_questions:
                del self.pending_questions[action_id]
            return "timeout"
        except Exception as e:
            # Clean up pending question
            if action_id in self.pending_questions:
                del self.pending_questions[action_id]
            logger.error(f"Error asking Slack question: {e}")
            raise
