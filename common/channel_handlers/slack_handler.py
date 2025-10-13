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
import requests

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
        self.bot_user_id = None  # Will be set when Slack app is initialized
        self.pending_questions = {}  # Track questions waiting for responses
        self.question_timeout = 300  # Default timeout for questions
        self.shutdown_flag = False  # Flag to signal shutdown
        self.session_active = False  # Track if WebSocket session is active
        self.shutdown_requested = False  # Runtime shutdown flag

        # Phase 2: Shared Session Pool
        self._session = None
        self._session_lock = None
        self._main_loop = None
        self._session_pool_size = self.config.get("session_pool_size", 5)
        self._use_shared_session = self.config.get("use_shared_session", True)

        # Validate configuration - don't disable here, let _validate_requirements handle it

    def _validate_requirements(self) -> bool:
        """Validate Slack configuration."""
        logger.info(
            f"Validating Slack configuration: bot_token={'***' if self.bot_token else None}, app_token={'***' if self.app_token else None}, webhook_url={'***' if self.webhook_url else None}"
        )

        # For bidirectional support, we need both bot token and app token
        if self.bot_token and self.app_token:
            logger.info("Slack WebSocket enabled: full bidirectional support available")
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

    def _get_requirements_error_message(self) -> str:
        """Get descriptive error message for missing Slack requirements."""
        if not self.bot_token and not self.webhook_url:
            return "missing SLACK_BOT_TOKEN environment variable or webhook_url config. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN for full WebSocket support, or webhook_url for basic notifications"
        elif self.bot_token and not self.app_token:
            return "missing SLACK_APP_TOKEN environment variable. Set SLACK_APP_TOKEN for bidirectional WebSocket support, or use webhook_url for unidirectional notifications"
        else:
            return "Slack configuration incomplete"

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

        # Enhance message with @ mention if user_id is provided
        user_id = metadata.get("user_id")
        enhanced_message = self._format_message_with_mention(message, user_id)

        # Check if we have buttons to add
        buttons = metadata.get("buttons", [])

        # Prepare the payload based on available credentials
        if self.webhook_url:
            return await self._send_via_webhook(
                enhanced_message, channel, message_format, buttons, metadata
            )
        elif self.bot_token:
            return await self._send_via_api(
                enhanced_message, channel, message_format, buttons, metadata
            )
        else:
            return {"success": False, "error": "No Slack credentials configured"}

    def _format_message_with_mention(self, message: str, user_id: Optional[str]) -> str:
        """Format message with @ mention if user_id is provided.

        Args:
            message: Original message content
            user_id: Slack user ID to mention (e.g., "U123456")

        Returns:
            Enhanced message with @ mention prepended if user_id provided
        """
        if user_id:
            return f"<@{user_id}> {message}"
        return message

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

    def _is_background_thread(self) -> bool:
        """Detect if we're running in a background thread context."""
        import threading

        current_thread = threading.current_thread()
        return current_thread.name != "MainThread"

    async def _get_or_create_session(self):
        """Get or create shared aiohttp session with proper event loop management."""
        if self._session is None or self._session.closed:
            if self._session_lock is None:
                self._session_lock = asyncio.Lock()

            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=10.0)
                    connector = aiohttp.TCPConnector(
                        limit=self._session_pool_size,
                        limit_per_host=self._session_pool_size,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    )
                    self._session = aiohttp.ClientSession(
                        timeout=timeout, connector=connector
                    )
                    logger.info(
                        f"Created shared aiohttp session with pool size {self._session_pool_size}"
                    )

        return self._session

    async def _send_via_shared_session(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send using shared session with proper event loop coordination."""
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
            logger.info(
                f"Shared session Slack API call starting with payload: {payload}"
            )
            session = await self._get_or_create_session()

            async with session.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self.bot_token}",
                },
            ) as response:
                logger.info(f"Received response with status: {response.status}")
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
        except asyncio.TimeoutError as e:
            logger.error(
                f"Shared session Slack API call timed out after 10 seconds: {str(e)}"
            )
            return {"success": False, "error": "Slack API timeout"}
        except Exception as e:
            logger.error(f"Error sending shared session Slack API request: {str(e)}")
            # If session is broken, reset it for next time
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except:
                    pass
                self._session = None
            return {"success": False, "error": str(e)}

    async def _send_via_api(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send message with automatic thread-safe detection and session management."""
        if self._is_background_thread():
            logger.info("Background thread detected, using thread-safe HTTP client")
            return await self._send_via_api_threadsafe(
                message, channel, message_format, buttons, metadata
            )
        else:
            if self._use_shared_session:
                logger.info("Main thread detected, using shared aiohttp session")
                return await self._send_via_shared_session(
                    message, channel, message_format, buttons, metadata
                )
            else:
                logger.info("Main thread detected, using standard aiohttp")
                return await self._send_via_api_aiohttp(
                    message, channel, message_format, buttons, metadata
                )

    async def _send_via_api_threadsafe(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Thread-safe version using requests library for cross-thread calls."""
        import asyncio

        import requests

        payload = {"channel": channel, "text": message}

        # Add thread_ts, blocks, attachments as before
        if "thread_ts" in metadata:
            payload["thread_ts"] = metadata["thread_ts"]
        if "blocks" in metadata:
            payload["blocks"] = metadata["blocks"]
        elif buttons:
            payload["blocks"] = self._create_blocks_with_buttons(message, buttons)
        if "attachments" in metadata:
            payload["attachments"] = metadata["attachments"]

        try:
            logger.info(f"Thread-safe Slack API call starting with payload: {payload}")

            # Run requests.post in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: requests.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "Authorization": f"Bearer {self.bot_token}",
                    },
                    timeout=10.0,
                ),
            )

            logger.info(f"Received response with status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
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
                return {
                    "success": False,
                    "error": f"Slack API error: {response.status_code} - {response.text}",
                }

        except Exception as e:
            logger.error(f"Error sending thread-safe Slack API request: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_via_api_aiohttp(
        self,
        message: str,
        channel: str,
        message_format: MessageFormat,
        buttons: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a message using the Slack API with aiohttp (original implementation)."""
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
            logger.info(f"Slack API call starting with payload: {payload}")
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=10.0)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info("Created aiohttp session, making POST request...")
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "Authorization": f"Bearer {self.bot_token}",
                    },
                ) as response:
                    logger.info(f"Received response with status: {response.status}")
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
        except asyncio.TimeoutError as e:
            logger.error(f"Slack API call timed out after 10 seconds: {str(e)}")
            return {"success": False, "error": "Slack API timeout"}
        except Exception as e:
            logger.error(f"Error sending Slack API request: {str(e)}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
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
        logger.info("🚀 Starting Slack WebSocket background session...")
        if not (self.bot_token and self.app_token):
            logger.warning("Slack WebSocket disabled: missing bot_token or app_token")
            return

        try:
            # Import slack_bolt here to avoid dependency issues if not installed
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            logger.info("✅ Successfully imported slack_bolt")
        except ImportError:
            logger.error(
                "slack_bolt library required for WebSocket support: pip install slack-bolt"
            )
            return

        # Create Slack app
        logger.info("🔧 Creating Slack app and socket handler...")
        self.slack_app = App(token=self.bot_token)
        self.socket_handler = SocketModeHandler(self.slack_app, self.app_token)

        # Get bot user ID for duplicate message filtering
        try:
            auth_response = self.slack_app.client.auth_test()
            self.bot_user_id = auth_response["user_id"]
            logger.info(f"🤖 Bot user ID: {self.bot_user_id}")
        except Exception as e:
            logger.warning(f"Failed to get bot user ID: {e}")
            self.bot_user_id = None

        # Unified message handler - routes based on message type
        def process_slack_message(event, message_type):
            """Unified message processing with proper routing."""
            # Ignore bot messages
            if event.get("bot_id"):
                logger.debug(f"🤖 Ignoring bot message: {event.get('bot_id')}")
                return

            text = event.get("text", "")
            user_id = event["user"]
            channel_id = event["channel"]
            timestamp = event.get("ts")

            # Determine message type for routing
            channel_type = event.get("channel_type", "")
            has_bot_mention = (
                text and self.bot_user_id and f"<@{self.bot_user_id}>" in text
            )

            # Route message based on type and context
            if message_type == "app_mention":
                # App mentions - always process regardless of channel
                logger.info(f"📢 Processing app mention from user {user_id}: {text}")
                queue_type = "app_mention"
            elif message_type == "message" and channel_type == "im":
                # Direct messages - process without bot mention requirement
                logger.info(f"📨 Processing direct message from user {user_id}: {text}")
                queue_type = "incoming_message"
            elif message_type == "message" and has_bot_mention:
                # Channel message with bot mention - ignore (will be handled by app_mention)
                logger.debug(
                    f"🔕 Ignoring channel message with bot mention (handled by app_mention): {text}"
                )
                return
            else:
                # Other channel messages without bot mention - ignore
                logger.debug(f"🔕 Ignoring channel message without bot mention: {text}")
                return

            # Queue the message for processing
            message_data = {
                "type": queue_type,
                "user_id": user_id,
                "channel_id": channel_id,
                "text": text,
                "timestamp": timestamp,
            }

            logger.debug(f"Queuing {queue_type} message: {message_data}")
            self.message_queue.put(message_data)

        # Handle incoming messages
        @self.slack_app.event("message")
        def handle_message(event, say):
            process_slack_message(event, "message")

        # Handle app mentions
        @self.slack_app.event("app_mention")
        def handle_app_mention(event, say):
            process_slack_message(event, "app_mention")

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

            # Send to main thread for general processing using thread-safe queue
            # Fixed: Use direct put() since message_queue is a thread-safe queue.Queue(), not asyncio queue
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
            )

        # Start WebSocket (blocks in this thread)
        logger.info("Starting Slack WebSocket connection...")
        # Use the correct method name for slack-bolt SocketModeHandler
        try:
            # Run the socket handler with shutdown monitoring
            await self._run_interruptible_socket_handler()
        except Exception as e:
            logger.error(f"Slack WebSocket connection failed: {e}")
            if not self.shutdown_requested:
                raise

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

    async def _run_interruptible_socket_handler(self):
        """Run socket handler with proper shutdown monitoring."""
        try:
            # Try async start first
            if hasattr(self.socket_handler, "start_async"):
                await self.socket_handler.start_async()
            else:
                # Fallback to synchronous start in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.socket_handler.start)
        except Exception as e:
            if not self.shutdown_requested:
                logger.error(f"Socket handler error: {e}")
                raise
            else:
                logger.info("Socket handler stopped due to shutdown")

    async def _cleanup_shared_session(self):
        """Clean up shared aiohttp session."""
        if self._session:
            if not self._session.closed:
                try:
                    await self._session.close()
                    logger.info("✅ Shared aiohttp session closed")
                except Exception as e:
                    logger.warning(f"Error closing shared session: {e}")
            # Always clear references regardless of close status
            self._session = None
            self._session_lock = None

    async def stop_session(self):
        """Stop Slack WebSocket session and cleanup resources."""
        logger.info("Stopping Slack WebSocket session...")
        self.shutdown_requested = True

        try:
            # Clean up shared session first
            await self._cleanup_shared_session()

            # Stop the socket handler if it exists
            if self.socket_handler:
                try:
                    # Try to stop synchronously first to avoid executor threads
                    if hasattr(self.socket_handler, "close"):
                        self.socket_handler.close()
                    elif hasattr(self.socket_handler, "stop"):
                        self.socket_handler.stop()
                except Exception as e:
                    logger.warning(f"Direct socket handler stop failed: {e}")
                    # Fallback to executor only if direct call fails
                    try:
                        if hasattr(self.socket_handler, "close"):
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.socket_handler.close
                            )
                        elif hasattr(self.socket_handler, "stop"):
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.socket_handler.stop
                            )
                    except Exception as e2:
                        logger.warning(
                            f"Executor socket handler stop also failed: {e2}"
                        )

            # Clear pending questions
            self.pending_questions.clear()

            # Mark session as inactive
            self.session_active = False

            logger.info("Slack WebSocket session stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping Slack session: {e}")
        finally:
            # Ensure cleanup even if there are errors
            self.slack_app = None
            self.socket_handler = None
            # Ensure shared session is cleaned up
            self._session = None
            self._session_lock = None
