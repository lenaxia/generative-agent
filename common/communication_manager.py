"""
Communication Manager for multi-channel notification delivery.

This module provides an abstraction layer for sending notifications
across multiple communication channels without tight coupling.
"""

import asyncio
import glob
import importlib
import inspect
import logging
import os
import pkgutil
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from common.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Supported notification channel types."""

    SLACK = "slack"
    SMS = "sms"
    EMAIL = "email"
    SONOS = "sonos"
    HOME_ASSISTANT = "home_assistant"
    WHATSAPP = "whatsapp"
    CONSOLE = "console"  # Default fallback channel


class DeliveryGuarantee(Enum):
    """Delivery guarantee levels for notifications."""

    BEST_EFFORT = "best_effort"  # Try primary channel only, no guarantees
    AT_LEAST_ONCE = (
        "at_least_once"  # Try fallbacks, may result in duplicate notifications
    )
    EXACTLY_ONCE = "exactly_once"  # Ensure delivery with deduplication


class MessageFormat(Enum):
    """Message format types supported by channels."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    RICH_TEXT = "rich_text"
    JSON = "json"


class ChannelHandler:
    """Base class for all channel handlers."""

    channel_type: ChannelType

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the channel handler with optional configuration."""
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    async def send_notification(
        self,
        message: str,
        recipient: Optional[str] = None,
        message_format: MessageFormat = MessageFormat.PLAIN_TEXT,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Send a notification through this channel.

        Args:
            message: The message content
            recipient: Channel-specific recipient identifier
            message_format: Format of the message content
            metadata: Additional channel-specific metadata

        Returns:
            Dict with status information about the delivery
        """
        if not self.enabled:
            return {"success": False, "error": "Channel disabled"}

        try:
            return await self._send(message, recipient, message_format, metadata or {})
        except Exception as e:
            logger.error(
                f"Error sending notification via {self.channel_type}: {str(e)}"
            )
            return {"success": False, "error": str(e)}

    async def _send(
        self,
        message: str,
        recipient: Optional[str],
        message_format: MessageFormat,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Implementation-specific sending logic.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Channel handlers must implement _send method")


class CommunicationManager:
    """
    Manages communication across multiple channels with fallback support.

    This class handles:
    - Registration of channel handlers
    - Routing notifications to appropriate channels
    - Fallback logic when primary channels fail
    - Message format conversion between channels
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "CommunicationManager":
        """Get or create the singleton instance of CommunicationManager."""
        if cls._instance is None:
            cls._instance = CommunicationManager()
        return cls._instance

    def __init__(self):
        """Initialize the communication manager."""
        if CommunicationManager._instance is not None:
            raise RuntimeError("Use get_instance() to get the CommunicationManager")

        self.channels: dict[ChannelType, ChannelHandler] = {}
        self.default_channel = ChannelType.CONSOLE
        self.message_bus = MessageBus()
        self.message_bus.start()  # Start the message bus

        # Subscribe to timer expired events
        self.message_bus.subscribe(
            self, MessageType.TIMER_EXPIRED, self._handle_timer_expired
        )

        # Auto-discover and register available channel handlers
        self._discover_channel_handlers()

    def _discover_channel_handlers(self):
        """Automatically discover and register available channel handlers."""
        try:
            # Get the directory where channel handlers are located
            channel_dir = os.path.join(os.path.dirname(__file__), "channel_handlers")

            if not os.path.exists(channel_dir):
                logger.warning(f"Channel handlers directory not found: {channel_dir}")
                return

            # Find all handler files in the directory
            handler_files = glob.glob(os.path.join(channel_dir, "*_handler.py"))

            for handler_file in handler_files:
                try:
                    # Extract module name from file path
                    module_name = os.path.basename(handler_file)[
                        :-3
                    ]  # Remove .py extension

                    # Import the module
                    module = importlib.import_module(
                        f"common.channel_handlers.{module_name}"
                    )

                    # Find handler classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, ChannelHandler)
                            and attr is not ChannelHandler
                        ):
                            # Create and register the handler
                            handler = attr()
                            if hasattr(handler, "channel_type"):
                                self.register_channel(handler)
                                logger.info(
                                    f"Auto-registered channel handler: {attr_name} from {module_name}"
                                )
                except Exception as e:
                    logger.error(
                        f"Error loading channel handler from {handler_file}: {str(e)}"
                    )
        except Exception as e:
            logger.error(f"Error during channel handler discovery: {str(e)}")

    def register_channel(self, handler: ChannelHandler) -> None:
        """
        Register a new channel handler.

        Args:
            handler: The channel handler instance to register
        """
        if not isinstance(handler, ChannelHandler):
            raise TypeError("Handler must be an instance of ChannelHandler")

        self.channels[handler.channel_type] = handler
        logger.info(f"Registered channel handler for {handler.channel_type.value}")

    def unregister_channel(self, channel_type: ChannelType) -> None:
        """
        Unregister a channel handler.

        Args:
            channel_type: The type of channel to unregister
        """
        if channel_type in self.channels:
            del self.channels[channel_type]
            logger.info(f"Unregistered channel handler for {channel_type.value}")

    async def send_notification(
        self,
        message: str,
        channel_type: Optional[ChannelType] = None,
        recipient: Optional[str] = None,
        message_format: MessageFormat = MessageFormat.PLAIN_TEXT,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.BEST_EFFORT,
        metadata: Optional[dict[str, Any]] = None,
        fallback_channels: Optional[list[ChannelType]] = None,
    ) -> dict[str, Any]:
        """
        Send a notification through the specified channel with fallback support.

        Args:
            message: The message content
            channel_type: Primary channel to use (defaults to default_channel)
            recipient: Channel-specific recipient identifier
            message_format: Format of the message content
            priority: Priority level of the notification
            metadata: Additional channel-specific metadata
            fallback_channels: List of fallback channels to try if primary fails

        Returns:
            Dict with status information about the delivery
        """
        metadata = metadata or {}
        channel_type = channel_type or self.default_channel
        # Set fallback behavior based on delivery guarantee
        fallback_channels = fallback_channels or []

        # For BEST_EFFORT, we don't use fallbacks unless explicitly provided
        if (
            delivery_guarantee == DeliveryGuarantee.BEST_EFFORT
            and not fallback_channels
        ):
            fallback_channels = []
        # For AT_LEAST_ONCE, we use all available channels as potential fallbacks
        elif (
            delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
            and not fallback_channels
        ):
            fallback_channels = [
                ch for ch in self.channels.keys() if ch != channel_type
            ]
        # For EXACTLY_ONCE, we use a subset of reliable channels as fallbacks
        elif (
            delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE
            and not fallback_channels
        ):
            # Prioritize more reliable channels for exactly-once delivery
            reliable_channels = [
                ChannelType.EMAIL,
                ChannelType.SLACK,
                ChannelType.CONSOLE,
            ]
            fallback_channels = [
                ch
                for ch in reliable_channels
                if ch in self.channels and ch != channel_type
            ]

        # Try primary channel first
        if channel_type in self.channels:
            result = await self.channels[channel_type].send_notification(
                message, recipient, message_format, metadata
            )

            if result.get("success", False):
                return result

            logger.warning(
                f"Primary channel {channel_type.value} failed: {result.get('error')}"
            )
        else:
            logger.warning(f"Channel {channel_type.value} not registered")

        # Try fallback channels in order
        for fallback in fallback_channels:
            if fallback in self.channels and fallback != channel_type:
                result = await self.channels[fallback].send_notification(
                    message, recipient, message_format, metadata
                )

                if result.get("success", False):
                    logger.info(
                        f"Notification sent via fallback channel {fallback.value}"
                    )
                    return result

        # For AT_LEAST_ONCE or EXACTLY_ONCE guarantees, try all available channels as last resort
        if delivery_guarantee in [
            DeliveryGuarantee.AT_LEAST_ONCE,
            DeliveryGuarantee.EXACTLY_ONCE,
        ]:
            for available_channel in self.channels:
                if (
                    available_channel != channel_type
                    and available_channel not in fallback_channels
                ):
                    result = await self.channels[available_channel].send_notification(
                        message, recipient, message_format, metadata
                    )

                    if result.get("success", False):
                        logger.info(
                            f"Critical notification sent via channel {available_channel.value}"
                        )
                        return result

        # If we get here, all channels failed
        logger.error(
            f"Failed to send notification through any channel: {message[:50]}..."
        )
        return {"success": False, "error": "All notification channels failed"}

    async def _handle_timer_expired(self, message: dict[str, Any]) -> None:
        """
        Handle timer expired events from the message bus.

        Args:
            message: The timer expired event message
        """
        timer_data = message.get("data", {})
        timer_id = timer_data.get("timer_id")
        timer_name = timer_data.get("name", "Timer")

        if not timer_id:
            logger.error("Received timer expired event without timer_id")
            return

        # Extract notification preferences if available
        notification_channel = timer_data.get("notification_channel")
        recipient = timer_data.get("notification_recipient")

        channel_type = None
        if notification_channel:
            try:
                channel_type = ChannelType(notification_channel)
            except ValueError:
                logger.warning(f"Unknown notification channel: {notification_channel}")

        # Send the notification
        notification_message = f"⏰ Timer expired: {timer_name}"
        await self.send_notification(
            message=notification_message,
            channel_type=channel_type,
            recipient=recipient,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
            metadata={"timer_id": timer_id, "timer_data": timer_data},
            additional_channels=[],  # Use default channels based on delivery guarantee
        )


def get_communication_manager() -> CommunicationManager:
    """Get the singleton instance of the CommunicationManager."""
    return CommunicationManager.get_instance()
