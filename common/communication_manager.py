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
import queue
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Supported notification channel types."""

    SLACK = "slack"
    SMS = "sms"
    EMAIL = "email"
    SONOS = "sonos"
    HOME_ASSISTANT = "home_assistant"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
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
    """Base class for all communication channel handlers."""

    channel_type: ChannelType  # Must be defined in subclasses

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize the channel handler with optional configuration."""
        self.config = config or {}
        self.enabled = False
        self.session_active = False
        self.background_task = None
        self.message_queue = None
        self.communication_manager = None  # Set by CommunicationManager
        self.shutdown_requested = False  # Flag for graceful shutdown

    async def validate_and_initialize(self) -> bool:
        """Validate requirements and initialize channel."""
        validation_result = self._validate_requirements()
        if not validation_result:
            # Get more descriptive error message
            error_msg = self._get_requirements_error_message()
            logger.warning(f"{self.channel_type.value} disabled: {error_msg}")
            return False

        try:
            if self.requires_background_thread():
                await self._start_background_thread()
            else:
                await self.start_session()

            self.enabled = True
            logger.info(f"{self.channel_type.value} channel initialized successfully")
            return True
        except Exception as e:
            logger.error(f"{self.channel_type.value} initialization failed: {e}")
            return False

    def _get_requirements_error_message(self) -> str:
        """Get descriptive error message for missing requirements. Override in subclasses."""
        return "requirements not met"

    def _validate_requirements(self) -> bool:
        """Validate channel requirements (env vars, hardware, etc.). Override in subclasses."""
        return True

    def requires_background_thread(self) -> bool:
        """Return True if channel needs background thread for persistent connections."""
        capabilities = self.get_capabilities()
        return capabilities.get("requires_session", False) and capabilities.get(
            "bidirectional", False
        )

    async def start_session(self):
        """Start channel session. Override for stateless channels."""
        self.session_active = True

    async def _start_background_thread(self):
        """Start background thread for stateful channels."""
        self.message_queue = queue.Queue()
        self.background_thread = threading.Thread(
            target=self._run_background_session,
            daemon=True,  # Ensure daemon mode for clean shutdown
            name=f"{self.channel_type.value}_thread",
        )
        self.background_thread.start()

        # Register queue and thread with CommunicationManager
        if self.communication_manager:
            self.communication_manager.channel_queues[
                self.channel_type.value
            ] = self.message_queue
            # LLM-SAFE: Track the background task for cleanup
            if not hasattr(self.communication_manager, "_background_tasks"):
                self.communication_manager._background_tasks = {}
            self.communication_manager._background_tasks[
                self.channel_type.value
            ] = self.background_task

    def _run_background_session(self):
        """Run background session in dedicated thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._background_session_loop())
        except Exception as e:
            logger.error(f"{self.channel_type.value} background session failed: {e}")
        finally:
            loop.close()

    async def _background_session_loop(self):
        """Background session loop. Override in stateful channels."""
        while True:
            await asyncio.sleep(1)  # Default: do nothing

    def get_capabilities(self) -> dict[str, Any]:
        """Return channel capabilities. Must override in subclasses."""
        return {
            "supports_rich_text": False,
            "supports_buttons": False,
            "supports_audio": False,
            "supports_images": False,
            "bidirectional": False,
            "requires_session": False,
            "max_message_length": 1000,
        }

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

    async def ask_question(
        self, question: str, options: Optional[list[str]] = None, timeout: int = 300
    ) -> str:
        """Ask user a question and wait for response. Override in bidirectional channels."""
        if not self.get_capabilities().get("bidirectional", False):
            raise NotImplementedError(
                f"{self.channel_type.value} doesn't support bidirectional communication"
            )
        return await self._ask_question_impl(question, options or [], timeout)

    async def _ask_question_impl(
        self, question: str, options: list[str], timeout: int
    ) -> str:
        """Question implementation. Override in bidirectional channels."""
        raise NotImplementedError(
            "Bidirectional channels must implement _ask_question_impl"
        )


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
        """Get or create the singleton instance of CommunicationManager.

        Note: This method is deprecated. Use dependency injection instead.
        """
        if cls._instance is None:
            # Create with a new MessageBus for backward compatibility
            from common.message_bus import MessageBus

            message_bus = MessageBus()
            message_bus.start()
            cls._instance = CommunicationManager(message_bus)
        return cls._instance

    def __init__(self, message_bus: MessageBus):
        """Initialize the communication manager with supervisor's MessageBus."""
        self.message_bus = message_bus  # Use supervisor's MessageBus
        self.channels: dict[str, ChannelHandler] = {}
        self.channel_queues: dict[str, asyncio.Queue] = {}  # Thread communication
        self.default_channel = ChannelType.CONSOLE
        self._shutdown_called = False  # Prevent duplicate shutdowns
        self._background_threads: dict[
            str, threading.Thread
        ] = {}  # Track all background threads
        self._initialized = False
        self._pending_requests: dict[
            str, dict
        ] = {}  # Track requests awaiting responses

        # Subscribe to communication events
        self._setup_message_subscriptions()

    async def initialize(self):
        """Initialize channels and start background tasks."""
        if self._initialized:
            return

        # Auto-discover and register available channel handlers
        await self._discover_and_initialize_channels()

        # Start queue processor for background thread communication
        asyncio.create_task(self._process_channel_queues())

        self._initialized = True

    def initialize_sync(self):
        """Initialize channels synchronously (for startup without event loop)."""
        if self._initialized:
            return

        # Discover and register channel handlers synchronously
        import asyncio

        # Create a temporary event loop for initialization
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async initialization
        loop.run_until_complete(self._discover_and_initialize_channels())

        self._initialized = True
        logger.info("Communication manager channels initialized synchronously")

        # Note: Queue processor will be started when supervisor starts its event loop

        # Start queue processor in background thread since we don't have an event loop
        self.start_queue_processor_thread()

    def start_queue_processor_thread(self):
        """Start the queue processor in a background thread."""
        if not self._initialized:
            logger.warning(
                "Communication manager not initialized, cannot start queue processor"
            )
            return

        logger.info("🚀 Starting channel queue processor in background thread...")

        def run_queue_processor():
            """Run the queue processor in its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._process_channel_queues())
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
            finally:
                loop.close()

        queue_thread = threading.Thread(
            target=run_queue_processor,
            daemon=True,
            name="communication_queue_processor",
        )
        queue_thread.start()
        logger.info("✅ Channel queue processor thread started")

    def _setup_message_subscriptions(self):
        """Subscribe to all communication-related MessageBus events."""
        subscriptions = [
            (MessageType.SEND_MESSAGE, self._handle_send_message),
            (MessageType.AGENT_QUESTION, self._handle_agent_question),
            (MessageType.TASK_RESPONSE, self._handle_task_response),
        ]

        for message_type, handler in subscriptions:
            self.message_bus.subscribe(self, message_type, handler)

    async def _process_channel_queues(self):
        """Process incoming messages from channel background threads."""
        logger.info("🔄 Starting channel queue processor...")
        while True:
            for channel_id, queue_obj in self.channel_queues.items():
                try:
                    # Process all available messages
                    while True:
                        try:
                            message = queue_obj.get_nowait()  # Non-blocking get
                            logger.debug(
                                f"Processing queued message from {channel_id}: {message}"
                            )
                            await self._handle_channel_message(channel_id, message)
                            queue_obj.task_done()
                        except queue.Empty:
                            break
                except Exception as e:
                    logger.error(f"Queue processing error for {channel_id}: {e}")
            await asyncio.sleep(0.1)  # Prevent busy loop

    async def _process_single_queue_iteration(self):
        """Process a single iteration of queue processing for testing."""
        for channel_id, queue_obj in self.channel_queues.items():
            try:
                # Process all available messages
                while True:
                    try:
                        if hasattr(queue_obj, "get_nowait"):
                            # Thread-safe queue
                            message = queue_obj.get_nowait()
                            await self._handle_channel_message(channel_id, message)
                            queue_obj.task_done()
                        else:
                            # Asyncio queue (current implementation)
                            message = await queue_obj.get()
                            await self._handle_channel_message(channel_id, message)
                    except (
                        queue.Empty
                        if hasattr(queue_obj, "get_nowait")
                        else asyncio.QueueEmpty
                    ):
                        break
            except Exception as e:
                logger.error(f"Queue processing error for {channel_id}: {e}")

    async def _handle_channel_message(self, channel_id: str, message: dict):
        """Handle incoming message from channel background thread."""
        logger.debug(f"Handling channel message from {channel_id}: {message}")

        if message["type"] == "incoming_message" or message["type"] == "app_mention":
            # Generate a unique request ID to track this request
            import uuid

            request_id = str(uuid.uuid4())

            # Store request info for response routing
            full_channel_id = f"{channel_id}:{message['channel_id']}"
            self._pending_requests[request_id] = {
                "channel_id": full_channel_id,
                "user_id": message["user_id"],
                "source_channel": channel_id,
                "original_message": message,
            }

            # Create proper RequestMetadata object for WorkflowEngine
            request_metadata = RequestMetadata(
                prompt=message["text"],
                source_id=channel_id,
                target_id="workflow_engine",
                metadata={
                    "user_id": message["user_id"],
                    "channel_id": full_channel_id,
                    "source": channel_id,
                    "request_id": request_id,
                },
                response_requested=True,
            )

            logger.debug(
                f"Publishing INCOMING_REQUEST to message bus: {request_metadata}"
            )

            # Route to supervisor for processing
            self.message_bus.publish(
                self,
                MessageType.INCOMING_REQUEST,
                request_metadata,
            )
        elif message["type"] == "user_response":
            # Handle response to agent question
            self.message_bus.publish(self, MessageType.USER_RESPONSE, message["data"])

    async def _handle_send_message(self, message: dict[str, Any]) -> None:
        """Handle send message events from the message bus."""
        logger.info(f"_handle_send_message called with: {message}")
        message_text = message.get("message", "")
        context = message.get("context", {})
        logger.info(f"Routing message: '{message_text}' with context: {context}")

        try:
            result = await self.route_message(message_text, context)
            logger.info(f"_handle_send_message completed successfully: {result}")
        except Exception as e:
            logger.error(f"Error in _handle_send_message: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _handle_agent_question(self, message: dict[str, Any]) -> None:
        """Handle agent question events from the message bus."""
        question_data = message.get("data", {})
        question = question_data.get("question", "")
        options = question_data.get("options", [])
        timeout = question_data.get("timeout", 300)
        origin_channel = question_data.get("channel_id", "console")

        # Route question to appropriate channel
        if origin_channel in self.channels:
            try:
                response = await self.channels[origin_channel].ask_question(
                    question, options, timeout
                )
                self.message_bus.publish(
                    self,
                    MessageType.USER_RESPONSE,
                    {
                        "response": response,
                        "question_id": question_data.get("question_id"),
                    },
                )
            except Exception as e:
                logger.error(f"Error asking question via {origin_channel}: {e}")
                self.message_bus.publish(
                    self,
                    MessageType.USER_RESPONSE,
                    {
                        "response": "error",
                        "error": str(e),
                        "question_id": question_data.get("question_id"),
                    },
                )

    async def _handle_task_response(self, message: dict[str, Any]) -> None:
        """Handle task response events from the message bus."""
        try:
            logger.info(f"📥 Received TASK_RESPONSE: {message}")

            # Extract response data (handle actual workflow engine structure)
            request_id = message.get("request_id")
            response_text = message.get("result", message.get("response", ""))
            task_id = message.get("task_id")
            status = message.get("status")

            if not request_id:
                logger.warning(f"Received task response without request_id: {message}")
                return

            if request_id not in self._pending_requests:
                logger.warning(
                    f"Received response for unknown request ID: {request_id}"
                )
                return

            # Get the original request info
            request_info = self._pending_requests.pop(request_id)
            source_channel = request_info["source_channel"]
            channel_id = request_info["channel_id"]
            user_id = request_info["user_id"]

            # Extract just the channel part (remove source prefix)
            target_channel = (
                channel_id.split(":", 1)[1] if ":" in channel_id else channel_id
            )

            logger.info(
                f"🎯 Routing response back to {source_channel}:{target_channel} for user {user_id}"
            )

            # Send response back to the originating channel
            if source_channel in self.channels:
                await self.channels[source_channel].send_notification(
                    f"🤖 {response_text}",
                    target_channel,
                    MessageFormat.PLAIN_TEXT,
                    {"user_id": user_id},
                )
                logger.info(
                    f"✅ Sent response back to {source_channel}:{target_channel}"
                )
            else:
                logger.warning(
                    f"❌ Source channel {source_channel} not available for response"
                )

        except Exception as e:
            logger.error(f"❌ Error handling task response: {e}")

    async def route_message(self, message: str, context: dict) -> list[dict]:
        """Route message to appropriate channels with fallback support."""
        origin_channel = context.get("channel_id", "console")
        delivery_guarantee = context.get(
            "delivery_guarantee", DeliveryGuarantee.BEST_EFFORT
        )
        message_type = context.get("message_type", "notification")

        logger.info(
            f"route_message called: origin_channel={origin_channel}, message_type={message_type}"
        )

        # Determine target channels
        target_channels = self._determine_target_channels(
            origin_channel, message_type, context
        )

        logger.info(f"Target channels determined: {target_channels}")

        # Send with appropriate delivery guarantee
        result = await self._send_with_delivery_guarantee(
            message, target_channels, context, delivery_guarantee
        )

        logger.info(f"_send_with_delivery_guarantee result: {result}")
        return result

    def _determine_target_channels(
        self, origin_channel: str, message_type: str, context: dict
    ) -> list[str]:
        """Determine which channels should receive the message."""
        # Extract channel type from channel_id (format: "channel_type:actual_id")
        if ":" in origin_channel:
            channel_type = origin_channel.split(":", 1)[0]
        else:
            channel_type = origin_channel

        # Default: return to origin channel type
        channels = [channel_type] if channel_type else ["console"]

        # Special routing rules
        if message_type == "timer_expired":
            # Timer notifications: origin + audio if user preferences allow
            if self._should_add_audio_notification(context):
                channels.append("sonos")
        elif message_type == "music_control":
            # Music control: always route to Sonos + origin for confirmation
            channels = ["sonos"] + (
                [origin_channel] if origin_channel != "sonos" else []
            )
        elif message_type == "smart_home_control":
            # Smart home: origin + Home Assistant for device status
            channels = [origin_channel, "home_assistant"]

        return [ch for ch in channels if ch and ch in self.channels]

    def _should_add_audio_notification(self, context: dict) -> bool:
        """Check if audio notification should be added based on context."""
        # Add logic based on user preferences, time of day, etc.
        return context.get("audio_enabled", False)

    async def _send_with_delivery_guarantee(
        self,
        message: str,
        target_channels: list[str],
        context: dict,
        delivery_guarantee: DeliveryGuarantee,
    ) -> list[dict]:
        """Send message with specified delivery guarantee."""
        logger.info(
            f"_send_with_delivery_guarantee called with target_channels: {target_channels}"
        )
        results = []
        successful_delivery = False

        # Try target channels first
        for channel_id in target_channels:
            logger.info(f"Trying to send to channel: {channel_id}")
            if channel_id in self.channels:
                logger.info(f"Channel {channel_id} found in self.channels")
                try:
                    logger.info(
                        f"Calling send_notification on {channel_id} with message: '{message}'"
                    )
                    logger.info(
                        f"send_notification context: recipient={context.get('recipient')}, format={context.get('message_format', MessageFormat.PLAIN_TEXT)}"
                    )

                    # Add timeout to prevent hanging
                    result = await asyncio.wait_for(
                        self.channels[channel_id].send_notification(
                            message,
                            context.get("recipient"),
                            context.get("message_format", MessageFormat.PLAIN_TEXT),
                            context.get("metadata", {}),
                        ),
                        timeout=10.0,  # 10 second timeout
                    )
                    logger.info(
                        f"Channel {channel_id} send_notification result: {result}"
                    )
                    results.append({"channel": channel_id, "result": result})
                    if result.get("success"):
                        successful_delivery = True
                        logger.info(f"Successful delivery via {channel_id}")
                except Exception as e:
                    logger.error(f"Error sending to channel {channel_id}: {e}")
                    import traceback

                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    results.append(
                        {
                            "channel": channel_id,
                            "result": {"success": False, "error": str(e)},
                        }
                    )
            else:
                logger.warning(
                    f"Channel {channel_id} not found in self.channels: {list(self.channels.keys())}"
                )

        # If no successful delivery and we need guarantees, try fallback channels
        if (
            not successful_delivery
            and delivery_guarantee != DeliveryGuarantee.BEST_EFFORT
        ):
            # Try all other available channels as fallbacks
            for channel_id, handler in self.channels.items():
                if channel_id not in target_channels and handler.enabled:
                    result = await handler.send_notification(
                        message,
                        context.get("recipient"),
                        context.get("message_format", MessageFormat.PLAIN_TEXT),
                        context.get("metadata", {}),
                    )
                    results.append({"channel": channel_id, "result": result})
                    if result.get("success"):
                        successful_delivery = True
                        logger.info(
                            f"Message delivered via fallback channel: {channel_id}"
                        )
                        break  # Stop after first successful fallback

        # If still no success and channel doesn't exist, try console as last resort
        if not successful_delivery and "console" not in [r["channel"] for r in results]:
            if "console" in self.channels:
                result = await self.channels["console"].send_notification(
                    message,
                    context.get("recipient"),
                    context.get("message_format", MessageFormat.PLAIN_TEXT),
                    context.get("metadata", {}),
                )
                results.append({"channel": "console", "result": result})

        return results

    async def _discover_and_initialize_channels(self):
        """Discover and initialize all available channel handlers."""
        await self._discover_channel_handlers()
        await self._initialize_all_channels()

    async def _discover_channel_handlers(self):
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

    async def _initialize_all_channels(self):
        """Initialize all registered channel handlers."""
        # Only initialize unique handlers (avoid duplicates from dual key storage)
        initialized_handlers = set()

        for channel_id, handler in self.channels.items():
            # Skip if we've already initialized this handler instance
            if id(handler) in initialized_handlers:
                continue

            handler.communication_manager = self
            success = await handler.validate_and_initialize()
            if not success:
                # Get descriptive error message
                error_msg = handler._get_requirements_error_message()
                logger.warning(f"{handler.channel_type.value} disabled: {error_msg}")

            initialized_handlers.add(id(handler))

    def register_channel(self, handler: ChannelHandler) -> None:
        """
        Register a new channel handler.

        Args:
            handler: The channel handler instance to register
        """
        if not isinstance(handler, ChannelHandler):
            raise TypeError("Handler must be an instance of ChannelHandler")

        # Store with both string key (new) and enum key (backward compatibility)
        self.channels[handler.channel_type.value] = handler
        self.channels[handler.channel_type] = handler
        logger.info(f"Registered channel handler for {handler.channel_type.value}")

    def unregister_channel(self, channel_type: ChannelType) -> None:
        """
        Unregister a channel handler.

        Args:
            channel_type: The type of channel to unregister
        """
        # Remove both string and enum keys
        channel_id = channel_type.value
        if channel_id in self.channels:
            del self.channels[channel_id]
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

        This method provides backward compatibility with the old API.
        New code should use route_message() instead.

        Args:
            message: The message content
            channel_type: Primary channel to use (defaults to default_channel)
            recipient: Channel-specific recipient identifier
            message_format: Format of the message content
            delivery_guarantee: Delivery guarantee level
            metadata: Additional channel-specific metadata
            fallback_channels: List of fallback channels to try if primary fails

        Returns:
            Dict with status information about the delivery
        """
        # Convert to new route_message format
        context = {
            "channel_id": (
                channel_type.value if channel_type else self.default_channel.value
            ),
            "recipient": recipient,
            "message_format": message_format,
            "delivery_guarantee": delivery_guarantee,
            "metadata": metadata or {},
            "message_type": "notification",
        }

        results = await self.route_message(message, context)

        # Convert results back to old format for compatibility
        if results and len(results) > 0:
            # Return the first successful result
            for result in results:
                if result["result"]["success"]:
                    return result["result"]
            # If no success, return the first result
            return results[0]["result"]
        else:
            return {"success": False, "error": "No channels available"}

    async def shutdown(self):
        """Shutdown all channel handlers and cleanup resources."""
        if self._shutdown_called:
            logger.debug("Communication manager shutdown already called, skipping...")
            return

        self._shutdown_called = True
        logger.info("Shutting down communication manager...")

        # First, set shutdown flag on all handlers to signal graceful shutdown
        for channel_type, handler in self.channels.items():
            if hasattr(handler, "shutdown_requested"):
                handler.shutdown_requested = True

        # Stop all channel handlers that have stop_session method
        for channel_type, handler in self.channels.items():
            try:
                # Handle both string keys and enum keys
                channel_name = (
                    channel_type.value
                    if hasattr(channel_type, "value")
                    else str(channel_type)
                )

                if hasattr(handler, "stop_session"):
                    logger.info(f"Stopping session for {channel_name} channel...")
                    await handler.stop_session()
                else:
                    logger.debug(f"Channel {channel_name} has no stop_session method")
            except Exception as e:
                channel_name = (
                    channel_type.value
                    if hasattr(channel_type, "value")
                    else str(channel_type)
                )
                logger.error(f"Error stopping {channel_name} channel: {e}")

        # Cancel any remaining background tasks
        await self._terminate_background_tasks()

        logger.info("Communication manager shutdown complete")

    async def _terminate_background_threads(self):
        """Forcefully terminate all tracked background threads."""
        if not self._background_threads:
            logger.debug("No background threads to terminate")
            return

        logger.info(
            f"Terminating {len(self._background_threads)} background threads..."
        )

        for channel_name, thread in self._background_threads.items():
            try:
                if thread.is_alive():
                    logger.info(f"Waiting for {channel_name} thread to terminate...")
                    # Give thread a very short time to terminate gracefully
                    thread.join(timeout=0.5)

                    if thread.is_alive():
                        logger.warning(
                            f"Thread {channel_name} did not terminate gracefully"
                        )
                        # Since Python doesn't have thread.terminate(), we rely on daemon=True
                        # The thread will be forcefully terminated when the main process exits
                    else:
                        logger.info(f"Thread {channel_name} terminated successfully")
                else:
                    logger.debug(f"Thread {channel_name} already terminated")
            except Exception as e:
                logger.error(f"Error terminating thread {channel_name}: {e}")

        # Clear the thread tracking
        self._background_threads.clear()
        logger.info("Background thread termination complete")

        # Force garbage collection to help with cleanup
        import gc

        gc.collect()


def get_communication_manager() -> CommunicationManager:
    """Get the singleton instance of the CommunicationManager."""
    return CommunicationManager.get_instance()
