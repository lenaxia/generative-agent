# Unified Communication Architecture for Multi-Channel AI Assistant

## Overview

This document defines the architecture for a unified communication system supporting multiple channels (Slack, WhatsApp, Sonos, Voice Assistants, Home Assistant) with bidirectional communication, session management, and intelligent message routing.

## Requirements

### **Supported Channels:**

- **Slack**: Bidirectional text/rich media with WebSocket
- **WhatsApp/Messengers**: Bidirectional text/media via API
- **OpenAI Compatible Endpoint**: API-based bidirectional communication
- **Sonos**: Output-only audio with device discovery
- **Raspberry Pi Voice Assistants**: Bidirectional voice with wake word detection
- **Home Assistant**: Bidirectional device control with WebSocket

### **Use Cases:**

- Smart home control (lights, temperature, devices)
- Timers/alarms with multi-channel notifications
- Weather/time information requests
- Music control/playback through Sonos
- Reminders/lists management
- Information queries (define, translate, explain)

## Current State Analysis

### **Existing Infrastructure:**

- ✅ **CommunicationManager**: Dynamic channel handler loading system
- ✅ **Base ChannelHandler**: Abstract class with `get_capabilities()` method
- ✅ **Existing Handlers**: `SlackChannelHandler`, `EmailChannelHandler`, `ConsoleChannelHandler`
- ✅ **Message Types**: `DeliveryGuarantee`, `MessageFormat`, `ChannelType` enums
- ❌ **MessageBus Isolation**: CommunicationManager creates own MessageBus instead of using supervisor's
- ❌ **No Supervisor Integration**: CommunicationManager never initialized by supervisor
- ❌ **No Session Management**: No support for persistent connections (WebSocket, device sessions)
- ❌ **No Bidirectional Support**: Only supports outbound notifications

## Architecture Design

### **Core Principles:**

1. **Single MessageBus**: All communication flows through supervisor's MessageBus
2. **Dynamic Loading**: Auto-discover channel handlers from `common/channel_handlers/`
3. **Self-Describing Channels**: Each handler defines its own capabilities
4. **Session Management**: Support both stateless and stateful channels
5. **Bidirectional Communication**: Support both inbound and outbound messages
6. **Intelligent Routing**: Default to origin channel with fallback support
7. **Background Threads**: Stateful channels run in dedicated threads with event loops

### **System Architecture:**

```
Supervisor Process
├── Main Thread
│   ├── Supervisor (orchestration)
│   ├── WorkflowEngine (task processing)
│   ├── UniversalAgent (AI processing)
│   ├── MessageBus (event coordination)
│   └── CommunicationManager (routing/coordination)
│       ├── ConsoleHandler (stateless)
│       ├── EmailHandler (stateless)
│       └── SMSHandler (stateless)
├── Slack Handler Thread
│   └── WebSocket + Rich Interactions
├── Voice Handler Thread
│   └── Wake Word + Audio Processing
├── Sonos Handler Thread
│   └── Device Discovery + Audio Streaming
└── Home Assistant Thread
    └── WebSocket + Device State Management
```

## Core Components

### **1. Enhanced CommunicationManager**

**Location**: `common/communication_manager.py`

**Key Changes:**

```python
class CommunicationManager:
    def __init__(self, message_bus: MessageBus):  # CHANGE: Accept supervisor's MessageBus
        self.message_bus = message_bus  # CHANGE: Don't create own MessageBus
        self.channels: Dict[str, ChannelHandler] = {}
        self.channel_queues: Dict[str, asyncio.Queue] = {}  # NEW: Thread communication

        # Subscribe to communication events
        self._setup_message_subscriptions()

        # Auto-discover and initialize channel handlers
        self._discover_channel_handlers()

        # Start queue processor for background thread communication
        asyncio.create_task(self._process_channel_queues())

    def _setup_message_subscriptions(self):
        """Subscribe to all communication-related MessageBus events."""
        subscriptions = [
            (MessageType.TIMER_EXPIRED, self._handle_timer_expired),
            (MessageType.SEND_MESSAGE, self._handle_send_message),
            (MessageType.AGENT_QUESTION, self._handle_agent_question),
        ]

        for message_type, handler in subscriptions:
            self.message_bus.subscribe(self, message_type, handler)

    async def _process_channel_queues(self):
        """Process incoming messages from channel background threads."""
        while True:
            for channel_id, queue in self.channel_queues.items():
                try:
                    while not queue.empty():
                        message = await queue.get()
                        await self._handle_channel_message(channel_id, message)
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0.1)  # Prevent busy loop

    async def _handle_channel_message(self, channel_id: str, message: dict):
        """Handle incoming message from channel background thread."""
        if message["type"] == "incoming_message":
            # Route to supervisor for processing
            self.message_bus.publish(
                self, MessageType.INCOMING_REQUEST, {
                    "request": message["text"],
                    "user_id": message["user_id"],
                    "channel_id": f"{channel_id}:{message['channel_id']}",
                    "source": channel_id
                }
            )
        elif message["type"] == "user_response":
            # Handle response to agent question
            self.message_bus.publish(
                self, MessageType.USER_RESPONSE, message["data"]
            )

    async def route_message(self, message: str, context: dict) -> List[dict]:
        """Route message to appropriate channels with fallback support."""
        origin_channel = context.get('channel_id', 'console')
        delivery_guarantee = context.get('delivery_guarantee', DeliveryGuarantee.BEST_EFFORT)
        message_type = context.get('message_type', 'notification')

        # Determine target channels
        target_channels = self._determine_target_channels(origin_channel, message_type, context)

        # Send with appropriate delivery guarantee
        return await self._send_with_delivery_guarantee(
            message, target_channels, context, delivery_guarantee
        )

    def _determine_target_channels(self, origin_channel: str, message_type: str, context: dict) -> List[str]:
        """Determine which channels should receive the message."""
        # Default: return to origin channel
        channels = [origin_channel] if origin_channel else ['console']

        # Special routing rules
        if message_type == 'timer_expired':
            # Timer notifications: origin + audio if user preferences allow
            if self._should_add_audio_notification(context):
                channels.append('sonos')
        elif message_type == 'music_control':
            # Music control: always route to Sonos + origin for confirmation
            channels = ['sonos'] + ([origin_channel] if origin_channel != 'sonos' else [])
        elif message_type == 'smart_home_control':
            # Smart home: origin + Home Assistant for device status
            channels = [origin_channel, 'home_assistant']

        return [ch for ch in channels if ch and ch in self.channels]
```

### **2. Enhanced ChannelHandler Base Class**

**Location**: `common/communication_manager.py`

**Key Enhancements:**

```python
class ChannelHandler:
    """Base class for all communication channel handlers."""

    channel_type: ChannelType  # Must be defined in subclasses

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled = False
        self.session_active = False
        self.background_thread = None
        self.message_queue = None
        self.communication_manager = None  # Set by CommunicationManager

    async def validate_and_initialize(self) -> bool:
        """Validate requirements and initialize channel."""
        if not self._validate_requirements():
            logger.warning(f"{self.channel_type.value} disabled: requirements not met")
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

    def _validate_requirements(self) -> bool:
        """Validate channel requirements (env vars, hardware, etc.). Override in subclasses."""
        return True

    def requires_background_thread(self) -> bool:
        """Return True if channel needs background thread for persistent connections."""
        capabilities = self.get_capabilities()
        return capabilities.get('requires_session', False) and capabilities.get('bidirectional', False)

    async def start_session(self):
        """Start channel session. Override for stateless channels."""
        self.session_active = True

    async def _start_background_thread(self):
        """Start background thread for stateful channels."""
        self.message_queue = asyncio.Queue()
        self.background_thread = threading.Thread(
            target=self._run_background_session,
            daemon=True,
            name=f"{self.channel_type.value}_thread"
        )
        self.background_thread.start()

        # Register queue with CommunicationManager
        if self.communication_manager:
            self.communication_manager.channel_queues[self.channel_type.value] = self.message_queue

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

    def get_capabilities(self) -> dict:
        """Return channel capabilities. Must override in subclasses."""
        return {
            "supports_rich_text": False,
            "supports_buttons": False,
            "supports_audio": False,
            "supports_images": False,
            "bidirectional": False,
            "requires_session": False,
            "max_message_length": 1000
        }

    async def send_notification(self, message: str, recipient: str = None,
                              message_format: MessageFormat = MessageFormat.PLAIN_TEXT,
                              metadata: dict = None) -> dict:
        """Send outbound notification. Override in subclasses."""
        if not self.enabled:
            return {"success": False, "error": "Channel disabled"}

        return await self._send(message, recipient, message_format, metadata or {})

    async def _send(self, message: str, recipient: str, message_format: MessageFormat, metadata: dict) -> dict:
        """Send implementation. Must override in subclasses."""
        raise NotImplementedError("Subclasses must implement _send method")

    async def ask_question(self, question: str, options: List[str] = None, timeout: int = 300) -> str:
        """Ask user a question and wait for response. Override in bidirectional channels."""
        if not self.get_capabilities().get('bidirectional', False):
            raise NotImplementedError(f"{self.channel_type.value} doesn't support bidirectional communication")
        return await self._ask_question_impl(question, options, timeout)

    async def _ask_question_impl(self, question: str, options: List[str], timeout: int) -> str:
        """Question implementation. Override in bidirectional channels."""
        raise NotImplementedError("Bidirectional channels must implement _ask_question_impl")
```

### **3. Example Stateful Channel Handler**

**Location**: `common/channel_handlers/slack_handler.py`

```python
class SlackChannelHandler(ChannelHandler):
    """Slack channel handler with WebSocket support and rich interactions."""

    channel_type = ChannelType.SLACK

    def _validate_requirements(self) -> bool:
        """Validate Slack configuration."""
        bot_token = os.environ.get('SLACK_BOT_TOKEN')
        app_token = os.environ.get('SLACK_APP_TOKEN')

        if not bot_token:
            logger.error("SLACK_BOT_TOKEN environment variable required")
            return False
        if not app_token:
            logger.error("SLACK_APP_TOKEN environment variable required")
            return False

        return True

    def get_capabilities(self) -> dict:
        """Slack channel capabilities."""
        return {
            "supports_rich_text": True,
            "supports_buttons": True,
            "supports_images": True,
            "bidirectional": True,
            "requires_session": True,
            "max_message_length": 4000
        }

    async def _background_session_loop(self):
        """Run Slack WebSocket in background thread."""
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        # Create Slack app
        app = App(token=os.environ['SLACK_BOT_TOKEN'])
        handler = SocketModeHandler(app, os.environ['SLACK_APP_TOKEN'])

        # Handle incoming messages
        @app.event("message")
        def handle_message(event):
            if not event.get('bot_id'):  # Ignore bot messages
                # Send to main thread via queue
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put({
                        "type": "incoming_message",
                        "user_id": event["user"],
                        "channel_id": event["channel"],
                        "text": event.get("text", ""),
                        "timestamp": event.get("ts")
                    }),
                    self._get_main_event_loop()
                )

        # Handle button interactions
        @app.action(".*")  # Match all button actions
        def handle_button_click(ack, body):
            ack()  # Acknowledge button click
            # Process button response
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({
                    "type": "user_response",
                    "data": {
                        "action_id": body["actions"][0]["action_id"],
                        "value": body["actions"][0]["value"],
                        "user_id": body["user"]["id"],
                        "channel_id": body["channel"]["id"]
                    }
                }),
                self._get_main_event_loop()
            )

        # Start WebSocket (blocks in this thread)
        await handler.start_async()

    def _get_main_event_loop(self):
        """Get reference to main thread's event loop."""
        # Implementation depends on how supervisor manages event loop
        return asyncio.get_event_loop()

    async def _send(self, message: str, recipient: str, message_format: MessageFormat, metadata: dict) -> dict:
        """Send message to Slack."""
        # Implementation for sending Slack messages
        # Can use existing SlackChannelHandler._send_via_api logic
        pass

    async def _ask_question_impl(self, question: str, options: List[str], timeout: int) -> str:
        """Ask question with Slack buttons and wait for response."""
        # Create Slack blocks with buttons
        blocks = self._create_question_blocks(question, options)

        # Send question
        await self._send(question, None, MessageFormat.RICH_TEXT, {"blocks": blocks})

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(self._wait_for_user_response(), timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return "timeout"
```

### **4. Example Stateless Channel Handler**

**Location**: `common/channel_handlers/sonos_handler.py`

```python
class SonosChannelHandler(ChannelHandler):
    """Sonos audio output handler with device discovery."""

    channel_type = ChannelType.SONOS

    def _validate_requirements(self) -> bool:
        """Check if Sonos devices are available on network."""
        try:
            import soco
            devices = list(soco.discover())
            return len(devices) > 0
        except ImportError:
            logger.error("soco library required for Sonos support: pip install soco")
            return False
        except Exception as e:
            logger.warning(f"Sonos device discovery failed: {e}")
            return False

    async def start_session(self):
        """Discover and cache Sonos devices."""
        import soco
        self.devices = {device.player_name: device for device in soco.discover()}
        self.session_active = len(self.devices) > 0
        logger.info(f"Discovered {len(self.devices)} Sonos devices: {list(self.devices.keys())}")

    def get_capabilities(self) -> dict:
        """Sonos channel capabilities."""
        return {
            "supports_audio": True,
            "bidirectional": False,  # Output only
            "requires_session": True,  # Need device discovery
            "max_message_length": 0  # Audio has no text length limit
        }

    async def _send(self, message: str, recipient: str, message_format: MessageFormat, metadata: dict) -> dict:
        """Send audio message to Sonos device."""
        device_name = recipient or "all"
        volume = metadata.get("volume", 0.7)

        try:
            # Convert text to speech
            audio_file = await self._text_to_speech(message)

            # Play on specified device(s)
            if device_name == "all":
                devices = list(self.devices.values())
            else:
                devices = [self.devices.get(device_name)]

            for device in devices:
                if device:
                    # Save current state
                    current_volume = device.volume

                    # Play announcement
                    device.volume = int(volume * 100)
                    device.play_uri(audio_file)

                    # Restore volume after playback
                    await asyncio.sleep(len(message) * 0.1)  # Estimate duration
                    device.volume = current_volume

            return {"success": True, "devices": len(devices)}

        except Exception as e:
            logger.error(f"Sonos playback failed: {e}")
            return {"success": False, "error": str(e)}
```

## Integration with Supervisor

### **Supervisor Initialization**

**Location**: `supervisor/supervisor.py`

```python
def initialize_components(self):
    """Initialize all supervisor components."""
    # ... existing initialization ...
    self._initialize_communication_manager()

def _initialize_communication_manager(self):
    """Initialize communication manager with supervisor's MessageBus."""
    from common.communication_manager import CommunicationManager

    self.communication_manager = CommunicationManager(self.message_bus)
    logger.info("Communication manager initialized with channel handlers")
```

## Message Flow Examples

### **Timer Expiry Notification:**

1. **Timer Monitor** detects expired timer
2. **Timer Monitor** publishes `MessageType.TIMER_EXPIRED` to MessageBus
3. **CommunicationManager** receives event, determines target channels
4. **CommunicationManager** routes to SlackHandler and SonosHandler
5. **SlackHandler** sends rich text notification with buttons
6. **SonosHandler** plays audio announcement

### **Incoming Slack Message:**

1. **SlackHandler** (background thread) receives WebSocket message
2. **SlackHandler** puts message in queue for main thread
3. **CommunicationManager** processes queue, publishes `MessageType.INCOMING_REQUEST`
4. **WorkflowEngine** receives request, processes with UniversalAgent
5. **Response** routed back through CommunicationManager to SlackHandler

### **Agent Question Flow:**

1. **UniversalAgent** needs user input, publishes `MessageType.AGENT_QUESTION`
2. **CommunicationManager** routes to appropriate channel (origin channel)
3. **SlackHandler** sends question with buttons, waits for response
4. **User** clicks button, SlackHandler receives via WebSocket
5. **SlackHandler** publishes `MessageType.USER_RESPONSE` via queue
6. **UniversalAgent** receives response, continues processing

## Configuration

### **Environment Variables (Channel Credentials):**

```bash
# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# WhatsApp (example)
WHATSAPP_API_KEY=your-whatsapp-api-key
WHATSAPP_PHONE_NUMBER=+1234567890

# Home Assistant
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=your-long-lived-access-token

# Sonos (auto-discovery, no config needed)
# Voice Assistant (hardware-based, no config needed)
```

### **Channel Configuration in config.yaml:**

```yaml
communication:
  default_channel: "console"
  delivery_guarantee: "best_effort" # best_effort, at_least_once, exactly_once

  # Channel-specific settings
  channels:
    slack:
      enabled: true
      default_channel: "#general"
    sonos:
      enabled: true
      default_volume: 0.7
      devices: ["Kitchen", "Living Room"]
    voice:
      enabled: true
      wake_word: "hey assistant"
      language: "en-US"
```

## Implementation Phases

### **Phase 1: Fix Core Integration (Week 1)**

1. **Modify CommunicationManager** to accept supervisor's MessageBus
2. **Add CommunicationManager initialization** to supervisor
3. **Add background thread support** to base ChannelHandler
4. **Test timer notifications** through CommunicationManager

### **Phase 2: Enhance Existing Handlers (Week 2)**

1. **Upgrade SlackChannelHandler** with background thread WebSocket
2. **Add requirement validation** to all existing handlers
3. **Add bidirectional support** to SlackChannelHandler
4. **Test bidirectional communication** flow

### **Phase 3: Add New Channel Handlers (Weeks 3-4)**

1. **SonosChannelHandler**: Audio output with device discovery
2. **VoiceChannelHandler**: RPi voice with wake word detection
3. **HomeAssistantChannelHandler**: Device integration with WebSocket
4. **WhatsAppChannelHandler**: Messaging with webhook/API support

### **Phase 4: Replace Standalone Services (Week 5)**

1. **Migrate functionality** from `slack.py` to enhanced `SlackChannelHandler`
2. **Remove standalone `slack.py`** once CommunicationManager handles all cases
3. **Add comprehensive testing** for all channel types
4. **Performance optimization** and monitoring

## Testing Strategy

### **Unit Tests:**

- Test each channel handler independently
- Mock external APIs and hardware
- Test requirement validation logic
- Test capability reporting

### **Integration Tests:**

- Test CommunicationManager routing logic
- Test background thread communication
- Test bidirectional message flow
- Test fallback mechanisms

### **End-to-End Tests:**

- Test complete timer notification flow
- Test agent question/response flow
- Test multi-channel delivery
- Test error recovery scenarios

This architecture leverages the existing excellent dynamic loading system while adding proper session management, bidirectional communication, and background thread support for complex stateful channels.
