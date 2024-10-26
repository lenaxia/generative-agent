import os
import threading
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from llm_provider.factory import LLMFactory
from shared_tools.message_bus import MessageBus, MessageType
from pydantic import BaseModel
import logging


class SlackAgentConfig(BaseModel):
    slack_channel: str
    status_channel: str
    monitored_event_types: List[str] = ["message"]
    online_message: str = "SlackAgent is online and ready to receive messages."

class SlackAgent(BaseAgent):
    slack_app_token = os.environ.get("SLACK_APP_TOKEN")
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app = App(token=slack_bot_token)

    def __init__(self, logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        self.logger = logger
        super().__init__(logger, llm_factory, message_bus, agent_id, config=config)
        self.config = SlackAgentConfig(**config)
        self.setup_middleware()
        self.setup_events()
        self.setup_error_handler()
        self.subscribe_to_messages(MessageType.SEND_MESSAGE, self.handle_send_message)
        self.initialize()

    def setup_middleware(self):
        @self.app.middleware
        def log_request(logger, body, next):
            logger.debug(body)
            return next()

    def setup_events(self):
        @self.app.event("app_mention")
        def handle_app_mention(event, say):
            metadata = self.event_metadata(event)
            self.publish_message(MessageType.INCOMING_REQUEST, {"event": event, "metadata": metadata})
            say(f"Event: {event}\n\nMetadata: {metadata}")

        @self.app.event("message")
        def handle_message(event, say, logger):
            pass

    def setup_error_handler(self):
        @self.app.error
        def global_error_handler(error, body, logger):
            logger.exception(error)
            logger.info(body)

    def initialize(self):
        # Start the SocketModeHandler in a separate thread
        socket_mode_thread = threading.Thread(target=self.start_socket_mode_handler)
        socket_mode_thread.start()

        # Post online message to the preset channel
        self.post_message_to_channel(self.config.status_channel, self.config.online_message)

    def start_socket_mode_handler(self):
        if not self.slack_app_token or not self.slack_bot_token:
            self.logger.error("Slack app token or bot token not provided")
            return

        handler = SocketModeHandler(self.app, self.slack_app_token)
        self.logger.info("Starting Socket Mode Handler")
        handler.start()

        # Add additional logging for socket events
        handler.socket_mode_handler.logger.setLevel(logging.DEBUG)
        handler.socket_mode_handler.logger.addHandler(logging.StreamHandler())

    def post_message_to_channel(self, channel, message):
        try:
            self.app.client.chat_postMessage(channel=channel, text=message)
        except Exception as e:
            self.logger.error(f"Failed to post message to Slack channel: {e}")

    @staticmethod
    def event_metadata(event):
        return {
            "event_type": event["type"],
            "channel": event.get("channel", None),
            "user": event.get("user", None),
            "timestamp": event.get("ts", None)
        }

    def handle_send_message(self, message):
        event = message["event"]
        metadata = message["metadata"]
        channel = metadata.get("channel", None)
        if channel:
            self.app.client.chat_postMessage(channel=channel, text=event["text"])

    def setup(self):
        pass

    def teardown(self):
        pass