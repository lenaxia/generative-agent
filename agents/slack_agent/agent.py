import json
import os
import threading
import time
from typing import Any, Dict, List
from agents.base_agent import AgentInput, BaseAgent
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import logging

from common.request_model import RequestMetadata
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus, MessageType


class SlackAgentConfig(BaseModel):
    slack_channel: str
    status_channel: str
    monitored_event_types: List[str] = ["message"]
    online_message: str = "SlackAgent is online and ready to receive messages."
    llm_class: str = "default"
    history_limit: int = 5
    
class SlackMessageOutput(BaseModel):
    text: str = Field(description="The text of the message")
    

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
        self.agent_description = "An agent which can send and receive messages to and from Slack. When receiving a request from Slack, send a response when it makes sense"

    def setup_middleware(self):
        @self.app.middleware
        def log_request(logger, body, next):
            logger.debug(body)
            return next()

    def setup_events(self):
        @self.app.event("app_mention")
        def handle_app_mention(event, say):
            metadata = self.event_metadata(event)
            client = self.app.client
            
            history_limit = self.config.history_limit
            result = client.conversations_history(channel=metadata.get("channel"), limit=history_limit)
            history = result.data["messages"]
            history.pop(0) # remove the most recent message from history becuase thats what we're processing right now
            
            messages = []
            
            for message in history:
                user = message["user"]
                text = message["text"]
                age = time.time() - float(message.get("ts", 0))
                age_minutes = round(age / 60)
                messages.insert(0, f"({age_minutes}min ago) {user}: {text}")
            
            compiled_history = "\n".join(messages)
            
            prompt = f"""
Here is the most recent {history_limit} messages in the channel for context when responding to the below message:
{compiled_history}

{metadata.get("user", "")} just mentioned your name and said the following:

{metadata.get("text", "")}
                    """
                                
            request = RequestMetadata(
                prompt=prompt,
                source_id=self.agent_id,
                target_id="supervisor",
                response_requested=True,
                callback_details={"channel": metadata["channel"], "timestamp": metadata["timestamp"], "user": event["user"]}
            )
            
            self.publish_message(MessageType.INCOMING_REQUEST, request)
            #say(f"Request received: {request}")

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
        text = event.get("text", "")
        if "<@" in text:
            text = text.split(">", 1)[1].strip()
        return {
            "event_type": event["type"],
            "channel": event.get("channel", None),
            "user": event.get("user", None),
            "timestamp": event.get("ts", None),
            "text": text
        }

    def handle_send_message(self, message):
        event = message["event"]
        metadata = message["metadata"]
        channel = metadata.get("channel", None)
        if channel:
            self.app.client.chat_postMessage(channel=channel, text=event["text"])

    def _run(self, input: AgentInput) -> Any:
        parser = PydanticOutputParser(pydantic_object=SlackMessageOutput)  # Replace YourOutputModel with the appropriate output model
        
        prompt_template = PromptTemplate(
            input_variables=["prompt", "history"],
            template="Given the following prompt and history, generate a response: {prompt}\n\n"
                     "History:\n{history}\n\n"
                     "Your output should follow the provided JSON schema:\n\n{format_instructions}\n\n"
                     "Your response should be conversational and be as if you are responding to the user."
                     "Do not include anything else besides the json output\n\n"
                     "Response:",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        llm_provider = self._select_llm_provider()
        
        chain = prompt_template | llm_provider | parser
        response = chain.invoke({"prompt": input.prompt, "history": input.history})
        
        self.post_message_to_channel(self.config.slack_channel, str(response.text))
        
        return response

    def setup(self):
        pass

    def teardown(self):
        pass