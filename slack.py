#!/usr/bin/env python3
"""Enhanced Slack Bot with Universal Agent Integration.

Connects your Slack bot to the StrandsAgent Universal Agent system for intelligent responses.
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from supervisor.supervisor import Supervisor
from supervisor.workflow_duration_logger import WorkflowSource

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentSlackBot:
    """Slack bot that integrates with the Universal Agent system for intelligent responses."""

    def __init__(self, app_token, bot_token=None, config_path="config.yaml"):
        """Initialize the intelligent Slack bot.

        Args:
            app_token: Slack app token for socket mode
            bot_token: Slack bot token (optional, can be set via environment)
            config_path: Path to the Universal Agent config file
        """
        self.app_token = app_token
        self.bot_token = bot_token
        self.config_path = config_path

        # Initialize Slack app
        self.app = App(token=bot_token) if bot_token else App()

        # Initialize Universal Agent system
        self.supervisor = None
        self.system_ready = False

        # Setup Slack handlers
        self.setup_middleware()
        self.setup_commands()
        self.setup_events()
        self.setup_error_handler()

        # Initialize AI system in background
        self._initialize_ai_system()

        # Cache for user and channel info
        self.user_cache = {}
        self.channel_cache = {}

    def _initialize_ai_system(self):
        """Initialize the Universal Agent system."""
        try:
            logger.info("🤖 Initializing Universal Agent system...")
            self.supervisor = Supervisor(self.config_path)
            self.supervisor.start()

            # Subscribe to timer expired events
            self._setup_timer_notifications()

            self.system_ready = True
            logger.info("✅ Universal Agent system ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Universal Agent system: {e}")
            self.system_ready = False

    def _setup_timer_notifications(self):
        """Set up timer expiry notifications."""
        try:
            from common.message_bus import MessageType

            # Timer notifications now handled via intent-based system
            logger.info("✅ Timer notifications handled via intent-based system")
        except Exception as e:
            logger.error(f"❌ Failed to setup timer notifications: {e}")

    async def _get_user_info(self, user_id: str, client) -> dict:
        """Get user information from Slack API with caching."""
        if user_id in self.user_cache:
            return self.user_cache[user_id]

        try:
            response = client.users_info(user=user_id)
            if response["ok"]:
                user_info = {
                    "id": user_id,
                    "name": response["user"]["name"],
                    "real_name": response["user"].get(
                        "real_name", response["user"]["name"]
                    ),
                    "display_name": response["user"]["profile"].get(
                        "display_name", response["user"]["name"]
                    ),
                }
                self.user_cache[user_id] = user_info
                return user_info
        except Exception as e:
            logger.warning(f"Could not get user info for {user_id}: {e}")

        return {
            "id": user_id,
            "name": user_id,
            "real_name": user_id,
            "display_name": user_id,
        }

    async def _get_channel_info(self, channel_id: str, client) -> dict:
        """Get channel information from Slack API with caching."""
        if channel_id in self.channel_cache:
            return self.channel_cache[channel_id]

        try:
            response = client.conversations_info(channel=channel_id)
            if response["ok"]:
                channel_info = {
                    "id": channel_id,
                    "name": response["channel"]["name"],
                    "is_private": response["channel"]["is_private"],
                }
                self.channel_cache[channel_id] = channel_info
                return channel_info
        except Exception as e:
            logger.warning(f"Could not get channel info for {channel_id}: {e}")

        return {"id": channel_id, "name": channel_id, "is_private": False}

    async def _resolve_slack_references(self, text: str, client) -> str:
        """Resolve Slack user and channel references to human-readable names.

        Args:
            text: Message text with Slack references like <@U123> or <#C123|channelname>
            client: Slack client for API calls

        Returns:
            Text with resolved references
        """
        import re

        # Resolve user mentions: <@U1234567890> -> @username
        user_pattern = r"<@([A-Z0-9]+)>"
        for match in re.finditer(user_pattern, text):
            user_id = match.group(1)
            user_info = await self._get_user_info(user_id, client)
            user_name = user_info.get("name", user_id)
            text = text.replace(match.group(0), f"@{user_name}")

        # Resolve channel mentions: <#C1234567890|channelname> -> #channelname
        channel_pattern = r"<#([A-Z0-9]+)\|([^>]+)>"
        for match in re.finditer(channel_pattern, text):
            channel_name = match.group(2)
            text = text.replace(match.group(0), f"#{channel_name}")

        # Resolve channel mentions without name: <#C1234567890> -> #channelname
        channel_pattern_no_name = r"<#([A-Z0-9]+)>"
        for match in re.finditer(channel_pattern_no_name, text):
            channel_id = match.group(1)
            channel_info = await self._get_channel_info(channel_id, client)
            channel_name = channel_info.get("name", channel_id)
            text = text.replace(match.group(0), f"#{channel_name}")

        return text

    async def _process_with_ai(
        self, message_text, user_id=None, channel=None, client=None
    ):
        """Process message with the Universal Agent system.

        Args:
            message_text: The message to process
            user_id: Slack user ID (optional)
            channel: Slack channel (optional)
            client: Slack client for API calls (optional)

        Returns:
            AI response string
        """
        if not self.system_ready:
            return "🤖 AI system is starting up, please try again in a moment..."

        # Handle slash commands BEFORE AI processing
        if message_text.startswith("/"):
            return self._handle_slash_command(message_text, user_id)

        try:
            context_message = await self._build_context_message(
                message_text, user_id, channel, client
            )
            workflow_id = self._start_workflow(context_message, user_id, channel)
            return await self._wait_for_workflow_completion(workflow_id)

        except Exception as e:
            logger.error(f"❌ AI processing error: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"

    async def _build_context_message(self, message_text, user_id, channel, client):
        """Build context message with user and channel information."""
        # Resolve Slack references in the message text
        resolved_message = message_text
        if client:
            resolved_message = await self._resolve_slack_references(
                message_text, client
            )

        # Build context message with human-readable names
        context_message = resolved_message

        if user_id and client:
            user_info = await self._get_user_info(user_id, client)
            user_name = self._extract_user_name(user_info)
            context_message = (
                f"User {user_name} (@{user_info.get('name')}) asks: {resolved_message}"
            )
        elif user_id:
            context_message = f"User {user_id} asks: {resolved_message}"

        # Add channel context if available
        if channel and client:
            channel_info = await self._get_channel_info(channel, client)
            channel_name = channel_info.get("name")
            if channel_name:
                context_message += f" [in #{channel_name}]"

        logger.info(f"🧠 Processing with AI: {context_message}")
        return context_message

    def _extract_user_name(self, user_info):
        """Extract the best available user name from user info."""
        return (
            user_info.get("display_name")
            or user_info.get("real_name")
            or user_info.get("name")
        )

    def _start_workflow(self, context_message, user_id, channel):
        """Start workflow and update source information."""
        workflow_id = self.supervisor.workflow_engine.start_workflow(context_message)

        # Update workflow source to Slack (duration tracking is handled in workflow engine)
        self.supervisor.workflow_engine.update_workflow_source(
            workflow_id, WorkflowSource.SLACK, user_id=user_id, channel_id=channel
        )
        return workflow_id

    async def _wait_for_workflow_completion(self, workflow_id):
        """Wait for workflow completion with polling."""
        max_wait = 30  # 30 seconds timeout
        wait_time = 0
        check_interval = 0.5  # Check every 500ms for faster response

        while wait_time < max_wait:
            status = self.supervisor.workflow_engine.get_request_status(workflow_id)

            result = await self._check_workflow_status(workflow_id, status)
            if result:
                return result

            await asyncio.sleep(check_interval)
            wait_time += check_interval

        return "⏰ AI is taking longer than expected. Please try a simpler request."

    async def _check_workflow_status(self, workflow_id, status):
        """Check workflow status and return result if ready."""
        if status is None:
            return await self._get_workflow_result(workflow_id)

        if status.get("is_completed", False):
            return await self._get_workflow_result(workflow_id)

        if status.get("error"):
            return f"❌ Error: {status.get('error')}"

        # Check if any tasks are completed even if workflow isn't marked complete
        task_statuses = status.get("task_statuses", {})
        if task_statuses:
            completed_tasks = [
                task_id
                for task_id, task_status in task_statuses.items()
                if task_status == "COMPLETED"
            ]
            if completed_tasks:
                # Try to get partial results
                partial_result = await self._get_workflow_result(workflow_id)
                if partial_result and "🤖" in partial_result:
                    return partial_result

        return None  # Continue waiting

    async def _get_workflow_result(self, workflow_id: str) -> str:
        """Get the actual result from a completed workflow.

        Args:
            workflow_id: The workflow ID to get results for

        Returns:
            The actual AI response or a fallback message
        """
        try:
            task_context = self.supervisor.workflow_engine.get_request_context(
                workflow_id
            )

            if not task_context:
                return "✅ Task completed successfully!"

            # Try different result extraction strategies
            result = (
                self._get_result_from_completed_nodes(task_context)
                or self._get_result_from_conversation_history(task_context)
                or self._get_result_from_progressive_summary(task_context)
            )

            return result or "✅ Task completed successfully!"

        except Exception as e:
            logger.error(f"❌ Error getting workflow result: {e}")
            return "✅ Task completed successfully!"

    def _get_result_from_completed_nodes(self, task_context):
        """Extract result from completed task nodes."""
        completed_results = []
        for node in task_context.task_graph.nodes.values():
            if node.status.value == "COMPLETED" and node.result:
                completed_results.append(node.result)

        if completed_results:
            final_result = completed_results[-1]
            logger.info(f"🎯 Retrieved AI result: {final_result[:100]}...")
            return f"🤖 {final_result}"

        return None

    def _get_result_from_conversation_history(self, task_context):
        """Extract result from conversation history."""
        conversation_history = task_context.get_conversation_history()
        if not conversation_history:
            return None

        # Look for the last assistant message
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg.get("content")
                if content and content.strip():
                    logger.info(f"🎯 Retrieved from conversation: {content[:100]}...")
                    return f"🤖 {content}"

        return None

    def _get_result_from_progressive_summary(self, task_context):
        """Extract result from progressive summary."""
        if not (
            hasattr(task_context.task_graph, "progressive_summary")
            and task_context.task_graph.progressive_summary
        ):
            return None

        summary = task_context.task_graph.progressive_summary[-1]
        if summary and summary.strip():
            logger.info(f"🎯 Retrieved from summary: {summary[:100]}...")
            return f"🤖 {summary}"

        return None

    def setup_middleware(self):
        """Setup Slack middleware for logging."""

        @self.app.middleware
        def log_request(logger, body, next):
            logger.debug(f"📨 Received Slack event: {body.get('type', 'unknown')}")
            return next()

    def _handle_slash_command(
        self, message_text: str, user_id: Optional[str] = None
    ) -> str:
        """Handle slash commands in messages (similar to CLI implementation).

        Args:
            message_text: The message text starting with /
            user_id: Slack user ID for personalized responses

        Returns:
            Response string for the slash command
        """
        command = message_text[1:].lower().split()[0]
        user_mention = f"<@{user_id}> " if user_id else ""

        if command == "status":
            if not self.system_ready:
                return f"{user_mention}🔄 AI system is starting up..."

            try:
                status = self.supervisor.status()
                if status:
                    status_msg = "🤖 *AI System Status*\n"
                    status_msg += f"• System: {status.get('status', 'unknown')}\n"
                    status_msg += (
                        f"• Running: {'✅' if status.get('running') else '❌'}\n"
                    )

                    if "workflow_engine" in status:
                        we_status = status["workflow_engine"]
                        status_msg += f"• Active Workflows: {we_status.get('active_workflows', 0)}\n"
                        status_msg += (
                            f"• Queue Size: {we_status.get('queued_tasks', 0)}\n"
                        )

                    if "universal_agent" in status:
                        ua_status = status["universal_agent"]
                        status_msg += (
                            f"• Framework: {ua_status.get('framework', 'unknown')}\n"
                        )
                        status_msg += f"• Agent Enabled: {'✅' if ua_status.get('universal_agent_enabled') else '❌'}\n"

                    # Add health status from our fixed health check
                    if "heartbeat" in status:
                        hb_status = status["heartbeat"]
                        health_status = hb_status.get("overall_status", "unknown")
                        health_emoji = (
                            "✅"
                            if health_status == "healthy"
                            else "⚠️" if health_status == "degraded" else "❓"
                        )
                        status_msg += (
                            f"• Health Status: {health_emoji} {health_status}\n"
                        )

                    return status_msg
                else:
                    return f"{user_mention}❌ Could not retrieve system status"

            except Exception as e:
                logger.error(f"❌ Status error: {e}")
                return f"{user_mention}❌ Error getting status: {str(e)}"

        elif command == "help":
            help_msg = f"{user_mention}🤖 *Available Commands:*\n"
            help_msg += "• `/status` - Show system status and health\n"
            help_msg += "• `/help` - Show this help message\n"
            help_msg += "• Any other text - Process as AI workflow\n"
            return help_msg

        else:
            return f"{user_mention}❌ Unknown command: `{message_text}`\n💡 Use `/help` for available commands"

    def setup_commands(self):
        """Setup Slack slash commands."""

        @self.app.command("/ai")
        def ai_command(ack, body, say, respond):
            """Handle /ai slash command for direct AI interaction."""
            ack()
            self._handle_ai_command(body, say, respond)

        @self.app.command("/status")
        def status_command(ack, body, say):
            """Handle /status command to show AI system status."""
            ack()
            self._handle_status_command(body, say)

    def _handle_ai_command(self, body, say, respond):
        """Handle /ai slash command logic."""
        user_id = body["user_id"]
        text = body.get("text", "").strip()

        if not text:
            say(f"Hi <@{user_id}>! Use `/ai your question` to chat with me. 🤖")
            return

        logger.info(f"📝 /ai command from {user_id}: {text}")
        respond(f"<@{user_id}> 🤔 Processing your request...")

        self._process_ai_command_async(text, user_id, respond)

    def _handle_status_command(self, body, say):
        """Handle /status command logic."""
        user_id = body["user_id"]

        if not self.system_ready:
            say(f"<@{user_id}> 🔄 AI system is starting up...")
            return

        try:
            status = self.supervisor.status()
            if status:
                status_msg = self._build_status_message_for_command(user_id, status)
                say(status_msg)
            else:
                say(f"<@{user_id}> ❌ Could not retrieve system status")
        except Exception as e:
            logger.error(f"❌ Status error: {e}")
            say(f"<@{user_id}> ❌ Error getting status: {str(e)}")

    def _process_ai_command_async(self, text, user_id, respond):
        """Process AI command in background thread."""

        def process_ai():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self._process_with_ai(text, user_id))
                respond(f"<@{user_id}> {response}")
            except Exception as e:
                logger.error(f"❌ Error in AI processing: {e}")
                respond(
                    f"<@{user_id}> Sorry, I encountered an error processing your request."
                )

        # LLM-SAFE: Use asyncio instead of threading
        asyncio.create_task(asyncio.to_thread(process_ai))

    def _build_status_message_for_command(self, user_id, status):
        """Build status message for slash command."""
        status_msg = f"<@{user_id}> 🤖 *AI System Status*\n"
        status_msg += f"• System: {status.get('status', 'unknown')}\n"
        status_msg += f"• Running: {'✅' if status.get('running') else '❌'}\n"

        if "workflow_engine" in status:
            we_status = status["workflow_engine"]
            status_msg += (
                f"• Active Workflows: {we_status.get('active_workflows', 0)}\n"
            )

        if "universal_agent" in status:
            ua_status = status["universal_agent"]
            roles = ua_status.get("available_roles", [])
            status_msg += f"• Available Roles: {', '.join(roles)}\n"

        return status_msg

    def setup_events(self):
        """Setup Slack event handlers."""

        @self.app.event("app_mention")
        def handle_mention(body, say, logger, client):
            """Handle @mentions of the bot."""
            self._handle_mention_event(body, say, logger, client)

        @self.app.event("message")
        def handle_direct_message(body, say, logger, client):
            """Handle direct messages to the bot."""
            self._handle_direct_message_event(body, say, logger, client)

    def _handle_mention_event(self, body, say, logger, client):
        """Handle @mentions of the bot."""
        event = body["event"]
        user_id = event["user"]
        text = event["text"]
        channel = event["channel"]

        clean_text = self._clean_mention_text(text)

        if not clean_text:
            say(f"Hi <@{user_id}>! How can I help you today? 🤖")
            return

        logger.info(f"📢 Mentioned by {user_id}: {clean_text}")
        initial_msg = say(f"<@{user_id}> 🤔 Processing your request...")

        self._process_mention_async(
            clean_text, user_id, channel, client, initial_msg, logger, say
        )

    def _handle_direct_message_event(self, body, say, logger, client):
        """Handle direct messages to the bot."""
        event = body["event"]

        if not self._should_process_dm(event):
            return

        user_id = event["user"]
        text = event.get("text", "").strip()
        channel = event["channel"]

        if not text:
            return

        if text.startswith("/"):
            self._handle_dm_command(text, user_id, say, logger)
            return

        logger.info(f"💬 Direct message from {user_id}: {text}")
        initial_msg = say("🤔 Processing your message...")

        self._process_dm_async(text, user_id, channel, client, initial_msg, logger, say)

    def _clean_mention_text(self, text):
        """Remove bot mention from text."""
        import re

        return re.sub(r"<@[A-Z0-9]+>", "", text).strip()

    def _should_process_dm(self, event):
        """Check if we should process this direct message."""
        return event.get("channel_type") == "im" and not event.get("bot_id")

    def _handle_dm_command(self, text, user_id, say, logger):
        """Handle slash commands in direct messages."""
        command = text[1:].lower().split()[0]

        if command == "status":
            self._handle_dm_status_command(say, logger)
        elif command == "help":
            self._handle_help_command(say)
        else:
            say(f"❌ Unknown command: `{text}`\n💡 Use `/help` for available commands")

    def _handle_dm_status_command(self, say, logger):
        """Handle /status command in direct messages."""
        if not self.system_ready:
            say("🔄 AI system is starting up...")
            return

        try:
            status = self.supervisor.status()
            if status:
                status_msg = self._build_status_message(status)
                say(status_msg)
            else:
                say("❌ Could not retrieve system status")
        except Exception as e:
            logger.error(f"❌ Status error: {e}")
            say(f"❌ Error getting status: {str(e)}")

    def _handle_help_command(self, say):
        """Handle /help command."""
        help_msg = "🤖 *Available Commands:*\n"
        help_msg += "• `/status` - Show system status and health\n"
        help_msg += "• `/help` - Show this help message\n"
        help_msg += "• Any other text - Process as AI workflow\n"
        say(help_msg)

    def _build_status_message(self, status):
        """Build system status message."""
        status_msg = "🤖 *AI System Status*\n"
        status_msg += f"• System: {status.get('status', 'unknown')}\n"
        status_msg += f"• Running: {'✅' if status.get('running') else '❌'}\n"

        if "workflow_engine" in status:
            we_status = status["workflow_engine"]
            status_msg += (
                f"• Active Workflows: {we_status.get('active_workflows', 0)}\n"
            )

        if "universal_agent" in status:
            ua_status = status["universal_agent"]
            roles = ua_status.get("available_roles", [])
            status_msg += f"• Available Roles: {', '.join(roles)}\n"

        if "heartbeat" in status:
            hb_status = status["heartbeat"]
            health_status = hb_status.get("overall_status", "unknown")
            health_emoji = self._get_health_emoji(health_status)
            status_msg += f"• Health Status: {health_emoji} {health_status}\n"

        return status_msg

    def _get_health_emoji(self, health_status):
        """Get emoji for health status."""
        if health_status == "healthy":
            return "✅"
        elif health_status == "degraded":
            return "⚠️"
        else:
            return "❓"

    def _process_mention_async(
        self, clean_text, user_id, channel, client, initial_msg, logger, say
    ):
        """Process mention in background thread."""

        def process_mention():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    self._process_with_ai(clean_text, user_id, channel, client)
                )
                self._update_message_or_fallback(
                    client,
                    channel,
                    initial_msg,
                    f"<@{user_id}> {response}",
                    say,
                    f"<@{user_id}> {response}",
                )
            except Exception as e:
                logger.error(f"❌ Error processing mention: {e}")
                self._update_message_or_fallback(
                    client,
                    channel,
                    initial_msg,
                    f"<@{user_id}> Sorry, I encountered an error processing your request.",
                    say,
                    f"<@{user_id}> Sorry, I encountered an error.",
                )

        # LLM-SAFE: Use asyncio instead of threading
        asyncio.create_task(asyncio.to_thread(process_mention))

    def _process_dm_async(
        self, text, user_id, channel, client, initial_msg, logger, say
    ):
        """Process direct message in background thread."""

        def process_dm():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    self._process_with_ai(text, user_id, channel, client)
                )
                self._update_message_or_fallback(
                    client, channel, initial_msg, response, say, response
                )
            except Exception as e:
                logger.error(f"❌ Error processing DM: {e}")
                self._update_message_or_fallback(
                    client,
                    channel,
                    initial_msg,
                    "Sorry, I encountered an error processing your message.",
                    say,
                    "Sorry, I encountered an error processing your message.",
                )

        # LLM-SAFE: Use asyncio instead of threading
        asyncio.create_task(asyncio.to_thread(process_dm))

    def _update_message_or_fallback(
        self, client, channel, initial_msg, update_text, say, fallback_text
    ):
        """Update message or fallback to new message."""
        if initial_msg and initial_msg.get("ts"):
            client.chat_update(
                channel=channel,
                ts=initial_msg["ts"],
                text=update_text,
            )
        else:
            say(fallback_text)

    def setup_error_handler(self):
        """Setup global error handler."""

        @self.app.error
        def global_error_handler(error, body, logger):
            logger.exception(f"❌ Slack app error: {error}")
            logger.info(f"📋 Error context: {body}")

    def run(self):
        """Start the Slack bot."""
        try:
            handler = SocketModeHandler(self.app, self.app_token)
            logger.info("🚀 Starting Intelligent Slack Bot...")
            logger.info("🤖 Bot features:")
            logger.info("   • @mention me for AI responses")
            logger.info("   • Send me direct messages")
            logger.info("   • Use /ai <question> command")
            logger.info("   • Use /status for system status")

            handler.start()

        except KeyboardInterrupt:
            logger.info("⚠️ Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
        finally:
            if self.supervisor and self.system_ready:
                logger.info("🛑 Stopping Universal Agent system...")
                self.supervisor.stop()

    def stop(self):
        """Stop the bot and cleanup."""
        if self.supervisor and self.system_ready:
            logger.info("🛑 Stopping Universal Agent system...")
            self.supervisor.stop()


def main():
    """Main entry point for the Slack bot."""
    # Get tokens from environment
    app_token = os.environ.get("SLACK_APP_TOKEN")
    bot_token = os.environ.get("SLACK_BOT_TOKEN")

    if not app_token:
        logger.error("❌ SLACK_APP_TOKEN environment variable is required")
        sys.exit(1)

    # Optional: custom config path
    config_path = os.environ.get("AI_CONFIG_PATH", "config.yaml")

    # Create and run bot
    bot = IntelligentSlackBot(
        app_token=app_token, bot_token=bot_token, config_path=config_path
    )

    bot.run()


if __name__ == "__main__":
    main()
