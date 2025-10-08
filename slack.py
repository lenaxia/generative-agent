#!/usr/bin/env python3
"""
Enhanced Slack Bot with Universal Agent Integration

Connects your Slack bot to the StrandsAgent Universal Agent system for intelligent responses.
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict

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
    """
    Slack bot that integrates with the Universal Agent system for intelligent responses.
    """

    def __init__(self, app_token, bot_token=None, config_path="config.yaml"):
        """
        Initialize the intelligent Slack bot.

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
            self.system_ready = True
            logger.info("✅ Universal Agent system ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Universal Agent system: {e}")
            self.system_ready = False

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
        """
        Resolve Slack user and channel references to human-readable names.

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
        """
        Process message with the Universal Agent system.

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
                user_name = (
                    user_info.get("display_name")
                    or user_info.get("real_name")
                    or user_info.get("name")
                )
                context_message = f"User {user_name} (@{user_info.get('name')}) asks: {resolved_message}"
            elif user_id:
                context_message = f"User {user_id} asks: {resolved_message}"

            # Add channel context if available
            if channel and client:
                channel_info = await self._get_channel_info(channel, client)
                channel_name = channel_info.get("name")
                if channel_name:
                    context_message += f" [in #{channel_name}]"

            logger.info(f"🧠 Processing with AI: {context_message}")

            # Start workflow with Universal Agent
            workflow_id = self.supervisor.workflow_engine.start_workflow(
                context_message
            )

            # Update workflow source to Slack (duration tracking is handled in workflow engine)
            self.supervisor.workflow_engine.update_workflow_source(
                workflow_id, WorkflowSource.SLACK, user_id=user_id, channel_id=channel
            )

            # Wait for completion with more frequent checks and immediate response
            max_wait = 30  # 30 seconds timeout
            wait_time = 0
            check_interval = 0.5  # Check every 500ms for faster response

            while wait_time < max_wait:
                status = self.supervisor.workflow_engine.get_request_status(workflow_id)

                if status is None:
                    # Workflow completed, try to get results from TaskContext
                    return await self._get_workflow_result(workflow_id)

                if status.get("is_completed", False):
                    # Workflow completed, get the actual result immediately
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

                await asyncio.sleep(check_interval)
                wait_time += check_interval

            return "⏰ AI is taking longer than expected. Please try a simpler request."

        except Exception as e:
            logger.error(f"❌ AI processing error: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"

    async def _get_workflow_result(self, workflow_id: str) -> str:
        """
        Get the actual result from a completed workflow.

        Args:
            workflow_id: The workflow ID to get results for

        Returns:
            The actual AI response or a fallback message
        """
        try:
            # Get the TaskContext for this workflow
            task_context = self.supervisor.workflow_engine.get_request_context(
                workflow_id
            )

            if not task_context:
                return "✅ Task completed successfully!"

            # Get all completed task nodes and their results
            completed_results = []
            for node in task_context.task_graph.nodes.values():
                if node.status.value == "COMPLETED" and node.result:
                    completed_results.append(node.result)

            # If we have results, return the most recent/relevant one
            if completed_results:
                # For simple workflows, return the last result
                final_result = completed_results[-1]
                logger.info(f"🎯 Retrieved AI result: {final_result[:100]}...")
                return f"🤖 {final_result}"

            # Try to get from conversation history
            conversation_history = task_context.get_conversation_history()
            if conversation_history:
                # Look for the last assistant message
                for msg in reversed(conversation_history):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content")
                        if content and content.strip():
                            logger.info(
                                f"🎯 Retrieved from conversation: {content[:100]}..."
                            )
                            return f"🤖 {content}"

            # Try to get from progressive summary
            if (
                hasattr(task_context.task_graph, "progressive_summary")
                and task_context.task_graph.progressive_summary
            ):
                summary = task_context.task_graph.progressive_summary[-1]
                if summary and summary.strip():
                    logger.info(f"🎯 Retrieved from summary: {summary[:100]}...")
                    return f"🤖 {summary}"

            return "✅ Task completed successfully!"

        except Exception as e:
            logger.error(f"❌ Error getting workflow result: {e}")
            return "✅ Task completed successfully!"

    def setup_middleware(self):
        """Setup Slack middleware for logging."""

        @self.app.middleware
        def log_request(logger, body, next):
            logger.debug(f"📨 Received Slack event: {body.get('type', 'unknown')}")
            return next()

    def _handle_slash_command(self, message_text: str, user_id: str = None) -> str:
        """
        Handle slash commands in messages (similar to CLI implementation).

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
                            else "⚠️"
                            if health_status == "degraded"
                            else "❓"
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

            user_id = body["user_id"]
            text = body.get("text", "").strip()

            if not text:
                say(f"Hi <@{user_id}>! Use `/ai your question` to chat with me. 🤖")
                return

            logger.info(f"📝 /ai command from {user_id}: {text}")

            # Send initial response
            respond(f"<@{user_id}> 🤔 Processing your request...")

            # Process with AI in a separate thread to avoid blocking
            def process_ai():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self._process_with_ai(text, user_id)
                    )
                    # Update the original message
                    respond(f"<@{user_id}> {response}")
                except Exception as e:
                    logger.error(f"❌ Error in AI processing: {e}")
                    respond(
                        f"<@{user_id}> Sorry, I encountered an error processing your request."
                    )

            threading.Thread(target=process_ai, daemon=True).start()

        @self.app.command("/status")
        def status_command(ack, body, say):
            """Handle /status command to show AI system status."""
            ack()

            user_id = body["user_id"]

            if not self.system_ready:
                say(f"<@{user_id}> 🔄 AI system is starting up...")
                return

            try:
                status = self.supervisor.status()
                if status:
                    status_msg = f"<@{user_id}> 🤖 *AI System Status*\n"
                    status_msg += f"• System: {status.get('status', 'unknown')}\n"
                    status_msg += (
                        f"• Running: {'✅' if status.get('running') else '❌'}\n"
                    )

                    if "workflow_engine" in status:
                        we_status = status["workflow_engine"]
                        status_msg += f"• Active Workflows: {we_status.get('active_workflows', 0)}\n"

                    if "universal_agent" in status:
                        ua_status = status["universal_agent"]
                        roles = ua_status.get("available_roles", [])
                        status_msg += f"• Available Roles: {', '.join(roles)}\n"

                    say(status_msg)
                else:
                    say(f"<@{user_id}> ❌ Could not retrieve system status")

            except Exception as e:
                logger.error(f"❌ Status error: {e}")
                say(f"<@{user_id}> ❌ Error getting status: {str(e)}")

    def setup_events(self):
        """Setup Slack event handlers."""

        @self.app.event("app_mention")
        def handle_mention(body, say, logger, client):
            """Handle @mentions of the bot."""
            event = body["event"]
            user_id = event["user"]
            text = event["text"]
            channel = event["channel"]

            # Remove bot mention from text
            # Slack mentions look like <@U1234567890>
            import re

            clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

            if not clean_text:
                say(f"Hi <@{user_id}>! How can I help you today? 🤖")
                return

            logger.info(f"📢 Mentioned by {user_id}: {clean_text}")

            # Send initial processing message
            initial_msg = say(f"<@{user_id}> 🤔 Processing your request...")

            # Process with AI
            def process_mention():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self._process_with_ai(clean_text, user_id, channel, client)
                    )

                    # Update the original message
                    if initial_msg and initial_msg.get("ts"):
                        client.chat_update(
                            channel=channel,
                            ts=initial_msg["ts"],
                            text=f"<@{user_id}> {response}",
                        )
                    else:
                        # Fallback to new message if update fails
                        say(f"<@{user_id}> {response}")

                except Exception as e:
                    logger.error(f"❌ Error processing mention: {e}")
                    # Update with error message
                    if initial_msg and initial_msg.get("ts"):
                        client.chat_update(
                            channel=channel,
                            ts=initial_msg["ts"],
                            text=f"<@{user_id}> Sorry, I encountered an error processing your request.",
                        )
                    else:
                        say(f"<@{user_id}> Sorry, I encountered an error.")

            threading.Thread(target=process_mention, daemon=True).start()

        @self.app.event("message")
        def handle_direct_message(body, say, logger, client):
            """Handle direct messages to the bot."""
            event = body["event"]

            # Only respond to direct messages (not channel messages)
            if event.get("channel_type") != "im":
                return

            # Ignore bot messages
            if event.get("bot_id"):
                return

            user_id = event["user"]
            text = event.get("text", "").strip()
            channel = event["channel"]

            if not text:
                return

            # Handle slash commands in direct messages
            if text.startswith("/"):
                command = text[1:].lower().split()[0]

                if command == "status":
                    # Handle /status command in DM
                    if not self.system_ready:
                        say("🔄 AI system is starting up...")
                        return

                    try:
                        status = self.supervisor.status()
                        if status:
                            status_msg = "🤖 *AI System Status*\n"
                            status_msg += (
                                f"• System: {status.get('status', 'unknown')}\n"
                            )
                            status_msg += (
                                f"• Running: {'✅' if status.get('running') else '❌'}\n"
                            )

                            if "workflow_engine" in status:
                                we_status = status["workflow_engine"]
                                status_msg += f"• Active Workflows: {we_status.get('active_workflows', 0)}\n"

                            if "universal_agent" in status:
                                ua_status = status["universal_agent"]
                                roles = ua_status.get("available_roles", [])
                                status_msg += f"• Available Roles: {', '.join(roles)}\n"

                            # Add health status from our fixed health check
                            if "heartbeat" in status:
                                hb_status = status["heartbeat"]
                                health_status = hb_status.get(
                                    "overall_status", "unknown"
                                )
                                health_emoji = (
                                    "✅"
                                    if health_status == "healthy"
                                    else "⚠️"
                                    if health_status == "degraded"
                                    else "❓"
                                )
                                status_msg += (
                                    f"• Health Status: {health_emoji} {health_status}\n"
                                )

                            say(status_msg)
                        else:
                            say("❌ Could not retrieve system status")

                    except Exception as e:
                        logger.error(f"❌ Status error: {e}")
                        say(f"❌ Error getting status: {str(e)}")
                    return

                elif command == "help":
                    # Handle /help command in DM
                    help_msg = "🤖 *Available Commands:*\n"
                    help_msg += "• `/status` - Show system status and health\n"
                    help_msg += "• `/help` - Show this help message\n"
                    help_msg += "• Any other text - Process as AI workflow\n"
                    say(help_msg)
                    return

                else:
                    # Unknown slash command
                    say(
                        f"❌ Unknown command: `{text}`\n💡 Use `/help` for available commands"
                    )
                    return

            logger.info(f"💬 Direct message from {user_id}: {text}")

            # Send initial processing message
            initial_msg = say("🤔 Processing your message...")

            # Process with AI
            def process_dm():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self._process_with_ai(text, user_id, channel, client)
                    )

                    # Update the original message
                    if initial_msg and initial_msg.get("ts"):
                        client.chat_update(
                            channel=channel, ts=initial_msg["ts"], text=response
                        )
                    else:
                        # Fallback to new message if update fails
                        say(response)

                except Exception as e:
                    logger.error(f"❌ Error processing DM: {e}")
                    # Update with error message
                    if initial_msg and initial_msg.get("ts"):
                        client.chat_update(
                            channel=channel,
                            ts=initial_msg["ts"],
                            text="Sorry, I encountered an error processing your message.",
                        )
                    else:
                        say("Sorry, I encountered an error processing your message.")

            threading.Thread(target=process_dm, daemon=True).start()

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
