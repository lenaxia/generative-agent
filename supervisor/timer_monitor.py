"""Timer Monitor for processing expired timers.

Integrates with the Heartbeat system to monitor and process timer events,
handling timer expiry, recurring timers, and notification dispatch.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from common.message_bus import MessageBus, MessageType
from roles.timer.lifecycle import get_timer_manager

logger = logging.getLogger(__name__)


class TimerMonitor:
    """Monitors and processes timer events integrated with Heartbeat system.

    This class is designed to be called from the Heartbeat system to check
    for expired timers and process them appropriately.
    """

    def __init__(self, message_bus: MessageBus):
        """Initialize TimerMonitor.

        Args:
            message_bus: MessageBus instance for publishing timer events
        """
        self.message_bus = message_bus
        self.timer_manager = get_timer_manager()
        self.last_check = 0
        self.check_interval = 5  # seconds - Fine-grained timer checking
        self.processing_timers = (
            set()
        )  # Track timers being processed to avoid duplicates

    async def check_expired_timers(self) -> list[dict]:
        """Check Redis for expired timers with rate limiting.

        Returns:
            List of expired timer data
        """
        current_time = int(time.time())

        # Rate limiting - only check if enough time has passed (minimum 5 seconds)
        if current_time - self.last_check < self.check_interval:
            return []

        self.last_check = current_time

        try:
            # Get expired timers from Redis sorted set
            expired_timers = await self.timer_manager.get_expiring_timers(current_time)

            # Filter out timers already being processed
            new_expired_timers = [
                timer
                for timer in expired_timers
                if timer["id"] not in self.processing_timers
            ]

            logger.debug(f"Found {len(new_expired_timers)} new expired timers")
            return new_expired_timers

        except Exception as e:
            logger.error(f"Failed to check expired timers: {e}")
            return []

    async def process_expired_timer(self, timer: dict) -> bool:
        """Process an expired timer with all configured actions.

        Args:
            timer: Timer data dictionary

        Returns:
            True if processed successfully, False otherwise
        """
        timer_id = timer["id"]

        # Mark timer as being processed
        self.processing_timers.add(timer_id)

        try:
            logger.info(f"Processing expired timer: {timer_id}")

            # Update timer status to completed
            await self.timer_manager.update_timer_status(timer_id, "completed")

            # Handle recurring timers first
            next_timer_id = None
            if timer.get("recurring"):
                next_timer_id = await self.timer_manager.create_recurring_instance(
                    timer
                )
                if next_timer_id:
                    logger.info(f"Created next recurring timer: {next_timer_id}")

            # Process all configured actions
            await self._process_timer_actions(timer)

            # Publish timer expired event to MessageBus
            self._publish_timer_expired_event(timer, next_timer_id)

            logger.info(f"Successfully processed expired timer: {timer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process expired timer {timer_id}: {e}")
            return False

        finally:
            # Remove from processing set
            self.processing_timers.discard(timer_id)

    async def _process_timer_actions(self, timer: dict):
        """Process all actions configured for the timer.

        Args:
            timer: Timer data dictionary
        """
        actions = timer.get("actions", [])

        for action in actions:
            try:
                await self._process_single_action(timer, action)
            except Exception as e:
                logger.error(
                    f"Failed to process action {action} for timer {timer['id']}: {e}"
                )

    async def _process_single_action(self, timer: dict, action: dict):
        """Process a single timer action.

        Args:
            timer: Timer data dictionary
            action: Action configuration
        """
        action_type = action.get("type")

        if action_type == "notify":
            await self._process_notification_action(timer, action)
        elif action_type == "trigger_request":
            await self._process_request_trigger_action(timer, action)
        elif action_type == "webhook":
            await self._process_webhook_action(timer, action)
        else:
            logger.warning(f"Unknown action type: {action_type}")

    async def _process_notification_action(self, timer: dict, action: dict):
        """Process notification action.

        Args:
            timer: Timer data dictionary
            action: Notification action configuration
        """
        # This will be handled by the Communication Manager in the next step
        # For now, just log the notification
        logger.info(f"Notification action for timer {timer['id']}: {action}")

    async def _process_request_trigger_action(self, timer: dict, action: dict):
        """Process request trigger action.

        Args:
            timer: Timer data dictionary
            action: Request trigger action configuration
        """
        request_text = action.get("request")
        context = action.get("context", {})

        if request_text:
            # Publish as incoming request to trigger workflow
            request_message = {
                "request": request_text,
                "source": "timer_trigger",
                "timer_id": timer["id"],
                "context": context,
                "user_id": timer.get("user_id"),
                "channel_id": timer.get("channel_id"),
            }

            self.message_bus.publish(
                self, MessageType.INCOMING_REQUEST, request_message
            )

            logger.info(f"Triggered request for timer {timer['id']}: {request_text}")

    async def _process_webhook_action(self, timer: dict, action: dict):
        """Process webhook action.

        Args:
            timer: Timer data dictionary
            action: Webhook action configuration
        """
        import aiohttp

        url = action.get("url")
        method = action.get("method", "POST").upper()

        if not url:
            logger.error("Webhook action missing URL")
            return

        payload = {
            "timer_id": timer["id"],
            "timer_name": timer.get("name"),
            "expired_at": int(time.time()),
            "user_id": timer.get("user_id"),
            "channel_id": timer.get("channel_id"),
            "custom_message": timer.get("custom_message"),
        }

        try:
            async with aiohttp.ClientSession() as session:
                if method == "POST":
                    async with session.post(url, json=payload) as response:
                        logger.info(f"Webhook POST to {url}: {response.status}")
                elif method == "GET":
                    async with session.get(url, params=payload) as response:
                        logger.info(f"Webhook GET to {url}: {response.status}")

        except Exception as e:
            logger.error(f"Webhook request failed for timer {timer['id']}: {e}")

    def _publish_timer_expired_event(self, timer: dict, next_timer_id: str = None):
        """Publish timer expired event to MessageBus.

        Args:
            timer: Expired timer data
            next_timer_id: ID of next recurring timer if created
        """
        # Extract notification metadata from timer
        metadata = timer.get("metadata", {})
        notification_channel = metadata.get("notification_channel")
        notification_recipient = metadata.get("notification_recipient")
        notification_priority = metadata.get("notification_priority")

        event_data = {
            "timer_id": timer["id"],
            "timer_name": timer.get("name"),
            "timer_type": timer.get("type"),
            "user_id": timer.get("user_id"),
            "channel_id": timer.get("channel_id"),
            "custom_message": timer.get("custom_message"),
            "notification_config": timer.get("notification_config", {}),
            "expired_at": int(time.time()),
            "next_timer_id": next_timer_id,
            # Include notification metadata
            "notification_channel": notification_channel,
            "notification_recipient": notification_recipient,
            "notification_priority": notification_priority,
            "metadata": metadata,
        }

        self.message_bus.publish(self, MessageType.TIMER_EXPIRED, event_data)

        logger.info(f"Published timer expired event for {timer['id']}")

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get timer monitoring statistics.

        Returns:
            Dictionary with monitoring statistics
        """
        return {
            "last_check": self.last_check,
            "check_interval": self.check_interval,
            "processing_timers_count": len(self.processing_timers),
            "processing_timer_ids": list(self.processing_timers),
        }

    async def cleanup_stale_processing_timers(self, max_age_seconds: int = 300):
        """Clean up stale processing timers (safety mechanism).

        Args:
            max_age_seconds: Maximum age for processing timers before cleanup
        """
        # This is a safety mechanism in case timers get stuck in processing state
        # In a real implementation, you'd track processing start times
        current_time = int(time.time())

        # For now, just clear the set if it gets too large
        if len(self.processing_timers) > 100:
            logger.warning("Clearing large processing timers set as safety measure")
            self.processing_timers.clear()

    async def force_check_timers(self) -> list[dict]:
        """Force check for expired timers regardless of rate limiting.

        Returns:
            List of expired timer data
        """
        # Temporarily reset last_check to force immediate check
        original_last_check = self.last_check
        self.last_check = 0

        try:
            return await self.check_expired_timers()
        finally:
            # Don't restore original time to avoid skipping next regular check
            pass
