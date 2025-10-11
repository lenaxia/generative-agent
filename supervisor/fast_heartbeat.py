"""
FastHeartbeat service for high-frequency monitoring.

Provides fast heartbeat events (5s interval) for time-sensitive monitoring
like timer expiry, while keeping the main heartbeat for system-level monitoring.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict

from common.message_bus import MessageBus

logger = logging.getLogger(__name__)


class FastHeartbeat:
    """Fast heartbeat service for high-frequency monitoring needs.

    Publishes FAST_HEARTBEAT_TICK events at configurable intervals (default 5s)
    for roles that need frequent monitoring like timers, alarms, etc.

    This complements the main Heartbeat service which handles system-level
    monitoring at longer intervals (30s).
    """

    def __init__(self, message_bus: MessageBus, interval: int = 5):
        """Initialize FastHeartbeat service.

        Args:
            message_bus: MessageBus for publishing heartbeat events
            interval: Heartbeat interval in seconds (default: 5)
        """
        self.message_bus = message_bus
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.is_running = False
        self.tick_count = 0

    def start(self):
        """Start the fast heartbeat service."""
        if self.is_running:
            logger.warning("FastHeartbeat already running")
            return

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="FastHeartbeat"
        )
        self.thread.start()
        self.is_running = True

        logger.info(f"FastHeartbeat started with {self.interval}s interval")

    def stop(self):
        """Stop the fast heartbeat service."""
        if not self.is_running:
            return

        self.stop_event.set()
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        logger.info("FastHeartbeat stopped")

    def _run(self):
        """Main heartbeat loop."""
        logger.info("FastHeartbeat loop started")

        while not self.stop_event.is_set():
            try:
                self._publish_fast_heartbeat_tick()
                self.tick_count += 1

                # Wait for next interval or stop event
                if self.stop_event.wait(timeout=self.interval):
                    break  # Stop event was set

            except Exception as e:
                logger.error(f"FastHeartbeat error: {e}")
                # Continue running even on errors

        logger.info(f"FastHeartbeat loop ended after {self.tick_count} ticks")

    def _publish_fast_heartbeat_tick(self):
        """Publish FAST_HEARTBEAT_TICK event."""
        if not self.message_bus or not self.message_bus.is_running():
            return

        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "interval": self.interval,
            "tick_count": self.tick_count,
            "event_source": "fast_heartbeat",
        }

        try:
            self.message_bus.publish(self, "FAST_HEARTBEAT_TICK", heartbeat_data)
            logger.debug(f"Published FAST_HEARTBEAT_TICK #{self.tick_count}")
        except Exception as e:
            logger.error(f"Failed to publish fast heartbeat tick: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get FastHeartbeat statistics.

        Returns:
            Dictionary with heartbeat statistics
        """
        return {
            "is_running": self.is_running,
            "interval": self.interval,
            "tick_count": self.tick_count,
            "thread_alive": self.thread.is_alive() if self.thread else False,
        }
