"""Intent Collector Module

Provides context-local intent collection for agent execution.
Tools register intents during execution which are collected and processed after completion.
"""

import contextvars
import logging
from typing import List, Optional

from common.intents import Intent

logger = logging.getLogger(__name__)


class IntentCollector:
    """
    Collects intents during agent execution.

    Tools register intents here instead of executing directly.
    After agent completes, intents are retrieved and processed by IntentProcessor.
    """

    def __init__(self):
        """Initialize the intent collector."""
        self._intents: List[Intent] = []

    def register(self, intent: Intent):
        """Register an intent for later processing.

        Args:
            intent: Intent object to register
        """
        self._intents.append(intent)
        logger.debug(f"Intent registered: {intent.__class__.__name__}")

    def get_intents(self) -> List[Intent]:
        """Get all collected intents.

        Returns:
            Copy of collected intents list
        """
        return self._intents.copy()

    def clear(self):
        """Clear all collected intents."""
        count = len(self._intents)
        self._intents.clear()
        logger.debug(f"Cleared {count} intents from collector")

    def count(self) -> int:
        """Get count of collected intents."""
        return len(self._intents)


# Context-local storage for current collector
# This ensures each async context has its own intent collector
_current_collector: contextvars.ContextVar[Optional[IntentCollector]] = (
    contextvars.ContextVar("intent_collector", default=None)
)


def set_current_collector(collector: IntentCollector):
    """Set the intent collector for current execution context.

    Args:
        collector: IntentCollector instance to set
    """
    _current_collector.set(collector)
    logger.debug("Intent collector set for current context")


def get_current_collector() -> Optional[IntentCollector]:
    """Get the current intent collector from context.

    Returns:
        IntentCollector instance or None if not set
    """
    return _current_collector.get()


def clear_current_collector():
    """Clear the current intent collector from context."""
    _current_collector.set(None)
    logger.debug("Intent collector cleared from current context")


async def register_intent(intent: Intent):
    """Register an intent with the current collector.

    This is the main function that tools call to register intents.

    Args:
        intent: Intent object to register
    """
    collector = get_current_collector()

    if collector is not None:
        collector.register(intent)
    else:
        # No collector available - log warning
        logger.warning(
            f"No intent collector available for {intent.__class__.__name__}. "
            f"Intent may not be processed!"
        )
        # Still log the intent for debugging
        logger.info(f"Orphaned intent: {intent}")
