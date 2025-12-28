"""Timer Domain

Provides timer management tools with intent-based execution.
"""

from .role import TimerRole
from .tools import create_timer_tools

__all__ = ["create_timer_tools", "TimerRole"]
