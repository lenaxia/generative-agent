"""Smart Home Domain

Provides Home Assistant integration tools for device control and monitoring.
"""

from .tools import create_smart_home_tools
from .role import SmartHomeRole

__all__ = ["create_smart_home_tools", "SmartHomeRole"]
