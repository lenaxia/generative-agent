"""Smart Home Domain

Provides Home Assistant integration tools for device control and monitoring.
"""

from .role import SmartHomeRole
from .tools import create_smart_home_tools

__all__ = ["create_smart_home_tools", "SmartHomeRole"]
