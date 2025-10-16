"""Interface definitions for Universal Agent System components.

This module provides abstract base classes and data models for key system
components to enable future extensibility and testing flexibility.
"""

from .context_interfaces import (
    ContextData,
    ContextProvider,
    EnvironmentProvider,
    LocationData,
    LocationProvider,
    MemoryEntry,
    MemoryProvider,
)

__all__ = [
    "ContextProvider",
    "MemoryProvider",
    "LocationProvider",
    "EnvironmentProvider",
    "ContextData",
    "MemoryEntry",
    "LocationData",
]
