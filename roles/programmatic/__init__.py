"""
Programmatic Roles Package

Contains programmatic role implementations for hybrid execution architecture.
These roles execute directly without LLM processing for optimal performance.
"""

from .search_data_collector_role import SearchDataCollectorRole

__all__ = [
    'SearchDataCollectorRole'
]