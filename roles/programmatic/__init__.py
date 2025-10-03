"""
Programmatic Roles Package

Contains programmatic role implementations for hybrid execution architecture.
These roles execute directly without LLM processing for optimal performance.
"""

from .search_data_collector_role import SearchDataCollectorRole
from .data_collection_role import DataCollectionRole

__all__ = [
    'SearchDataCollectorRole',
    'DataCollectionRole'
]