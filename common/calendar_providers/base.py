"""Base calendar provider abstract class.

Defines the interface that all calendar providers must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class CalendarProvider(ABC):
    """Abstract base class for calendar providers.

    All calendar providers must implement these methods to ensure
    a consistent interface regardless of the underlying calendar service.

    Event Format:
        All methods that return events use a standardized dict format:
        {
            'id': str,              # Unique event identifier
            'title': str,           # Event title/summary
            'start': datetime,      # Start time (timezone-aware)
            'end': datetime,        # End time (timezone-aware)
            'location': str | None, # Event location (optional)
            'description': str | None  # Event description (optional)
        }
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the calendar provider.

        Returns:
            bool: True if authentication successful, False otherwise
        """
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with the provider.

        Returns:
            bool: True if authenticated, False otherwise
        """
        pass

    @abstractmethod
    def get_events(self, start_date: datetime, end_date: datetime) -> list[dict]:
        """Get calendar events in the specified date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of event dicts in standardized format

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    def add_event(
        self,
        title: str,
        start: datetime,
        duration: int,
        location: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Add a new calendar event.

        Args:
            title: Event title/summary
            start: Event start time (timezone-aware)
            duration: Event duration in minutes
            location: Optional event location
            description: Optional event description

        Returns:
            Event dict in standardized format

        Raises:
            Exception: If event creation fails
        """
        pass

    @abstractmethod
    def update_event(self, event_id: str, **kwargs) -> dict:
        """Update an existing calendar event.

        Args:
            event_id: Unique identifier of event to update
            **kwargs: Fields to update (title, start, duration, location, description)

        Returns:
            Updated event dict in standardized format

        Raises:
            Exception: If update fails or event not found
        """
        pass

    @abstractmethod
    def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event.

        Args:
            event_id: Unique identifier of event to delete

        Returns:
            bool: True if deletion successful, False otherwise

        Raises:
            Exception: If deletion fails
        """
        pass
