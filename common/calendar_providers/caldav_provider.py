"""CalDAV calendar provider implementation.

Supports iCloud, Nextcloud, and any CalDAV-compatible calendar server.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from .base import CalendarProvider

logger = logging.getLogger(__name__)


class CalDAVProvider(CalendarProvider):
    """CalDAV provider for iCloud, Nextcloud, and other CalDAV servers.

    This provider uses the caldav library to interact with CalDAV servers.
    It supports basic calendar operations: get events, add events, update, delete.

    Setup for iCloud:
        1. Go to appleid.apple.com
        2. Generate app-specific password
        3. Use iCloud email and app-specific password
        4. URL: https://caldav.icloud.com
    """

    def __init__(self, url: str, username: str, password: str):
        """Initialize CalDAV provider.

        Args:
            url: CalDAV server URL (e.g., https://caldav.icloud.com)
            username: Username or email address
            password: Password or app-specific password
        """
        self.url = url
        self.username = username
        self.password = password
        self.client = None
        self.calendar = None
        self._authenticated = False

    def authenticate(self) -> bool:
        """Authenticate with CalDAV server."""
        try:
            from caldav import DAVClient

            self.client = DAVClient(
                url=self.url, username=self.username, password=self.password
            )

            principal = self.client.principal()
            calendars = principal.calendars()

            if calendars:
                self.calendar = calendars[0]
                self._authenticated = True
                logger.info(f"Connected to CalDAV calendar: {self.calendar.name}")
                return True

            logger.error("No calendars found on CalDAV server")
            return False

        except ImportError:
            logger.error(
                "caldav library not installed. Install with: pip install caldav"
            )
            return False
        except Exception as e:
            logger.error(f"CalDAV authentication failed: {e}")
            return False

    def is_authenticated(self) -> bool:
        """Check if authenticated."""
        return (
            self._authenticated
            and self.client is not None
            and self.calendar is not None
        )

    def get_events(self, start_date: datetime, end_date: datetime) -> list[dict]:
        """Get events from CalDAV calendar.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of event dicts in standardized format
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return []

        try:
            events = self.calendar.date_search(
                start=start_date, end=end_date, expand=True
            )

            parsed_events = []
            for event in events:
                parsed = self._parse_event(event)
                if parsed:
                    parsed_events.append(parsed)

            logger.info(f"Retrieved {len(parsed_events)} events from CalDAV")
            return parsed_events

        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def add_event(
        self,
        title: str,
        start: datetime,
        duration: int,
        location: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Add event to CalDAV calendar.

        Args:
            title: Event title
            start: Event start time
            duration: Duration in minutes
            location: Optional location
            description: Optional description

        Returns:
            Event dict in standardized format
        """
        if not self.is_authenticated():
            if not self.authenticate():
                raise Exception("Not authenticated with CalDAV server")

        try:
            import pytz
            from icalendar import Calendar
            from icalendar import Event as ICalEvent

            # Create iCalendar event
            cal = Calendar()
            event = ICalEvent()

            # Generate unique ID
            event_id = str(uuid.uuid4())
            event.add("uid", event_id)
            event.add("summary", title)

            # Ensure timezone-aware datetimes
            if start.tzinfo is None:
                start = pytz.utc.localize(start)

            end = start + timedelta(minutes=duration)

            event.add("dtstart", start)
            event.add("dtend", end)
            event.add("dtstamp", datetime.now(pytz.utc))

            if location:
                event.add("location", location)
            if description:
                event.add("description", description)

            # Add to calendar
            cal.add_component(event)

            # Save to server
            self.calendar.save_event(cal.to_ical())

            logger.info(f"Added event: {title}")

            return {
                "id": event_id,
                "title": title,
                "start": start,
                "end": end,
                "location": location,
                "description": description,
            }

        except ImportError:
            logger.error(
                "icalendar library not installed. Install with: pip install icalendar"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to add event: {e}")
            raise

    def update_event(self, event_id: str, **kwargs) -> dict:
        """Update existing event.

        Args:
            event_id: Event ID to update
            **kwargs: Fields to update

        Returns:
            Updated event dict

        Note:
            CalDAV update is complex - requires fetching, modifying, and saving.
            This is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "Event update not yet implemented for CalDAV. "
            "Delete and recreate the event as a workaround."
        )

    def delete_event(self, event_id: str) -> bool:
        """Delete event from calendar.

        Args:
            event_id: Event ID to delete

        Returns:
            True if successful

        Note:
            CalDAV delete requires finding the event first.
            This is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "Event deletion not yet implemented for CalDAV. "
            "Use your calendar app to delete events."
        )

    def _parse_event(self, caldav_event) -> dict:
        """Parse CalDAV event to standardized format.

        Args:
            caldav_event: CalDAV event object

        Returns:
            Event dict in standardized format, or empty dict if parsing fails
        """
        try:
            from icalendar import Calendar

            ical = Calendar.from_ical(caldav_event.data)

            for component in ical.walk():
                if component.name == "VEVENT":
                    # Extract event data
                    event_id = str(component.get("uid", ""))
                    title = str(component.get("summary", "Untitled"))
                    start = component.get("dtstart").dt
                    end = component.get("dtend").dt
                    location = str(component.get("location", "")) or None
                    description = str(component.get("description", "")) or None

                    # Handle date-only events (convert to datetime)
                    if not isinstance(start, datetime):
                        from datetime import date

                        if isinstance(start, date):
                            start = datetime.combine(start, datetime.min.time())

                    if not isinstance(end, datetime):
                        from datetime import date

                        if isinstance(end, date):
                            end = datetime.combine(end, datetime.min.time())

                    return {
                        "id": event_id,
                        "title": title,
                        "start": start,
                        "end": end,
                        "location": location,
                        "description": description,
                    }
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")
            return {}

        return {}
