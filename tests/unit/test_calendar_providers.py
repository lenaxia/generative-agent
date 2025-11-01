"""Tests for calendar provider abstraction layer."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from common.calendar_providers import CalendarProvider, get_calendar_provider
from common.calendar_providers.caldav_provider import CalDAVProvider


class TestCalendarProviderFactory:
    """Test calendar provider factory function."""

    def test_get_caldav_provider(self):
        """Test creating CalDAV provider from config."""
        config = {
            "provider": "caldav",
            "caldav": {
                "url": "https://caldav.example.com",
                "username": "test@example.com",
                "password": "test-password",
            },
        }

        provider = get_calendar_provider(config)

        assert isinstance(provider, CalDAVProvider)
        assert provider.url == "https://caldav.example.com"
        assert provider.username == "test@example.com"
        assert provider.password == "test-password"

    def test_get_provider_unknown_type(self):
        """Test error handling for unknown provider type."""
        config = {
            "provider": "unknown",
        }

        with pytest.raises(ValueError, match="Unknown calendar provider"):
            get_calendar_provider(config)

    def test_get_provider_default_caldav(self):
        """Test default provider is CalDAV."""
        config = {
            "caldav": {
                "url": "https://caldav.example.com",
                "username": "test@example.com",
                "password": "test-password",
            }
        }

        provider = get_calendar_provider(config)

        assert isinstance(provider, CalDAVProvider)


class TestCalDAVProvider:
    """Test CalDAV provider implementation."""

    @pytest.fixture
    def provider(self):
        """Create CalDAV provider instance."""
        return CalDAVProvider(
            url="https://caldav.example.com",
            username="test@example.com",
            password="test-password",
        )

    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.url == "https://caldav.example.com"
        assert provider.username == "test@example.com"
        assert provider.password == "test-password"
        assert provider.client is None
        assert provider.calendar is None
        assert not provider.is_authenticated()

    @patch("caldav.DAVClient")
    def test_authenticate_success(self, mock_dav_client, provider):
        """Test successful authentication."""
        # Setup mocks
        mock_calendar = Mock()
        mock_calendar.name = "Test Calendar"

        mock_principal = Mock()
        mock_principal.calendars.return_value = [mock_calendar]

        mock_client_instance = Mock()
        mock_client_instance.principal.return_value = mock_principal

        mock_dav_client.return_value = mock_client_instance

        # Test authentication
        result = provider.authenticate()

        assert result is True
        assert provider.is_authenticated()
        assert provider.calendar == mock_calendar

        # Verify DAVClient was called correctly
        mock_dav_client.assert_called_once_with(
            url="https://caldav.example.com",
            username="test@example.com",
            password="test-password",
        )

    @patch("caldav.DAVClient")
    def test_authenticate_no_calendars(self, mock_dav_client, provider):
        """Test authentication failure when no calendars found."""
        # Setup mocks
        mock_principal = Mock()
        mock_principal.calendars.return_value = []

        mock_client_instance = Mock()
        mock_client_instance.principal.return_value = mock_principal

        mock_dav_client.return_value = mock_client_instance

        # Test authentication
        result = provider.authenticate()

        assert result is False
        assert not provider.is_authenticated()

    def test_authenticate_import_error(self, provider):
        """Test authentication when caldav library not installed."""
        # Simulate import error by temporarily removing the import
        with patch.dict("sys.modules", {"caldav": None}):
            result = provider.authenticate()

        assert result is False
        assert not provider.is_authenticated()

    @patch("caldav.DAVClient")
    @patch("icalendar.Calendar")
    def test_get_events(self, mock_calendar_class, mock_dav_client, provider):
        """Test getting events from calendar."""
        # Setup authentication mocks
        mock_calendar = Mock()
        mock_calendar.name = "Test Calendar"

        mock_principal = Mock()
        mock_principal.calendars.return_value = [mock_calendar]

        mock_client_instance = Mock()
        mock_client_instance.principal.return_value = mock_principal

        mock_dav_client.return_value = mock_client_instance

        # Setup event mocks
        mock_event = Mock()
        mock_event.data = b"VCALENDAR_DATA"

        mock_calendar.date_search.return_value = [mock_event]

        # Setup iCalendar parsing mock
        mock_ical = Mock()
        mock_vevent = Mock()
        mock_vevent.name = "VEVENT"
        mock_vevent.get.side_effect = lambda key, default=None: {
            "uid": "test-uid-123",
            "summary": "Test Event",
            "dtstart": Mock(dt=datetime(2024, 3, 15, 14, 0)),
            "dtend": Mock(dt=datetime(2024, 3, 15, 15, 0)),
            "location": "Test Location",
            "description": "Test Description",
        }.get(key, default)

        mock_ical.walk.return_value = [mock_vevent]
        mock_calendar_class.from_ical.return_value = mock_ical

        # Authenticate first
        provider.authenticate()

        # Test get_events
        start = datetime(2024, 3, 15)
        end = datetime(2024, 3, 16)
        events = provider.get_events(start, end)

        assert len(events) == 1
        assert events[0]["id"] == "test-uid-123"
        assert events[0]["title"] == "Test Event"
        assert events[0]["location"] == "Test Location"

    @patch("caldav.DAVClient")
    @patch("icalendar.Calendar")
    @patch("icalendar.Event")
    def test_add_event(
        self, mock_ical_event, mock_calendar_class, mock_dav_client, provider
    ):
        """Test adding event to calendar."""
        # Setup authentication mocks
        mock_calendar = Mock()
        mock_calendar.name = "Test Calendar"
        mock_calendar.save_event = Mock()

        mock_principal = Mock()
        mock_principal.calendars.return_value = [mock_calendar]

        mock_client_instance = Mock()
        mock_client_instance.principal.return_value = mock_principal

        mock_dav_client.return_value = mock_client_instance

        # Authenticate first
        provider.authenticate()

        # Test add_event
        start = datetime(2024, 3, 15, 14, 0)
        event = provider.add_event(
            title="Test Event",
            start=start,
            duration=60,
            location="Test Location",
            description="Test Description",
        )

        assert event["title"] == "Test Event"
        assert event["start"] == start
        assert event["end"] == start + timedelta(minutes=60)
        assert event["location"] == "Test Location"
        assert event["description"] == "Test Description"
        assert "id" in event

    def test_update_event_not_implemented(self, provider):
        """Test that update_event raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider.update_event("test-id", title="New Title")

    def test_delete_event_not_implemented(self, provider):
        """Test that delete_event raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider.delete_event("test-id")


class TestCalendarProviderInterface:
    """Test that CalDAVProvider implements CalendarProvider interface."""

    def test_implements_interface(self):
        """Test that CalDAVProvider implements all required methods."""
        provider = CalDAVProvider("url", "user", "pass")

        # Check all abstract methods are implemented
        assert hasattr(provider, "authenticate")
        assert hasattr(provider, "is_authenticated")
        assert hasattr(provider, "get_events")
        assert hasattr(provider, "add_event")
        assert hasattr(provider, "update_event")
        assert hasattr(provider, "delete_event")

        # Check they are callable
        assert callable(provider.authenticate)
        assert callable(provider.is_authenticated)
        assert callable(provider.get_events)
        assert callable(provider.add_event)
        assert callable(provider.update_event)
        assert callable(provider.delete_event)
