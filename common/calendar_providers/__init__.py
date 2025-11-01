"""Calendar provider abstraction layer.

This module provides a unified interface for different calendar providers
(CalDAV, Google Calendar, Outlook, etc.) with a clean abstraction that
allows easy switching between providers.

Currently implemented:
- CalDAV (iCloud, Nextcloud, self-hosted)

Future providers can be added by:
1. Creating a new provider class inheriting from CalendarProvider
2. Implementing all abstract methods
3. Adding to the factory function
"""

from .base import CalendarProvider
from .caldav_provider import CalDAVProvider


def get_calendar_provider(config: dict) -> CalendarProvider:
    """Factory function to create calendar provider from config.

    Args:
        config: Calendar configuration dict with 'provider' key

    Returns:
        CalendarProvider instance

    Raises:
        ValueError: If provider type is unknown or config is invalid

    Example:
        config = {
            'provider': 'caldav',
            'caldav': {
                'url': 'https://caldav.icloud.com',
                'username': 'user@icloud.com',
                'password': 'app-specific-password'
            }
        }
        provider = get_calendar_provider(config)
    """
    provider_type = config.get("provider", "caldav")

    if provider_type == "caldav":
        caldav_config = config.get("caldav", {})
        return CalDAVProvider(
            url=caldav_config["url"],
            username=caldav_config["username"],
            password=caldav_config["password"],
        )

    # Future providers can be added here:
    # elif provider_type == 'google':
    #     return GoogleCalendarProvider(...)
    # elif provider_type == 'outlook':
    #     return OutlookCalendarProvider(...)

    else:
        raise ValueError(f"Unknown calendar provider: {provider_type}")


__all__ = ["CalendarProvider", "CalDAVProvider", "get_calendar_provider"]
