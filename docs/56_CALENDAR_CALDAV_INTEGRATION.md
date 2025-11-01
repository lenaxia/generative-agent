# Calendar CalDAV Integration

## Overview

This document describes the CalDAV calendar integration added to the Universal Agent System. The implementation provides a clean provider abstraction layer that supports CalDAV-compatible calendar services (iCloud, Nextcloud, self-hosted) with the ability to add Google Calendar and Outlook support in the future.

## Architecture

### Provider Abstraction Layer

The calendar integration uses a provider abstraction pattern that separates the calendar role logic from the underlying calendar service implementation:

```
roles/core_calendar.py (User Interface)
         ↓
common/calendar_providers/ (Abstraction Layer)
    ├── base.py (CalendarProvider interface)
    ├── caldav_provider.py (CalDAV implementation)
    └── __init__.py (Factory function)
```

### Key Components

1. **CalendarProvider (Abstract Base Class)**

   - Defines interface all providers must implement
   - Methods: authenticate(), is_authenticated(), get_events(), add_event(), update_event(), delete_event()
   - Standardized event format across all providers

2. **CalDAVProvider**

   - Implements CalendarProvider for CalDAV protocol
   - Supports iCloud, Nextcloud, and any CalDAV-compatible server
   - Uses `caldav` and `icalendar` Python libraries

3. **Calendar Role**
   - Updated to use provider abstraction
   - Tools: get_schedule(), add_calendar_event()
   - Context-aware with location and memory integration

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install calendar dependencies
pip install -r requirements.txt
```

This installs:

- `caldav>=1.3.9` - CalDAV protocol client
- `icalendar>=5.0.11` - iCalendar format parsing
- `pytz>=2024.1` - Timezone support

### 2. Configure Calendar Provider

#### For iCloud Calendar

1. Go to [appleid.apple.com](https://appleid.apple.com)
2. Sign in with your Apple ID
3. Go to "Security" section
4. Under "App-Specific Passwords", click "Generate Password"
5. Enter a label (e.g., "Universal Agent Calendar")
6. Copy the generated password

Add to your environment or `.env` file:

```bash
export CALDAV_URL="https://caldav.icloud.com"
export CALDAV_USERNAME="your-email@icloud.com"
export CALDAV_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # App-specific password
```

#### For Nextcloud

```bash
export CALDAV_URL="https://your-nextcloud.com/remote.php/dav"
export CALDAV_USERNAME="your-username"
export CALDAV_PASSWORD="your-password"
```

#### For Other CalDAV Servers

```bash
export CALDAV_URL="https://your-caldav-server.com"
export CALDAV_USERNAME="your-username"
export CALDAV_PASSWORD="your-password"
```

### 3. Update config.yaml (Optional)

The calendar configuration is already in `config.yaml`:

```yaml
calendar:
  enabled: true
  provider: "caldav"

  caldav:
    url: "${CALDAV_URL}"
    username: "${CALDAV_USERNAME}"
    password: "${CALDAV_PASSWORD}"
```

## Usage

### Getting Schedule

```python
# Via CLI
python cli.py
> Get my schedule for the next 7 days

# Via calendar role tool
get_schedule(user_id="user123", days_ahead=7)
```

### Adding Events

```python
# Via CLI
python cli.py
> Add a meeting tomorrow at 2pm for 1 hour

# Via calendar role tool
add_calendar_event(
    title="Team Meeting",
    start_time="2024-03-15T14:00:00",
    duration=60,
    location="Conference Room A"
)
```

## Testing

### Run Calendar Provider Tests

```bash
# Run all calendar tests
python -m pytest tests/unit/test_calendar_providers.py -v

# Run specific test
python -m pytest tests/unit/test_calendar_providers.py::TestCalDAVProvider::test_initialization -v
```

### Test Coverage

- Factory pattern tests (3 tests)
- CalDAV provider tests (7 tests)
- Interface compliance tests (2 tests)

**Note**: Some tests require `caldav` and `icalendar` libraries installed. Install with `pip install -r requirements.txt`.

## Implementation Details

### Event Format

All providers return events in a standardized format:

```python
{
    'id': str,              # Unique event identifier
    'title': str,           # Event title/summary
    'start': datetime,      # Start time (timezone-aware)
    'end': datetime,        # End time (timezone-aware)
    'location': str | None, # Event location (optional)
    'description': str | None  # Event description (optional)
}
```

### Error Handling

The CalDAV provider handles common errors:

- **Import Error**: Returns False if `caldav` library not installed
- **Authentication Error**: Logs error and returns False
- **No Calendars**: Returns False if no calendars found on server
- **Network Errors**: Caught and logged, returns empty list or raises exception

### Lazy Imports

The CalDAV provider uses lazy imports to avoid requiring the libraries when not in use:

```python
def authenticate(self) -> bool:
    try:
        from caldav import DAVClient  # Import only when needed
        # ... authentication logic
    except ImportError:
        logger.error("caldav library not installed")
        return False
```

## Future Enhancements

### Adding Google Calendar Support

1. Create `common/calendar_providers/google_provider.py`
2. Implement `CalendarProvider` interface
3. Use `google-api-python-client` library
4. Add OAuth2 authentication flow
5. Update factory in `__init__.py`

### Adding Outlook Support

1. Create `common/calendar_providers/outlook_provider.py`
2. Implement `CalendarProvider` interface
3. Use `O365` library
4. Add OAuth2 authentication flow
5. Update factory in `__init__.py`

### Planned Features

- **Update Events**: Implement update_event() in CalDAVProvider
- **Delete Events**: Implement delete_event() in CalDAVProvider
- **Recurring Events**: Support for recurring event patterns
- **Event Reminders**: Integration with timer role for event notifications
- **Multiple Calendars**: Support for selecting specific calendars
- **Free/Busy Queries**: Check availability for scheduling

## Troubleshooting

### "No module named 'caldav'"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### "No calendars found"

**Possible causes**:

1. Incorrect CalDAV URL
2. Invalid credentials
3. No calendars in account

**Solution**: Verify credentials and URL, check calendar exists in web interface

### "Authentication failed"

**For iCloud**:

- Ensure you're using an app-specific password, not your Apple ID password
- Generate new app-specific password if needed

**For Nextcloud**:

- Verify username and password
- Check CalDAV URL format: `https://your-nextcloud.com/remote.php/dav`

### Import errors in tests

**Solution**: The tests that require `caldav` and `icalendar` will fail until you install the dependencies. This is expected. Install with:

```bash
pip install caldav icalendar pytz
```

## Files Modified/Created

### New Files

- `common/calendar_providers/__init__.py` - Provider factory
- `common/calendar_providers/base.py` - Abstract base class
- `common/calendar_providers/caldav_provider.py` - CalDAV implementation
- `tests/unit/test_calendar_providers.py` - Comprehensive tests
- `docs/56_CALENDAR_CALDAV_INTEGRATION.md` - This document

### Modified Files

- `roles/core_calendar.py` - Updated to use provider abstraction
- `requirements.txt` - Added caldav, icalendar, pytz
- `config.yaml` - Added calendar configuration section

## References

- CalDAV Protocol: [RFC 4791](https://tools.ietf.org/html/rfc4791)
- iCalendar Format: [RFC 5545](https://tools.ietf.org/html/rfc5545)
- Python caldav library: [https://github.com/python-caldav/caldav](https://github.com/python-caldav/caldav)
- Python icalendar library: [https://github.com/collective/icalendar](https://github.com/collective/icalendar)

## Summary

The CalDAV calendar integration provides:

- ✅ Clean provider abstraction for future extensibility
- ✅ CalDAV support (iCloud, Nextcloud, self-hosted)
- ✅ Standardized event format across providers
- ✅ Comprehensive test coverage
- ✅ Simple configuration via environment variables
- ✅ No OAuth2 complexity (for CalDAV)
- ✅ Ready for Google Calendar and Outlook additions

The implementation follows the single-file role architecture and maintains consistency with the Universal Agent System's design principles.
