# Roles Directory

This directory contains all AI agent roles for the Universal Agent System. Each role is implemented as a single Python file following the single-file role architecture.

## Current Roles

### Core Roles (Single-File Architecture)

All roles follow the pattern: `core_<role_name>.py`

1. **Router** ([`core_router.py`](core_router.py))

   - Intelligent request routing with context selection
   - Determines which role should handle each request
   - Selects which context to gather (location, memory, presence, schedule)
   - 95%+ confidence for fast-reply routing

2. **Planning** ([`core_planning.py`](core_planning.py))

   - Complex task planning and workflow generation
   - Breaks down multi-step tasks into executable workflows
   - Creates WorkflowIntents with task graphs
   - Uses STRONG LLM model for complex reasoning

3. **Conversation** ([`core_conversation.py`](core_conversation.py))

   - General conversation with memory
   - Topic-based conversation tracking
   - Memory importance assessment
   - Context-aware responses

4. **Timer** ([`core_timer.py`](core_timer.py))

   - Timer, alarm, and reminder management
   - Heartbeat-driven expiry checking
   - Redis-backed persistence
   - Intent-based notifications

5. **Weather** ([`core_weather.py`](core_weather.py))

   - Weather information and forecasts
   - Location-aware queries
   - MCP integration for weather data

6. **Smart Home** ([`core_smart_home.py`](core_smart_home.py))

   - Device control via Home Assistant MCP
   - Location-aware device commands
   - Scene and automation support

7. **Calendar** ([`core_calendar.py`](core_calendar.py))

   - Calendar and scheduling
   - Event management
   - Time-based queries

8. **Search** ([`core_search.py`](core_search.py))

   - Web search capabilities
   - MCP integration for search providers

9. **Summarizer** ([`core_summarizer.py`](core_summarizer.py))
   - Synthesize and analyze information from multiple sources
   - Create structured outputs (summaries, reports, itineraries, analysis)
   - Consolidate predecessor task results
   - Factual, concise presentation without conversational elements

## Role Structure

Each role file contains:

```python
# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "role_name",
    "version": "1.0.0",
    "description": "What this role does",
    "llm_type": "WEAK|DEFAULT|STRONG",
    "fast_reply": True|False,
    "when_to_use": "When to use this role"
}

# 2. TOOLS (optional)
@tool
def my_tool(param: str) -> dict:
    """Tool function."""
    return {"result": param}

# 3. EVENT HANDLERS (optional)
def handle_event(event_data, context) -> list[Intent]:
    """Pure function returning intents."""
    return [NotificationIntent(...)]

# 4. LIFECYCLE FUNCTIONS (optional)
def pre_processing(instruction, context, parameters):
    """Pre-processing before LLM call."""
    return {"extra_context": "data"}

def post_processing(llm_result, context, pre_data):
    """Post-processing after LLM call."""
    return llm_result

# 5. ROLE REGISTRATION
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {...},
        "tools": [...],
        "intents": [...]
    }
```

## Shared Tools

### Redis Tools ([`shared_tools/redis_tools.py`](shared_tools/redis_tools.py))

Shared Redis utilities used by multiple roles:

- `redis_read()` - Read from Redis
- `redis_write()` - Write to Redis
- `redis_get_keys()` - Get keys matching pattern
- `_get_redis_client()` - Get Redis client instance

Used by:

- Timer role (timer persistence)
- Conversation role (memory storage)
- Context providers (location, memory)

## Adding New Roles

1. Create `core_<name>.py` in this directory
2. Follow the single-file role structure above
3. Implement required functions
4. Role will be auto-discovered by RoleRegistry

See main README.md for detailed role creation guide.

## Role Discovery

Roles are automatically discovered by RoleRegistry using:

- Pattern: `core_*.py`
- Auto-discovery: Enabled by default
- Validation: Structure validated on load

## Architecture Notes

### Single-File Design

- Each role is ~300 lines (vs 1800+ in old architecture)
- All role code in one file for easy understanding
- No complex directory structures
- LLM-friendly for AI-assisted development

### Intent-Based Processing

- Event handlers return declarative intents
- No direct execution in handlers
- Pure functions for LLM-safe architecture
- IntentProcessor handles execution

### Fast-Reply Routing

- Router determines role and confidence
- High confidence (>95%) = instant execution
- Low confidence = escalate to planning role
- Context selection based on role needs

## Migration Complete

The old multi-file role architecture has been fully migrated to single-file roles. The `archived/` folder has been removed as it contained only obsolete code from the previous architecture.
