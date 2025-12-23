# Core Tools

**DO NOT MODIFY THESE FILES**

These are system infrastructure tools maintained by the core system.
Modifying these tools may break core functionality.

For custom tools, use `tools/custom/` instead.

## Available Core Tools

### Memory (`memory.py`)
Memory storage and retrieval infrastructure.

**Tools:**
- `search_memory()` - Search unified memory across all types
- `get_recent_memories()` - Get recent memories, optionally filtered by type

**Used by:** Calendar, Conversation, Planning roles

### Notification (`notification.py`)
Notification dispatch infrastructure.

**Tools:**
- `send_notification()` - Send notification to user via configured channels

**Used by:** Timer, Calendar, Smart Home roles

## Architecture

Core tools follow this pattern:
1. `create_*_tools(provider)` - Factory function that receives infrastructure provider
2. Global provider reference stored for tool functions
3. Tools decorated with `@tool` from strands framework
4. Query tools (read-only) vs Action tools (side effects) clearly marked

## Adding Core Tools

To add a new core tool:
1. Create `tools/core/your_tool.py`
2. Follow the existing pattern (factory function + global provider)
3. Update `tools/core/__init__.py` to export
4. Update `llm_provider/tool_registry.py` to load
5. Document in this README

**Remember:** Core tools should only be added for system infrastructure, not domain-specific functionality. Domain-specific tools belong in domain roles (`roles/{domain}/tools.py`).
