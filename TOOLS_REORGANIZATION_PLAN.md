# Tools Reorganization Plan

**Date:** 2025-12-22
**Goal:** Separate infrastructure tools from domain roles
**Status:** Planning → Implementation

---

## Problem Statement

Currently we have "tool-only domains" in `roles/` that aren't actually roles:
- `roles/memory/` - Infrastructure for memory storage
- `roles/notification/` - Infrastructure for notifications
- `roles/planning/` - Empty placeholder
- `roles/search/` - Web search tools (but `core_search.py` also exists)

**Issue:** The `roles/` directory should only contain actual roles (agents), not utility tools.

---

## New Architecture

```
generative-agent/
├── tools/
│   ├── core/                    ← Core infrastructure (don't modify)
│   │   ├── __init__.py
│   │   ├── memory.py           ← From roles/memory/tools.py
│   │   ├── notification.py     ← From roles/notification/tools.py
│   │   └── README.md           ← "Do not modify core tools"
│   │
│   └── custom/                  ← User-extensible
│       ├── __init__.py
│       ├── README.md           ← Instructions for adding custom tools
│       └── example.py          ← Template for custom tools
│
├── roles/
│   ├── calendar/               ← Domain roles ONLY
│   ├── timer/
│   ├── weather/
│   ├── smart_home/
│   ├── search/                 ← Decision: Make full role OR remove
│   │   ├── role.py            ← If keeping as role
│   │   ├── handlers.py
│   │   └── tools.py
│   │
│   └── shared_tools/           ← Keep (lifecycle helpers)
│       ├── conversation_analysis.py
│       ├── lifecycle_helpers.py
│       ├── memory_tools.py
│       └── redis_tools.py
│
└── roles_old/                   ← Legacy single-file roles
    ├── core_router.py
    ├── core_search.py          ← Decision: Migrate or remove
    ├── core_summarizer.py
    └── core_conversation.py
```

---

## Analysis of Current Tool-Only Domains

### 1. Memory (`roles/memory/`)

**Current:**
- `tools.py` with 2 tools: `search_memory`, `get_recent_memories`
- Used BY other roles (calendar, conversation, etc.)
- Infrastructure for memory storage/retrieval

**Decision:** ✅ **Move to `tools/core/memory.py`**

**Reasoning:**
- Infrastructure tool, not a domain role
- Users don't say "hey assistant, memory something"
- Used by multiple roles as a utility
- Core system functionality

### 2. Notification (`roles/notification/`)

**Current:**
- `tools.py` with 1 tool: `send_notification`
- Used BY other roles (timer, calendar, smart_home)
- Infrastructure for dispatching notifications

**Decision:** ✅ **Move to `tools/core/notification.py`**

**Reasoning:**
- Infrastructure tool, not a domain role
- Users don't invoke directly
- Service layer for sending messages
- Core system functionality

### 3. Planning (`roles/planning/`)

**Current:**
- `tools.py` - Empty placeholder (Phase 4)
- Previously had `core_planning.py` (deleted during migration)

**Decision:** 🤔 **Need to investigate**

**Questions:**
1. Where did planning logic go after deleting `core_planning.py`?
2. Is it in WorkflowEngine now?
3. Should planning become a full domain role?

**If it's for meta-planning (Document 65):**
- User: "plan a trip to Thailand"
- Router → PlanningRole
- PlanningRole creates agent configuration
- **This SHOULD be a full domain role**

**Action:** Investigate and decide after memory/notification migration.

### 4. Search (`roles/search/`)

**Current:**
- `roles/search/tools.py` - Has `web_search`, `search_news` tools
- `roles/core_search.py` - Legacy single-file role
- **CONFLICT:** Both exist!

**Decision:** 🚨 **CRITICAL - Needs decision**

**Options:**

**Option A: Make Search a Full Domain Role**
```
User: "search for best pizza in Seattle"
→ Router selects: search
→ SearchRole executes with web_search tools
```

**Option B: Search is Infrastructure**
```
User: "search for best pizza in Seattle"
→ Router selects: conversation
→ ConversationRole uses web_search tool
```

**Recommendation:** **Option A - Full Domain Role**

**Reasoning:**
- Users explicitly request searches
- Search has specialized behavior
- `core_search.py` already exists as a role
- Should migrate `core_search.py` → `roles/search/{role.py,handlers.py}`

---

## Migration Steps

### Phase 1: Create New Structure ✅

**Step 1.1:** Create `tools/core/` directory
```bash
mkdir -p tools/core
touch tools/core/__init__.py
```

**Step 1.2:** Create `tools/custom/` directory
```bash
mkdir -p tools/custom
touch tools/custom/__init__.py
```

### Phase 2: Move Memory Tools

**Step 2.1:** Move memory tools
```bash
mv roles/memory/tools.py tools/core/memory.py
```

**Step 2.2:** Update imports in moved file
- Change module path references
- Update docstrings

**Step 2.3:** Remove old directory
```bash
rm -rf roles/memory/
```

### Phase 3: Move Notification Tools

**Step 3.1:** Move notification tools
```bash
mv roles/notification/tools.py tools/core/notification.py
```

**Step 3.2:** Update imports in moved file

**Step 3.3:** Remove old directory
```bash
rm -rf roles/notification/
```

### Phase 4: Update ToolRegistry

**File:** `llm_provider/tool_registry.py`

**Changes:**
```python
# OLD
domain_modules = [
    ("memory", "roles.memory.tools", providers.memory),
    ("notification", "roles.notification.tools", providers.communication),
]

# NEW
core_tool_modules = [
    ("memory", "tools.core.memory", providers.memory),
    ("notification", "tools.core.notification", providers.communication),
]

# Add custom tool discovery
custom_tool_modules = self._discover_custom_tools()
```

### Phase 5: Handle Search Role

**Decision Required:** Migrate `core_search.py` to domain role pattern

**If migrating:**
1. Create `roles/search/role.py` (from core_search.py)
2. Create `roles/search/handlers.py` (extract handlers)
3. Keep `roles/search/tools.py` (already exists)
4. Delete `roles/core_search.py`

### Phase 6: Handle Planning

**Decision Required:** Investigate where planning logic went

**If recreating as domain role:**
1. Create `roles/planning/role.py`
2. Create `roles/planning/handlers.py`
3. Create `roles/planning/tools.py` (currently empty)

**If planning is in WorkflowEngine:**
- Remove `roles/planning/` entirely
- Planning is orchestration, not a domain

### Phase 7: Documentation

**Create `tools/core/README.md`:**
```markdown
# Core Tools

These are system infrastructure tools. **DO NOT MODIFY.**

Core tools are maintained by the system and should not be changed.
For custom tools, use `tools/custom/`.

## Available Core Tools

- `memory.py` - Memory storage and retrieval
- `notification.py` - Notification dispatch
```

**Create `tools/custom/README.md`:**
```markdown
# Custom Tools

Add your own tools here for the agent to use.

## How to Add a Custom Tool

1. Create a new Python file in this directory
2. Use the `@tool` decorator from strands
3. Register in `__init__.py`

See `example.py` for a template.
```

**Create `tools/custom/example.py`:**
```python
"""Example custom tool - delete or modify this."""

from strands import tool

@tool
def my_custom_tool(input: str) -> dict:
    """Example custom tool.

    Args:
        input: Some input string

    Returns:
        Dict with result
    """
    return {
        "success": True,
        "result": f"Processed: {input}"
    }
```

---

## Impact Analysis

### Files to Modify

1. **llm_provider/tool_registry.py**
   - Change: Update module paths for memory/notification
   - Risk: Medium (core system component)
   - Testing: Integration tests

2. **Any files importing memory/notification tools**
   - Change: Update import paths
   - Risk: Low (Python will error if wrong)
   - Testing: Run full test suite

3. **roles/search/** (if migrating)
   - Change: Create role.py and handlers.py
   - Risk: Medium (active role)
   - Testing: Search functionality tests

### Files to Create

- `tools/core/__init__.py`
- `tools/core/memory.py` (moved)
- `tools/core/notification.py` (moved)
- `tools/core/README.md` (new)
- `tools/custom/__init__.py`
- `tools/custom/README.md` (new)
- `tools/custom/example.py` (new)

### Files to Delete

- `roles/memory/` (entire directory)
- `roles/notification/` (entire directory)
- `roles/planning/` (TBD based on investigation)
- `roles/core_search.py` (if migrating to domain role)

---

## Benefits

### For Users
✅ Clear location for custom tools (`tools/custom/`)
✅ Simple instructions for adding tools
✅ Separation of system vs user tools

### For Developers
✅ Clear architecture (roles vs tools)
✅ Infrastructure tools grouped together
✅ Easier to understand what can/cannot be modified

### For System
✅ Cleaner `roles/` directory (only actual roles)
✅ Proper separation of concerns
✅ More maintainable structure

---

## Risks & Mitigations

### Risk 1: Import Path Changes
**Risk:** Breaking imports across the codebase
**Mitigation:**
- Use grep to find all imports
- Update systematically
- Run full test suite

### Risk 2: ToolRegistry Changes
**Risk:** Breaking tool loading mechanism
**Mitigation:**
- Test with existing tools first
- Gradual migration (one tool type at a time)
- Validation tests

### Risk 3: Search Role Migration
**Risk:** Breaking existing search functionality
**Mitigation:**
- Keep `core_search.py` until migration complete
- Test thoroughly before deleting
- Have rollback plan

---

## Decision Points

### 1. Search Role Migration 🚨
**Question:** Migrate `core_search.py` to domain role or keep as-is?
**Recommendation:** Migrate (consistent with other domain roles)
**Timeline:** After memory/notification migration

### 2. Planning Role 🤔
**Question:** Recreate as domain role or remove entirely?
**Depends On:** Where planning logic currently lives
**Timeline:** Requires investigation first

### 3. Shared Tools
**Question:** Keep `roles/shared_tools/` or move to `tools/shared/`?
**Recommendation:** Keep (different purpose - lifecycle helpers for roles)
**Reasoning:** These are role-specific utilities, not general tools

---

## Success Criteria

- ✅ `tools/core/` contains infrastructure tools
- ✅ `tools/custom/` ready for user tools
- ✅ `roles/` contains ONLY actual roles
- ✅ All imports updated and working
- ✅ ToolRegistry loads from new locations
- ✅ All tests pass
- ✅ Documentation created

---

## Timeline Estimate

- Phase 1 (Structure): 15 minutes
- Phase 2 (Memory): 30 minutes
- Phase 3 (Notification): 30 minutes
- Phase 4 (ToolRegistry): 1 hour
- Phase 5 (Search): 2 hours (if migrating)
- Phase 6 (Planning): 1 hour (investigation + decision)
- Phase 7 (Documentation): 30 minutes

**Total:** ~6 hours for complete migration

---

## Next Steps

1. ✅ Get user approval on this plan
2. 🚀 Start with Phase 1-3 (low risk, high value)
3. 🔍 Investigate planning role status
4. 🤔 Decide on search role migration
5. ✅ Complete remaining phases

---

**Awaiting approval to proceed with implementation.**
