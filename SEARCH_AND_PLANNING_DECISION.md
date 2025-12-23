# Search and Planning - Migration Decision Needed

**Status:** Tools reorganization complete, but search and planning need decisions

---

## Summary of Completed Work ✅

**Completed:**
- ✅ Created `tools/core/` for infrastructure tools
- ✅ Created `tools/custom/` for user-extensible tools
- ✅ Moved `roles/memory/` → `tools/core/memory.py`
- ✅ Moved `roles/notification/` → `tools/core/notification.py`
- ✅ Updated ToolRegistry to load from new paths
- ✅ Removed old directories
- ✅ Created documentation and examples
- ✅ Tested imports - all working

**Result:**
```
tools/
├── core/
│   ├── memory.py          ✅ Working
│   ├── notification.py    ✅ Working
│   └── README.md
└── custom/
    ├── example.py         ✅ Working
    └── README.md
```

---

## Remaining Decisions

### 1. Search Role 🔍

**Current State:**
- `roles/core_search.py` - 424 lines (legacy single-file role)
- `roles/search/tools.py` - 192 lines (domain tools only)
- **CONFLICT:** Both exist!

**core_search.py contains:**
- Complete role implementation
- Fast-reply configuration
- Tavily web search integration
- News search functionality

**roles/search/tools.py contains:**
- Just the tool implementations
- `web_search()` and `search_news()` functions
- Extracted from core_search.py

**Decision Options:**

#### Option A: Migrate to Full Domain Role ⭐ **RECOMMENDED**
```
1. Create roles/search/role.py (from core_search.py)
2. Create roles/search/handlers.py (extract handlers)
3. Keep roles/search/tools.py (already exists)
4. Delete roles/core_search.py

Result: Consistent with other domain roles (timer, calendar, weather, smart_home)
```

**Pros:**
- ✅ Consistent architecture
- ✅ Users can invoke search directly
- ✅ Follows Phase 3 pattern
- ✅ Tools already extracted

**Cons:**
- ⚠️ Requires migration work (~2 hours)
- ⚠️ Need to test search functionality

#### Option B: Keep Legacy Role
```
1. Keep roles/core_search.py as-is
2. Remove roles/search/ directory
3. Update ToolRegistry to not load search tools separately

Result: Search stays as legacy system role
```

**Pros:**
- ✅ No migration work
- ✅ Already working

**Cons:**
- ❌ Inconsistent with new architecture
- ❌ Duplicate code (tools exist twice)
- ❌ Technical debt

**Recommendation:** **Option A - Migrate**

This is consistent with timer/calendar/weather/smart_home migrations. Search is a domain role (users say "search for X"), not infrastructure.

---

### 2. Planning Role 📋

**Current State:**
- `roles/planning/tools.py` - Empty placeholder (Phase 4)
- `roles/core_planning.py` - **DELETED** during previous migration
- No current planning role

**Question:** Where did planning logic go after deleting `core_planning.py`?

**Investigation Needed:**
```bash
# Check if planning is in WorkflowEngine
grep -n "planning\|plan_task\|create_task_graph" supervisor/workflow_engine.py

# Check git history
git log --oneline --all -- roles/core_planning.py | head -5
```

**Decision Options:**

#### Option A: Planning is Now in WorkflowEngine
If planning logic moved to WorkflowEngine:
```
Action: Remove roles/planning/ directory entirely
Reason: Planning is orchestration, not a domain role
```

#### Option B: Recreate Planning as Domain Role
If planning should be user-facing (Doc 65 meta-planning):
```
Action: Create roles/planning/{role.py, handlers.py, tools.py}
Purpose: Meta-planning agent for dynamic agent creation
Example: User: "plan a trip to Thailand"
```

#### Option C: Planning is Future Work
If planning is Phase 4 placeholder:
```
Action: Keep roles/planning/ as empty placeholder
Status: Wait for Phase 4 implementation
```

**Recommendation:** **Investigate first, then decide**

Check where planning logic went, then choose based on findings.

---

## Recommended Next Steps

### Immediate (Do Now)

1. **Search Migration** (~2 hours)
   ```bash
   # Migrate core_search.py to domain role pattern
   # Following same approach as timer/calendar/weather
   ```

2. **Planning Investigation** (~30 minutes)
   ```bash
   # Find where planning logic went
   # Decide whether to remove directory or recreate role
   ```

### After Migration

3. **Test Search Functionality**
   ```bash
   python3 cli.py
   > search for best pizza in Seattle
   # Should use domain search role
   ```

4. **Update Documentation**
   - Document tools/ structure in README
   - Update architecture diagrams
   - Create migration guide for future roles

5. **Clean Up Test Files**
   - Update any tests referencing old paths
   - Remove obsolete test files

---

## Current Directory Structure

```
generative-agent/
├── tools/                      ← NEW
│   ├── core/                   ← Infrastructure tools
│   │   ├── memory.py          ✅ Complete
│   │   ├── notification.py    ✅ Complete
│   │   └── README.md
│   └── custom/                 ← User tools
│       ├── example.py         ✅ Complete
│       └── README.md
│
├── roles/
│   ├── calendar/              ✅ Full domain role
│   ├── timer/                 ✅ Full domain role
│   ├── weather/               ✅ Full domain role
│   ├── smart_home/            ✅ Full domain role
│   │
│   ├── search/                🤔 Decision needed
│   │   └── tools.py           (tools only, but core_search.py also exists)
│   │
│   ├── planning/              🤔 Decision needed
│   │   └── tools.py           (empty placeholder)
│   │
│   ├── core_search.py         🚨 Migrate or remove
│   ├── core_conversation.py   ✅ Keep (system service)
│   ├── core_router.py         ✅ Keep (system service)
│   ├── core_summarizer.py     ✅ Keep (system service)
│   │
│   └── shared_tools/          ✅ Keep (lifecycle helpers)
```

---

## Questions for User

1. **Search Role:** Should I migrate `core_search.py` to domain role pattern (like timer/calendar/weather)?

2. **Planning Role:** What happened to planning logic after `core_planning.py` was deleted? Should planning be recreated as a domain role for meta-planning, or is it in WorkflowEngine now?

3. **Timeline:** If migrating search, when should I do it? (I can do it now or after other priorities)

---

## Benefits of Completed Work

**For Users:**
- ✅ Clear `tools/custom/` location for adding custom tools
- ✅ Comprehensive documentation with examples
- ✅ Example templates to copy from

**For Developers:**
- ✅ Clean separation: roles vs infrastructure tools
- ✅ Consistent architecture patterns
- ✅ Easier to find and modify code

**For System:**
- ✅ `roles/` directory now only contains actual roles
- ✅ Core infrastructure grouped in `tools/core/`
- ✅ Extensibility built in with `tools/custom/`

---

**Awaiting decisions on search and planning before proceeding.**
