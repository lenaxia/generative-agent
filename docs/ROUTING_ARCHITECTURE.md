# Routing Architecture: Phase 3 Fast-Reply vs Phase 4 Meta-Planning

**Date**: 2025-12-27
**Status**: Production
**Architecture**: Hybrid Phase 3 + Phase 4

---

## Overview

The Universal Agent System uses a **hybrid routing architecture** that intelligently directs requests to either fast-reply domain roles (Phase 3) or meta-planning workflows (Phase 4) based on complexity and confidence.

This document explains the routing logic, decision criteria, and when each pathway is used.

---

## Architecture Diagram

```
User Request
    ↓
Router (core_router.py)
    ↓
LLM Analysis → (role, confidence, parameters)
    ↓
    ├─────────────────────┬─────────────────────┐
    ↓                     ↓                     ↓
Confidence ≥ 0.95    0.70 ≤ Conf < 0.95   Confidence < 0.70
    ↓                     ↓                     ↓
Phase 3              Ambiguous           Phase 4
Fast-Reply           (Use judgment)      Meta-Planning
~600ms               Varies              8-16s
    ↓                     ↓                     ↓
Domain Role          May use either      Dynamic Agent
Direct execution     pathway             Structured planning
Single tool          Context-dependent   Multiple tools
```

---

## Phase 3: Fast-Reply Domain Roles

### When to Use

**Characteristics:**

- Single, clear intent
- Maps to one domain
- No multi-step orchestration needed
- User explicitly requests domain capability

**Confidence Threshold**: ≥ 0.95

**Examples:**

```
✅ "Set a timer for 5 minutes" → timer role
✅ "What's the weather in Seattle?" → weather role
✅ "Search for best pizza in Portland" → search role
✅ "Add event to my calendar" → calendar role
✅ "Turn on living room lights" → smart_home role
✅ "Tell me about quantum computing" → conversation role
```

### Execution Flow

1. **Router** identifies single domain role with high confidence
2. **UniversalAgent** assumes the domain role
3. **Tool execution** using role's specialized tools
4. **Fast response** (~600ms for simple operations)
5. **Return result** to user

### Advantages

- **Speed**: ~600ms response time
- **Efficiency**: Uses WEAK/DEFAULT LLM models
- **Cost**: Lower token usage
- **Specialized**: Role-specific system prompts
- **Direct**: No planning overhead

### Available Fast-Reply Roles

| Role             | Domain       | LLM Type | Description            |
| ---------------- | ------------ | -------- | ---------------------- |
| **timer**        | timer        | WEAK     | Set/cancel/list timers |
| **calendar**     | calendar     | DEFAULT  | Schedule events        |
| **weather**      | weather      | WEAK     | Weather queries        |
| **smart_home**   | smart_home   | DEFAULT  | Device control         |
| **search**       | search       | WEAK     | Web/news search        |
| **conversation** | conversation | DEFAULT  | Generic chat, Q&A      |

---

## Phase 4: Meta-Planning Workflows

### When to Use

**Characteristics:**

- Multi-domain coordination needed
- Multiple distinct operations
- Complex workflow or dependencies
- Requires orchestration between tools
- No single domain fits

**Confidence Threshold**: < 0.70

**Examples:**

```
✅ "Check weather in Seattle AND set timer for 10 minutes"
   → Needs: weather + timer

✅ "Search for restaurants in Portland, check the weather, and add to calendar"
   → Needs: search + weather + calendar

✅ "Find the latest AI news and summarize it"
   → Needs: search + summarization

✅ "Turn on lights, set timer, and check weather"
   → Needs: smart_home + timer + weather
```

### Execution Flow

1. **Router** identifies low confidence → routes to "planning"
2. **Meta-Planning** (plan_and_configure_agent):
   - LLM analyzes request (STRONG model)
   - Selects 2-3 tools from registry
   - Creates **ExecutionPlan** with structured steps ⭐
   - Adds **replan tool** for dynamic adjustment ⭐
3. **Agent Creation** (RuntimeAgentFactory):
   - Creates custom agent with selected tools
   - Provides execution plan in prompt ⭐
4. **Autonomous Execution**:
   - Agent follows execution plan steps
   - Can call replan() if steps fail ⭐
   - Executes 10-15 iterations
5. **Intent Processing**:
   - Collects intents from tool calls
   - Processes side effects (notifications, etc.)
6. **Return result** to user

### Advantages

- **Flexible**: Any tool combination
- **Intelligent**: LLM-driven tool selection
- **Adaptive**: Can replan on failures ⭐
- **Structured**: Step-by-step execution guidance ⭐
- **No code changes**: New workflows work automatically

### Performance

- **Latency**: 8-16 seconds end-to-end
- **Meta-planning**: ~5.5s (tool selection)
- **Execution**: ~11s (tool calls + synthesis)
- **Model**: STRONG for planning, DEFAULT/WEAK for execution
- **Tools per workflow**: 2-3 average

---

## Ambiguous Cases (0.70 ≤ Confidence < 0.95)

When confidence falls in the middle range, the system uses judgment:

### Factors Considered

1. **Request complexity**: Multiple verbs/actions?
2. **Domain overlap**: Requires multiple domains?
3. **Context**: History suggests complex workflow?
4. **User pattern**: Typically simple or complex requests?

### Resolution Strategy

- **Lean toward fast-reply** if single domain can handle it
- **Use meta-planning** if any orchestration needed
- **Router LLM** makes final decision based on analysis

### Examples

```
Confidence 0.85: "Set a timer for when the weather changes"
→ Complex condition → Meta-planning

Confidence 0.80: "What's the weather and should I bring an umbrella?"
→ Single domain can answer → Fast-reply (weather)

Confidence 0.75: "Search for pizza places and make a reservation"
→ Reservation needs additional tools → Meta-planning
```

---

## Routing Configuration

### Router Role (core_router.py)

```python
ROUTING_RULES:
- For multi-task requests → ALWAYS route to "planning"
- For single-task requests → Choose best matching role
- If confidence < 0.7 → Route to "planning"
- Consider role priorities: timer > weather > smart_home > search > planning
- Use lowercase role names
```

### Fast-Reply Configuration

**In domain role's `get_role_config()`:**

```python
return {
    "fast_reply": True,  # Enable fast-reply
    "llm_type": "WEAK",  # Or DEFAULT, STRONG
    "when_to_use": "Description of when to use this role"
}
```

### Meta-Planning Configuration

**In config.yaml:**

```yaml
feature_flags:
  enable_phase4_meta_planning: true # Enable Phase 4

fast_path:
  enabled: true
  confidence_threshold: 0.7 # Threshold for fast-reply
```

---

## Decision Matrix

| Request Type           | Confidence | Pathway | Latency | Tools    | Model          |
| ---------------------- | ---------- | ------- | ------- | -------- | -------------- |
| Single domain, clear   | ≥ 0.95     | Phase 3 | ~600ms  | 1        | WEAK/DEFAULT   |
| Single domain, unclear | 0.70-0.95  | Phase 3 | ~600ms  | 1        | WEAK/DEFAULT   |
| Multi-domain           | < 0.70     | Phase 4 | 8-16s   | 2-3      | STRONG+DEFAULT |
| Complex workflow       | < 0.70     | Phase 4 | 8-16s   | 2-4      | STRONG+DEFAULT |
| Ambiguous intent       | < 0.70     | Phase 4 | 8-16s   | Variable | STRONG+DEFAULT |

---

## Best Practices

### For Fast-Reply Roles

1. **Keep single-purpose**: One domain, one capability
2. **Use WEAK models**: For simple operations
3. **Specialized prompts**: Role-specific guidance
4. **Fast tools**: Optimize for speed

### For Meta-Planning

1. **Let LLM decide**: Trust tool selection
2. **Provide context**: Memory and conversation history
3. **Monitor plans**: Log execution plan IDs
4. **Enable replanning**: Add replan tool to toolset ⭐

### For Routing

1. **Clear when_to_use**: Help router classify correctly
2. **Descriptive parameters**: Extract from natural language
3. **Confidence tracking**: Monitor routing accuracy
4. **Fallback gracefully**: Handle low-confidence cases

---

## Monitoring and Debugging

### Fast-Reply Path

**Log markers:**

```
"✨ Using Phase 3 domain role: {role_name}"
"Fast-reply role execution"
```

**Metrics to track:**

- Response time (target: <1s)
- Tool usage count
- Success rate per role

### Meta-Planning Path

**Log markers:**

```
"Phase 4 meta-planning for request"
"Meta-planning complete: {n} tools selected"
"Created execution plan {plan_id} with {n} steps"  ⭐
"Using execution plan {plan_id}"  ⭐
```

**Metrics to track:**

- Meta-planning latency
- Tool selection accuracy
- Execution plan completion rate ⭐
- Replan invocation count ⭐
- End-to-end latency

---

## Migration Path

### Adding New Fast-Reply Role

1. Create domain role in `roles/{domain}/`
2. Set `fast_reply: True` in `get_role_config()`
3. Register with RoleRegistry (auto-discovery)
4. Update router's role descriptions

### Converting Tool to Fast-Reply Role

1. Assess if users directly invoke capability
2. Create role wrapper around tool
3. Add specialized system prompt
4. Test routing with sample requests

### Deprecating Fast-Reply Role

1. Remove `fast_reply: True` from config
2. Tools remain available to meta-planning
3. Router redirects to planning pathway
4. No code changes needed

---

## Future Enhancements

### Planned Improvements

1. **Adaptive thresholds**: Learn optimal confidence levels
2. **User preferences**: Remember user patterns
3. **Parallel execution**: Run independent steps concurrently
4. **Caching**: Cache meta-planning for similar requests
5. **Monitoring dashboard**: Real-time routing analytics

### Under Consideration

1. **Hybrid execution**: Start fast-reply, escalate if needed
2. **Cost optimization**: Prefer fast-reply when possible
3. **Quality scoring**: Track which pathway performs better
4. **Auto-routing**: Skip router for obvious cases

---

## Summary

The hybrid architecture provides:

- **✅ Speed** for simple requests (Phase 3 fast-reply)
- **✅ Flexibility** for complex workflows (Phase 4 meta-planning)
- **✅ Intelligent routing** based on confidence
- **✅ Structured execution** with planning and replanning ⭐
- **✅ Graceful degradation** on errors
- **✅ No breaking changes** to existing workflows

Both pathways coexist permanently, each serving its purpose in the system architecture.

---

**See also:**

- [EXECUTION_PLANNING_GUIDE.md](./EXECUTION_PLANNING_GUIDE.md) - How execution plans work
- [CLAUDE.md](../CLAUDE.md) - Complete development guide
- [Phase 4 documentation](../PHASE4_FINAL_STATUS.md) - Phase 4 implementation details
