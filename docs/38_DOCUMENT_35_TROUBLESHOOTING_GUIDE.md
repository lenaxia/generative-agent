# Document 35 Troubleshooting Guide

**Document ID:** 38
**Created:** 2025-10-28
**Status:** Production Support
**Priority:** Medium
**Context:** Document 35 Intent-Based Workflow Lifecycle Management

## Rules

- Regularly run `make lint` to validate that your code is healthy
- Always use the venv at ./venv/bin/activate
- ALWAYS use test driven development, write tests first
- Never assume tests pass, run the tests and positively verify that the test passed
- ALWAYS run all tests after making any change to ensure they are still all passing, do not move on until relevant tests are passing
- If a test fails, reflect deeply about why the test failed and fix it or fix the code
- Always write multiple tests, including happy, unhappy path and corner cases
- Always verify interfaces and data structures before writing code, do not assume the definition of a interface or data structure
- When performing refactors, ALWAYS use grep to find all instances that need to be refactored
- If you are stuck in a debugging cycle and can't seem to make forward progress, either ask for user input or take a step back and reflect on the broader scope of the code you're working on
- ALWAYS make sure your tests are meaningful, do not mock excessively, only mock where ABSOLUTELY necessary.
- Make a git commit after major changes have been completed
- When refactoring an object, refactor it in place, do not create a new file just for the sake of preserving the old version, we have git for that reason. For instance, if refactoring RequestManager, do NOT create an EnhancedRequestManager, just refactor or rewrite RequestManager
- ALWAYS Follow development and language best practices
- Use the Context7 MCP server if you need documentation for something, make sure you're looking at the right version
- Remember we are migrating AWAY from langchain TO strands agent
- Do not worry about backwards compatibility unless it is PART of a migration process and you will remove the backwards compatibility later
- Do not use fallbacks. Fallbacks tend to be brittle and fragile. Do implement fallbacks of any kind.
- Whenever you complete a phase, make sure to update this checklist
- Don't just blindly implement changes. Reflect on them to make sure they make sense within the larger project. Pull in other files if additional context is needed
- When you complete the implementation of a project add new todo items addressing outstanding technical debt related to what you just implemented, such as removing old code, updating documentation, searching for additional references, etc. Fix these issues, do not accept technical debt for the project being implemented.

## Executive Summary

This guide provides troubleshooting information for Document 35 intent-based workflow lifecycle management implementation. It covers common issues, diagnostic procedures, and resolution strategies for the new architecture.

## Architecture Overview

### Document 35 Implementation

- **Planning Role**: Creates WorkflowExecutionIntent (pure function, LLM-safe)
- **Universal Agent**: Detects intents and schedules processing (no asyncio)
- **Intent Processor**: Processes WorkflowExecutionIntent via scheduled tasks
- **Communication Manager**: Event-driven lifecycle tracking
- **Workflow Engine**: Executes workflows via scheduled tasks with event publishing

### Key Components

- **WorkflowExecutionIntent**: Declarative intent for multi-step workflows
- **Scheduled Tasks**: LLM-safe execution without asyncio or background threads
- **Event-Driven Lifecycle**: Automatic request tracking through workflow events
- **Pure Function Architecture**: Stateless, predictable functions throughout

## Common Issues and Solutions

### 1. Planning Role Issues

#### Issue: "Cannot create WorkflowExecutionIntent from error message"

**Symptoms:**

```
ValueError: Cannot create WorkflowExecutionIntent from error message: Invalid role references: Task 'task_1' uses unavailable role 'nonexistent'
```

**Root Cause:** Planning role received validation error message instead of valid JSON

**Diagnosis:**

1. Check if `validate_task_graph` is returning error messages
2. Verify role references in TaskGraph JSON
3. Check available roles in role registry

**Resolution:**

```python
# Check available roles
from llm_provider.role_registry import RoleRegistry
role_registry = RoleRegistry.get_global_registry()
available_roles = list(role_registry.roles.keys())
print(f"Available roles: {available_roles}")

# Verify TaskGraph uses only available roles
```

#### Issue: "Invalid JSON in TaskGraph"

**Symptoms:**

```
ValueError: Invalid JSON in TaskGraph: Expecting value: line 1 column 1 (char 0)
```

**Root Cause:** LLM generated invalid JSON or mixed content without extractable JSON

**Diagnosis:**

1. Check LLM output for JSON structure
2. Verify JSON extraction regex is working
3. Check for malformed JSON syntax

**Resolution:**

```python
# Test JSON extraction manually
import re
import json

mixed_content = "Your LLM output here"
json_match = re.search(r'\{.*\}', mixed_content, re.DOTALL)
if json_match:
    clean_json = json_match.group(0)
    try:
        parsed = json.loads(clean_json)
        print("JSON extraction successful")
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
else:
    print("No JSON found in content")
```

#### Issue: WorkflowExecutionIntent validation fails

**Symptoms:**

```python
intent.validate() returns False
```

**Root Cause:** Intent missing required fields or has invalid structure

**Diagnosis:**

```python
# Check intent structure
print(f"Request ID: {intent.request_id}")
print(f"Tasks: {len(intent.tasks)}")
print(f"Task roles: {[task.get('role') for task in intent.tasks]}")
print(f"Task IDs: {[task.get('id') for task in intent.tasks]}")
```

**Resolution:**

- Ensure request_id is not empty
- Ensure tasks array is not empty
- Ensure all tasks have 'id' and 'role' fields

### 2. Universal Agent Issues

#### Issue: "Workflow planned but cannot execute - no supervisor available"

**Symptoms:**

```
"Workflow planned but cannot execute - no supervisor available"
```

**Root Cause:** Universal agent doesn't have supervisor reference for scheduling

**Diagnosis:**

```python
# Check supervisor reference
print(f"Universal agent has supervisor: {hasattr(universal_agent, 'supervisor')}")
print(f"Supervisor value: {getattr(universal_agent, 'supervisor', 'NOT_SET')}")
```

**Resolution:**

```python
# Set supervisor reference during initialization
universal_agent.supervisor = supervisor_instance
universal_agent.intent_processor = intent_processor_instance
```

#### Issue: "Workflow planned but cannot execute - no intent processor"

**Symptoms:**

```
"Workflow planned but cannot execute - no intent processor available"
```

**Root Cause:** Universal agent doesn't have intent processor reference

**Resolution:**

```python
# Ensure intent processor is set
universal_agent.intent_processor = intent_processor_instance
```

### 3. Test Issues

#### Issue: Tests expecting string returns fail

**Symptoms:**

```
AssertionError: assert 'Multi-step workflow initiated' in WorkflowExecutionIntent(...)
```

**Root Cause:** Tests written for old string-based behavior, function now returns WorkflowExecutionIntent

**Resolution:**

```python
# Update test assertions
# OLD:
assert "Multi-step workflow initiated" in result

# NEW:
assert isinstance(result, WorkflowExecutionIntent)
assert len(result.tasks) == expected_task_count
```

#### Issue: Import errors for create_workflow_execution_intent

**Symptoms:**

```
ImportError: cannot import name 'create_workflow_execution_intent' from 'roles.core_planning'
```

**Root Cause:** Function was renamed to execute_task_graph during cleanup

**Resolution:**

```python
# Update imports
# OLD:
from roles.core_planning import create_workflow_execution_intent

# NEW:
from roles.core_planning import execute_task_graph
```

### 4. Architecture Compliance Issues

#### Issue: AsyncIO calls detected

**Symptoms:**

```
RuntimeError: This event loop is already running
```

**Root Cause:** Code using asyncio.create_task instead of scheduled tasks

**Diagnosis:**

```bash
# Search for asyncio usage
grep -r "asyncio\." --include="*.py" .
grep -r "create_task" --include="*.py" .
```

**Resolution:**

```python
# Replace asyncio calls with scheduled tasks
# OLD:
asyncio.create_task(some_function())

# NEW:
supervisor.add_scheduled_task({
    "type": "function_execution",
    "handler": some_function
})
```

#### Issue: Background threads detected

**Symptoms:**

```
Threading violations in LLM-safe architecture
```

**Diagnosis:**

```bash
# Search for threading usage
grep -r "threading\." --include="*.py" .
grep -r "Thread(" --include="*.py" .
```

**Resolution:**

- Replace all background threads with scheduled tasks
- Use supervisor's single event loop for all execution

## Diagnostic Procedures

### 1. Intent Creation Diagnostics

```python
def diagnose_intent_creation(llm_result, context):
    """Diagnose intent creation issues."""
    try:
        from roles.core_planning import execute_task_graph

        print(f"Input length: {len(llm_result)}")
        print(f"Input preview: {llm_result[:100]}...")
        print(f"Context ID: {getattr(context, 'context_id', 'MISSING')}")

        result = execute_task_graph(llm_result, context, {})

        print(f"Result type: {type(result)}")
        print(f"Intent valid: {result.validate()}")
        print(f"Task count: {len(result.tasks)}")
        print(f"Dependency count: {len(result.dependencies)}")

        return result

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        return None
```

### 2. Workflow ID Generation Diagnostics

```python
def diagnose_workflow_ids(intent):
    """Diagnose workflow ID generation."""
    try:
        workflow_ids = intent.get_expected_workflow_ids()
        print(f"Generated workflow IDs: {workflow_ids}")
        print(f"ID count: {len(workflow_ids)}")

        # Verify ID format
        for wf_id in workflow_ids:
            parts = wf_id.split("_task_")
            print(f"ID: {wf_id} -> Request: {parts[0]}, Task: {parts[1] if len(parts) > 1 else 'INVALID'}")

    except Exception as e:
        print(f"Workflow ID generation error: {e}")
```

### 3. Architecture Compliance Diagnostics

```python
def diagnose_architecture_compliance():
    """Check LLM-Safe architecture compliance."""
    import ast
    import os

    violations = []

    # Check for asyncio usage
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if "asyncio.create_task" in content:
                            violations.append(f"{filepath}: asyncio.create_task usage")
                        if "threading.Thread" in content:
                            violations.append(f"{filepath}: threading.Thread usage")
                except Exception:
                    continue

    if violations:
        print("Architecture violations found:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print("No architecture violations detected")
```

## Performance Monitoring

### 1. Intent Creation Performance

```python
def monitor_intent_creation_performance():
    """Monitor intent creation performance."""
    import time

    # Test with various workflow sizes
    for task_count in [1, 5, 10, 20, 50]:
        # Create test workflow
        tasks = [
            {
                "id": f"task_{i}",
                "name": f"Task {i}",
                "description": f"Description {i}",
                "role": "search",
                "parameters": {}
            }
            for i in range(task_count)
        ]

        workflow_json = json.dumps({"tasks": tasks, "dependencies": []})

        # Measure performance
        start_time = time.time()
        result = execute_task_graph(workflow_json, mock_context, {})
        execution_time = time.time() - start_time

        print(f"Tasks: {task_count}, Time: {execution_time:.4f}s")
```

### 2. Memory Usage Monitoring

```python
def monitor_memory_usage():
    """Monitor memory usage during intent processing."""
    import psutil
    import os
    import gc

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Create many intents
    for i in range(100):
        result = execute_task_graph(sample_workflow_json, mock_context, {})
        if i % 10 == 0:
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            print(f"Iteration {i}: Memory increase: {memory_increase / 1024 / 1024:.2f}MB")
            gc.collect()
```

## Recovery Procedures

### 1. Reset Planning Role State

```python
def reset_planning_role():
    """Reset planning role to clean state."""
    # Clear any cached role definitions
    from llm_provider.role_registry import RoleRegistry
    role_registry = RoleRegistry.get_global_registry()
    role_registry.reload_role("planning")

    print("Planning role reset complete")
```

### 2. Clear Intent Processing Queue

```python
def clear_intent_processing_queue(supervisor):
    """Clear any stuck intent processing tasks."""
    if hasattr(supervisor, '_scheduled_tasks'):
        intent_tasks = [
            task for task in supervisor._scheduled_tasks
            if task.get("type") == "process_workflow_intent"
        ]

        for task in intent_tasks:
            supervisor._scheduled_tasks.remove(task)

        print(f"Cleared {len(intent_tasks)} intent processing tasks")
```

### 3. Validate System State

```python
def validate_system_state(supervisor):
    """Validate overall system state."""
    checks = []

    # Check supervisor
    checks.append(("Supervisor initialized", supervisor is not None))

    # Check workflow engine
    workflow_engine = getattr(supervisor, 'workflow_engine', None)
    checks.append(("Workflow engine available", workflow_engine is not None))

    # Check universal agent
    universal_agent = getattr(workflow_engine, 'universal_agent', None)
    checks.append(("Universal agent available", universal_agent is not None))

    # Check intent processor
    intent_processor = getattr(supervisor, 'intent_processor', None)
    checks.append(("Intent processor available", intent_processor is not None))

    # Print results
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    return all(result for _, result in checks)
```

## Testing and Validation

### 1. Run Document 35 Test Suite

```bash
# Run all Document 35 related tests
./venv/bin/python -m pytest tests/integration/test_document_35_end_to_end.py -v
./venv/bin/python -m pytest tests/integration/test_intent_based_planning_integration.py -v
./venv/bin/python -m pytest tests/unit/test_planning_role_intent_creation.py -v

# Run specific test categories
./venv/bin/python -m pytest -k "intent" -v
./venv/bin/python -m pytest -k "workflow" -v
```

### 2. Validate Architecture Compliance

```bash
# Check for asyncio violations
grep -r "asyncio\." --include="*.py" . | grep -v test | grep -v archived

# Check for threading violations
grep -r "threading\." --include="*.py" . | grep -v test | grep -v archived

# Verify no background threads
grep -r "Thread(" --include="*.py" . | grep -v test | grep -v archived
```

### 3. Performance Validation

```bash
# Run performance tests
./venv/bin/python -m pytest tests/integration/test_document_35_end_to_end.py::TestDocument35EndToEnd::test_performance_with_large_workflows -v

# Run memory stability tests
./venv/bin/python -m pytest tests/integration/test_document_35_end_to_end.py::TestDocument35EndToEnd::test_memory_usage_stability -v

# Run concurrent execution tests
./venv/bin/python -m pytest tests/integration/test_document_35_end_to_end.py::TestDocument35EndToEnd::test_concurrent_intent_creation -v
```

## Monitoring and Observability

### 1. Intent Creation Monitoring

```python
# Monitor intent creation in logs
grep "Created WorkflowExecutionIntent" logs/application.log

# Monitor intent validation
grep "WorkflowExecutionIntent.*validate" logs/application.log
```

### 2. Workflow Lifecycle Monitoring

```python
# Monitor workflow events (when Phase 2 is implemented)
grep "WORKFLOW_STARTED\|WORKFLOW_COMPLETED\|WORKFLOW_FAILED" logs/application.log

# Monitor request lifecycle
grep "request.*completed\|request.*expired" logs/application.log
```

### 3. Performance Monitoring

```python
# Monitor execution times
grep "Created WorkflowExecutionIntent.*tasks" logs/application.log

# Monitor memory usage
grep "Memory.*MB" logs/application.log
```

## Known Limitations

### Current Implementation (Phase 1)

- **Intent Creation Only**: Only creates intents, doesn't execute workflows yet
- **No Lifecycle Management**: Communication manager lifecycle not implemented yet
- **Test Coverage**: Some legacy tests still need updates

### Planned Improvements (Phase 2)

- **Event-Driven Execution**: Full workflow execution via scheduled tasks
- **Lifecycle Management**: Automatic request tracking and cleanup
- **Communication Manager Integration**: Eliminate request ID warnings

## Support Contacts

### Development Team

- **Architecture Questions**: Refer to Documents 25, 26, 35
- **Implementation Issues**: Check Document 36 (Technical Debt Resolution)
- **Testing Problems**: Review comprehensive test suites

### Escalation Path

1. **Level 1**: Check this troubleshooting guide
2. **Level 2**: Review implementation documents (35, 36, 37)
3. **Level 3**: Run diagnostic procedures
4. **Level 4**: Contact development team with diagnostic results

## Appendix: Quick Reference

### Key Files

- **Planning Role**: `roles/core_planning.py`
- **Universal Agent**: `llm_provider/universal_agent.py`
- **WorkflowExecutionIntent**: `common/workflow_intent.py`
- **Tests**: `tests/integration/test_document_35_end_to_end.py`

### Key Functions

- **execute_task_graph()**: Creates WorkflowExecutionIntent
- **get_expected_workflow_ids()**: Generates workflow IDs for tracking
- **\_handle_potential_workflow_intent()**: Detects and schedules intents

### Key Commands

```bash
# Run all Document 35 tests
./venv/bin/python -m pytest -k "document_35 or intent" -v

# Check architecture compliance
grep -r "asyncio\|threading" --include="*.py" . | grep -v test

# Validate code quality
make lint
```

This troubleshooting guide provides comprehensive support for the Document 35 implementation, covering common issues, diagnostic procedures, and resolution strategies for production use.
