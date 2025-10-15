# Universal Agent Role Development Guide

**Complete LLM-Friendly Guide for Creating Single-File Roles**

This comprehensive guide provides everything needed to create new roles in the Universal Agent System. After reading this document, an LLM should be able to implement a complete, production-ready role with minimal additional guidance.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Single-File Role Structure](#single-file-role-structure)
3. [Role Metadata Configuration](#role-metadata-configuration)
4. [Intent System Integration](#intent-system-integration)
5. [Event Handler Implementation](#event-handler-implementation)
6. [Tool Development Patterns](#tool-development-patterns)
7. [Context Management and Traceability](#context-management-and-traceability)
8. [Pre/Post Processing with LLM Integration](#prepost-processing-with-llm-integration)
9. [Communication and Notification System](#communication-and-notification-system)
10. [Testing and Validation](#testing-and-validation)
11. [Complete Role Examples](#complete-role-examples)
12. [Integration with Universal Agent System](#integration-with-universal-agent-system)

## Architecture Overview

### Core Principles

The Universal Agent System uses a **single-file role architecture** optimized for LLM development:

- **LLM-Safe Design**: Each role is completely self-contained in one Python file (~300 lines)
- **Intent-Based Processing**: Pure function event handlers return declarative intents
- **Single Event Loop**: No threading complexity, no race conditions
- **Enhanced Context Storage**: Complete request traceability with `LLMSafeEventContext`
- **Unified Communication**: All notifications route through `IntentProcessor` → `CommunicationManager`

### System Flow

```
User Request → Router Role → Target Role → Tools/LLM → Intents → IntentProcessor → CommunicationManager → Channels
```

### Key Components

- **UniversalAgent**: Single agent interface with role specialization
- **RoleRegistry**: Auto-discovers and manages single-file roles
- **IntentProcessor**: Processes declarative intents from roles
- **MessageBus**: Event-driven communication with intent processing
- **CommunicationManager**: Multi-channel message routing

## Single-File Role Structure

Every role must follow this exact 8-section structure:

```python
"""Role Name - LLM-friendly single file implementation.

Brief description of role functionality and purpose.

Architecture: Single Event Loop + Intent-Based + [Tool-Based|JSON-Response|Pre-Processing]
Created: [Date]
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool  # For tool-based roles
from pydantic import BaseModel, Field  # For JSON response roles
import requests  # For pre-processing roles

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
# 2. ROLE-SPECIFIC INTENTS (owned by this role)
# 3. EVENT HANDLERS (pure functions returning intents)
# 4. TOOLS OR PROCESSING FUNCTIONS (pattern-specific)
# 5. HELPER FUNCTIONS (minimal, focused)
# 6. INTENT HANDLER REGISTRATION (processes role-specific intents)
# 7. PRE/POST PROCESSORS (optional, for LLM integration)
# 8. ROLE REGISTRATION (auto-discovery)
```

## Role Metadata Configuration

### Section 1: ROLE_CONFIG (Required)

```python
# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "your_role_name",  # Must match filename without _single_file.py
    "version": "1.0.0",
    "description": "Clear description of role capabilities and purpose",
    "llm_type": "WEAK",  # WEAK|DEFAULT|STRONG - choose based on complexity
    "fast_reply": True,  # True for quick responses, False for complex processing
    "when_to_use": "Specific criteria for when this role should be selected",

    # Tool Configuration
    "tools": {
        "automatic": True,  # Include custom @tool functions?
        "shared": ["redis_tools"],  # List of shared tools (redis_tools commonly used)
        "include_builtin": False,  # CRITICAL: Usually False (excludes calculator, file_read, shell)
        "fast_reply": {
            "enabled": True,  # Enable tools in fast-reply mode, you may not want this if you can fetch all relevant data in pre-
        },
    },

    # System Prompt Configuration
    "prompts": {
        "system": """You are a specialized [role type] agent.

Available tools:
- tool_name(param): Description of what the tool does

Architecture:
- All operations are intent-based and thread-safe
- Use available tools to perform actions
- Provide clear confirmation of actions taken

When users request [role-specific actions]:
1. Parse the request parameters
2. Use appropriate tools to perform actions
3. Provide clear confirmation of results"""
    },

    # Parameter Schema (for router integration)
    "parameters": {
        "action": {
            "type": "string",
            "required": True,
            "description": "Action to perform",
            "examples": ["create", "update", "delete"],
            "enum": ["create", "update", "delete"],
        },
        "target": {
            "type": "string",
            "required": False,
            "description": "Target for the action",
            "examples": ["item1", "item2"],
        },
    },
}
```

### LLM Type Selection Guide

- **WEAK**: Simple operations, JSON responses, basic tool usage
- **DEFAULT**: Standard operations, moderate complexity
- **STRONG**: Complex reasoning, multi-step operations, advanced processing

### Tool Configuration Options

```python
# No tools (JSON response roles)
"tools": {
    "automatic": False,
    "shared": [],
    "include_builtin": False,
}

# Custom tools only (most common)
"tools": {
    "automatic": True,
    "shared": ["redis_tools"],  # Include Redis for data persistence
    "include_builtin": False,   # Exclude calculator, file_read, shell
}

# Custom + shared tools
"tools": {
    "automatic": True,
    "shared": ["redis_tools", "other_shared_tool"],
    "include_builtin": False,
}

# Include built-in tools (rare - only if specifically needed)
"tools": {
    "automatic": True,
    "shared": ["redis_tools"],
    "include_builtin": True,  # Adds calculator, file_read, shell
}
```

## Intent System Integration

### Section 2: Role-Specific Intents

```python
# 2. ROLE-SPECIFIC INTENTS (owned by this role)
@dataclass
class YourRoleActionIntent(Intent):
    """Role-specific intent for performing actions."""

    action: str  # "create", "update", "delete"
    target_id: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    event_context: Optional[Dict[str, Any]] = None  # Store complete LLMSafeEventContext

    def validate(self) -> bool:
        """Validate intent parameters."""
        return (
            bool(self.action and self.target_id)
            and isinstance(self.parameters, dict)
            and self.action in ["create", "update", "delete"]
        )

@dataclass
class YourRoleDataIntent(Intent):
    """Role-specific intent for data operations."""

    operation: str  # "store", "retrieve", "process"
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    event_context: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate data intent parameters."""
        return (
            bool(self.operation and isinstance(self.data, dict))
            and self.operation in ["store", "retrieve", "process"]
        )
```

### Universal Intents (Available to All Roles)

```python
# Available from common.intents
NotificationIntent(
    message="User-facing message",
    channel="slack:C123456",  # Full channel ID
    user_id="U123456",
    priority="low|medium|high",
    notification_type="info|warning|error"
)

AuditIntent(
    action="action_performed",
    details={"key": "value"},
    user_id="U123456",
    severity="info|warning|error"
)

WorkflowIntent(
    workflow_type="workflow_name",
    parameters={"param": "value"},
    priority=1
)
```

## Event Handler Implementation

### Section 3: Event Handlers (Pure Functions)

```python
# 3. EVENT HANDLERS (pure functions returning intents)
def handle_role_specific_event(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """LLM-SAFE: Pure function for role-specific events."""
    try:
        # Parse event data safely
        parsed_data = _parse_event_data(event_data)

        # Extract context information
        user_id = context.user_id
        channel_id = context.channel_id
        source = context.source
        metadata = context.get_all_metadata()

        # Create intents based on event (no side effects)
        intents = []

        # Always create audit trail
        intents.append(
            AuditIntent(
                action="event_processed",
                details={
                    "event_type": "role_specific_event",
                    "data": parsed_data,
                    "source": source,
                    "processed_at": time.time()
                },
                user_id=user_id,
                severity="info"
            )
        )

        # Create notification if needed
        if _should_notify_user(parsed_data, context):
            intents.append(
                NotificationIntent(
                    message=f"Event processed: {parsed_data.get('summary', 'Unknown event')}",
                    channel=channel_id or "console",
                    user_id=user_id,
                    priority="medium"
                )
            )

        # Create role-specific intent for further processing
        if _requires_action(parsed_data):
            intents.append(
                YourRoleActionIntent(
                    action=parsed_data.get("action", "process"),
                    target_id=parsed_data.get("target", "unknown"),
                    parameters=parsed_data,
                    user_id=user_id,
                    channel_id=channel_id,
                    event_context=context.to_dict()  # Store complete context
                )
            )

        return intents

    except Exception as e:
        logger.error(f"Event handler error: {e}")
        return [
            NotificationIntent(
                message=f"Event processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error"
            )
        ]

def handle_heartbeat_monitoring(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """LLM-SAFE: Handle periodic heartbeat events for monitoring."""
    try:
        # Perform periodic checks (database health, external services, etc.)
        health_status = _check_role_health()

        if not health_status["healthy"]:
            return [
                NotificationIntent(
                    message=f"Role health issue: {health_status['issue']}",
                    channel="console",
                    priority="high",
                    notification_type="warning"
                )
            ]

        return []  # No action needed if healthy

    except Exception as e:
        logger.error(f"Heartbeat monitoring error: {e}")
        return []
```

### Event Handler Best Practices

1. **Pure Functions**: No side effects, only return intents
2. **Error Handling**: Always wrap in try/catch with error intents
3. **Context Usage**: Extract user_id, channel_id from context parameter
4. **Intent Creation**: Use appropriate intent types for different actions
5. **Logging**: Use appropriate log levels (debug/info/warning/error)

## Tool Development Patterns

### Section 4: Tool Functions (Tool-Based Pattern)

```python
# 4. TOOLS (declarative, LLM-friendly, intent-based)
@tool
def create_item(name: str, category: str = "default") -> Dict[str, Any]:
    """LLM-SAFE: Create a new item - returns intent for processing."""
    try:
        # Validate parameters
        if not name or len(name.strip()) == 0:
            return {"success": False, "error": "Name is required"}

        # Generate unique ID
        item_id = f"item_{uuid.uuid4().hex[:8]}"

        # Return intent data for processing by infrastructure
        return {
            "success": True,
            "item_id": item_id,
            "message": f"Item '{name}' created in category '{category}'",
            "intent": {
                "type": "YourRoleActionIntent",
                "action": "create",
                "target_id": item_id,
                "parameters": {
                    "name": name,
                    "category": category,
                    "created_at": time.time()
                }
                # user_id, channel_id, event_context will be injected by UniversalAgent
            }
        }
    except Exception as e:
        logger.error(f"Item creation error: {e}")
        return {"success": False, "error": str(e)}

@tool
def update_item(item_id: str, **updates) -> Dict[str, Any]:
    """LLM-SAFE: Update an existing item - returns intent for processing."""
    return {
        "success": True,
        "message": f"Item {item_id} updated",
        "intent": {
            "type": "YourRoleActionIntent",
            "action": "update",
            "target_id": item_id,
            "parameters": updates
        }
    }

@tool
def delete_item(item_id: str) -> Dict[str, Any]:
    """LLM-SAFE: Delete an item - returns intent for processing."""
    return {
        "success": True,
        "message": f"Item {item_id} deleted",
        "intent": {
            "type": "YourRoleActionIntent",
            "action": "delete",
            "target_id": item_id,
            "parameters": {"deleted_at": time.time()}
        }
    }
```

### Tool Design Principles

1. **Declarative**: Tools return intents, don't perform I/O directly
2. **Validation**: Validate parameters and return clear error messages
3. **Intent-Based**: Return intent objects for infrastructure processing
4. **Context Injection**: UniversalAgent injects user_id, channel_id, event_context
5. **Error Handling**: Return error objects instead of raising exceptions

## Context Management and Traceability

### Enhanced Context Storage

The system now stores complete `LLMSafeEventContext` for full request traceability:

```python
# When storing data (in intent handlers), preserve complete context
async def process_your_role_action_intent(intent: YourRoleActionIntent):
    """Process role-specific intents - handles actual operations."""
    logger.info(f"Processing {intent.action} for {intent.target_id}")

    try:
        # Store data with complete context for traceability
        item_data = {
            "id": intent.target_id,
            "action": intent.action,
            "parameters": intent.parameters,
            "created_at": time.time(),
            "status": "active",
            "event_context": intent.event_context,  # Complete context for traceability
        }

        # Use Redis for persistence with context
        from roles.shared_tools.redis_tools import redis_write
        redis_result = redis_write(
            f"item:data:{intent.target_id}",
            item_data,
            ttl=3600  # 1 hour TTL
        )

        if redis_result.get("success"):
            logger.info(f"Item {intent.target_id} {intent.action} completed successfully")
        else:
            logger.error(f"Failed to store item: {redis_result.get('error')}")

    except Exception as e:
        logger.error(f"Intent processing failed: {e}")
```

### Context Structure

The `event_context` field contains:

```json
{
  "user_id": "U52L1U8M6",
  "channel_id": "slack:C52L1UK5E",
  "timestamp": 1760477995.428,
  "source": "slack_message",
  "metadata": {
    "request_id": "req_123",
    "workflow_type": "your_role",
    "priority": "high",
    "original_message": "User's original request"
  }
}
```

### Context Extraction Patterns

```python
# Extract context from stored data
def _extract_context_from_stored_data(stored_data: Dict[str, Any]) -> Dict[str, str]:
    """Extract user and channel context from stored data."""
    event_context = stored_data.get("event_context", {})
    return {
        "user_id": event_context.get("user_id"),
        "channel_id": event_context.get("channel_id"),
        "source": event_context.get("source", "unknown"),
        "request_id": event_context.get("metadata", {}).get("request_id")
    }

# Use context for notifications
def _create_notification_with_context(message: str, stored_data: Dict[str, Any]) -> NotificationIntent:
    """Create notification using stored context."""
    context = _extract_context_from_stored_data(stored_data)
    return NotificationIntent(
        message=message,
        channel=context["channel_id"] or "console",
        user_id=context["user_id"],
        priority="medium"
    )
```

## Event Handler Implementation

### Section 3: Event Handlers

Event handlers are pure functions that process events and return intents:

```python
# 3. EVENT HANDLERS (pure functions returning intents)
def handle_data_processing_event(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Handle data processing events."""
    try:
        # Parse event data
        if isinstance(event_data, dict):
            data_id = event_data.get("data_id")
            operation = event_data.get("operation", "process")
        else:
            return []  # Invalid event data

        # Create processing intent
        return [
            YourRoleDataIntent(
                operation=operation,
                data={"data_id": data_id, "timestamp": time.time()},
                metadata=context.get_all_metadata(),
                event_context=context.to_dict()
            )
        ]

    except Exception as e:
        logger.error(f"Data processing event error: {e}")
        return [
            NotificationIntent(
                message=f"Data processing failed: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error"
            )
        ]

def handle_periodic_maintenance(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Handle periodic maintenance tasks."""
    try:
        # Check for items that need maintenance
        maintenance_items = _get_items_needing_maintenance()

        intents = []
        for item in maintenance_items:
            intents.append(
                YourRoleActionIntent(
                    action="maintain",
                    target_id=item["id"],
                    parameters={"maintenance_type": "periodic"},
                    event_context=context.to_dict()
                )
            )

        if maintenance_items:
            intents.append(
                NotificationIntent(
                    message=f"Performing maintenance on {len(maintenance_items)} items",
                    channel="console",
                    priority="low"
                )
            )

        return intents

    except Exception as e:
        logger.error(f"Maintenance handler error: {e}")
        return []
```

## Pre/Post Processing with LLM Integration

### Section 7: Pre/Post Processors (Optional)

Pre and post processors allow roles to integrate with LLMs for complex processing:

```python
# 7. PRE/POST PROCESSORS (optional, for LLM integration)
def pre_process_request(instruction: str, parameters: Dict[str, Any], context) -> Dict[str, Any]:
    """Pre-process request with external data or LLM analysis."""
    try:
        # Fetch external data if needed
        external_data = _fetch_external_data(parameters)

        # Use LLM for analysis if needed
        analysis_result = _analyze_with_llm(instruction, external_data)

        # Return enhanced instruction and context
        return {
            "enhanced_instruction": f"""Original Request: {instruction}

External Data:
{_format_external_data(external_data)}

Analysis:
{analysis_result}

Process this request with the provided context.""",
            "pre_data": {
                "external_data": external_data,
                "analysis": analysis_result,
                "processed_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Pre-processing failed: {e}")
        return {
            "enhanced_instruction": instruction,
            "pre_data": {"error": str(e)}
        }

def post_process_result(result: str, pre_data: Dict[str, Any], context) -> str:
    """Post-process LLM result with additional formatting or validation."""
    try:
        # Extract pre-processing data
        external_data = pre_data.get("external_data", {})
        analysis = pre_data.get("analysis", {})

        # Use LLM for result enhancement if needed
        enhanced_result = _enhance_result_with_llm(result, external_data, analysis)

        # Format final result
        return f"""Result: {enhanced_result}

Context: Based on {len(external_data)} external data points
Analysis: {analysis.get('summary', 'No analysis available')}
Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}"""

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return result  # Return original result if post-processing fails

def _analyze_with_llm(instruction: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM for analysis within pre-processing."""
    try:
        # Get UniversalAgent instance (injected during role loading)
        universal_agent = getattr(_analyze_with_llm, "_universal_agent", None)
        if not universal_agent:
            return {"error": "No LLM available for analysis"}

        # Create analysis instruction
        analysis_instruction = f"""Analyze this request and data:

Request: {instruction}
Data: {data}

Provide analysis in JSON format:
{{
  "complexity": "low|medium|high",
  "required_actions": ["action1", "action2"],
  "confidence": 0.95,
  "summary": "Brief analysis summary"
}}"""

        # Execute with appropriate LLM type
        from llm_provider.factory import LLMType
        result = universal_agent.execute_task(
            instruction=analysis_instruction,
            role="router",  # Use router for JSON analysis
            llm_type=LLMType.WEAK
        )

        # Parse JSON result
        import json
        return json.loads(result)

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {"error": str(e), "confidence": 0.0}

def _enhance_result_with_llm(result: str, external_data: Dict, analysis: Dict) -> str:
    """Use LLM to enhance result with additional context."""
    try:
        universal_agent = getattr(_enhance_result_with_llm, "_universal_agent", None)
        if not universal_agent:
            return result

        enhancement_instruction = f"""Enhance this result with additional context:

Original Result: {result}
External Data: {external_data}
Analysis: {analysis}

Provide an enhanced, more comprehensive response that incorporates the additional context."""

        from llm_provider.factory import LLMType
        enhanced = universal_agent.execute_task(
            instruction=enhancement_instruction,
            role="your_role_name",
            llm_type=LLMType.DEFAULT
        )

        return enhanced

    except Exception as e:
        logger.error(f"Result enhancement failed: {e}")
        return result
```

### LLM Integration in Pre/Post Processors

**Key Points:**

- UniversalAgent instance is injected as `_universal_agent` attribute on functions
- Use appropriate LLM types: WEAK for analysis, DEFAULT for enhancement, STRONG for complex reasoning
- Always handle LLM failures gracefully with fallbacks
- Use existing roles (like "router") for JSON analysis tasks

## Intent Handler Registration

### Section 6: Intent Handlers

Intent handlers process role-specific intents and perform actual operations:

```python
# 6. INTENT HANDLER REGISTRATION
async def process_your_role_action_intent(intent: YourRoleActionIntent):
    """Process action intents - handles actual operations."""
    logger.info(f"Processing {intent.action} action for {intent.target_id}")

    try:
        if intent.action == "create":
            await _handle_create_action(intent)
        elif intent.action == "update":
            await _handle_update_action(intent)
        elif intent.action == "delete":
            await _handle_delete_action(intent)
        else:
            logger.warning(f"Unknown action: {intent.action}")

    except Exception as e:
        logger.error(f"Action intent processing failed: {e}")

async def process_your_role_data_intent(intent: YourRoleDataIntent):
    """Process data intents - handles data operations."""
    logger.info(f"Processing {intent.operation} operation")

    try:
        if intent.operation == "store":
            await _store_data(intent.data, intent.event_context)
        elif intent.operation == "retrieve":
            await _retrieve_data(intent.data, intent.event_context)
        elif intent.operation == "process":
            await _process_data(intent.data, intent.event_context)
        else:
            logger.warning(f"Unknown operation: {intent.operation}")

    except Exception as e:
        logger.error(f"Data intent processing failed: {e}")

# Intent handler implementation examples
async def _handle_create_action(intent: YourRoleActionIntent):
    """Handle create action with Redis storage."""
    from roles.shared_tools.redis_tools import redis_write

    # Create item data with complete context
    item_data = {
        "id": intent.target_id,
        "name": intent.parameters.get("name"),
        "category": intent.parameters.get("category", "default"),
        "created_at": time.time(),
        "status": "active",
        "event_context": intent.event_context  # Store complete context
    }

    # Store in Redis
    redis_result = redis_write(
        f"item:data:{intent.target_id}",
        item_data,
        ttl=86400  # 24 hours
    )

    if redis_result.get("success"):
        logger.info(f"Item {intent.target_id} created successfully")

        # Send success notification using stored context
        if intent.event_context:
            context = intent.event_context
            # Create notification intent and process it
            notification = NotificationIntent(
                message=f"✅ Item '{intent.parameters.get('name')}' created successfully",
                channel=context.get("channel_id", "console"),
                user_id=context.get("user_id"),
                priority="medium"
            )

            # Process notification through IntentProcessor
            from llm_provider.role_registry import RoleRegistry
            role_registry = RoleRegistry.get_global_registry()
            if role_registry and role_registry.intent_processor:
                await role_registry.intent_processor._process_notification(notification)
    else:
        logger.error(f"Failed to create item: {redis_result.get('error')}")
```

## Communication and Notification System

### Notification Patterns

```python
# Standard notification (processed by IntentProcessor)
notification_intent = NotificationIntent(
    message="User-facing message",
    channel="slack:C123456",  # Use full channel ID from context
    user_id="U123456",
    priority="medium",  # low|medium|high
    notification_type="info"  # info|warning|error
)

# Audit trail (for compliance and debugging)
audit_intent = AuditIntent(
    action="action_performed",
    details={
        "target": "item_123",
        "parameters": {"key": "value"},
        "result": "success",
        "timestamp": time.time()
    },
    user_id="U123456",
    severity="info"  # info|warning|error
)

# Workflow trigger (start another workflow)
workflow_intent = WorkflowIntent(
    workflow_type="data_processing",
    parameters={"data_id": "data_123"},
    priority=1
)
```

### Channel Context Best Practices

1. **Always use full channel IDs**: `"slack:C123456"`, not just `"C123456"`
2. **Store complete context**: Use `event_context` field for traceability
3. **Extract from context**: Get user_id/channel_id from `event_context` when available
4. **Fallback gracefully**: Use `"console"` as fallback channel

## Role Registration

### Section 8: Role Registration (Required)

```python
# 8. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    registration = {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "ROLE_SPECIFIC_EVENT": handle_role_specific_event,
            "DATA_PROCESSING": handle_data_processing_event,
            "HEARTBEAT_TICK": handle_periodic_maintenance,  # Optional: periodic tasks
        },
        "tools": [],  # Will be populated based on ROLE_CONFIG["tools"]["automatic"]
        "intents": {
            YourRoleActionIntent: process_your_role_action_intent,
            YourRoleDataIntent: process_your_role_data_intent,
        }
    }

    # Add tools if automatic is enabled
    if ROLE_CONFIG["tools"]["automatic"]:
        registration["tools"] = [create_item, update_item, delete_item]

    # Add pre/post processors if implemented
    if "pre_process_request" in globals():
        registration["pre_processors"] = [pre_process_request]
    if "post_process_result" in globals():
        registration["post_processors"] = [post_process_result]

    return registration
```

### Event Registration

Common event types to handle:

- **Role-specific events**: Custom events for your role
- **HEARTBEAT_TICK**: Periodic maintenance (every 60 seconds)
- **FAST_HEARTBEAT_TICK**: Frequent checks (every 5 seconds)
- **DATA_PROCESSING**: Data processing events
- **USER_REQUEST**: Direct user requests

## Complete Role Examples

### Tool-Based Role (Timer Pattern)

```python
"""Timer Role - Complete example of tool-based role with enhanced context storage."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool
from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "timer",
    "version": "2.0.0",
    "description": "Timer and alarm management with enhanced context storage",
    "llm_type": "WEAK",
    "fast_reply": True,
    "when_to_use": "Set timers, alarms, manage time-based reminders",
    "tools": {
        "automatic": True,
        "shared": ["redis_tools"],
        "include_builtin": False,
    },
    "prompts": {
        "system": """You are a timer management specialist. You can set, cancel, and list timers.

Available tools:
- set_timer(duration, label): Set a new timer with duration (e.g., "5s", "2m", "1h") and optional label
- cancel_timer(timer_id): Cancel an existing timer by ID
- list_timers(): List all active timers

Always use the timer tools to perform timer operations."""
    },
}

# 2. ROLE-SPECIFIC INTENTS
@dataclass
class TimerCreationIntent(Intent):
    """Timer-specific intent: Create a timer with enhanced context."""
    timer_id: str
    duration: str
    duration_seconds: int
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    event_context: Optional[Dict[str, Any]] = None  # Complete context for traceability

    def validate(self) -> bool:
        return (
            bool(self.timer_id and self.duration)
            and isinstance(self.duration_seconds, (int, float))
            and self.duration_seconds > 0
        )

@dataclass
class TimerExpiryIntent(Intent):
    """Timer-specific intent: Handle timer expiry notification."""
    timer_id: str
    original_duration: str
    label: str = ""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    event_context: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        return bool(self.timer_id and self.original_duration)

# 3. EVENT HANDLERS
def handle_heartbeat_monitoring(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Check for expired timers every 5 seconds."""
    try:
        from roles.shared_tools.redis_tools import _get_redis_client

        current_time = int(time.time())

        # Get expired timers from Redis sorted set
        client = _get_redis_client()
        expired_timer_ids = client.zrangebyscore("timer:active_queue", 0, current_time)

        if expired_timer_ids:
            # Remove expired timers from queue
            client.zremrangebyscore("timer:active_queue", 0, current_time)

            # Create expiry intents
            intents = []
            for timer_id in expired_timer_ids:
                # Get timer data for context
                timer_result = redis_read(f"timer:data:{timer_id}")
                if timer_result.get("success"):
                    timer_data = timer_result.get("value", {})
                    stored_context = timer_data.get("event_context", {})

                    intents.append(
                        TimerExpiryIntent(
                            timer_id=timer_id,
                            original_duration=timer_data.get("duration", "unknown"),
                            label=timer_data.get("label", ""),
                            user_id=stored_context.get("user_id"),
                            channel_id=stored_context.get("channel_id"),
                            event_context=stored_context
                        )
                    )

            return intents

        return []

    except Exception as e:
        logger.error(f"Heartbeat monitoring error: {e}")
        return []

# 4. TOOLS
@tool
def set_timer(duration: str, label: str = "") -> Dict[str, Any]:
    """Create a timer with specified duration and optional label."""
    try:
        duration_seconds = _parse_duration(duration)
        timer_id = f"timer_{uuid.uuid4().hex[:8]}"

        return {
            "success": True,
            "timer_id": timer_id,
            "message": f"Timer set for {duration}" + (f" ({label})" if label else ""),
            "intent": {
                "type": "TimerCreationIntent",
                "timer_id": timer_id,
                "duration": duration,
                "duration_seconds": duration_seconds,
                "label": label,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 5. HELPER FUNCTIONS
def _parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds."""
    duration_str = duration_str.strip().lower()
    if duration_str.endswith("s"):
        return int(duration_str[:-1])
    elif duration_str.endswith("m"):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith("h"):
        return int(duration_str[:-1]) * 3600
    else:
        return int(duration_str)  # Assume seconds

# 6. INTENT HANDLERS
async def process_timer_creation_intent(intent: TimerCreationIntent):
    """Process timer creation - store in Redis with complete context."""
    from roles.shared_tools.redis_tools import redis_write, _get_redis_client

    try:
        expiry_time = time.time() + intent.duration_seconds

        # Store timer data with complete context
        timer_data = {
            "id": intent.timer_id,
            "duration": intent.duration,
            "duration_seconds": intent.duration_seconds,
            "label": intent.label,
            "created_at": time.time(),
            "expires_at": expiry_time,
            "status": "active",
            "event_context": intent.event_context,  # Complete context for traceability
        }

        # Store metadata
        redis_result = redis_write(f"timer:data:{intent.timer_id}", timer_data, ttl=intent.duration_seconds + 60)

        if redis_result.get("success"):
            # Add to sorted set for expiry checking
            client = _get_redis_client()
            client.zadd("timer:active_queue", {intent.timer_id: expiry_time})
            logger.info(f"Timer {intent.timer_id} created successfully")

    except Exception as e:
        logger.error(f"Timer creation failed: {e}")

async def process_timer_expiry_intent(intent: TimerExpiryIntent):
    """Process timer expiry - send notification to original channel."""
    try:
        # Create notification message
        message = f"⏰ Timer expired: {intent.original_duration}"
        if intent.label:
            message += f" ({intent.label})"

        # Create notification intent with stored context
        notification_intent = NotificationIntent(
            message=message,
            channel=intent.channel_id or "console",
            user_id=intent.user_id,
            priority="medium"
        )

        # Process notification through IntentProcessor
        from llm_provider.role_registry import RoleRegistry
        role_registry = RoleRegistry.get_global_registry()
        if role_registry and role_registry.intent_processor:
            await role_registry.intent_processor._process_notification(notification_intent)
            logger.info(f"Timer expiry notification sent")

    except Exception as e:
        logger.error(f"Timer expiry processing failed: {e}")

# 8. ROLE REGISTRATION
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring,
        },
        "tools": [set_timer],
        "intents": {
            TimerCreationIntent: process_timer_creation_intent,
            TimerExpiryIntent: process_timer_expiry_intent,
        }
    }
```

## JSON Response Role Example

````python
"""Router Role - Complete example of JSON response role."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from common.enhanced_event_context import LLMSafeEventContext
from common.intents import Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. PYDANTIC MODELS
class RoutingDecision(BaseModel):
    """Pydantic model for parsing routing decisions."""
    route: str = Field(..., description="Target role name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "router",
    "version": "1.0.0",
    "description": "Request routing and classification with JSON responses",
    "llm_type": "WEAK",
    "fast_reply": True,
    "when_to_use": "Route and classify user requests to appropriate roles",
    "tools": {
        "automatic": False,  # No custom tools
        "shared": [],
        "include_builtin": False,
    },
    "prompts": {
        "system": """You are a request router that responds with ONLY valid JSON.

Analyze user requests and route them to appropriate roles:
- timer: Set timers, alarms, reminders
- weather: Weather information and forecasts
- smart_home: Device control and automation
- planning: Complex task planning and analysis

CRITICAL: Respond with ONLY valid JSON in this exact format:
{
  "route": "role_name",
  "confidence": 0.95,
  "parameters": {"key": "value"}
}

Always validate your JSON before responding."""
    },
}

# 2. ROLE-SPECIFIC INTENTS
@dataclass
class RoutingRequestIntent(Intent):
    """Intent for processing routing requests."""
    request: str
    routing_decision: Dict[str, Any]
    event_context: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        return bool(self.request and self.routing_decision)

# 3. EVENT HANDLERS
def handle_external_routing_request(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Handle external routing requests."""
    try:
        if isinstance(event_data, dict):
            request_text = event_data.get("request", "")
            if request_text:
                return [
                    RoutingRequestIntent(
                        request=request_text,
                        routing_decision={"pending": True},
                        event_context=context.to_dict()
                    )
                ]
        return []
    except Exception as e:
        logger.error(f"Routing event handler error: {e}")
        return []

# 4. JSON PROCESSING FUNCTIONS
def parse_routing_response(response_text: str) -> Dict[str, Any]:
    """Parse routing response with Pydantic validation."""
    try:
        # Clean response text
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        # Parse and validate
        routing_decision = RoutingDecision.model_validate_json(cleaned_response)
        return {
            "valid": True,
            "data": routing_decision.model_dump(),
        }
    except ValidationError as e:
        logger.error(f"Routing response validation failed: {e}")
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "fallback_data": {"route": "planning", "confidence": 0.1, "parameters": {}}
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {
            "valid": False,
            "error": f"JSON error: {str(e)}",
            "fallback_data": {"route": "planning", "confidence": 0.1, "parameters": {}}
        }

# 6. INTENT HANDLERS
async def process_routing_request_intent(intent: RoutingRequestIntent):
    """Process routing requests - analyze and classify."""
    logger.info(f"Processing routing request: {intent.request[:50]}...")
    # Implementation would analyze request and update routing decision

# 8. ROLE REGISTRATION
def register_role():
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "EXTERNAL_ROUTING_REQUEST": handle_external_routing_request,
        },
        "tools": [],
        "intents": {
            RoutingRequestIntent: process_routing_request_intent,
        }
    }
````

## Pre-Processing Role Example

```python
"""Weather Role - Complete example of pre-processing role with external data."""

import logging
import requests
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "weather",
    "version": "1.0.0",
    "description": "Weather information with external API integration",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "when_to_use": "Get weather information, forecasts, weather-related queries",
    "tools": {
        "automatic": False,  # No custom tools - uses pre-processing
        "shared": [],
        "include_builtin": False,
    },
    "prompts": {
        "system": """You are a weather specialist with access to current weather data.

The weather data has been pre-fetched and injected into your context.
Use this data to provide accurate, helpful weather information.

Format responses clearly with current conditions, forecasts, and relevant advice."""
    },
}

# 2. ROLE-SPECIFIC INTENTS
@dataclass
class WeatherIntent(Intent):
    """Intent for weather data requests."""
    location: str
    request_type: str  # "current", "forecast", "alerts"
    event_context: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        return bool(self.location and self.request_type)

# 3. EVENT HANDLERS
def handle_weather_request(event_data: Any, context: LLMSafeEventContext) -> List[Intent]:
    """Handle weather request events."""
    try:
        if isinstance(event_data, dict):
            location = event_data.get("location", "")
            request_type = event_data.get("type", "current")

            if location:
                return [
                    WeatherIntent(
                        location=location,
                        request_type=request_type,
                        event_context=context.to_dict()
                    )
                ]
        return []
    except Exception as e:
        logger.error(f"Weather event handler error: {e}")
        return []

# 7. PRE-PROCESSORS
def pre_process_weather_request(instruction: str, parameters: Dict[str, Any], context) -> Dict[str, Any]:
    """Pre-process weather request by fetching external data."""
    try:
        # Extract location from parameters or instruction
        location = parameters.get("location", "")
        if not location:
            # Try to extract from instruction using LLM
            location = _extract_location_with_llm(instruction)

        # Fetch weather data from external API
        weather_data = _fetch_weather_data(location)

        # Build enhanced instruction with weather data
        enhanced_instruction = f"""USER REQUEST: "{instruction}"

CURRENT WEATHER DATA FOR {location.upper()}:
Temperature: {weather_data.get('temperature', 'N/A')}°F
Conditions: {weather_data.get('conditions', 'N/A')}
Humidity: {weather_data.get('humidity', 'N/A')}%
Wind: {weather_data.get('wind_speed', 'N/A')} mph {weather_data.get('wind_direction', '')}

FORECAST:
{_format_forecast_data(weather_data.get('forecast', []))}

Based on this current weather data, provide a comprehensive response to the user's request."""

        return {
            "enhanced_instruction": enhanced_instruction,
            "pre_data": {
                "weather_data": weather_data,
                "location": location,
                "fetched_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Weather pre-processing failed: {e}")
        return {
            "enhanced_instruction": instruction,
            "pre_data": {"error": str(e)}
        }

def _fetch_weather_data(location: str) -> Dict[str, Any]:
    """Fetch weather data from external API."""
    try:
        # Example API call (replace with actual weather service)
        api_key = "your_weather_api_key"
        url = f"https://api.weather.com/v1/current?location={location}&key={api_key}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        logger.error(f"Weather API call failed: {e}")
        return {"error": str(e), "location": location}

def _extract_location_with_llm(instruction: str) -> str:
    """Use LLM to extract location from instruction."""
    try:
        universal_agent = getattr(_extract_location_with_llm, "_universal_agent", None)
        if not universal_agent:
            return "unknown"

        extraction_prompt = f"""Extract the location from this weather request:
"{instruction}"

Respond with ONLY the location name (city, state/country if specified).
If no location is found, respond with "unknown"."""

        from llm_provider.factory import LLMType
        result = universal_agent.execute_task(
            instruction=extraction_prompt,
            role="router",
            llm_type=LLMType.WEAK
        )

        return result.strip()

    except Exception as e:
        logger.error(f"Location extraction failed: {e}")
        return "unknown"

# 8. ROLE REGISTRATION
def register_role():
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "WEATHER_REQUEST": handle_weather_request,
        },
        "tools": [],
        "intents": {
            WeatherIntent: process_weather_intent,
        },
        "pre_processors": [pre_process_weather_request]  # Enable pre-processing
    }
```

## Testing and Validation

### Basic Role Testing Template

```python
# tests/test_your_role.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from roles.your_role_single_file import (
    ROLE_CONFIG,
    YourRoleActionIntent,
    handle_role_specific_event,
    your_role_tool,
    process_your_role_action_intent,
    register_role
)

class TestYourRole:
    """Test your role implementation."""

    def test_role_config_structure(self):
        """Test role configuration follows required structure."""
        required_keys = ["name", "version", "description", "llm_type", "fast_reply", "when_to_use", "tools", "prompts"]
        for key in required_keys:
            assert key in ROLE_CONFIG, f"Missing required config key: {key}"

        # Validate tool configuration
        tools_config = ROLE_CONFIG["tools"]
        assert "automatic" in tools_config
        assert "shared" in tools_config
        assert "include_builtin" in tools_config

        # Validate prompts
        assert "system" in ROLE_CONFIG["prompts"]
        assert len(ROLE_CONFIG["prompts"]["system"]) > 50

    def test_intent_validation(self):
        """Test intent validation logic."""
        # Valid intent
        valid_intent = YourRoleActionIntent(
            action="create",
            target_id="test_123",
            parameters={"name": "test"}
        )
        assert valid_intent.validate() is True

        # Invalid intent
        invalid_intent = YourRoleActionIntent(
            action="",  # Empty action
            target_id="test_123",
            parameters={}
        )
        assert invalid_intent.validate() is False

    def test_event_handler_returns_intents(self):
        """Test event handlers return proper intents."""
        from common.enhanced_event_context import LLMSafeEventContext

        context = LLMSafeEventContext(
            user_id="test_user",
            channel_id="test_channel",
            source="test"
        )

        event_data = {"action": "test", "target": "item_123"}
        intents = handle_role_specific_event(event_data, context)

        assert isinstance(intents, list)
        assert len(intents) > 0
        assert all(hasattr(intent, 'validate') for intent in intents)

    @pytest.mark.asyncio
    async def test_intent_processing(self):
        """Test intent processing with mocked dependencies."""
        intent = YourRoleActionIntent(
            action="create",
            target_id="test_item",
            parameters={"name": "Test Item"},
            event_context={"user_id": "test_user", "channel_id": "test_channel"}
        )

        with patch("roles.shared_tools.redis_tools.redis_write") as mock_write:
            mock_write.return_value = {"success": True}

            await process_your_role_action_intent(intent)

            # Verify Redis write was called
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            assert "item:data:test_item" in str(call_args)

    def test_role_registration_structure(self):
        """Test role registration returns proper structure."""
        registration = register_role()

        required_keys = ["config", "event_handlers", "tools", "intents"]
        for key in required_keys:
            assert key in registration

        assert registration["config"] == ROLE_CONFIG
        assert len(registration["event_handlers"]) > 0
        assert len(registration["intents"]) > 0
```

## Integration with Universal Agent System

### How Roles Integrate

1. **Auto-Discovery**: RoleRegistry finds `*_single_file.py` files and calls `register_role()`
2. **Configuration**: Role metadata configures LLM type, tools, and prompts
3. **Tool Assembly**: UniversalAgent assembles tools based on role configuration
4. **Execution**: LLM executes with role-specific context and tools
5. **Intent Processing**: Tool results generate intents processed by IntentProcessor
6. **Communication**: Notifications route through CommunicationManager to channels

### Universal Agent Integration Points

```python
# Role is discovered and loaded
role_registry = RoleRegistry("roles")
role_def = role_registry.get_role("your_role_name")

# UniversalAgent assumes role
universal_agent = UniversalAgent(llm_factory, role_registry)
agent = universal_agent.assume_role("your_role_name")

# Execute task with role
result = universal_agent.execute_task(
    instruction="User request",
    role="your_role_name",
    context=task_context,
    event_context=llm_safe_context
)

# Tool results generate intents
# Intents are processed by IntentProcessor
# Notifications are sent via CommunicationManager
```

### MessageBus Integration

```python
# Roles subscribe to events via event_handlers
message_bus.subscribe("your_role", "ROLE_SPECIFIC_EVENT", handle_role_specific_event)

# Event handlers return intents
intents = handle_role_specific_event(event_data, context)

# MessageBus processes intents via IntentProcessor
await message_bus._process_callback_intents(intents, "handle_role_specific_event")
```

## Role Development Checklist

### Pre-Development

- [ ] Choose implementation pattern (Tool-Based, JSON Response, Pre-Processing)
- [ ] Define role purpose and scope clearly
- [ ] Identify required tools and shared dependencies
- [ ] Plan intent types and event handlers needed

### Implementation

- [ ] Create `roles/your_role_name_single_file.py`
- [ ] Implement all 8 required sections
- [ ] Add role-specific intents with `event_context` field
- [ ] Implement event handlers as pure functions
- [ ] Add intent handlers for actual operations
- [ ] Include complete context storage in data operations
- [ ] Set appropriate logging levels

### Configuration

- [ ] Set correct `llm_type` based on complexity
- [ ] Configure tools appropriately (usually `include_builtin: False`)
- [ ] Write clear system prompt with tool descriptions
- [ ] Add parameter schema for router integration
- [ ] Test role configuration with RoleRegistry

### Testing

- [ ] Write comprehensive unit tests
- [ ] Test intent validation logic
- [ ] Test event handlers return proper intents
- [ ] Test intent processing with mocked dependencies
- [ ] Test role registration structure
- [ ] Test integration with UniversalAgent

### Integration

- [ ] Verify auto-discovery by RoleRegistry
- [ ] Test role execution via UniversalAgent
- [ ] Verify intent processing pipeline
- [ ] Test notification routing to correct channels
- [ ] Validate context preservation and traceability

### Production Readiness

- [ ] Clean up debug logging
- [ ] Add error handling for all edge cases
- [ ] Validate performance with realistic data
- [ ] Document any external dependencies
- [ ] Add monitoring and health checks if needed

## Best Practices Summary

1. **Single File**: Keep everything in one file for LLM comprehension
2. **Intent-Based**: Use intents for all side effects and I/O operations
3. **Context Storage**: Always store complete `event_context` for traceability
4. **Pure Functions**: Event handlers should have no side effects
5. **Error Handling**: Graceful error handling with appropriate notifications
6. **Logging**: Use appropriate log levels for production monitoring
7. **Testing**: Comprehensive test coverage for all functionality
8. **Documentation**: Clear docstrings and comments for LLM understanding

This guide provides everything needed to create production-ready roles that integrate seamlessly with the Universal Agent System's intent-based, thread-safe architecture.
