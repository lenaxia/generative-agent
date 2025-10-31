# Threading Architecture Implementation for LLM Development

**Document ID:** 25
**Created:** 2025-10-12
**Status:** Low-Level Implementation Guide for AI Development
**Priority:** High
**Context:** 100% LLM Development Environment
**Companion:** Document 26 - High-Level Architecture Patterns

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

This document provides detailed implementation specifications for eliminating threading issues through simplified, LLM-optimized architecture. All changes focus on creating predictable patterns that AI agents can reliably implement while eliminating threading complexity entirely.

**LLM Development Principle:** Create the simplest possible implementation that eliminates error classes while providing clear templates that AI agents can follow consistently.

## Simplified Implementation Strategy

### **Core Architecture Changes**

#### **1. Single Event Loop Implementation**

**Goal:** Eliminate all background threads and threading complexity.
**Implementation:** Modify existing `Supervisor` to use scheduled tasks instead of background threads.

#### **2. Single-File Role Architecture**

**Goal:** Consolidate each role into a single, LLM-friendly Python file.
**Implementation:** Migrate from multi-file role structure to single-file pattern.

#### **3. Intent-Based Processing**

**Goal:** Separate declarative intents from imperative I/O operations.
**Implementation:** Pure function event handlers returning intents, processed by infrastructure.

## Phase 1: Foundation Implementation (Week 1)

### **Task 1.1: Create Core Intent System**

**File:** `common/intents.py` (NEW FILE)

```python
# LLM-SAFE: Core intent system with universal intents only
from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time

@dataclass
class Intent(ABC):
    """LLM-SAFE: Base class for all intents."""
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @abstractmethod
    def validate(self) -> bool:
        """All intents must implement validation."""
        pass

@dataclass
class NotificationIntent(Intent):
    """Universal intent: Any role can send notifications."""
    message: str
    channel: str
    user_id: Optional[str] = None
    priority: str = "medium"  # "low", "medium", "high"

    def validate(self) -> bool:
        return (
            bool(self.message and self.channel) and
            self.priority in ["low", "medium", "high"]
        )

@dataclass
class AuditIntent(Intent):
    """Universal intent: Any role can audit actions."""
    action: str
    details: Dict[str, Any]
    user_id: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.action and isinstance(self.details, dict))

@dataclass
class WorkflowIntent(Intent):
    """Universal intent: Any role can start workflows."""
    workflow_type: str
    parameters: Dict[str, Any]
    priority: int = 1

    def validate(self) -> bool:
        return bool(self.workflow_type and isinstance(self.parameters, dict))
```

### **Task 1.2: Create Intent Processor**

**File:** `common/intent_processor.py` (NEW FILE)

```python
# LLM-SAFE: Intent processor with dynamic role registration
import logging
from typing import List, Dict, Any, Callable
from common.intents import Intent, NotificationIntent, AuditIntent, WorkflowIntent

logger = logging.getLogger(__name__)

class IntentProcessor:
    """LLM-SAFE: Processes intents with role-specific handler registration."""

    def __init__(self, communication_manager=None, workflow_engine=None):
        self.communication_manager = communication_manager
        self.workflow_engine = workflow_engine

        # Core intent handlers (built-in)
        self._core_handlers = {
            NotificationIntent: self._process_notification,
            AuditIntent: self._process_audit,
            WorkflowIntent: self._process_workflow
        }

        # Role-specific handlers (registered dynamically)
        self._role_handlers: Dict[type, Callable] = {}

    def register_role_intent_handler(self, intent_type: type, handler: Callable):
        """Allow roles to register their own intent handlers."""
        self._role_handlers[intent_type] = handler
        logger.info(f"Registered handler for {intent_type.__name__}")

    async def process_intents(self, intents: List[Intent]) -> Dict[str, Any]:
        """Process list of intents with comprehensive error handling."""
        results = {"processed": 0, "failed": 0, "errors": []}

        for intent in intents:
            try:
                if not intent.validate():
                    results["errors"].append(f"Invalid intent: {intent}")
                    results["failed"] += 1
                    continue

                await self._process_single_intent(intent)
                results["processed"] += 1

            except Exception as e:
                logger.error(f"Intent processing failed: {e}")
                results["errors"].append(str(e))
                results["failed"] += 1

        return results

    async def _process_single_intent(self, intent: Intent):
        """Process single intent with type-specific handling."""
        intent_type = type(intent)

        # Check core handlers first
        if intent_type in self._core_handlers:
            await self._core_handlers[intent_type](intent)
        # Check role-specific handlers
        elif intent_type in self._role_handlers:
            await self._role_handlers[intent_type](intent)
        else:
            logger.warning(f"No handler for intent type: {intent_type}")

    async def _process_notification(self, intent: NotificationIntent):
        """Process notification intent."""
        if self.communication_manager:
            await self.communication_manager.send_notification(
                message=intent.message,
                channel=intent.channel,
                user_id=intent.user_id
            )

    async def _process_audit(self, intent: AuditIntent):
        """Process audit intent."""
        logger.info(f"Audit: {intent.action} - {intent.details}")

    async def _process_workflow(self, intent: WorkflowIntent):
        """Process workflow intent."""
        if self.workflow_engine:
            await self.workflow_engine.start_workflow(
                request=f"Execute {intent.workflow_type}",
                parameters=intent.parameters
            )
```

### **Task 1.3: Single-File Role Template**

**File:** `roles/timer.py` (NEW FILE - Replaces timer/ directory)

```python
"""Timer role - LLM-friendly single file implementation."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import logging
from common.intents import Intent, NotificationIntent, AuditIntent
from strands import tool

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "timer",
    "version": "3.0.0",
    "description": "Timer and alarm management with event-driven workflows",
    "llm_type": "WEAK",
    "fast_reply": True,
    "when_to_use": "Set timers, alarms, manage time-based reminders"
}

# 2. ROLE-SPECIFIC INTENTS (owned by timer role)
@dataclass
class TimerIntent(Intent):
    """Timer-specific intent - owned by timer role."""
    action: str  # "create", "cancel", "check"
    timer_id: Optional[str] = None
    duration: Optional[int] = None
    label: Optional[str] = None

    def validate(self) -> bool:
        return bool(self.action and self.action in ["create", "cancel", "check"])

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_timer_expiry(event_data: Any, context) -> List[Intent]:
    """LLM-SAFE: Pure function for timer expiry events."""
    try:
        # Parse event data
        timer_id, request = _parse_timer_event_data(event_data)

        # Create intents
        return [
            NotificationIntent(
                message=f"⏰ Timer expired: {request}",
                channel=context.channel_id or "general",
                user_id=context.user_id,
                priority="medium"
            ),
            AuditIntent(
                action="timer_expired",
                details={
                    "timer_id": timer_id,
                    "original_request": request,
                    "processed_at": time.time()
                },
                user_id=context.user_id
            )
        ]

    except Exception as e:
        logger.error(f"Timer handler error: {e}")
        return [
            NotificationIntent(
                message=f"Timer processing error: {e}",
                channel=context.channel_id or "general",
                priority="high"
            )
        ]

def handle_heartbeat_monitoring(event_data: Any, context) -> List[Intent]:
    """LLM-SAFE: Pure function for heartbeat monitoring."""
    return [
        TimerIntent(action="check", timer_id=None)
    ]

# 4. TOOLS (simplified, LLM-friendly)
@tool
def set_timer(duration: str, label: str = "") -> Dict[str, Any]:
    """LLM-SAFE: Set a timer - returns intent for processing."""
    try:
        duration_seconds = _parse_duration(duration)
        return {
            "success": True,
            "message": f"Timer set for {duration}",
            "intent": TimerIntent(
                action="create",
                duration=duration_seconds,
                label=label
            )
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def cancel_timer(timer_id: str) -> Dict[str, Any]:
    """LLM-SAFE: Cancel a timer - returns intent for processing."""
    return {
        "success": True,
        "message": f"Timer {timer_id} cancelled",
        "intent": TimerIntent(action="cancel", timer_id=timer_id)
    }

@tool
def list_timers() -> Dict[str, Any]:
    """LLM-SAFE: List timers - returns intent for processing."""
    return {
        "success": True,
        "message": "Listing active timers",
        "intent": TimerIntent(action="check")
    }

# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_timer_event_data(event_data: Any) -> tuple[str, str]:
    """LLM-SAFE: Parse timer event data with error handling."""
    try:
        if isinstance(event_data, list) and len(event_data) >= 2:
            return str(event_data[0]), str(event_data[1])
        elif isinstance(event_data, dict):
            return (
                event_data.get('timer_id', 'unknown'),
                event_data.get('original_request', 'Unknown timer')
            )
        else:
            return 'unknown', f'Unparseable data: {event_data}'
    except Exception as e:
        return 'parse_error', f'Parse error: {e}'

def _parse_duration(duration_str: str) -> int:
    """LLM-SAFE: Simple duration parsing that LLMs can understand."""
    try:
        if duration_str.endswith('m'):
            return int(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return int(duration_str[:-1]) * 3600
        elif duration_str.endswith('s'):
            return int(duration_str[:-1])
        else:
            return int(duration_str)  # Assume seconds
    except ValueError:
        raise ValueError(f"Invalid duration format: {duration_str}")

# 6. INTENT HANDLER REGISTRATION
async def process_timer_intent(intent: TimerIntent):
    """Process timer-specific intents - called by IntentProcessor."""
    # This would interact with actual timer infrastructure
    # For now, just log the intent
    logger.info(f"Processing timer intent: {intent.action}")

    # In full implementation, this would:
    # - Create/cancel/check timers using timer manager
    # - Return additional intents if needed

# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "TIMER_EXPIRED": handle_timer_expiry,
            "FAST_HEARTBEAT_TICK": handle_heartbeat_monitoring
        },
        "tools": [set_timer, cancel_timer, list_timers],
        "intents": {
            TimerIntent: process_timer_intent
        }
    }
```

## Phase 2: Component Evolution (Week 2)

### **Task 2.1: Enhanced MessageBus for Single-File Roles**

**File:** `common/message_bus.py` (MODIFY EXISTING)

```python
# Enhanced MessageBus to work with single-file roles
class MessageBus:
    def __init__(self):
        # Existing initialization preserved
        self._subscribers = {}
        self._running = False

        # NEW: Intent processing for single-file roles
        self._intent_processor = None
        self._enable_intent_processing = True

    def start(self):
        """Enhanced start with intent processing."""
        self._running = True
        logger.info("MessageBus started in LLM-safe mode")

        # Initialize intent processor when dependencies available
        if self._enable_intent_processing:
            self._initialize_intent_processor()

    def _initialize_intent_processor(self):
        """Initialize intent processor with dependencies."""
        from common.intent_processor import IntentProcessor

        self._intent_processor = IntentProcessor(
            communication_manager=getattr(self, 'communication_manager', None),
            workflow_engine=getattr(self, 'workflow_engine', None)
        )

    async def publish(self, publisher, event_type: str, message: Any):
        """LLM-SAFE: Enhanced publish with intent processing."""
        if not self._running:
            return

        # Create explicit context
        context = self._create_event_context(publisher)

        # Process subscribers
        if event_type in self._subscribers:
            for role_name, handlers in self._subscribers[event_type].items():
                for handler in handlers:
                    try:
                        result = await handler(message, context)

                        # Process intents if returned
                        if self._is_intent_list(result):
                            await self._process_intents(result)

                    except Exception as e:
                        logger.error(f"Handler error in {role_name}: {e}")

    def _create_event_context(self, publisher):
        """Create explicit event context."""
        return type('EventContext', (), {
            'channel_id': getattr(publisher, 'channel_id', None),
            'user_id': getattr(publisher, 'user_id', None),
            'timestamp': time.time(),
            'source': publisher.__class__.__name__ if publisher else "unknown"
        })()

    def _is_intent_list(self, result) -> bool:
        """Check if result is list of intents."""
        return (
            isinstance(result, list) and
            len(result) > 0 and
            all(hasattr(item, 'validate') for item in result)
        )

    async def _process_intents(self, intents: List[Intent]):
        """Process intents using intent processor."""
        if self._intent_processor:
            await self._intent_processor.process_intents(intents)
```

### **Task 2.2: Enhanced Supervisor for Single Event Loop**

**File:** `supervisor/supervisor.py` (MODIFY EXISTING)

```python
# Enhanced Supervisor with single event loop
import asyncio
from typing import List

class Supervisor:
    def __init__(self, config_file: Optional[str] = None):
        # Existing initialization preserved
        logger.info("Initializing LLM-safe Supervisor...")

        # NEW: Single event loop management
        self.scheduled_tasks: List[asyncio.Task] = []
        self._use_single_event_loop = True

        # Existing initialization continues
        self.config_file = config_file
        self.initialize_config_manager(config_file)
        self._set_environment_variables()
        self.initialize_components()
        logger.info("LLM-safe Supervisor initialization complete.")

    def initialize_components(self):
        """Enhanced component initialization."""
        # Existing component initialization preserved
        self.initialize_llm_factory()
        self.initialize_workflow_engine()
        self.initialize_communication_manager()

        # NEW: Use scheduled tasks instead of background threads
        if self._use_single_event_loop:
            self._initialize_scheduled_tasks()

    def _initialize_scheduled_tasks(self):
        """LLM-SAFE: Replace background threads with scheduled tasks."""
        try:
            # Heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.scheduled_tasks.append(heartbeat_task)

            # Timer monitoring task
            timer_task = asyncio.create_task(self._timer_monitoring_loop())
            self.scheduled_tasks.append(timer_task)

            logger.info(f"Initialized {len(self.scheduled_tasks)} scheduled tasks")

        except Exception as e:
            logger.error(f"Failed to initialize scheduled tasks: {e}")
            raise

    async def _heartbeat_loop(self):
        """LLM-SAFE: Heartbeat as scheduled task."""
        while True:
            try:
                await self._perform_heartbeat()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _timer_monitoring_loop(self):
        """LLM-SAFE: Timer monitoring as scheduled task."""
        while True:
            try:
                await self._perform_timer_monitoring()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Timer monitoring error: {e}")
                await asyncio.sleep(1)

    async def _perform_heartbeat(self):
        """LLM-SAFE: Heartbeat operations in main event loop."""
        if self.workflow_engine:
            await self.workflow_engine.cleanup_old_workflows()

        if self.message_bus:
            self.message_bus.publish(self, "HEARTBEAT_TICK", {
                "timestamp": time.time(),
                "active_workflows": len(getattr(self.workflow_engine, 'active_workflows', {}))
            })

    async def _perform_timer_monitoring(self):
        """LLM-SAFE: Timer monitoring in main event loop."""
        if self.message_bus:
            self.message_bus.publish(self, "FAST_HEARTBEAT_TICK", {
                "timestamp": time.time()
            })
```

### **Task 2.3: Enhanced RoleRegistry for Single-File Roles**

**File:** `llm_provider/role_registry.py` (MODIFY EXISTING)

```python
# Enhanced RoleRegistry for single-file role discovery
import importlib
import os
from pathlib import Path

class RoleRegistry:
    def __init__(self, roles_directory: str = "roles", message_bus=None):
        # Existing initialization preserved
        self.roles_directory = roles_directory
        self.message_bus = message_bus
        self.roles = {}

        # NEW: Intent processor integration
        self.intent_processor = None

    def set_intent_processor(self, intent_processor):
        """Set intent processor for role intent registration."""
        self.intent_processor = intent_processor
        self._register_all_role_intents()

    def discover_roles(self):
        """LLM-SAFE: Discover single-file roles."""
        roles_path = Path(self.roles_directory)

        # Look for Python files (not directories)
        for role_file in roles_path.glob("*.py"):
            if role_file.name.startswith("_"):
                continue  # Skip private files

            role_name = role_file.stem
            try:
                self._load_single_file_role(role_name)
            except Exception as e:
                logger.error(f"Failed to load role {role_name}: {e}")

    def _load_single_file_role(self, role_name: str):
        """Load role from single Python file."""
        try:
            # Import role module
            module = importlib.import_module(f"roles.{role_name}")

            # Get role registration
            if hasattr(module, 'register_role'):
                role_info = module.register_role()

                # Register role
                self.roles[role_name] = role_info

                # Register event handlers
                self._register_event_handlers(role_name, role_info)

                # Register intent handlers
                self._register_intent_handlers(role_name, role_info)

                logger.info(f"Loaded single-file role: {role_name}")

        except Exception as e:
            logger.error(f"Failed to load role {role_name}: {e}")

    def _register_event_handlers(self, role_name: str, role_info: dict):
        """Register event handlers from single-file role."""
        event_handlers = role_info.get('event_handlers', {})

        for event_type, handler in event_handlers.items():
            if self.message_bus:
                self.message_bus.subscribe(role_name, event_type, handler)

    def _register_intent_handlers(self, role_name: str, role_info: dict):
        """Register intent handlers from single-file role."""
        intent_handlers = role_info.get('intents', {})

        if self.intent_processor:
            for intent_type, handler in intent_handlers.items():
                self.intent_processor.register_role_intent_handler(intent_type, handler)
```

## Phase 3: Testing & Validation (Week 3)

### **Task 3.1: LLM-Safe Testing Patterns**

**File:** `tests/llm_development/test_single_file_roles.py` (NEW FILE)

```python
# LLM-SAFE: Testing patterns for single-file roles
import pytest
from roles.timer import handle_timer_expiry, TimerIntent, ROLE_CONFIG

class TestSingleFileRoles:
    """Test templates for single-file role patterns."""

    def test_role_config_structure(self):
        """Verify role config follows expected structure."""
        required_fields = ["name", "version", "description", "llm_type"]
        for field in required_fields:
            assert field in ROLE_CONFIG

        assert ROLE_CONFIG["name"] == "timer"
        assert ROLE_CONFIG["llm_type"] in ["WEAK", "DEFAULT", "STRONG"]

    def test_timer_intent_validation(self):
        """Test timer intent validation."""
        valid_intent = TimerIntent(action="create", duration=300)
        assert valid_intent.validate() == True

        invalid_intent = TimerIntent(action="invalid_action")
        assert invalid_intent.validate() == False

    def test_pure_event_handler(self):
        """Test event handler is pure function."""
        event_data = ["timer_123", "Test reminder"]
        context = type('Context', (), {
            'channel_id': 'C123',
            'user_id': 'U456'
        })()

        # Should return intents
        intents = handle_timer_expiry(event_data, context)

        assert isinstance(intents, list)
        assert len(intents) >= 2
        assert all(hasattr(intent, 'validate') for intent in intents)
        assert all(intent.validate() for intent in intents)

    def test_role_registration(self):
        """Test role registration structure."""
        from roles.timer import register_role

        role_info = register_role()

        required_keys = ["config", "event_handlers", "tools", "intents"]
        for key in required_keys:
            assert key in role_info

        assert "TIMER_EXPIRED" in role_info["event_handlers"]
        assert len(role_info["tools"]) > 0
```

## Implementation Validation

### **Week 1 Validation Commands**

```bash
# Test intent system
python -c "
from common.intents import NotificationIntent
intent = NotificationIntent(message='test', channel='C123')
assert intent.validate() == True
print('✅ Intent system working')
"

# Test single-file role
python -c "
from roles.timer import register_role, ROLE_CONFIG
role_info = register_role()
assert 'config' in role_info
assert role_info['config']['name'] == 'timer'
print('✅ Single-file role working')
"
```

### **Week 2 Validation Commands**

```bash
# Test no background threads
python -c "
import threading
from supervisor.supervisor import Supervisor
initial_count = threading.active_count()
supervisor = Supervisor('config.yaml')
final_count = threading.active_count()
assert final_count == initial_count
print('✅ No background threads created')
"

# Test intent processing
python -c "
from common.intent_processor import IntentProcessor
from common.intents import NotificationIntent
from unittest.mock import AsyncMock
import asyncio

async def test():
    processor = IntentProcessor(communication_manager=AsyncMock())
    intents = [NotificationIntent(message='test', channel='C123')]
    results = await processor.process_intents(intents)
    assert results['processed'] == 1
    print('✅ Intent processing working')

asyncio.run(test())
"
```

## Success Criteria

### **Simplification Success**

- ✅ Each role in single Python file
- ✅ Role files under 300 lines each
- ✅ No complex multi-file dependencies
- ✅ Clear, consistent structure across all roles

### **Threading Success**

- ✅ Zero background threads in Supervisor
- ✅ All operations in main event loop
- ✅ No cross-thread async operations
- ✅ Predictable, deterministic behavior

### **LLM Development Success**

- ✅ All roles follow same template pattern
- ✅ Pure function event handlers
- ✅ Explicit dependencies and validation
- ✅ Clear error handling in all functions

This simplified implementation eliminates threading complexity while creating an architecture that AI agents can reliably understand, modify, and extend.
