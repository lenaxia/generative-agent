"""
Intent Processor for LLM-Safe Declarative Event Processing

This module processes declarative intents returned by pure function event handlers,
eliminating threading complexity by separating "what should happen" from "how to make it happen".

Created: 2025-10-12
Part of: Threading Architecture Improvements (Documents 25, 26, 27)
"""

import logging
from collections.abc import Callable
from typing import Any

from common.intents import (
    AuditIntent,
    ErrorIntent,
    Intent,
    MemoryWriteIntent,
    NotificationIntent,
    WorkflowIntent,
)
from common.request_model import RequestMetadata

logger = logging.getLogger(__name__)


class IntentProcessor:
    """
    LLM-SAFE: Processes declarative intents with comprehensive error handling.

    This processor handles the "how to make it happen" part of the intent architecture,
    allowing event handlers to be pure functions that return declarative intents.
    """

    def __init__(
        self, communication_manager=None, workflow_engine=None, message_bus=None
    ):
        """
        Initialize the intent processor.

        Args:
            communication_manager: Optional communication manager for notifications
            workflow_engine: Optional workflow engine for starting workflows
            message_bus: Optional message bus for event publishing
        """
        self.communication_manager = communication_manager
        self.workflow_engine = workflow_engine
        self.message_bus = message_bus

        # Core intent handlers (built-in)
        self._core_handlers = {
            NotificationIntent: self._process_notification,
            AuditIntent: self._process_audit,
            WorkflowIntent: self._process_workflow,
            ErrorIntent: self._process_error,
            MemoryWriteIntent: self._process_memory_write,
        }

        # Role-specific intent handlers (registered dynamically)
        self._role_handlers: dict[type, dict[str, Any]] = {}

        # Metrics tracking
        self._processed_count = 0

    def register_role_intent_handler(
        self, intent_type: type, handler_func: Callable, role_name: str
    ):
        """
        Allow roles to register their own intent handlers.

        Args:
            intent_type: The intent class type to handle
            handler_func: Async function to handle the intent
            role_name: Name of the role registering the handler
        """
        self._role_handlers[intent_type] = {"handler": handler_func, "role": role_name}
        logger.info(f"Registered {intent_type.__name__} handler for {role_name} role")

    async def process_intents(self, intents: list[Intent]) -> dict[str, Any]:
        """
        Process list of intents with comprehensive error handling.

        Args:
            intents: List of intent objects to process

        Returns:
            Dict with processing results: processed count, failed count, errors
        """
        results = {"processed": 0, "failed": 0, "errors": []}

        for intent in intents:
            try:
                # Validate before processing
                if not intent.validate():
                    results["errors"].append(f"Invalid intent: {intent}")
                    results["failed"] += 1
                    continue

                # Process the intent
                await self._process_single_intent(intent)
                results["processed"] += 1
                self._processed_count += 1

            except Exception as e:
                logger.error(f"Intent processing failed: {e}")
                results["errors"].append(str(e))
                results["failed"] += 1

        return results

    async def _process_single_intent(self, intent: Intent):
        """
        Process single intent with type-specific handling.

        Handles both sync and async handlers for LLM-safe architecture compatibility.

        Args:
            intent: Intent object to process
        """
        import inspect

        intent_type = type(intent)
        intent_type_name = f"{intent_type.__module__}.{intent_type.__qualname__}"

        # Check core handlers first
        if intent_type in self._core_handlers:
            handler = self._core_handlers[intent_type]
            if inspect.iscoroutinefunction(handler):
                await handler(intent)
            else:
                handler(intent)
        # Check role-specific handlers by class identity first, then by name
        elif intent_type in self._role_handlers:
            handler_info = self._role_handlers[intent_type]
            handler = handler_info["handler"]
            if inspect.iscoroutinefunction(handler):
                await handler(intent)
            else:
                handler(intent)
        else:
            # Fallback: search by class name for class identity issues
            found_handler = None
            for registered_type, handler_info in self._role_handlers.items():
                registered_name = (
                    f"{registered_type.__module__}.{registered_type.__qualname__}"
                )
                if registered_name == intent_type_name:
                    found_handler = handler_info
                    logger.info(f"Found handler by name match: {intent_type_name}")
                    break

            if found_handler:
                handler = found_handler["handler"]
                if inspect.iscoroutinefunction(handler):
                    await handler(intent)
                else:
                    handler(intent)
            else:
                # Log warning but don't fail - allows for unknown intent types
                logger.warning(
                    f"No handler registered for intent type: {intent_type} (name: {intent_type_name})"
                )

    async def _process_notification(self, intent: NotificationIntent):
        """
        Process notification intent.

        Args:
            intent: NotificationIntent to process
        """
        if not self.communication_manager:
            logger.error("No communication manager available for notification")
            return

        try:
            # Call route_message method as expected by tests and API
            await self.communication_manager.route_message(
                message=intent.message,
                context={
                    "channel_id": intent.channel,
                    "user_id": intent.user_id,
                    "message_type": "notification",
                    "priority": getattr(intent, "priority", "medium"),
                    "notification_type": getattr(intent, "notification_type", "info"),
                },
            )
            logger.info(
                f"Notification sent successfully: {intent.message} to {intent.channel}"
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise

    async def _process_audit(self, intent: AuditIntent):
        """
        Process audit intent.

        Args:
            intent: AuditIntent to process
        """
        audit_entry = {
            "action": intent.action,
            "details": intent.details,
            "user_id": intent.user_id,
            "severity": intent.severity,
            "timestamp": intent.created_at,
        }

        # Log audit entry (in production, this might go to a dedicated audit system)
        log_level = getattr(logging, intent.severity.upper(), logging.INFO)
        logger.log(log_level, f"Audit: {intent.action} - {intent.details}")

    def _process_workflow(self, intent: WorkflowIntent):
        """
        Process workflow intent - handles both simple and task graph workflows.

        Document 35: Made synchronous following Documents 25 & 26 LLM-safe architecture.

        Args:
            intent: WorkflowIntent to process (simple or with task graph)
        """
        if not self.workflow_engine:
            logger.error("No workflow engine available for workflow")
            return

        try:
            # Check if this is a task graph workflow (Document 35 enhancement)
            if intent.is_task_graph_workflow():
                logger.info(
                    f"Processing task graph workflow for request {intent.request_id} with {len(intent.tasks or [])} tasks"
                )
                # Execute via workflow engine's task graph execution (synchronous)
                self.workflow_engine.execute_workflow_intent(intent)
            else:
                # Simple workflow - create RequestMetadata with user_id and channel_id from intent
                # Use original_instruction if available (for deferred workflows), otherwise use workflow_type
                instruction = intent.original_instruction or intent.workflow_type

                # Only add "Execute" prefix if this looks like a workflow type, not a full instruction
                if intent.original_instruction:
                    # This is a full instruction (e.g., from deferred workflow), use as-is
                    prompt = instruction
                else:
                    # This is a workflow type, add Execute prefix
                    prompt = f"Execute {instruction}"

                request = RequestMetadata(
                    prompt=prompt,
                    source_id="intent_processor",
                    target_id="workflow_engine",
                    user_id=intent.user_id,
                    channel_id=intent.channel_id,
                    metadata=intent.parameters or {},
                    response_requested=True,  # Ensure result is sent back to user
                )
                workflow_id = self.workflow_engine.handle_request(request)
                logger.info(
                    f"Started workflow {workflow_id} from intent: {intent.workflow_type}"
                )
        except Exception as e:
            logger.error(f"Failed to process workflow intent: {e}")

    async def _process_error(self, intent: ErrorIntent):
        """
        Process error intent.

        Args:
            intent: ErrorIntent to process
        """
        error_entry = {
            "error_type": intent.error_type,
            "error_message": intent.error_message,
            "error_details": intent.error_details,
            "recoverable": intent.recoverable,
            "user_id": intent.user_id,
            "timestamp": intent.created_at,
        }

        # Log error with appropriate level
        if intent.recoverable:
            logger.warning(
                f"Recoverable error: {intent.error_type} - {intent.error_message}"
            )
        else:
            logger.error(
                f"Non-recoverable error: {intent.error_type} - {intent.error_message}"
            )

        # In production, this might trigger alerts, create tickets, etc.
        logger.debug(f"Error details: {intent.error_details}")

    async def _process_memory_write(self, intent: MemoryWriteIntent):
        """
        Process memory write intent.

        Args:
            intent: MemoryWriteIntent to process
        """
        try:
            from common.providers.universal_memory_provider import (
                UniversalMemoryProvider,
            )

            provider = UniversalMemoryProvider()

            memory_id = provider.write_memory(
                user_id=intent.user_id,
                memory_type=intent.memory_type,
                content=intent.content,
                source_role=intent.source_role,
                importance=intent.importance,
                metadata=intent.metadata,
                tags=intent.tags,
                related_memories=intent.related_memories,
            )

            if memory_id:
                logger.info(
                    f"Memory written successfully: {memory_id} for user {intent.user_id}"
                )
            else:
                logger.warning(
                    f"Failed to write memory for user {intent.user_id}: {intent.memory_type}"
                )

        except Exception as e:
            logger.error(f"Error processing memory write intent: {e}")

    def get_processed_count(self) -> int:
        """
        Get total number of intents processed.

        Returns:
            Total count of successfully processed intents
        """
        return self._processed_count

    def get_registered_handlers(self) -> dict[str, list[str]]:
        """
        Get information about registered intent handlers.

        Returns:
            Dict mapping handler types to role names
        """
        handlers_info = {
            "core_handlers": [
                handler.__name__ for handler in self._core_handlers.keys()
            ],
            "role_handlers": [
                f"{handler_type.__name__} ({info['role']})"
                for handler_type, info in self._role_handlers.items()
            ],
        }
        return handlers_info

    def reset_metrics(self):
        """Reset processing metrics (useful for testing)."""
        self._processed_count = 0
