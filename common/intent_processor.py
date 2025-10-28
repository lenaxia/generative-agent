"""
Intent Processor for LLM-Safe Declarative Event Processing

This module processes declarative intents returned by pure function event handlers,
eliminating threading complexity by separating "what should happen" from "how to make it happen".

Created: 2025-10-12
Part of: Threading Architecture Improvements (Documents 25, 26, 27)
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from common.intents import (
    AuditIntent,
    ErrorIntent,
    Intent,
    NotificationIntent,
    WorkflowIntent,
)
from common.workflow_intent import WorkflowExecutionIntent

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
            WorkflowExecutionIntent: self._process_workflow_execution,  # Document 35
            ErrorIntent: self._process_error,
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

        Args:
            intent: Intent object to process
        """
        intent_type = type(intent)
        intent_type_name = f"{intent_type.__module__}.{intent_type.__qualname__}"

        # Check core handlers first
        if intent_type in self._core_handlers:
            await self._core_handlers[intent_type](intent)
        # Check role-specific handlers by class identity first, then by name
        elif intent_type in self._role_handlers:
            handler_info = self._role_handlers[intent_type]
            await handler_info["handler"](intent)
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
                await found_handler["handler"](intent)
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

    async def _process_workflow(self, intent: WorkflowIntent):
        """
        Process workflow intent.

        Args:
            intent: WorkflowIntent to process
        """
        if not self.workflow_engine:
            logger.error("No workflow engine available for workflow")
            return

        try:
            workflow_id = await self.workflow_engine.start_workflow(
                request=f"Execute {intent.workflow_type}", parameters=intent.parameters
            )
            logger.info(
                f"Started workflow {workflow_id} from intent: {intent.workflow_type}"
            )
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")

    def _process_workflow_execution(self, intent: WorkflowExecutionIntent):
        """Document 35: Process WorkflowExecutionIntent (LLM-SAFE, synchronous).

        Following Documents 25 & 26 LLM-safe architecture - no async/await.

        Args:
            intent: WorkflowExecutionIntent to process
        """
        if not self.workflow_engine:
            logger.error(
                "No workflow engine available for WorkflowExecutionIntent processing"
            )
            return

        try:
            logger.info(
                f"Processing WorkflowExecutionIntent for request {intent.request_id} with {len(intent.tasks)} tasks"
            )

            # LLM-SAFE: Execute workflow via workflow engine (synchronous)
            self.workflow_engine.execute_workflow_intent(intent)

        except Exception as e:
            logger.error(f"Failed to process WorkflowExecutionIntent: {e}")
            # Don't re-raise - intent processor should handle errors gracefully

    def _process_workflow_execution(self, intent: WorkflowExecutionIntent):
        """Document 35: Process WorkflowExecutionIntent (LLM-SAFE, synchronous).

        Following Documents 25 & 26 LLM-safe architecture - no async/await.

        Args:
            intent: WorkflowExecutionIntent to process
        """
        if not self.workflow_engine:
            logger.error(
                "No workflow engine available for WorkflowExecutionIntent processing"
            )
            return

        try:
            logger.info(
                f"Processing WorkflowExecutionIntent for request {intent.request_id} with {len(intent.tasks)} tasks"
            )

            # LLM-SAFE: Execute workflow via workflow engine (synchronous)
            self.workflow_engine.execute_workflow_intent(intent)

        except Exception as e:
            logger.error(f"Failed to process WorkflowExecutionIntent: {e}")
            # Don't re-raise - intent processor should handle errors gracefully

            raise

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
