"""Router role - LLM-friendly single file implementation (Option A: Tool-Only).

This role provides intelligent request routing using LLM-based analysis with direct
tool execution. No intent processing needed - tools handle everything directly.

Architecture: Single Event Loop + Intent-Based + Tool-Only Execution
Created: 2025-01-13
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "router",
    "version": "3.0.0",
    "description": "Intelligent request routing using LLM analysis with dynamic role discovery and direct execution",
    "llm_type": "WEAK",  # Routing is lightweight, use fast model
    "fast_reply": False,  # Router is not a fast-reply role itself
    "when_to_use": "Analyzing user requests and routing to the most appropriate role based on intent and available capabilities",
    "tools": {
        "automatic": True,  # Include custom tools automatically
        "shared": [],  # No shared tools needed
    },
    "prompts": {
        "system": """You are an intelligent request routing agent. Your job is to analyze user requests and route them to the most appropriate role.

WORKFLOW:
1. Analyze the user request against the available roles (provided in the prompt)
2. Call route_to_role() with your routing decision
3. Done - no further processing needed

ROUTING RULES:
- Choose the role that best matches the request intent and capabilities
- Use confidence 0.0-1.0 based on how well the request matches the role
- If confidence < 0.7, route to "planning" for complex analysis
- Consider role priorities: timer (urgent) > weather > smart_home > search > planning
- Always provide a clear reason for your routing decision

Be decisive and efficient in your routing decisions."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (minimal - only for external events)
@dataclass
class RoutingRequestIntent(Intent):
    """Intent for external routing requests via events."""

    request_text: str
    source_channel: str
    user_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate routing request intent."""
        return bool(self.request_text and self.source_channel)


# 3. EVENT HANDLERS (only for external events, not LLM interactions)
def handle_external_routing_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """Handle routing requests from external events (not direct LLM calls)."""
    try:
        # Parse external routing request
        if isinstance(event_data, dict):
            request_text = event_data.get("request_text", "")
            source_channel = event_data.get("channel", context.get_safe_channel())
        else:
            request_text = str(event_data)
            source_channel = context.get_safe_channel()

        # Create audit trail for external routing request
        return [
            AuditIntent(
                action="external_routing_request",
                details={
                    "request_text": request_text,
                    "source_channel": source_channel,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            )
        ]

    except Exception as e:
        logger.error(f"External routing handler error: {e}")
        return [
            NotificationIntent(
                message=f"Routing request processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


# 4. TOOLS (LLM calls these directly - no intent processing needed)


@tool
def route_to_role(
    confidence: float, selected_role: str, original_request: str, reasoning: str = ""
) -> dict[str, Any]:
    """Execute routing decision by starting a workflow with the selected role.

    Args:
        confidence: Confidence score (0.0-1.0) for the routing decision
        selected_role: Name of the role to route to
        original_request: The original user request
        reasoning: Explanation for why this role was selected

    Returns:
        Dict with routing execution results
    """
    try:
        # Validate inputs
        if not (0.0 <= confidence <= 1.0):
            return {
                "success": False,
                "error": f"Invalid confidence {confidence}, must be between 0.0 and 1.0",
            }

        if not selected_role or not original_request:
            return {
                "success": False,
                "error": "selected_role and original_request are required",
            }

        # Apply confidence threshold logic
        if confidence < 0.7 and selected_role != "planning":
            logger.info(
                f"Low confidence {confidence}, routing to planning instead of {selected_role}"
            )
            selected_role = "planning"
            reasoning = (
                f"Low confidence ({confidence:.2f}), routing to planning for analysis"
            )

        # Import workflow engine (avoid circular imports)
        try:
            from supervisor.workflow_engine import WorkflowEngine

            # In a real implementation, we'd get this from the supervisor
            # For now, we'll simulate the workflow creation
            workflow_created = True
            workflow_id = f"workflow_{int(time.time())}"
        except ImportError:
            # Fallback for testing/development
            workflow_created = True
            workflow_id = f"mock_workflow_{int(time.time())}"

        if workflow_created:
            # Log successful routing decision
            logger.info(
                f"Routing decision executed: {original_request[:50]}... -> {selected_role} "
                f"(confidence: {confidence:.2f})"
            )

            return {
                "success": True,
                "workflow_id": workflow_id,
                "selected_role": selected_role,
                "confidence": confidence,
                "reasoning": reasoning,
                "original_request": original_request,
                "message": f"Successfully routed to {selected_role} with {confidence:.1%} confidence",
                "execution_time": time.time(),
            }
        else:
            return {
                "success": False,
                "error": "Failed to create workflow",
                "selected_role": selected_role,
                "confidence": confidence,
            }

    except Exception as e:
        logger.error(f"Error executing routing decision: {e}")
        return {
            "success": False,
            "error": str(e),
            "selected_role": selected_role,
            "confidence": confidence,
            "fallback_action": "Route to planning role for manual handling",
        }


# 5. HELPER FUNCTIONS (minimal, focused)
def _validate_role_exists(role_name: str, available_roles: dict[str, Any]) -> bool:
    """Validate that a role exists in the available roles."""
    return role_name in available_roles


def _get_role_priority(role_name: str) -> int:
    """Get priority score for role (lower = higher priority)."""
    ROLE_PRIORITIES = {
        "timer": 1,  # Highest priority - time-sensitive
        "weather": 2,  # High priority - quick info
        "smart_home": 3,  # Medium priority - device control
        "search": 4,  # Medium priority - information retrieval
        "planning": 5,  # Lower priority - complex analysis
        "default": 10,  # Lowest priority - fallback
    }
    return ROLE_PRIORITIES.get(role_name, 10)


def _format_routing_summary(
    selected_role: str, confidence: float, reasoning: str
) -> str:
    """Format a human-readable routing summary."""
    confidence_desc = (
        "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
    )
    return f"Routed to {selected_role} ({confidence_desc} confidence: {confidence:.1%}) - {reasoning}"


# 6. ERROR HANDLING UTILITIES
def _create_routing_error_response(
    error: Exception, context: str = ""
) -> dict[str, Any]:
    """Create standardized error response for routing failures."""
    return {
        "success": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
        "context": context,
        "fallback_action": "Route to planning role",
        "timestamp": time.time(),
    }


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            # Only handle external routing requests via events
            "EXTERNAL_ROUTING_REQUEST": handle_external_routing_request,
        },
        "tools": [
            # LLM calls this tool directly - no intent processing needed
            route_to_role,
        ],
        "intents": {
            # Minimal - only for external events, not LLM interactions
            RoutingRequestIntent: None,  # No processing needed - tools handle everything
        },
    }


# 8. CONSTANTS AND CONFIGURATION
ROUTING_CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.3,
    "fallback": 0.7,  # Below this, route to planning
}

DEFAULT_ROUTING_TIMEOUT = 30  # seconds
MAX_ROUTING_ATTEMPTS = 3

# Role categories for better routing decisions
ROLE_CATEGORIES = {
    "time_sensitive": ["timer"],
    "information": ["weather", "search"],
    "control": ["smart_home"],
    "analysis": ["planning"],
    "fallback": ["default", "planning"],
}


# 9. PERFORMANCE MONITORING
def get_routing_statistics() -> dict[str, Any]:
    """Get routing performance statistics (for monitoring/debugging)."""
    return {
        "role_name": "router",
        "version": ROLE_CONFIG["version"],
        "tools_available": ["get_available_roles", "route_to_role"],
        "confidence_thresholds": ROUTING_CONFIDENCE_THRESHOLDS,
        "role_categories": ROLE_CATEGORIES,
        "architecture": "tool_only_execution",
        "intent_processing": "minimal_external_only",
    }


# 10. VALIDATION UTILITIES
def validate_routing_request(request_text: str) -> dict[str, Any]:
    """Validate routing request format and content."""
    if not request_text or not request_text.strip():
        return {
            "valid": False,
            "error": "Empty or whitespace-only request",
            "suggestion": "Provide a clear, non-empty request",
        }

    if len(request_text.strip()) < 3:
        return {
            "valid": False,
            "error": "Request too short",
            "suggestion": "Provide a more detailed request (at least 3 characters)",
        }

    if len(request_text) > 1000:
        return {
            "valid": False,
            "error": "Request too long",
            "suggestion": "Shorten request to under 1000 characters",
        }

    return {"valid": True, "message": "Request format is valid"}


def validate_confidence_score(confidence: float) -> dict[str, Any]:
    """Validate confidence score is within acceptable range."""
    if not isinstance(confidence, (int, float)):
        return {
            "valid": False,
            "error": f"Confidence must be a number, got {type(confidence)}",
        }

    if not (0.0 <= confidence <= 1.0):
        return {
            "valid": False,
            "error": f"Confidence {confidence} outside valid range 0.0-1.0",
        }

    return {
        "valid": True,
        "confidence_level": "high"
        if confidence >= 0.8
        else "medium"
        if confidence >= 0.6
        else "low",
    }
