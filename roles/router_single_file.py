"""Router role - LLM-friendly single file implementation.

This role consolidates all router functionality into a single file following
the new LLM-safe architecture patterns from Documents 25, 26, and 27.

Migrated from: roles/router/ (definition.yaml only)
Total reduction: ~76 lines → ~150 lines (expanded for LLM-safe patterns)
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "router",
    "version": "2.0.0",
    "description": "Specialized role for intelligent request routing and task classification using LLM-safe architecture",
    "llm_type": "WEAK",
    "fast_reply": False,  # Not a fast-reply role, used for routing decisions
    "when_to_use": "Making routing decisions, classifying user intents, determining best role for requests, analyzing request complexity",
}


# 2. ROLE-SPECIFIC INTENTS (owned by router role)
@dataclass
class RoutingIntent(Intent):
    """Routing-specific intent - owned by router role."""

    action: str  # "classify", "route", "analyze"
    request_text: str
    target_role: Optional[str] = None
    confidence: Optional[float] = None

    def validate(self) -> bool:
        """Validate routing intent parameters."""
        valid_actions = ["classify", "route", "analyze"]
        confidence_valid = self.confidence is None or (0.0 <= self.confidence <= 1.0)
        return bool(
            self.action
            and self.action in valid_actions
            and self.request_text
            and confidence_valid
        )


@dataclass
class RouteDecisionIntent(Intent):
    """Route decision intent - owned by router role."""

    original_request: str
    selected_role: str
    confidence_score: float
    reasoning: str

    def validate(self) -> bool:
        """Validate route decision intent parameters."""
        return bool(
            self.original_request
            and self.selected_role
            and 0.0 <= self.confidence_score <= 1.0
            and self.reasoning
        )


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_routing_request(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for routing request events."""
    try:
        # Parse event data
        request_text, routing_type = _parse_routing_event_data(event_data)

        # Create intents
        return [
            RoutingIntent(
                action="classify",
                request_text=request_text,
                target_role=None,
                confidence=None,
            ),
            AuditIntent(
                action="routing_request",
                details={
                    "request_text": request_text,
                    "routing_type": routing_type,
                    "processed_at": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Routing handler error: {e}")
        return [
            NotificationIntent(
                message=f"Routing processing error: {e}",
                channel=context.get_safe_channel(),
                priority="high",
                notification_type="error",
            )
        ]


def handle_route_decision(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """LLM-SAFE: Pure function for route decision events."""
    try:
        # Parse route decision data
        decision_data = _parse_route_decision_event(event_data)

        return [
            RouteDecisionIntent(
                original_request=decision_data.get("request", "unknown"),
                selected_role=decision_data.get("role", "default"),
                confidence_score=decision_data.get("confidence", 0.5),
                reasoning=decision_data.get("reasoning", "Automatic routing"),
            ),
            AuditIntent(
                action="route_decision",
                details=decision_data,
                user_id=context.user_id,
                severity="info",
            ),
        ]

    except Exception as e:
        logger.error(f"Route decision error: {e}")
        return [
            NotificationIntent(
                message=f"Route decision error: {e}",
                channel=context.get_safe_channel(),
                priority="medium",
                notification_type="warning",
            )
        ]


# 4. TOOLS (simplified, LLM-friendly)
def classify_request(request_text: str) -> dict[str, Any]:
    """LLM-SAFE: Classify user request for routing."""
    logger.info(f"Classifying request: {request_text[:50]}...")

    try:
        # Simple classification logic
        request_lower = request_text.lower()

        # Weather-related keywords
        if any(
            word in request_lower
            for word in ["weather", "temperature", "forecast", "rain", "sunny"]
        ):
            return {
                "classification": "weather",
                "confidence": 0.9,
                "reasoning": "Contains weather-related keywords",
                "suggested_role": "weather",
            }

        # Timer-related keywords
        elif any(
            word in request_lower
            for word in ["timer", "alarm", "remind", "minutes", "hours"]
        ):
            return {
                "classification": "timer",
                "confidence": 0.9,
                "reasoning": "Contains timer-related keywords",
                "suggested_role": "timer",
            }

        # Smart home keywords
        elif any(
            word in request_lower
            for word in ["lights", "thermostat", "temperature", "device", "home"]
        ):
            return {
                "classification": "smart_home",
                "confidence": 0.8,
                "reasoning": "Contains smart home keywords",
                "suggested_role": "smart_home",
            }

        # Planning keywords
        elif any(
            word in request_lower
            for word in ["plan", "strategy", "analyze", "complex", "steps"]
        ):
            return {
                "classification": "planning",
                "confidence": 0.7,
                "reasoning": "Contains planning-related keywords",
                "suggested_role": "planning",
            }

        # Default classification
        else:
            return {
                "classification": "general",
                "confidence": 0.5,
                "reasoning": "No specific domain detected",
                "suggested_role": "default",
            }

    except Exception as e:
        logger.error(f"Error classifying request: {e}")
        return {
            "classification": "error",
            "confidence": 0.0,
            "error": str(e),
            "suggested_role": "default",
        }


def route_request(
    request_text: str, available_roles: list[str] = None
) -> dict[str, Any]:
    """LLM-SAFE: Route request to appropriate role."""
    logger.info(f"Routing request: {request_text[:50]}...")

    try:
        # Get classification
        classification = classify_request(request_text)

        # Determine final route
        suggested_role = classification.get("suggested_role", "default")
        confidence = classification.get("confidence", 0.5)

        # Check if suggested role is available
        if available_roles and suggested_role not in available_roles:
            suggested_role = "default"
            confidence = max(0.3, confidence - 0.2)  # Reduce confidence

        # Format as expected JSON response
        route_decision = {"route": suggested_role, "confidence": confidence}

        return {
            "success": True,
            "route_decision": route_decision,
            "classification": classification,
            "message": f"Routed to {suggested_role} with {confidence:.1f} confidence",
        }

    except Exception as e:
        logger.error(f"Error routing request: {e}")
        return {
            "success": False,
            "error": str(e),
            "route_decision": {"route": "default", "confidence": 0.1},
        }


# 5. HELPER FUNCTIONS (minimal, focused)
def _parse_routing_event_data(event_data: Any) -> tuple[str, str]:
    """LLM-SAFE: Parse routing event data with error handling."""
    try:
        if isinstance(event_data, dict):
            return (
                event_data.get("request_text", "unknown request"),
                event_data.get("routing_type", "standard"),
            )
        elif isinstance(event_data, str):
            return event_data, "standard"
        else:
            return str(event_data), "standard"
    except Exception as e:
        return f"parse_error: {e}", "error"


def _parse_route_decision_event(event_data: Any) -> dict[str, Any]:
    """LLM-SAFE: Parse route decision event data."""
    try:
        if isinstance(event_data, dict):
            return event_data
        elif isinstance(event_data, str):
            # Try to parse as JSON
            try:
                return json.loads(event_data)
            except json.JSONDecodeError:
                return {"request": event_data, "role": "default", "confidence": 0.5}
        else:
            return {"request": str(event_data), "role": "default", "confidence": 0.5}
    except Exception as e:
        return {"error": str(e), "role": "default", "confidence": 0.1}


# 6. INTENT HANDLER REGISTRATION
async def process_routing_intent(intent: RoutingIntent):
    """Process routing-specific intents - called by IntentProcessor."""
    logger.info(f"Processing routing intent: {intent.action}")

    # In full implementation, this would:
    # - Perform request classification
    # - Make routing decisions
    # - Update routing metrics
    # For now, just log the intent processing


async def process_route_decision_intent(intent: RouteDecisionIntent):
    """Process route decision intents - called by IntentProcessor."""
    logger.info(
        f"Processing route decision: {intent.selected_role} (confidence: {intent.confidence_score})"
    )

    # In full implementation, this would:
    # - Execute the routing decision
    # - Update routing statistics
    # - Handle routing failures
    # For now, just log the intent processing


# 7. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "ROUTING_REQUEST": handle_routing_request,
            "ROUTE_DECISION": handle_route_decision,
        },
        "tools": [],  # Router should have NO tools - routing decisions are made via LLM prompts only
        "intents": {
            RoutingIntent: process_routing_intent,
            RouteDecisionIntent: process_route_decision_intent,
        },
    }


# 8. CONSTANTS AND CONFIGURATION
ROUTING_CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}

DEFAULT_ROUTING_TIMEOUT = 30  # seconds
MAX_ROUTING_ATTEMPTS = 3

# Role priority mapping for routing decisions
ROLE_PRIORITIES = {
    "timer": 1,  # High priority for time-sensitive requests
    "weather": 2,  # High priority for weather requests
    "smart_home": 3,  # Medium priority for device control
    "search": 4,  # Medium priority for information requests
    "planning": 5,  # Lower priority for complex planning
    "default": 10,  # Lowest priority fallback
}


def get_role_priority(role_name: str) -> int:
    """Get priority score for role (lower = higher priority)."""
    return ROLE_PRIORITIES.get(role_name, 10)


# 9. ENHANCED ERROR HANDLING
def create_routing_error_intent(
    error: Exception, context: LLMSafeEventContext
) -> list[Intent]:
    """Create error intents for routing operations."""
    return [
        NotificationIntent(
            message=f"Routing error: {error}",
            channel=context.get_safe_channel(),
            user_id=context.user_id,
            priority="high",
            notification_type="error",
        ),
        AuditIntent(
            action="routing_error",
            details={"error": str(error), "context": context.to_dict()},
            user_id=context.user_id,
            severity="error",
        ),
    ]


# 10. ROUTING UTILITIES
def parse_routing_response(response_text: str) -> dict[str, Any]:
    """Parse routing response from LLM."""
    try:
        # Try to parse as JSON
        parsed = json.loads(response_text.strip())

        # Validate required fields
        if "route" not in parsed or "confidence" not in parsed:
            raise ValueError("Missing required fields: route, confidence")

        # Validate confidence range
        confidence = float(parsed["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence {confidence} outside valid range 0.0-1.0")

        return {"route": str(parsed["route"]), "confidence": confidence, "valid": True}

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse routing response: {e}")
        return {"route": "default", "confidence": 0.1, "valid": False, "error": str(e)}
