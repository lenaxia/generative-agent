"""Router role - LLM-friendly single file implementation (JSON Response with Pydantic).

This role provides intelligent request routing using LLM-based analysis with direct
JSON output parsed by Pydantic. No tools needed - LLM outputs structured JSON.

Architecture: Single Event Loop + Intent-Based + JSON Response + Pydantic Parsing
Created: 2025-01-13
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)


# 1. PYDANTIC MODELS FOR JSON PARSING
class RoutingResponse(BaseModel):
    """Pydantic model for parsing LLM routing responses."""

    route: str = Field(..., description="Selected role name")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Optional parameters for the selected role"
    )

    class Config:
        extra = "forbid"  # Don't allow extra fields


# 2. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "router",
    "version": "4.0.0",
    "description": "Intelligent request routing using LLM analysis with JSON response format",
    "llm_type": "WEAK",  # Routing is lightweight, use fast model
    "fast_reply": False,  # Router is not a fast-reply role itself
    "when_to_use": "Analyzing user requests and routing to the most appropriate role based on intent and available capabilities",
    "tools": {
        "automatic": False,  # No tools - JSON response only
        "shared": [],  # No shared tools needed
        "include_builtin": False,  # No built-in tools (calculator, file_read, shell)
    },
    "prompts": {
        "system": """You are an intelligent request routing agent. Your job is to analyze user requests and respond with ONLY valid JSON.

CRITICAL: Respond with ONLY valid JSON. No explanations, no additional text, no markdown formatting.

Available roles will be provided in the prompt. Analyze the user request and respond with JSON in this EXACT format:

<routing_response> ::= "{" <route_field> "," <confidence_field> "," <parameters_field> "}"
<route_field> ::= '"route":' <role_name>
<confidence_field> ::= '"confidence":' <confidence_value>
<parameters_field> ::= '"parameters":' <parameters_object>
<role_name> ::= '"' <string> '"'
<confidence_value> ::= <number_between_0_and_1>
<parameters_object> ::= "{" "}"

Example:
{
  "route": "weather",
  "confidence": 0.95,
  "parameters": {}
}

ROUTING RULES:
- Choose the role that best matches the request intent and capabilities
- Use confidence 0.0-1.0 based on how well the request matches the role
- If confidence < 0.7, route to "planning" for complex analysis
- Consider role priorities: timer (urgent) > weather > smart_home > search > planning
- Respond with ONLY the JSON object, nothing else"""
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


# 4. NO TOOLS - JSON Response Only
# Router role uses direct JSON output instead of tool calls


# 5. ROUTING FUNCTIONS (owned by router role)
def route_request_with_available_roles(
    request_text: str, role_registry
) -> dict[str, Any]:
    """Route request using available roles - owned by router role.

    Args:
        request_text: The user request to route
        role_registry: RoleRegistry instance for getting available roles

    Returns:
        Dict with routing decision: {route, confidence, parameters, valid}
    """
    try:
        # Get available fast-reply roles
        fast_reply_roles = role_registry.get_fast_reply_roles()

        # Build role information for injection into prompt
        available_roles_info = {}
        for role_def in fast_reply_roles:
            role_config = role_def.config.get("role", {})
            role_name = role_def.name

            # Get parameter schema if available
            parameters_schema = role_config.get("parameters", {})

            available_roles_info[role_name] = {
                "description": role_config.get("description", ""),
                "when_to_use": role_config.get("when_to_use", ""),
                "capabilities": role_config.get("capabilities", []),
                "fast_reply": role_config.get("fast_reply", False),
                "parameters": parameters_schema,
            }

        # Always add planning as fallback
        if "planning" not in available_roles_info:
            available_roles_info["planning"] = {
                "description": "Complex task planning and analysis for multi-step workflows",
                "when_to_use": "Complex requests requiring planning, analysis, or multi-step execution",
                "capabilities": ["planning", "analysis", "complex_workflows"],
                "fast_reply": False,
            }

        # Build routing instruction with pre-injected role information and parameter schemas
        roles_description_parts = []
        for role_name, info in available_roles_info.items():
            role_desc = f"- {role_name}: {info['description']} (Use when: {info['when_to_use']})"

            # Add parameter schema if available
            if info.get("parameters"):
                param_examples = []
                for param_name, param_info in info["parameters"].items():
                    required_str = (
                        "required" if param_info.get("required") else "optional"
                    )
                    examples = param_info.get("examples", [])
                    example_str = (
                        f" (e.g., {', '.join(examples[:2])})" if examples else ""
                    )
                    param_examples.append(f"{param_name} ({required_str}){example_str}")

                if param_examples:
                    role_desc += f"\n  Parameters: {'; '.join(param_examples)}"

            roles_description_parts.append(role_desc)

        roles_description = "\n".join(roles_description_parts)

        routing_instruction = f"""USER REQUEST: "{request_text}"

AVAILABLE ROLES:
{roles_description}

Extract relevant parameters from the user request based on the role's parameter schema.
Respond with ONLY valid JSON in this exact format:
{{
  "route": "role_name",
  "confidence": 0.95,
  "parameters": {{
    "param_name": "extracted_value"
  }}
}}"""

        # Execute with router role - this will output JSON
        from llm_provider.factory import LLMType
        from llm_provider.universal_agent import UniversalAgent

        # Get universal agent from role registry (avoid circular imports)
        universal_agent = getattr(role_registry, "_universal_agent", None)
        if not universal_agent:
            # Fallback: create minimal universal agent
            logger.warning("No universal agent available, using fallback routing")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "valid": False,
                "error": "No universal agent available",
            }

        result = universal_agent.execute_task(
            instruction=routing_instruction, role="router", llm_type=LLMType.WEAK
        )

        # Parse routing JSON response using Pydantic
        return parse_routing_response(result)

    except Exception as e:
        logger.error(f"Router role routing failed: {e}")
        return {
            "route": "PLANNING",
            "confidence": 0.0,
            "parameters": {},
            "valid": False,
            "error": str(e),
        }


# 6. HELPER FUNCTIONS (minimal, focused)
def parse_routing_response(response_text: str) -> dict[str, Any]:
    """Parse routing response JSON from LLM using Pydantic validation."""
    try:
        # Clean the response text
        response_text = response_text.strip()

        # Parse JSON and validate with Pydantic
        routing_response = RoutingResponse.model_validate_json(response_text)

        # Apply confidence threshold logic
        selected_role = routing_response.route.lower()
        confidence = routing_response.confidence

        if confidence < 0.7 and selected_role != "planning":
            logger.info(
                f"Low confidence {confidence}, routing to planning instead of {selected_role}"
            )
            selected_role = "planning"
            confidence = 0.6  # Set reasonable confidence for planning fallback

        logger.info(
            f"Parsed routing decision: {selected_role} with confidence {confidence:.2f}"
        )

        return {
            "route": selected_role,  # Keep original case for role registry compatibility
            "confidence": confidence,
            "parameters": routing_response.parameters,
            "valid": True,
        }

    except ValidationError as e:
        logger.error(
            f"Pydantic validation failed for routing response '{response_text}': {e}"
        )
        return {
            "route": "PLANNING",
            "confidence": 0.0,
            "parameters": {},
            "valid": False,
            "error": f"Validation error: {str(e)}",
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed for routing response '{response_text}': {e}")
        return {
            "route": "PLANNING",
            "confidence": 0.0,
            "parameters": {},
            "valid": False,
            "error": f"JSON decode error: {str(e)}",
        }
    except Exception as e:
        logger.error(
            f"Unexpected error parsing routing response '{response_text}': {e}"
        )
        return {
            "route": "PLANNING",
            "confidence": 0.0,
            "parameters": {},
            "valid": False,
            "error": f"Unexpected error: {str(e)}",
        }


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


# 6. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            # Only handle external routing requests via events
            "EXTERNAL_ROUTING_REQUEST": handle_external_routing_request,
        },
        "tools": [],  # No tools - JSON response only
        "intents": {
            # Minimal - only for external events, not LLM interactions
            RoutingRequestIntent: None,  # No processing needed - JSON parsing handles everything
        },
    }


# 7. CONSTANTS AND CONFIGURATION
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


# 8. PERFORMANCE MONITORING
def get_routing_statistics() -> dict[str, Any]:
    """Get routing performance statistics (for monitoring/debugging)."""
    return {
        "role_name": "router",
        "version": ROLE_CONFIG["version"],
        "response_format": "json_only",
        "confidence_thresholds": ROUTING_CONFIDENCE_THRESHOLDS,
        "role_categories": ROLE_CATEGORIES,
        "architecture": "json_response_parsing",
        "tools": "none_json_only",
    }


# 9. HELPER FUNCTIONS
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
    selected_role: str, confidence: float, reasoning: str = ""
) -> str:
    """Format a human-readable routing summary."""
    confidence_desc = (
        "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
    )
    return (
        f"Routed to {selected_role} ({confidence_desc} confidence: {confidence:.1%})"
        + (f" - {reasoning}" if reasoning else "")
    )


# 10. ERROR HANDLING UTILITIES
def _create_routing_error_response(
    error: Exception, context: str = ""
) -> dict[str, Any]:
    """Create standardized error response for routing failures."""
    return {
        "route": "PLANNING",
        "confidence": 0.0,
        "parameters": {},
        "valid": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
        "context": context,
        "timestamp": time.time(),
    }
