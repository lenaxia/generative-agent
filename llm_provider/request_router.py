"""Request Router Module

Implements fast-path routing for the StrandsAgent Universal Agent System.
Routes incoming requests to either fast-reply roles or complex workflow planning
based on request complexity and available fast-reply capabilities.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleDefinition, RoleRegistry

# Import Agent from strands (same as UniversalAgent)
try:
    from strands import Agent
except ImportError:
    # Fallback for testing when strands is not available
    from llm_provider.factory import LLMFactory

    Agent = LLMFactory.Agent

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of request routing decision."""

    route: str
    confidence: float
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class RequestRouter:
    """Routes incoming requests to fast-reply roles or complex workflow planning.

    Uses existing WEAK model via LLMFactory for fast, cost-effective routing decisions.
    Integrates with RoleRegistry to identify available fast-reply roles.
    """

    def __init__(
        self, llm_factory: LLMFactory, role_registry: RoleRegistry, universal_agent
    ):
        """Initialize RequestRouter with LLM factory and role registry.

        Args:
            llm_factory: LLMFactory for creating routing models
            role_registry: RoleRegistry for fast-reply role discovery
            universal_agent: UniversalAgent for LLM calls (required)
        """
        if universal_agent is None:
            raise ValueError("UniversalAgent is required for RequestRouter")

        self.llm_factory = llm_factory
        self.role_registry = role_registry
        self.universal_agent = universal_agent

        # Performance optimization: Cache routing decisions for similar queries
        self._routing_cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size = 100  # Limit cache size to prevent memory issues

        logger.info(
            "RequestRouter initialized with fast-path routing capabilities and caching"
        )

    def route_request(self, instruction: str) -> dict[str, Any]:
        """Enhanced routing with parameter extraction in single LLM call.

        Returns:
            Dict containing route, confidence, and extracted parameters
        """
        import time

        start_time = time.time()

        # Handle None or empty instruction
        if instruction is None:
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": "No instruction provided",
            }

        try:
            # Check cache first
            cache_key = self._create_cache_key(instruction)
            if cache_key in self._routing_cache:
                cached_result = self._routing_cache[cache_key].copy()
                cached_result["execution_time_ms"] = 0.1
                return cached_result

            # Get fast-reply roles with parameter schemas
            fast_reply_roles = self.role_registry.get_fast_reply_roles()
            if not fast_reply_roles:
                return {"route": "PLANNING", "confidence": 0.0, "parameters": {}}

            # Build enhanced routing prompt with parameter schemas
            routing_prompt = self._build_enhanced_routing_prompt(
                instruction, fast_reply_roles
            )

            # Single LLM call for routing AND parameter extraction
            result = self.universal_agent.execute_task(
                instruction=routing_prompt, role="router", llm_type=LLMType.WEAK
            )

            # Parse routing result with parameters
            parsed_result = self._parse_routing_and_parameters(result)

            # Cache the result
            if len(self._routing_cache) >= self._cache_max_size:
                self._routing_cache.clear()
            self._routing_cache[cache_key] = parsed_result.copy()

            execution_time_ms = (time.time() - start_time) * 1000
            parsed_result["execution_time_ms"] = execution_time_ms

            logger.info(
                f"Enhanced routing: '{parsed_result['route']}' with confidence {parsed_result['confidence']:.2f} and {len(parsed_result.get('parameters', {}))} parameters in {execution_time_ms:.1f}ms"
            )
            return parsed_result

        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": str(e),
            }

    def _create_cache_key(self, instruction: str) -> str:
        """Create a cache key for routing decisions.

        Args:
            instruction: User instruction

        Returns:
            Cache key string
        """
        import hashlib

        # Handle None instruction
        if instruction is None:
            logger.warning("Received None instruction for cache key generation")
            instruction = ""

        # Normalize instruction for better cache hits
        normalized = str(instruction).lower().strip()
        # Create hash for consistent key
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _cache_routing_result(self, cache_key: str, result: dict[str, Any]):
        """Cache a routing result.

        Args:
            cache_key: Cache key
            result: Routing result to cache
        """
        # Manage cache size
        if len(self._routing_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._routing_cache))
            del self._routing_cache[oldest_key]
            logger.debug(f"Removed oldest cache entry: {oldest_key}")

        # Cache the result (exclude execution_time_ms)
        cached_result = {k: v for k, v in result.items() if k != "execution_time_ms"}
        self._routing_cache[cache_key] = cached_result
        logger.debug(f"Cached routing result for key: {cache_key}")

    def _build_routing_prompt(
        self, instruction: str, roles: list[RoleDefinition]
    ) -> str:
        """Build routing prompt with available fast-reply roles.

        Args:
            instruction: User instruction to route
            roles: List of available fast-reply roles

        Returns:
            Formatted prompt for routing decision
        """
        roles_list = "\n".join(
            [
                f"- {role.name}: {role.config.get('role', {}).get('description', '')}"
                for role in roles
            ]
        )

        return f"""Route this user request to the best option:

USER REQUEST: "{instruction}"

OPTIONS:
{roles_list}
- PLANNING: Multi-step task requiring planning and coordination

Respond with JSON only:
{{"route": "<role_name_or_PLANNING>", "confidence": <0.0-1.0>}}"""

    def _parse_routing_response(self, response: str) -> dict[str, Any]:
        """Parse LLM routing response with fallback handling.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed routing result with route and confidence
        """
        try:
            # Clean and parse JSON response
            cleaned_response = response.strip()

            # Handle cases where response might have extra text around JSON
            if "{" in cleaned_response and "}" in cleaned_response:
                start_idx = cleaned_response.find("{")
                end_idx = cleaned_response.rfind("}") + 1
                json_part = cleaned_response[start_idx:end_idx]
            else:
                json_part = cleaned_response

            result = json.loads(json_part)
            logger.debug(f"Parsed routing result: {result}")

            # Validate required fields
            if "route" not in result:
                logger.warning("Missing route field in LLM response")
                return {
                    "route": "PLANNING",
                    "confidence": 0.0,
                    "error": "Missing route field",
                }

            # Ensure confidence is present and valid
            confidence = result.get("confidence", 0.0)
            if (
                not isinstance(confidence, (int, float))
                or confidence < 0
                or confidence > 1
            ):
                logger.warning(f"Invalid confidence value: {confidence}")
                confidence = 0.0

            return {"route": result["route"], "confidence": confidence}

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to planning on parsing errors
            logger.warning(f"Failed to parse routing response '{response}': {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "error": f"Failed to parse routing response: {e}",
            }

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get routing statistics and performance metrics.

        Returns:
            Dictionary with routing statistics
        """
        fast_reply_roles = self.role_registry.get_fast_reply_roles()

        return {
            "available_fast_reply_roles": len(fast_reply_roles),
            "fast_reply_role_names": [role.name for role in fast_reply_roles],
            "routing_enabled": True,
            "llm_type_used": LLMType.WEAK.value,
        }

    def validate_routing_setup(self) -> dict[str, Any]:
        """Validate that routing is properly configured.

        Returns:
            Validation result with status and any issues
        """
        issues = []
        warnings = []

        # Check LLM factory
        try:
            model = self.llm_factory.create_strands_model(LLMType.WEAK)
            if model is None:
                issues.append("WEAK model not available from LLMFactory")
        except Exception as e:
            issues.append(f"Failed to create WEAK model: {e}")

        # Check role registry
        try:
            fast_reply_roles = self.role_registry.get_fast_reply_roles()
            if not fast_reply_roles:
                warnings.append("No fast-reply roles available")
            else:
                logger.info(f"Found {len(fast_reply_roles)} fast-reply roles")
        except Exception as e:
            issues.append(f"Failed to get fast-reply roles: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "fast_reply_roles_count": (
                len(fast_reply_roles) if "fast_reply_roles" in locals() else 0
            ),
        }

    def _build_enhanced_routing_prompt(
        self, instruction: str, fast_reply_roles: list
    ) -> str:
        """Build routing prompt that includes parameter extraction."""
        # Build role schemas for parameter extraction
        role_schemas = {}
        for role_def in fast_reply_roles:
            role_name = role_def.name
            parameters = self.role_registry.get_role_parameters(role_name)
            if parameters:
                role_schemas[role_name] = {
                    "description": role_def.config.get("role", {}).get(
                        "description", ""
                    ),
                    "parameters": parameters,
                }

        return f"""Route this request to the best role AND extract parameters for that role.

Request: "{instruction}"

Available roles and their parameters:
{json.dumps(role_schemas, indent=2)}

Analyze the request and respond with JSON in this exact format:
{{
  "route": "role_name",
  "confidence": 0.95,
  "parameters": {{
    "param_name": "extracted_value"
  }}
}}

Rules:
- Choose the role that best matches the request intent
- Extract only the parameters defined for the chosen role
- For enum parameters, pick from the allowed values only
- Use examples as guidance for parameter format
- Use confidence 0.0-1.0 based on how well the request matches the role
- If no role matches well, use "PLANNING" with confidence < 0.7
- Ensure all required parameters are extracted if possible"""

    def _parse_routing_and_parameters(self, llm_result: str) -> dict[str, Any]:
        """Parse LLM result to extract route and parameters."""
        try:
            # Clean the result and parse JSON
            cleaned_result = llm_result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:-3].strip()
            elif cleaned_result.startswith("```"):
                cleaned_result = cleaned_result[3:-3].strip()

            parsed = json.loads(cleaned_result)

            return {
                "route": parsed.get("route", "PLANNING"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "parameters": parsed.get("parameters", {}),
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse routing result: {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "parameters": {},
                "error": f"Parse error: {str(e)}",
            }


class FastPathRoutingConfig:
    """Configuration for fast-path routing behavior."""

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.7,
        max_response_time_ms: int = 3000,
        fallback_on_error: bool = True,
        log_routing_decisions: bool = True,
        track_performance_metrics: bool = True,
    ):
        """Initialize fast-path routing configuration.

        Args:
            enabled: Whether fast-path routing is enabled
            confidence_threshold: Minimum confidence for fast-path routing
            max_response_time_ms: Target response time in milliseconds
            fallback_on_error: Fall back to planning on fast-path errors
            log_routing_decisions: Log routing decisions for monitoring
            track_performance_metrics: Track performance metrics
        """
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.max_response_time_ms = max_response_time_ms
        self.fallback_on_error = fallback_on_error
        self.log_routing_decisions = log_routing_decisions
        self.track_performance_metrics = track_performance_metrics

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FastPathRoutingConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            FastPathRoutingConfig instance
        """
        return cls(
            enabled=config_dict.get("enabled", True),
            confidence_threshold=config_dict.get("confidence_threshold", 0.7),
            max_response_time_ms=config_dict.get("max_response_time", 3000),
            fallback_on_error=config_dict.get("fallback_on_error", True),
            log_routing_decisions=config_dict.get("log_routing_decisions", True),
            track_performance_metrics=config_dict.get(
                "track_performance_metrics", True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "max_response_time": self.max_response_time_ms,
            "fallback_on_error": self.fallback_on_error,
            "log_routing_decisions": self.log_routing_decisions,
            "track_performance_metrics": self.track_performance_metrics,
        }
