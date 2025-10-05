"""
Request Router Module

Implements fast-path routing for the StrandsAgent Universal Agent System.
Routes incoming requests to either fast-reply roles or complex workflow planning
based on request complexity and available fast-reply capabilities.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry, RoleDefinition

# Import Agent from strands (same as UniversalAgent)
from strands import Agent

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of request routing decision."""
    route: str
    confidence: float
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class RequestRouter:
    """
    Routes incoming requests to fast-reply roles or complex workflow planning.
    
    Uses existing WEAK model via LLMFactory for fast, cost-effective routing decisions.
    Integrates with RoleRegistry to identify available fast-reply roles.
    """
    
    def __init__(self, llm_factory: LLMFactory, role_registry: RoleRegistry, universal_agent):
        """
        Initialize RequestRouter with LLM factory and role registry.
        
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
        self._routing_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100  # Limit cache size to prevent memory issues
        
        logger.info("RequestRouter initialized with fast-path routing capabilities and caching")
    
    def route_request(self, instruction: str) -> Dict[str, Any]:
        """
        Route request using existing WEAK model from LLMFactory with caching.
        
        Args:
            instruction: User instruction to route
            
        Returns:
            Dict containing route, confidence, and optional error information
        """
        import time
        start_time = time.time()
        
        try:
            # Check cache first for performance
            cache_key = self._create_cache_key(instruction)
            if cache_key in self._routing_cache:
                cached_result = self._routing_cache[cache_key].copy()
                cached_result["execution_time_ms"] = 0.1  # Cache hit time
                logger.info(f"Cache hit: Routed request to '{cached_result['route']}' with confidence {cached_result['confidence']:.2f} in 0.1ms (cached)")
                return cached_result
            
            # Get fast-reply capable roles
            fast_reply_roles = self.role_registry.get_fast_reply_roles()
            logger.debug(f"Found {len(fast_reply_roles)} fast-reply roles for routing")
            
            # If no fast-reply roles are available, fall back to planning immediately
            if not fast_reply_roles:
                logger.warning("No fast-reply roles available, routing to PLANNING")
                execution_time = (time.time() - start_time) * 1000
                return {
                    "route": "PLANNING",
                    "confidence": 0.0,
                    "error": "No fast-reply roles available",
                    "execution_time_ms": execution_time
                }
            
            # Build routing prompt
            prompt = self._build_routing_prompt(instruction, fast_reply_roles)
            logger.debug(f"Built routing prompt: {prompt[:200]}...")
            
            # Use UniversalAgent for LLM calls (proper architecture)
            response = self.universal_agent.execute_task(
                instruction=prompt,
                role="router",  # Use specialized router role for routing decisions
                llm_type=LLMType.WEAK,
                context=None
            )
            
            logger.debug(f"LLM routing response: {response}")
            
            # Parse response
            result = self._parse_routing_response(response)
            
            # Add execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            result["execution_time_ms"] = execution_time
            
            # Cache the result for future use
            self._cache_routing_result(cache_key, result)
            
            logger.info(f"Routed request to '{result['route']}' with confidence {result['confidence']:.2f} in {execution_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during request routing: {e}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "error": f"Routing failed: {str(e)}",
                "execution_time_ms": execution_time
            }
    
    def _create_cache_key(self, instruction: str) -> str:
        """
        Create a cache key for routing decisions.
        
        Args:
            instruction: User instruction
            
        Returns:
            Cache key string
        """
        import hashlib
        # Normalize instruction for better cache hits
        normalized = instruction.lower().strip()
        # Create hash for consistent key
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _cache_routing_result(self, cache_key: str, result: Dict[str, Any]):
        """
        Cache a routing result.
        
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
    
    def _build_routing_prompt(self, instruction: str, roles: List[RoleDefinition]) -> str:
        """
        Build routing prompt with available fast-reply roles.
        
        Args:
            instruction: User instruction to route
            roles: List of available fast-reply roles
            
        Returns:
            Formatted prompt for routing decision
        """
        roles_list = "\n".join([
            f"- {role.name}: {role.config.get('role', {}).get('description', '')}"
            for role in roles
        ])
        
        return f"""Route this user request to the best option:

USER REQUEST: "{instruction}"

OPTIONS:
{roles_list}
- PLANNING: Multi-step task requiring planning and coordination

Respond with JSON only:
{{"route": "<role_name_or_PLANNING>", "confidence": <0.0-1.0>}}"""
    
    def _parse_routing_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM routing response with fallback handling.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed routing result with route and confidence
        """
        try:
            # Clean and parse JSON response
            cleaned_response = response.strip()
            
            # Handle cases where response might have extra text around JSON
            if '{' in cleaned_response and '}' in cleaned_response:
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
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
                    "error": "Missing route field"
                }
            
            # Ensure confidence is present and valid
            confidence = result.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                logger.warning(f"Invalid confidence value: {confidence}")
                confidence = 0.0
            
            return {
                "route": result["route"],
                "confidence": confidence
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback to planning on parsing errors
            logger.warning(f"Failed to parse routing response '{response}': {e}")
            return {
                "route": "PLANNING",
                "confidence": 0.0,
                "error": f"Failed to parse routing response: {e}"
            }
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics and performance metrics.
        
        Returns:
            Dictionary with routing statistics
        """
        fast_reply_roles = self.role_registry.get_fast_reply_roles()
        
        return {
            "available_fast_reply_roles": len(fast_reply_roles),
            "fast_reply_role_names": [role.name for role in fast_reply_roles],
            "routing_enabled": True,
            "llm_type_used": LLMType.WEAK.value
        }
    
    def validate_routing_setup(self) -> Dict[str, Any]:
        """
        Validate that routing is properly configured.
        
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
            "fast_reply_roles_count": len(fast_reply_roles) if 'fast_reply_roles' in locals() else 0
        }


class FastPathRoutingConfig:
    """Configuration for fast-path routing behavior."""
    
    def __init__(self, 
                 enabled: bool = True,
                 confidence_threshold: float = 0.7,
                 max_response_time_ms: int = 3000,
                 fallback_on_error: bool = True,
                 log_routing_decisions: bool = True,
                 track_performance_metrics: bool = True):
        """
        Initialize fast-path routing configuration.
        
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FastPathRoutingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            FastPathRoutingConfig instance
        """
        return cls(
            enabled=config_dict.get('enabled', True),
            confidence_threshold=config_dict.get('confidence_threshold', 0.7),
            max_response_time_ms=config_dict.get('max_response_time', 3000),
            fallback_on_error=config_dict.get('fallback_on_error', True),
            log_routing_decisions=config_dict.get('log_routing_decisions', True),
            track_performance_metrics=config_dict.get('track_performance_metrics', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'enabled': self.enabled,
            'confidence_threshold': self.confidence_threshold,
            'max_response_time': self.max_response_time_ms,
            'fallback_on_error': self.fallback_on_error,
            'log_routing_decisions': self.log_routing_decisions,
            'track_performance_metrics': self.track_performance_metrics
        }