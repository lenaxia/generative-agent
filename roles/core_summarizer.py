"""Summarizer role - LLM-friendly single file implementation.

This role provides synthesis, analysis, and structured presentation of information.
Designed to consolidate results from multiple sources and present them clearly.

Key principles:
- Receives predecessor task results automatically via workflow engine
- Focuses on synthesis and structured output, not conversation
- Supports multiple output formats (summary, report, itinerary, analysis)
- Factual and concise presentation
- No conversational personality

Architecture: Single Event Loop + Intent-Based + Lifecycle Functions
Created: 2025-10-31
"""

import logging
from dataclasses import dataclass
from typing import Any

from common.intents import Intent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "summarizer",
    "version": "1.0.0",
    "description": "Synthesize, analyze, and present information in structured formats from multiple sources",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "when_to_use": "Summarize results, create reports, synthesize information from multiple sources, present findings, create structured outputs, consolidate task results, generate itineraries or plans. When used in planning, this will often (but not always) be the final step.",
    "parameters": {
        "format": {
            "type": "string",
            "required": False,
            "description": "Output format for the summary",
            "examples": [
                "summary",
                "report",
                "itinerary",
                "analysis",
                "structured",
                "bullet_points",
            ],
            "enum": [
                "summary",
                "report",
                "itinerary",
                "analysis",
                "structured",
                "bullet_points",
            ],
            "default": "summary",
        },
        "focus": {
            "type": "string",
            "required": False,
            "description": "What aspect to focus on",
            "examples": [
                "key_points",
                "actionable_items",
                "comprehensive",
                "executive",
            ],
        },
        "length": {
            "type": "string",
            "required": False,
            "description": "Desired length of output",
            "examples": ["brief", "detailed", "comprehensive"],
            "enum": ["brief", "detailed", "comprehensive"],
            "default": "detailed",
        },
    },
    "tools": {
        "automatic": False,
        "shared": [],
        "include_builtin": False,
    },
    "lifecycle": {
        "pre_processing": {"enabled": True, "functions": ["load_predecessor_results"]},
        "post_processing": {"enabled": False},
    },
    "prompts": {
        "system": """You are a synthesis and analysis specialist. Your job is to consolidate information from multiple sources and present it clearly.

PREDECESSOR TASK RESULTS:
{{predecessor_results}}

FORMAT REQUESTED: {{format}}
FOCUS: {{focus}}
LENGTH: {{length}}

Your responsibilities:
1. Synthesize information from all predecessor task results
2. Create clear, structured output in the requested format
3. Be concise yet comprehensive
4. Focus on actionable information
5. Present findings logically and coherently
6. Remove redundancy and consolidate related information

Output format guidelines:

**summary**: Concise overview with key points and main findings
**report**: Structured document with sections, findings, and recommendations
**itinerary**: Day-by-day or step-by-step plan with timing and details
**analysis**: Deep dive with insights, patterns, and implications
**structured**: Organized with clear headings, bullet points, and sections
**bullet_points**: Key information in easy-to-scan bullet format

CRITICAL RULES:
- DO NOT engage in casual conversation or add conversational filler
- DO NOT say things like "I'm impressed" or "looking at this"
- DO NOT add personal opinions unless specifically requested
- DO focus on presenting information clearly and factually
- DO structure your output appropriately for the requested format
- DO consolidate and synthesize, don't just repeat what was provided
- DO highlight the most important and actionable information

Provide your synthesis now, formatted according to the request.""",
    },
}


# 2. ROLE-SPECIFIC INTENTS
@dataclass
class SynthesisIntent(Intent):
    """Intent for synthesizing information from multiple sources."""

    sources: list[dict[str, Any]]
    format_type: str = "summary"
    focus: str | None = None
    user_id: str | None = None

    def validate(self) -> bool:
        """Validate synthesis intent parameters."""
        return bool(
            self.sources
            and isinstance(self.sources, list)
            and len(self.sources) > 0
            and self.format_type
            in [
                "summary",
                "report",
                "itinerary",
                "analysis",
                "structured",
                "bullet_points",
            ]
        )


# 3. LIFECYCLE FUNCTIONS
def load_predecessor_results(instruction: str, context, parameters: dict) -> dict:
    """Pre-processor: Extract predecessor results from the instruction.

    The workflow engine adds predecessor results to the instruction text in the format:
    'Previous task results available for context:\n- Result1\n- Result2\n\nCurrent task: ...'

    This function extracts format parameters for prompt injection.
    """
    try:
        # Get format parameters
        format_type = parameters.get("format", "summary")
        focus = parameters.get("focus", "comprehensive")
        length = parameters.get("length", "detailed")

        # Check if instruction contains predecessor results
        has_predecessors = "Previous task results available for context:" in instruction

        if has_predecessors:
            logger.info("Predecessor results detected in instruction")
            predecessor_note = (
                "Predecessor task results are included in your instruction above."
            )
        else:
            logger.info("No predecessor results in instruction")
            predecessor_note = "No predecessor results available. Synthesize based on the current instruction."

        return {
            "predecessor_results": predecessor_note,
            "format": format_type,
            "focus": focus,
            "length": length,
            "has_predecessors": has_predecessors,
        }

    except Exception as e:
        logger.error(f"Failed to load predecessor results: {e}")
        return {
            "predecessor_results": "Error loading predecessor results. Using user instruction only.",
            "format": parameters.get("format", "summary"),
            "focus": parameters.get("focus", "comprehensive"),
            "length": parameters.get("length", "detailed"),
            "has_predecessors": False,
        }


# 4. INTERNAL HELPERS
def _format_predecessor_results(results: list[dict[str, Any]]) -> str:
    """Format predecessor task results for prompt injection.

    Args:
        results: List of predecessor task results with task names and outputs

    Returns:
        Formatted string for prompt injection
    """
    try:
        formatted_parts = []

        for i, result in enumerate(results, 1):
            task_name = result.get("task_name", f"Task {i}")
            task_result = result.get("result", "No result available")

            # Format each result with clear separation
            formatted_parts.append(f"## {task_name}\n\n{task_result}\n")

        return "\n".join(formatted_parts)

    except Exception as e:
        logger.error(f"Error formatting predecessor results: {e}")
        return "Error formatting results. Raw data available in context."


def _extract_key_points(text: str, max_points: int = 5) -> list[str]:
    """Extract key points from text for summarization.

    Simple extraction based on sentence structure and length.
    Could be enhanced with NLP/embeddings in the future.
    """
    try:
        # Split into sentences
        sentences = text.split(". ")

        # Filter for substantial sentences (>20 chars)
        substantial = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Return first N substantial sentences as key points
        return substantial[:max_points]

    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        return []


# 5. EVENT HANDLERS (if needed for external events)
def handle_synthesis_request(event_data: Any, context) -> list[Intent]:
    """Handle external synthesis requests via events.

    This is for event-driven synthesis, not the primary LLM-based workflow.
    """
    try:
        if isinstance(event_data, dict):
            sources = event_data.get("sources", [])
            format_type = event_data.get("format", "summary")
            focus = event_data.get("focus")
        else:
            # Fallback for simple event data
            sources = [{"content": str(event_data)}]
            format_type = "summary"
            focus = None

        return [
            SynthesisIntent(
                sources=sources,
                format_type=format_type,
                focus=focus,
                user_id=getattr(context, "user_id", None),
            )
        ]

    except Exception as e:
        logger.error(f"Synthesis event handler error: {e}")
        from common.intents import NotificationIntent

        return [
            NotificationIntent(
                message=f"Synthesis request processing error: {e}",
                channel=getattr(context, "channel_id", "console"),
                priority="high",
                notification_type="error",
            )
        ]


# 6. ROLE REGISTRATION
def register_role():
    """Auto-discovered by RoleRegistry.

    Returns role configuration, event handlers, tools, and intents.
    """
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "SYNTHESIS_REQUEST": handle_synthesis_request,
        },
        "tools": [],  # No tools needed - synthesis done via LLM
        "intents": {},  # No intent handlers needed - synthesis is LLM-driven
        "lifecycle": {
            "pre_processing": load_predecessor_results,
        },
    }
