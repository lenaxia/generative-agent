"""Memory importance assessor using LLM for structured assessment.

This module provides the MemoryImportanceAssessor class that uses an LLM
(WEAK model for speed and cost) to assess the importance of interactions
and generate structured metadata including summary, tags, and topics.
"""

import asyncio
import json
import logging
from typing import Optional

from pydantic import ValidationError

from common.memory_assessment import MemoryAssessment
from llm_provider.factory import LLMFactory, LLMType

logger = logging.getLogger(__name__)


class MemoryImportanceAssessor:
    """Assesses memory importance using LLM with structured JSON output.

    This class uses a WEAK LLM model to quickly assess the importance of
    interactions and generate metadata for storage in the unified memory system.
    """

    def __init__(
        self,
        llm_factory: LLMFactory,
        permanent_threshold: float = 0.7,
        timeout_seconds: int = 5,
    ):
        """Initialize the memory importance assessor.

        Args:
            llm_factory: Factory for creating LLM agents
            permanent_threshold: Importance threshold for permanent storage (default: 0.7)
            timeout_seconds: Timeout for LLM assessment calls (default: 5)
        """
        self.llm_factory = llm_factory
        self.permanent_threshold = permanent_threshold
        self.timeout_seconds = timeout_seconds
        self.agent = None

    async def initialize(self):
        """Initialize the assessor with WEAK model for fast, cheap assessments."""
        try:
            self.agent = self.llm_factory.get_agent(LLMType.WEAK)
            logger.info("Memory importance assessor initialized with WEAK model")
        except Exception as e:
            logger.error(f"Failed to initialize memory assessor: {e}")
            raise

    async def assess_memory(
        self,
        user_message: str,
        assistant_response: str,
        source_role: str,
        context: dict | None = None,
    ) -> MemoryAssessment | None:
        """Assess memory importance and generate metadata.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            source_role: Role that generated the response
            context: Optional additional context

        Returns:
            MemoryAssessment if successful, None if assessment fails
        """
        if not self.agent:
            logger.warning("Assessor not initialized, cannot assess memory")
            return None

        try:
            # Build assessment prompt
            prompt = self._build_prompt(
                user_message, assistant_response, source_role, context or {}
            )

            # LLM call with timeout
            result = await asyncio.wait_for(
                self.agent.execute(prompt), timeout=self.timeout_seconds
            )

            # Parse JSON response
            assessment_data = json.loads(result)

            # Validate with Pydantic
            assessment = MemoryAssessment(**assessment_data)

            logger.debug(
                f"Memory assessed: importance={assessment.importance}, tags={assessment.tags}"
            )
            return assessment

        except asyncio.TimeoutError:
            logger.warning(f"Memory assessment timed out after {self.timeout_seconds}s")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse assessment JSON: {e}")
            return None
        except ValidationError as e:
            logger.warning(f"Assessment validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Memory assessment failed: {e}")
            return None

    def calculate_ttl(self, importance: float) -> int | None:
        """Calculate TTL based on importance with graduated retention.

        Args:
            importance: Importance score between 0.0 and 1.0

        Returns:
            TTL in seconds, or None for permanent storage
        """
        if importance >= self.permanent_threshold:
            return None  # Permanent storage

        # Graduated TTL for lower importance
        if importance >= 0.5:
            return 30 * 24 * 60 * 60  # 1 month
        elif importance >= 0.3:
            return 7 * 24 * 60 * 60  # 1 week
        else:
            return 3 * 24 * 60 * 60  # 3 days

    def _build_prompt(
        self, user_msg: str, assistant_resp: str, role: str, ctx: dict
    ) -> str:
        """Build assessment prompt for LLM.

        Args:
            user_msg: User's message
            assistant_resp: Assistant's response
            role: Source role
            ctx: Additional context

        Returns:
            Complete prompt string
        """
        context_str = json.dumps(ctx) if ctx else "{}"

        return f"""You are a memory importance assessor. Analyze interactions and respond with ONLY valid JSON.

CRITICAL: Respond with ONLY valid JSON. No explanations, no additional text.

Your task is to assess the importance of storing this interaction as a long-term memory.

IMPORTANCE SCORING GUIDELINES:
0.9-1.0: Critical (explicit preferences, important decisions, key facts to remember)
0.7-0.8: High (significant conversations, important events, useful information)
0.5-0.6: Medium (normal conversations, regular events, general information)
0.3-0.4: Low (casual chat, minor events, transient information)
0.0-0.2: Minimal (greetings, acknowledgments, trivial exchanges)

SUMMARY GUIDELINES:
- Concise but complete
- Self-contained (understandable without full content)
- Include key details (who, what, when, where if relevant)
- Actionable (what was decided/discussed)

TAGGING GUIDELINES:
- Use lowercase only
- Be specific: "project_meeting" not just "meeting"
- Include context: "work", "personal", "urgent"
- 1-10 tags maximum
- Common tags: work, personal, urgent, project, meeting, planning, decision, preference

TOPIC EXTRACTION:
- Main subjects discussed
- Key entities mentioned (people, places, projects)
- Important concepts
- 0-5 topics maximum

Analyze this interaction:

User: {user_msg}
Assistant: {assistant_resp}

Role: {role}
Context: {context_str}

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "importance": 0.7,
  "summary": "User scheduled team meeting for tomorrow at 2pm to discuss Q4 budget",
  "tags": ["meeting", "work", "team", "budget", "q4"],
  "topics": ["Q4 Budget Planning", "Team Meetings"],
  "reasoning": "Important work event with specific time, participants, and agenda"
}}

NO EXPLANATIONS. ONLY JSON."""
