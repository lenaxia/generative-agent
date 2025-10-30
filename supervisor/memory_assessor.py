"""Memory assessor module for evaluating and storing important memories.

This module provides the MemoryAssessor class that uses LLM to evaluate
the importance of conversations and interactions, storing significant
memories for future context retrieval.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from common.interfaces.context_interfaces import MemoryProvider
from llm_provider.factory import LLMFactory
from llm_provider.universal_agent import UniversalAgent

logger = logging.getLogger(__name__)


class MemoryAssessor:
    """Assesses conversation importance and stores significant memories."""

    def __init__(self, memory_provider: MemoryProvider, llm_factory: LLMFactory):
        """Initialize memory assessor with provider and LLM factory.

        Args:
            memory_provider: Provider for storing and retrieving memories
            llm_factory: Factory for creating LLM agents
        """
        self.memory_provider = memory_provider
        self.llm_factory = llm_factory
        self.agent: UniversalAgent | None = None
        self.importance_threshold = 0.3  # Store memories with importance > 0.3

    async def initialize(self):
        """Initialize the memory assessor and its dependencies."""
        try:
            # Create Universal Agent for importance assessment
            self.agent = UniversalAgent(
                llm_factory=self.llm_factory,
                role_registry=None,  # Use default registry
            )

            # Initialize memory provider
            if hasattr(self.memory_provider, "initialize"):
                await self.memory_provider.initialize()

            logger.info("Memory assessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory assessor: {e}")
            raise

    async def assess_and_store_if_important(
        self,
        user_id: str,
        prompt: str,
        response: str,
        location: str | None = None,
        workflow_id: str | None = None,
    ) -> bool:
        """Assess conversation importance and store if significant.

        Args:
            user_id: ID of the user
            prompt: User's input/prompt
            response: System's response
            location: Optional location context
            workflow_id: Optional workflow identifier

        Returns:
            True if memory was stored, False otherwise
        """
        try:
            # Assess importance using LLM
            importance_score = await self._assess_importance(prompt, response, location)

            # Store if above threshold
            if importance_score > self.importance_threshold:
                memory_entry = {
                    "user_id": user_id,
                    "prompt": prompt,
                    "response": response,
                    "location": location,
                    "workflow_id": workflow_id,
                    "importance": importance_score,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                success = await self.memory_provider.store_memory(
                    user_id=user_id,
                    memory_type="conversation",
                    content=json.dumps(memory_entry),
                    importance=importance_score,
                    location=location,
                )

                if success:
                    logger.info(
                        f"Stored important memory for user {user_id} (importance: {importance_score})"
                    )
                    return True
                else:
                    logger.warning(f"Failed to store memory for user {user_id}")

            return False

        except Exception as e:
            logger.error(f"Error assessing memory importance: {e}")
            return False

    async def _assess_importance(
        self, prompt: str, response: str, location: str | None = None
    ) -> float:
        """Assess the importance of a conversation using LLM.

        Args:
            prompt: User's input
            response: System's response
            location: Optional location context

        Returns:
            Importance score between 0.0 and 1.0
        """
        try:
            assessment_prompt = self._build_assessment_prompt(
                prompt, response, location
            )

            if not self.agent:
                logger.warning("Agent not initialized, returning default importance")
                return 0.1

            result = await self.agent.execute(assessment_prompt)

            # Parse JSON response
            try:
                parsed = json.loads(result)
                importance = float(parsed.get("importance", 0.1))
                return max(0.0, min(1.0, importance))  # Clamp to [0, 1]
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.warning(f"Failed to parse importance assessment: {result}")
                return 0.1

        except Exception as e:
            logger.error(f"Error in importance assessment: {e}")
            return 0.1

    def _build_assessment_prompt(
        self, prompt: str, response: str, location: str | None = None
    ) -> str:
        """Build prompt for LLM importance assessment.

        Args:
            prompt: User's input
            response: System's response
            location: Optional location context

        Returns:
            Formatted assessment prompt
        """
        location_context = f"\nLocation: {location}" if location else ""

        return f"""Assess the importance of this conversation for future reference.
Consider factors like:
- Personal information shared
- Preferences expressed
- Important events or decisions
- Emotional significance
- Actionable items or commitments

Conversation:
User: {prompt}
Assistant: {response}{location_context}

Respond with JSON only:
{{"importance": <float between 0.0 and 1.0>}}

Where:
- 0.0-0.2: Trivial (weather, basic greetings)
- 0.3-0.5: Somewhat important (preferences, casual info)
- 0.6-0.8: Important (personal details, commitments)
- 0.9-1.0: Very important (major life events, critical decisions)"""
