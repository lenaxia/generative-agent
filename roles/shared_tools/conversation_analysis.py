"""Conversation analysis tool for triggering memory importance assessment.

This tool allows roles to trigger analysis of unanalyzed messages in the
realtime log, creating assessed memories with importance scores and metadata.
"""

import logging
from typing import Any

from strands import tool

from common.intent_processor import IntentProcessor
from common.intents import MemoryWriteIntent
from common.memory_importance_assessor import MemoryImportanceAssessor
from common.realtime_log import get_unanalyzed_messages, mark_as_analyzed
from llm_provider.factory import LLMFactory

logger = logging.getLogger(__name__)

# Global intent processor instance
_intent_processor = None


def get_intent_processor() -> IntentProcessor:
    """Get or create the global intent processor instance.

    Returns:
        IntentProcessor instance
    """
    global _intent_processor
    if _intent_processor is None:
        _intent_processor = IntentProcessor()
    return _intent_processor


@tool
async def analyze_conversation(user_id: str) -> dict[str, Any]:
    """Trigger analysis of unanalyzed messages in realtime log.

    This tool analyzes recent unanalyzed messages using an LLM to assess
    their importance and generate structured metadata. Assessed memories
    are stored with graduated TTL based on importance.

    Args:
        user_id: User identifier for the conversation to analyze

    Returns:
        Dict with analysis results including count of memories created
    """
    try:
        # Get unanalyzed messages
        messages = get_unanalyzed_messages(user_id)

        if not messages:
            logger.info(f"No unanalyzed messages for user {user_id}")
            return {
                "success": True,
                "analyzed_count": 0,
                "memories_created": 0,
                "message": "No unanalyzed messages found",
            }

        logger.info(f"Analyzing {len(messages)} unanalyzed messages for user {user_id}")

        # Initialize assessor
        llm_factory = LLMFactory({})
        assessor = MemoryImportanceAssessor(llm_factory)
        await assessor.initialize()

        # Process each message
        intents = []
        message_ids = []
        memories_created = 0

        for msg in messages:
            message_ids.append(msg["id"])

            # Assess memory importance
            assessment = await assessor.assess_memory(
                user_message=msg["user"],
                assistant_response=msg["assistant"],
                source_role=msg["role"],
                context=msg.get("metadata", {}),
            )

            if assessment:
                # Calculate TTL based on importance
                ttl = assessor.calculate_ttl(assessment.importance)

                # Create memory write intent
                intent = MemoryWriteIntent(
                    user_id=user_id,
                    memory_type="conversation",
                    content=assessment.summary,
                    source_role=msg["role"],
                    importance=assessment.importance,
                    metadata={
                        "original_user_message": msg["user"],
                        "original_assistant_response": msg["assistant"],
                        "timestamp": msg["timestamp"],
                        "ttl": ttl,
                        "reasoning": assessment.reasoning,
                    },
                    tags=assessment.tags,
                    related_memories=[],
                )
                intents.append(intent)
                memories_created += 1

                logger.debug(
                    f"Created memory intent: importance={assessment.importance}, "
                    f"summary={assessment.summary[:50]}..."
                )
            else:
                logger.warning(f"Failed to assess message {msg['id']}")

        # Mark messages as analyzed
        if message_ids:
            mark_as_analyzed(user_id, message_ids)
            logger.info(f"Marked {len(message_ids)} messages as analyzed")

        # Process intents if any were created
        if intents:
            processor = get_intent_processor()
            await processor.process_intents(intents)
            logger.info(f"Processed {len(intents)} memory write intents")

        return {
            "success": True,
            "analyzed_count": len(messages),
            "memories_created": memories_created,
            "message": f"Analyzed {len(messages)} messages, created {memories_created} memories",
        }

    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}", exc_info=True)
        return {
            "success": False,
            "analyzed_count": 0,
            "memories_created": 0,
            "error": str(e),
        }
