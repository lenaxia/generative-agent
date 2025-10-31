"""Conversation role - Simplified topic-based knowledge graph with global message log.

This role provides natural conversation capabilities with topic-based memory management.
Uses a simplified approach with a global message log and periodic LLM-triggered analysis.

Key principles:
- Global message log for all conversation role messages
- Single analyze_conversation tool triggered by LLM
- Pointer tracking last analysis position
- Topic-based knowledge graph built from analysis
- Natural conversational responses

Architecture: Global Message Log + LLM-Triggered Topic Analysis
Created: 2025-01-17
Redesigned: 2025-01-17 (Simplified global log approach)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from strands import tool

from common.intents import Intent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "conversation",
    "version": "5.0.0",
    "description": "Natural conversation with global message log and topic-based knowledge extraction",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "memory_enabled": True,
    "location_aware": False,
    "presence_aware": False,
    "schedule_aware": False,
    "when_to_use": "Natural conversation, questions, dialogue, follow-up discussions, casual chat, explanations, you can also use this to summarize or analyze information",
    "parameters": {
        # No specific parameters needed - LLM handles conversation naturally
    },
    "tools": {
        "automatic": True,  # Include analyze_conversation and search_topics tools
        "shared": ["redis_tools"],  # Need Redis for message storage
        "include_builtin": False,
        "fast_reply": {
            "enabled": True,
        },
    },
    "lifecycle": {
        "pre_processing": {"enabled": True, "functions": ["load_conversation_context"]},
        "post_processing": {
            "enabled": True,
            "functions": ["save_message_to_log"],
        },
    },
    "prompts": {
        "system": """You are a conversational AI assistant with topic-based memory.

Available tools:
- analyze_conversation(): Call this when you feel a conversation topic has concluded or shifted significantly
- search_topics(query): Search for relevant past topics when you need specific information about previous discussions

RECENT CONVERSATION CONTEXT:
{{recent_messages}}

RECENT TOPICS (auto-injected):
{{recent_topics}}

Recent message count: {{message_count}}
Unanalyzed messages: {{unanalyzed_count}}
Current topics: {{current_topics}}

Provide natural conversational responses. You're not here to explicitly help, but to engage in dialogue.
Have opinions, and don't shy away from standing your ground on them when appropriate.

Use the recent conversation context and recent topics to maintain continuity and reference previous discussions.

If you decide to make a tool call: DO NOT GENERATE ANY TEXT PRIOR TO MAKING A TOOL CALL

Call analyze_conversation() when:
- The conversation topic seems to have concluded
- The user shifts to a completely different topic
- You sense a natural break in the conversation flow
- There are many unanalyzed messages building up

Call search_topics(query) when:
- You need to recall specific information about past discussions
- The user asks about something that might have been discussed before
- You want to check if a topic has been covered previously

IMPORTANT: After calling any tool, ALWAYS provide a clear response to the user:
1. Answer the user's original question or request
2. Never mention the tool call you made, but you may use the context returned to enhance your response
3. Always end with a direct response to the user's most recent message
4. Don't offer or ask to help, you are here to engage in natural conversation

Never end your response after just calling a tool - always provide follow-up text for the user.""",
    },
}


# 2. ROLE-SPECIFIC INTENTS
@dataclass
class TopicAnalysisIntent(Intent):
    """Intent for analyzing conversation topics and extracting knowledge."""

    user_id: str
    analysis_trigger: str = "llm_triggered"  # "llm_triggered", "daily", "manual"

    def validate(self) -> bool:
        """Validate topic analysis intent parameters."""
        return bool(self.user_id) and len(self.user_id.strip()) > 0


@dataclass
class TopicSearchIntent(Intent):
    """Intent for searching relevant past topics with high relevance threshold."""

    user_id: str
    query: str
    relevance_threshold: float = 0.8

    def validate(self) -> bool:
        """Validate topic search intent parameters."""
        return (
            bool(self.user_id and self.query)
            and len(self.user_id.strip()) > 0
            and len(self.query.strip()) > 0
            and 0.0 <= self.relevance_threshold <= 1.0
        )


# 3. CONVERSATION TOOLS
@tool
def analyze_conversation() -> dict[str, Any]:
    """Trigger analysis of unanalyzed conversation messages for topic extraction and summarization."""
    try:
        logger.info("LLM triggered conversation analysis")

        result = {
            "success": True,
            "message": "Conversation analysis triggered - unanalyzed messages will be processed for topic extraction",
            "intent": {
                "type": "TopicAnalysisIntent",
                "analysis_trigger": "llm_triggered",
                # user_id will be injected by lifecycle functions
            },
        }
        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def search_topics(query: str) -> dict[str, Any]:
    """Search for relevant past topics based on query with high relevance threshold."""
    try:
        result = {
            "success": True,
            "message": f"Searching for topics related to: {query}",
            "query": query,
            "intent": {
                "type": "TopicSearchIntent",
                "query": query,
                "relevance_threshold": 0.8,
                # user_id will be injected by lifecycle functions
            },
        }

        return result

    except Exception as e:
        logger.error(f"Error searching topics: {e}")
        return {"success": False, "error": str(e)}


# 4. LIFECYCLE FUNCTIONS
def load_conversation_context(instruction: str, context, parameters: dict) -> dict:
    """Pre-processor: Load recent messages and cached recent topics (no heavy search)."""
    try:
        user_id = getattr(context, "user_id", "unknown")

        # Load recent messages (last 30 messages)
        recent_messages = _load_recent_messages(user_id, limit=30)

        # Load cached recent topics (lightweight, with TTL)
        recent_topics = _load_recent_topics_cache(user_id)

        # Count unanalyzed messages
        unanalyzed_count = _count_unanalyzed_messages(user_id)

        # Extract current topics from recent messages
        current_topics = _extract_current_topics_simple(recent_messages)

        return {
            "recent_messages": recent_messages,
            "recent_topics": recent_topics,
            "message_count": len(recent_messages),
            "unanalyzed_count": unanalyzed_count,
            "current_topics": current_topics,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Failed to load conversation context for {user_id}: {e}")
        return {
            "recent_messages": [],
            "recent_topics": {},
            "message_count": 0,
            "unanalyzed_count": 0,
            "current_topics": [],
            "user_id": getattr(context, "user_id", "unknown"),
        }


def save_message_to_log(llm_result: str, context, pre_data: dict) -> str:
    """Post-processing function to save conversation message to global log (sync version)."""
    try:
        user_id = getattr(context, "user_id", "unknown")
        channel_id = getattr(context, "channel_id", "unknown")
        user_message = getattr(context, "original_prompt", "unknown")

        # Save message to global conversation log (synchronous)
        _save_message_to_global_log(user_id, user_message, llm_result, channel_id)

        logger.info(f"Saved conversation message for {user_id}")

        return llm_result

    except Exception as e:
        logger.error(f"Post-processing message save failed: {e}")
        return llm_result


# 5. INTERNAL HELPERS
def _load_recent_messages(user_id: str, limit: int = 30) -> list[dict[str, Any]]:
    """Load recent messages from global conversation log."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        messages_data = redis_read(f"conversation:messages:{user_id}")
        if messages_data.get("success") and messages_data.get("value"):
            all_messages = messages_data["value"]  # Already parsed by redis_read
            # Ensure it's a list
            if not isinstance(all_messages, list):
                logger.warning(
                    f"Expected list but got {type(all_messages)} for messages"
                )
                return []
            # Return last N messages
            return all_messages[-limit:] if len(all_messages) > limit else all_messages
        else:
            return []

    except Exception as e:
        logger.error(f"Error loading recent messages for {user_id}: {e}")
        return []


def _count_unanalyzed_messages(user_id: str) -> int:
    """Count messages that haven't been analyzed yet."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        # Get last analysis pointer
        analysis_data = redis_read(f"conversation:last_analysis:{user_id}")
        if analysis_data.get("success") and analysis_data.get("value"):
            last_analysis = analysis_data["value"]  # Already parsed by redis_read
            if isinstance(last_analysis, dict):
                last_message_index = last_analysis.get("last_message_index", 0)
            else:
                last_message_index = 0
        else:
            last_message_index = 0

        # Get total message count
        messages_data = redis_read(f"conversation:messages:{user_id}")
        if messages_data.get("success") and messages_data.get("value"):
            all_messages = messages_data["value"]  # Already parsed by redis_read
            if isinstance(all_messages, list):
                total_messages = len(all_messages)
                return max(0, total_messages - last_message_index)
            else:
                return 0
        else:
            return 0

    except Exception as e:
        logger.error(f"Error counting unanalyzed messages for {user_id}: {e}")
        return 0


def _load_recent_topics_cache(user_id: str) -> dict[str, Any]:
    """Load cached recent topics (lightweight, with TTL)."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        # Load recent topics cache (has TTL of 1 hour)
        cache_data = redis_read(f"conversation:recent_topics_cache:{user_id}")
        if cache_data.get("success") and cache_data.get("value"):
            cached_topics = cache_data["value"]  # Already parsed by redis_read
            # Ensure it's a dict
            if not isinstance(cached_topics, dict):
                logger.warning(
                    f"Expected dict but got {type(cached_topics)} for topics cache"
                )
                return {}
            return cached_topics
        else:
            return {}

    except Exception as e:
        logger.error(f"Error loading recent topics cache for {user_id}: {e}")
        return {}


def _search_topics_with_relevance(
    user_id: str, query: str, threshold: float = 0.8
) -> dict[str, Any]:
    """Search topics with relevance scoring and high threshold filtering."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        # Load all topics
        topics_data = redis_read(f"conversation:topics:{user_id}")
        if not topics_data.get("success") or not topics_data.get("value"):
            return {}

        all_topics = topics_data["value"]  # Already parsed by redis_read
        if not isinstance(all_topics, dict):
            logger.warning(f"Expected dict but got {type(all_topics)} for topics")
            return {}
        relevant_topics = {}
        query_lower = query.lower()

        # Simple relevance scoring (could be enhanced with embeddings)
        for topic_name, topic_data in all_topics.items():
            relevance_score = 0.0

            # Topic name match (high weight)
            if topic_name.lower() in query_lower:
                relevance_score += 0.6

            # Summary match (medium weight)
            summary = topic_data.get("summary", "").lower()
            if any(word in summary for word in query_lower.split()):
                relevance_score += 0.3

            # Key details match (medium weight)
            key_details = topic_data.get("key_details", [])
            for detail in key_details:
                if any(word in detail.lower() for word in query_lower.split()):
                    relevance_score += 0.2
                    break

            # Only include if above threshold
            if relevance_score >= threshold:
                topic_data_with_score = topic_data.copy()
                topic_data_with_score["relevance_score"] = relevance_score
                relevant_topics[topic_name] = topic_data_with_score

        # Update recent topics cache with found topics (TTL 1 hour)
        if relevant_topics:
            _update_recent_topics_cache(user_id, relevant_topics)

        return relevant_topics

    except Exception as e:
        logger.error(f"Error searching topics for {user_id}: {e}")
        return {}


def _update_recent_topics_cache(user_id: str, topics: dict[str, Any]):
    """Update recent topics cache with TTL."""
    try:
        from roles.shared_tools.redis_tools import redis_write

        # Cache for 1 hour (3600 seconds)
        redis_write(f"recent_topics_cache:{user_id}", json.dumps(topics), ttl=3600)
        logger.info(
            f"Updated recent topics cache for {user_id} with {len(topics)} topics"
        )

    except Exception as e:
        logger.error(f"Error updating recent topics cache for {user_id}: {e}")


def _extract_current_topics_simple(messages: list[dict[str, Any]]) -> list[str]:
    """Extract simple topic keywords from recent messages."""
    try:
        # Simple keyword extraction - could be enhanced with NLP
        topics = set()
        for message in messages[-5:]:  # Look at last 5 messages
            content = message.get("content", "").lower()
            # Simple topic detection based on common patterns
            if any(word in content for word in ["dog", "puppy", "pet"]):
                topics.add("pets")
            if any(word in content for word in ["college", "university", "school"]):
                topics.add("education")
            if any(word in content for word in ["work", "job", "career"]):
                topics.add("career")

        return list(topics)

    except Exception as e:
        logger.error(f"Error extracting current topics: {e}")
        return []


def _save_message_to_global_log(
    user_id: str, user_message: str, assistant_response: str, channel_id: str
):
    """Save message exchange to global conversation log."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        # Load existing messages
        messages_data = redis_read(f"conversation:messages:{user_id}")
        if messages_data.get("success") and messages_data.get("value"):
            messages = messages_data["value"]
        else:
            messages = []

        # Add user message
        messages.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "role": "user",
                "content": user_message,
                "channel_id": channel_id,
            }
        )

        # Add assistant response
        messages.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "role": "assistant",
                "content": assistant_response,
                "channel_id": channel_id,
            }
        )

        # Keep only last 200 messages to prevent unbounded growth
        if len(messages) > 200:
            messages = messages[-200:]

        # Save back to Redis
        redis_write(f"conversation:messages:{user_id}", json.dumps(messages))

        logger.info(
            f"Saved message exchange for {user_id}: {len(messages)} total messages"
        )

    except Exception as e:
        logger.error(f"Error saving message to global log for {user_id}: {e}")


# 6. INTENT HANDLER REGISTRATION
async def process_topic_analysis_intent(intent: TopicAnalysisIntent):
    """Process topic analysis intents - extract topics from unanalyzed messages."""
    try:
        logger.info(f"Processing topic analysis for {intent.user_id}")

        # Get unanalyzed messages
        unanalyzed_messages = _get_unanalyzed_messages(intent.user_id)

        if not unanalyzed_messages:
            logger.info(f"No unanalyzed messages for {intent.user_id}")
            return

        # Analyze topics using LLM (this would be a separate LLM call)
        topic_analysis = await _analyze_topics_with_llm(unanalyzed_messages)

        # Update topic knowledge base
        _update_topic_knowledge_base(intent.user_id, topic_analysis)

        # Update analysis pointer
        _update_analysis_pointer(intent.user_id, len(unanalyzed_messages))

        logger.info(
            f"Completed topic analysis for {intent.user_id}: {len(topic_analysis.get('topics', []))} topics"
        )

    except Exception as e:
        logger.error(f"Topic analysis failed for {intent.user_id}: {e}")


async def process_topic_search_intent(intent: TopicSearchIntent):
    """Process topic search intents - search for relevant topics and update cache."""
    try:
        logger.info(f"Processing topic search for {intent.user_id}: {intent.query}")

        # Search topics with high relevance threshold
        relevant_topics = _search_topics_with_relevance(
            intent.user_id, intent.query, intent.relevance_threshold
        )

        if relevant_topics:
            logger.info(
                f"Found {len(relevant_topics)} relevant topics for query: {intent.query}"
            )
            # Topics are automatically cached by _search_topics_with_relevance
        else:
            logger.info(
                f"No topics found above threshold {intent.relevance_threshold} for query: {intent.query}"
            )

    except Exception as e:
        logger.error(f"Topic search failed for {intent.user_id}: {e}")


def _get_unanalyzed_messages(user_id: str) -> list[dict[str, Any]]:
    """Get messages that haven't been analyzed yet."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        # Get last analysis pointer
        analysis_data = redis_read(f"last_analysis:{user_id}")
        if analysis_data.get("success") and analysis_data.get("value"):
            last_analysis = analysis_data["value"]
            last_message_index = last_analysis.get("last_message_index", 0)
        else:
            last_message_index = 0

        # Get all messages
        messages_data = redis_read(f"conversation:messages:{user_id}")
        if messages_data.get("success") and messages_data.get("value"):
            all_messages = messages_data["value"]
            # Return messages after last analysis
            return all_messages[last_message_index:]
        else:
            return []

    except Exception as e:
        logger.error(f"Error getting unanalyzed messages for {user_id}: {e}")
        return []


def _update_analysis_pointer(user_id: str, analyzed_count: int):
    """Update the pointer to track last analysis position."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        # Get current total message count
        messages_data = redis_read(f"global_messages:{user_id}")
        if messages_data.get("success") and messages_data.get("value"):
            all_messages = messages_data["value"]
            total_messages = len(all_messages)
        else:
            total_messages = 0

        # Update analysis pointer
        analysis_pointer = {
            "last_analysis_timestamp": time.time(),
            "last_message_index": total_messages,  # Point to end of current messages
            "analyzed_message_count": analyzed_count,
        }

        redis_write(
            f"conversation:last_analysis:{user_id}", json.dumps(analysis_pointer)
        )

        logger.info(
            f"Updated analysis pointer for {user_id}: analyzed {analyzed_count} messages"
        )

    except Exception as e:
        logger.error(f"Error updating analysis pointer for {user_id}: {e}")


async def _analyze_topics_with_llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze messages with LLM to extract topics and key information."""
    # This would make an LLM call to analyze the conversation
    # For now, return a simple mock analysis
    try:
        # Create conversation text for analysis
        conversation_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

        # Mock analysis - in real implementation, this would be an LLM call
        mock_analysis = {
            "topics": ["general_conversation"],
            "key_details": ["User engaged in natural conversation"],
            "importance": 5,
            "summary": f"Conversation with {len(messages)} messages",
        }

        logger.info(f"Analyzed {len(messages)} messages")
        return mock_analysis

    except Exception as e:
        logger.error(f"LLM topic analysis failed: {e}")
        return {
            "topics": [],
            "key_details": [],
            "importance": 1,
            "summary": "Analysis failed",
        }


def _update_topic_knowledge_base(user_id: str, analysis: dict[str, Any]):
    """Update the topic knowledge base with analysis results."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        # Load existing topics
        topics_data = redis_read(f"conversation:topics:{user_id}")
        if topics_data.get("success") and topics_data.get("value"):
            topics = topics_data["value"]
        else:
            topics = {}

        # Update topics with new analysis
        for topic in analysis.get("topics", []):
            if topic in topics:
                # Update existing topic
                topics[topic]["last_discussed"] = time.time()
                topics[topic]["key_details"].extend(analysis.get("key_details", []))
                # Keep only unique details
                topics[topic]["key_details"] = list(set(topics[topic]["key_details"]))
            else:
                # Create new topic
                topics[topic] = {
                    "summary": analysis.get("summary", f"Discussion about {topic}"),
                    "key_details": analysis.get("key_details", []),
                    "last_discussed": time.time(),
                    "importance": analysis.get("importance", 5),
                    "related_topics": [],
                }

        # Save updated topics
        redis_write(f"conversation:topics:{user_id}", json.dumps(topics))

        logger.info(f"Updated topic knowledge base for {user_id}")

    except Exception as e:
        logger.error(f"Error updating topic knowledge base for {user_id}: {e}")


# 7. ROLE REGISTRATION
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers needed
        "tools": [analyze_conversation, search_topics],
        "intents": {
            TopicAnalysisIntent: process_topic_analysis_intent,
            TopicSearchIntent: process_topic_search_intent,
        },
        "pre_processors": [load_conversation_context],
        "post_processors": [save_message_to_log],
    }


# 8. CONVERSATION UTILITIES
def get_conversation_statistics() -> dict[str, Any]:
    """Get conversation role statistics."""
    return {
        "role_name": "conversation",
        "version": ROLE_CONFIG["version"],
        "architecture": "global_message_log_with_topic_analysis",
        "features": [
            "global_message_log",
            "topic_extraction",
            "memory_importance_ranking",
            "llm_triggered_analysis",
            "analysis_pointer_tracking",
            "natural_conversation",
        ],
    }


# 9. ERROR HANDLING
def create_conversation_error_response(
    error: Exception, context: str = ""
) -> dict[str, Any]:
    """Create standardized error response for conversation failures."""
    return {
        "success": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
        "context": context,
        "suggestion": "Try rephrasing your message or calling analyze_conversation()",
    }
