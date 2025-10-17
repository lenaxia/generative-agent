"""Conversation role - Simple natural dialogue following timer/weather pattern.

This role provides natural conversation capabilities with memory context awareness
provided by the router. Uses NotificationIntent pattern like timer role for responses.

Key principles:
- Natural conversational responses using NotificationIntent
- Memory context provided by router when needed
- Simple tools: respond_to_user (with save) + start_new_conversation
- Fast-reply optimized

Architecture: Simple + Natural + Router-Driven Context + NotificationIntent
Created: 2025-01-17
Simplified: 2025-01-17
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from strands import tool

from common.intents import Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (simple like weather role)
ROLE_CONFIG = {
    "name": "conversation",
    "version": "3.0.0",
    "description": "Natural conversation and dialogue with router-provided memory context",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "memory_enabled": True,  # Router provides memory context when needed
    "location_aware": False,
    "presence_aware": False,
    "schedule_aware": False,
    "when_to_use": "Natural conversation, questions, dialogue, follow-up discussions, casual chat, explanations",
    "parameters": {
        # No specific parameters needed - LLM handles conversation naturally
    },
    "tools": {
        "automatic": True,  # Include conversation tools
        "shared": [],  # No shared tools needed
        "include_builtin": False,
        "fast_reply": {
            "enabled": True,
        },
    },
    "prompts": {
        "system": """You are a conversational AI assistant focused on natural dialogue.

Available tools:
- respond_to_user(user_id, user_message, response_text, channel): Send your response and save conversation
- start_new_conversation(user_id, new_topic, reason): Start fresh conversation when topic shifts

IMPORTANT: Always use respond_to_user() to send your responses. This saves conversation history and delivers the message.

When memory context is provided by the router, use it to maintain conversation continuity and reference previous discussions naturally.

Call start_new_conversation() only when the user shifts to a completely different topic.

Provide helpful, engaging responses and use respond_to_user() to deliver them."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (minimal)
@dataclass
class ConversationIntent(Intent):
    """Conversation-specific intent for logging conversation interactions."""

    interaction_type: str = "conversation"
    user_message: Optional[str] = None

    def validate(self) -> bool:
        """Validate conversation intent parameters."""
        return bool(self.interaction_type)


# 3. CONVERSATION TOOLS (simplified)


@tool
def respond_to_user(
    user_id: str, user_message: str, response_text: str, channel: str
) -> str:
    """Send response to user and save conversation history. Returns response text for delivery."""
    try:
        # Save conversation exchange to Redis
        _save_conversation_exchange(user_id, user_message, response_text, channel)

        # Return the response text for the system to deliver (like timer tools return data)
        return response_text
    except Exception as e:
        logger.error(f"Error in respond_to_user for {user_id}: {e}")
        return f"I apologize, but I encountered an error: {e}"


@tool
def start_new_conversation(
    user_id: str, new_topic: str, reason: str = "topic_shift"
) -> dict[str, Any]:
    """Start a new conversation, archiving the current one automatically."""
    try:
        # Archive current conversation if it exists (internal helper)
        archive_result = _archive_current_conversation(user_id)

        # Initialize fresh conversation
        _initialize_fresh_conversation(user_id, new_topic)

        logger.info(
            f"Started new conversation for {user_id}: {new_topic} (reason: {reason})"
        )

        return {
            "success": True,
            "new_topic": new_topic,
            "reason": reason,
            "archived_previous": archive_result.get("success", False),
        }
    except Exception as e:
        logger.error(f"Error starting new conversation for {user_id}: {e}")
        return {"success": False, "error": str(e)}


# 4. INTERNAL HELPERS (not exposed as tools)


def _save_conversation_exchange(
    user_id: str, user_message: str, response_text: str, channel: str
):
    """Internal helper to save conversation exchange to Redis."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        # Load existing conversation
        conversation_data = redis_read(f"conversation:{user_id}")
        if conversation_data.get("success") and conversation_data.get("data"):
            conversation = json.loads(conversation_data["data"])
        else:
            conversation = {
                "messages": [],
                "topic": None,
                "started_at": None,
                "last_active": None,
                "message_count": 0,
            }

        # Add user message
        conversation["messages"].append(
            {
                "timestamp": time.time(),
                "channel": channel,
                "role": "user",
                "content": user_message,
            }
        )

        # Add assistant response
        conversation["messages"].append(
            {
                "timestamp": time.time(),
                "channel": channel,
                "role": "assistant",
                "content": response_text,
            }
        )

        conversation["last_active"] = time.time()
        conversation["message_count"] = len(conversation["messages"])

        if not conversation["started_at"]:
            conversation["started_at"] = time.time()

        # Save back to Redis
        redis_write(f"conversation:{user_id}", json.dumps(conversation))

        logger.info(
            f"Saved conversation exchange for {user_id}: {len(conversation['messages'])} total messages"
        )

    except Exception as e:
        logger.error(f"Error saving conversation exchange for {user_id}: {e}")


def _archive_current_conversation(user_id: str) -> dict[str, Any]:
    """Internal helper to archive current conversation (not exposed as tool)."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        # Load current conversation
        conversation_data = redis_read(f"conversation:{user_id}")
        if not conversation_data.get("success") or not conversation_data.get("data"):
            return {"success": False, "reason": "no_conversation"}

        conversation = json.loads(conversation_data["data"])

        if len(conversation.get("messages", [])) < 3:
            return {"success": False, "reason": "too_short"}

        # Create simple archive entry
        archive_entry = {
            "archived_at": time.time(),
            "message_count": len(conversation["messages"]),
            "topic": conversation.get("topic", "general"),
            "summary": f"Conversation with {len(conversation['messages'])} messages",
        }

        # Save to archive (simplified)
        archive_key = f"conversation_archive:{user_id}"
        archive_data = redis_read(archive_key)
        if archive_data.get("success") and archive_data.get("data"):
            archive = json.loads(archive_data["data"])
        else:
            archive = {"archived_conversations": []}

        archive["archived_conversations"].append(archive_entry)
        archive["archived_conversations"] = archive["archived_conversations"][
            -10:
        ]  # Keep last 10

        redis_write(archive_key, json.dumps(archive))

        logger.info(
            f"Archived conversation for {user_id}: {len(conversation['messages'])} messages"
        )
        return {"success": True, "archived": True}

    except Exception as e:
        logger.error(f"Error archiving conversation for {user_id}: {e}")
        return {"success": False, "error": str(e)}


def _initialize_fresh_conversation(user_id: str, new_topic: str):
    """Internal helper to initialize fresh conversation."""
    try:
        from roles.shared_tools.redis_tools import redis_write

        new_conversation = {
            "messages": [],
            "topic": new_topic,
            "started_at": time.time(),
            "last_active": time.time(),
            "message_count": 0,
        }

        redis_write(f"conversation:{user_id}", json.dumps(new_conversation))

    except Exception as e:
        logger.error(f"Error initializing fresh conversation for {user_id}: {e}")


# 5. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {},  # No event handlers
        "tools": [respond_to_user, start_new_conversation],  # Simple tools
        "intents": [ConversationIntent],
    }


# 6. CONVERSATION UTILITIES (helper functions)
def get_conversation_statistics() -> dict[str, Any]:
    """Get conversation role statistics (for monitoring/debugging)."""
    return {
        "role_name": "conversation",
        "version": ROLE_CONFIG["version"],
        "memory_enabled": ROLE_CONFIG["memory_enabled"],
        "fast_reply": ROLE_CONFIG["fast_reply"],
        "tools_required": True,
        "context_types": ["recent_memory"],  # Router provides memory context
        "features": ["natural_conversation", "memory_aware", "notification_intent"],
    }


# 7. CONSTANTS AND CONFIGURATION
DEFAULT_RESPONSE_TIMEOUT = 10  # seconds


# 8. ERROR HANDLING UTILITIES
def create_conversation_error_response(
    error: Exception, context: str = ""
) -> dict[str, Any]:
    """Create standardized error response for conversation failures."""
    return {
        "success": False,
        "error": str(error),
        "error_type": error.__class__.__name__,
        "context": context,
        "suggestion": "Try rephrasing your message or starting a new conversation",
    }
