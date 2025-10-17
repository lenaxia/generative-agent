"""Conversation role - Enhanced with intelligent conversation management.

This role provides natural conversation with persistent memory, automatic topic detection,
and intelligent conversation archiving. Designed for homelab environments with natural
conversation flow across topics and channels.

Key features:
- Persistent conversation state in Redis
- Automatic topic boundary detection
- LLM-driven conversation archiving
- Cross-channel conversation continuity
- Searchable conversation archive

Architecture: Single Event Loop + Intent-Based + Self-Managing Conversation State
Created: 2025-01-17
Enhanced: 2025-01-17
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool

from common.event_context import LLMSafeEventContext
from common.intents import AuditIntent, Intent, NotificationIntent

logger = logging.getLogger(__name__)

# 1. ROLE METADATA (replaces definition.yaml)
ROLE_CONFIG = {
    "name": "conversation",
    "version": "2.0.0",
    "description": "Natural conversation with persistent memory and intelligent topic management",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "memory_enabled": False,  # Manages its own conversation state
    "location_aware": False,
    "presence_aware": False,
    "schedule_aware": False,
    "when_to_use": "Natural conversation, questions, dialogue, follow-up discussions, casual chat, explanations, and continuing conversations across topics",
    "parameters": {
        # No specific parameters needed - LLM handles conversation naturally
    },
    "tools": {
        "automatic": True,  # Include conversation management tools
        "shared": ["redis_tools"],  # Need Redis for conversation storage
        "include_builtin": False,
        "fast_reply": {
            "enabled": True,
        },
    },
    "prompts": {
        "system": """You are a conversational AI with persistent memory and intelligent topic management.

Available tools:
- load_conversation(user_id): Load current conversation history
- save_message(user_id, role, content, channel): Save messages to conversation
- start_new_conversation(user_id, new_topic, reason): Start fresh conversation when topic shifts
- search_archive(user_id, query): Find relevant past conversations

CONVERSATION MANAGEMENT:
- Always load conversation history at the start to maintain context
- Save both user messages and your responses to maintain conversation state
- Call start_new_conversation() when you detect the user is shifting to a significantly different topic

Examples of when to start new conversations:
- User was asking about Docker, now asks about Home Assistant setup
- User was discussing vacation plans, now asks about work projects
- User says "let's talk about something else" or "new topic"
- User asks about a completely different subject area

When starting new conversations:
- Choose a clear, descriptive topic name (e.g., "Docker Setup", "Home Assistant Configuration")
- Provide a brief reason for the topic shift
- The previous conversation will be automatically archived

CONVERSATION FLOW:
1. Load conversation history to understand context
2. Save the user's message
3. Respond naturally using conversation history
4. Save your response
5. If topic shifts significantly, call start_new_conversation()

Use conversation history to maintain natural flow and reference previous discussions within the current topic."""
    },
}


# 2. ROLE-SPECIFIC INTENTS (minimal for conversation role)
@dataclass
class ConversationIntent(Intent):
    """Conversation-specific intent for logging conversation interactions."""

    interaction_type: str = "conversation"
    user_message: Optional[str] = None

    def validate(self) -> bool:
        """Validate conversation intent parameters."""
        return bool(self.interaction_type)


# 3. EVENT HANDLERS (pure functions returning intents)
def handle_conversation_start(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """Handle conversation start events."""
    try:
        conversation_data = (
            event_data if isinstance(event_data, dict) else {"message": str(event_data)}
        )

        return [
            AuditIntent(
                action="conversation_started",
                details={
                    "message": conversation_data.get("message", str(event_data)),
                    "channel": context.get_safe_channel(),
                    "timestamp": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            )
        ]

    except Exception as e:
        logger.error(f"Conversation start handler error: {e}")
        return [
            NotificationIntent(
                message=f"Error starting conversation: {e}",
                channel=context.get_safe_channel(),
                priority="medium",
                notification_type="error",
            )
        ]


def handle_conversation_end(
    event_data: Any, context: LLMSafeEventContext
) -> list[Intent]:
    """Handle conversation end events."""
    try:
        conversation_data = (
            event_data if isinstance(event_data, dict) else {"message": str(event_data)}
        )

        return [
            AuditIntent(
                action="conversation_ended",
                details={
                    "channel": context.get_safe_channel(),
                    "timestamp": time.time(),
                },
                user_id=context.user_id,
                severity="info",
            )
        ]

    except Exception as e:
        logger.error(f"Conversation end handler error: {e}")
        return [
            NotificationIntent(
                message=f"Error ending conversation: {e}",
                channel=context.get_safe_channel(),
                priority="low",
                notification_type="error",
            )
        ]


# 4. CONVERSATION MANAGEMENT TOOLS


@tool
def load_conversation(user_id: str) -> dict[str, Any]:
    """Load active conversation history for user."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        conversation_data = redis_read(f"conversation:{user_id}")
        if conversation_data.get("success"):
            conversation_data = conversation_data.get("data")
        else:
            conversation_data = None
        if not conversation_data:
            return {
                "messages": [],
                "topic": None,
                "started_at": None,
                "last_active": None,
                "message_count": 0,
            }

        return json.loads(conversation_data)
    except Exception as e:
        logger.error(f"Error loading conversation for {user_id}: {e}")
        return {
            "messages": [],
            "topic": None,
            "started_at": None,
            "last_active": None,
            "message_count": 0,
        }


@tool
def save_message(user_id: str, role: str, content: str, channel: str) -> dict[str, Any]:
    """Save message to active conversation."""
    try:
        from roles.shared_tools.redis_tools import redis_write

        conversation = load_conversation(user_id)

        message = {
            "timestamp": time.time(),
            "channel": channel,
            "role": role,
            "content": content,
        }

        conversation["messages"].append(message)
        conversation["last_active"] = time.time()
        conversation["message_count"] = len(conversation["messages"])

        if not conversation["started_at"]:
            conversation["started_at"] = time.time()

        redis_write(f"conversation:{user_id}", json.dumps(conversation))

        return {
            "success": True,
            "message_count": conversation["message_count"],
            "topic": conversation.get("topic"),
        }
    except Exception as e:
        logger.error(f"Error saving message for {user_id}: {e}")
        return {"success": False, "error": str(e)}


@tool
def start_new_conversation(
    user_id: str, new_topic: str, reason: str = "topic_shift"
) -> dict[str, Any]:
    """Start a new conversation, archiving the current one automatically."""
    try:
        from roles.shared_tools.redis_tools import redis_write

        # Archive current conversation if it exists and has messages
        current_conversation = load_conversation(user_id)
        archived_previous = False

        if current_conversation["messages"]:
            archive_result = archive_conversation(user_id, auto_archive=True)
            archived_previous = archive_result.get("success", False)

        # Initialize fresh conversation
        new_conversation = {
            "messages": [],
            "topic": new_topic,
            "started_at": time.time(),
            "last_active": time.time(),
            "message_count": 0,
        }

        redis_write(f"conversation:{user_id}", json.dumps(new_conversation))

        logger.info(
            f"Started new conversation for {user_id}: {new_topic} (reason: {reason})"
        )

        return {
            "success": True,
            "new_topic": new_topic,
            "reason": reason,
            "archived_previous": archived_previous,
        }
    except Exception as e:
        logger.error(f"Error starting new conversation for {user_id}: {e}")
        return {"success": False, "error": str(e)}


@tool
def search_archive(user_id: str, query: str) -> list[dict[str, Any]]:
    """Search archived conversations for relevant context."""
    try:
        from roles.shared_tools.redis_tools import redis_read

        archive_data = redis_read(f"conversation_archive:{user_id}")
        if archive_data.get("success"):
            archive_data = archive_data.get("data")
        else:
            archive_data = None
        if not archive_data:
            return []

        archived_conversations = json.loads(archive_data).get(
            "archived_conversations", []
        )

        # Simple keyword search in summaries and topics
        query_lower = query.lower()
        relevant = []

        for conv in archived_conversations:
            summary_match = query_lower in conv.get("summary", "").lower()
            topic_match = any(
                query_lower in topic.lower() for topic in conv.get("key_topics", [])
            )

            if summary_match or topic_match:
                relevant.append(conv)

        # Return most recent matches first
        return sorted(relevant, key=lambda x: x.get("archived_at", 0), reverse=True)[:3]
    except Exception as e:
        logger.error(f"Error searching archive for {user_id}: {e}")
        return []


@tool
def archive_conversation(user_id: str, auto_archive: bool = False) -> dict[str, Any]:
    """Archive current conversation with summary."""
    try:
        from roles.shared_tools.redis_tools import redis_read, redis_write

        conversation = load_conversation(user_id)

        if len(conversation["messages"]) < 3:
            return {"success": False, "reason": "conversation_too_short"}

        # Extract key topics from messages (simplified version)
        key_topics = _extract_topics_from_messages(conversation["messages"])

        # Create archive entry
        archive_entry = {
            "period": f"{conversation.get('started_at', time.time())} to {conversation.get('last_active', time.time())}",
            "summary": f"Discussion about {conversation.get('topic', 'various topics')} - {len(conversation['messages'])} messages exchanged",
            "key_topics": key_topics,
            "message_count": len(conversation["messages"]),
            "archived_at": time.time(),
            "auto_archived": auto_archive,
            "topic": conversation.get("topic"),
        }

        # Load existing archive
        archive_result = redis_read(f"conversation_archive:{user_id}")
        if archive_result.get("success") and archive_result.get("data"):
            archive = json.loads(archive_result["data"])
        else:
            archive = {"archived_conversations": []}

        archive["archived_conversations"].append(archive_entry)

        # Keep only last 20 archived conversations for homelab efficiency
        archive["archived_conversations"] = archive["archived_conversations"][-20:]

        redis_write(f"conversation_archive:{user_id}", json.dumps(archive))

        logger.info(
            f"Archived conversation for {user_id}: {conversation.get('topic')} ({len(conversation['messages'])} messages)"
        )

        return {
            "success": True,
            "archived_messages": len(conversation["messages"]),
            "topic": conversation.get("topic"),
        }
    except Exception as e:
        logger.error(f"Error archiving conversation for {user_id}: {e}")
        return {"success": False, "error": str(e)}


def _extract_topics_from_messages(messages: list[dict]) -> list[str]:
    """Extract key topics from conversation messages (simplified version)."""
    topics = set()

    # Common homelab/tech topics to detect
    common_topics = [
        "docker",
        "home-assistant",
        "proxmox",
        "networking",
        "automation",
        "plex",
        "media-server",
        "mqtt",
        "smart-home",
        "linux",
        "server",
        "backup",
        "security",
        "vpn",
        "dns",
        "firewall",
        "kubernetes",
    ]

    # Combine all message content
    all_content = " ".join([msg.get("content", "").lower() for msg in messages])

    # Simple keyword detection
    for topic in common_topics:
        if topic in all_content or topic.replace("-", " ") in all_content:
            topics.add(topic)

    return list(topics)[:5]  # Limit to 5 topics


# 5. ROLE REGISTRATION (auto-discovery)
def register_role():
    """Auto-discovered by RoleRegistry - LLM can modify this."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {
            "CONVERSATION_START": handle_conversation_start,
            "CONVERSATION_END": handle_conversation_end,
        },
        "tools": [
            load_conversation,
            save_message,
            start_new_conversation,
            search_archive,
            archive_conversation,
        ],
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
        "context_types": ["conversation_history"],
        "features": ["persistent_conversations", "topic_detection", "auto_archiving"],
    }


# 7. CONSTANTS AND CONFIGURATION
# Performance settings
DEFAULT_RESPONSE_TIMEOUT = 10  # seconds
MAX_CONVERSATION_MESSAGES = 50  # Before suggesting archival
ARCHIVE_RETENTION_COUNT = 20  # Number of archived conversations to keep


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
        "timestamp": time.time(),
        "suggestion": "Try rephrasing your message or starting a new conversation",
    }
