# Conversation Role Architecture

## Overview

The conversation role implements a **topic-based knowledge graph** with **global message log** architecture for natural conversation management with memory continuity.

## Key Features

- **Global Message Log**: Single chronological log per user (`global_messages:{user_id}`)
- **LLM-Triggered Analysis**: `analyze_conversation()` tool for topic extraction
- **Analysis Pointer Tracking**: Tracks last analysis position (`last_analysis:{user_id}`)
- **Topic Knowledge Base**: Extracted topics with importance ranking (`topics:{user_id}`)
- **Context-Aware Responses**: Loads relevant topic knowledge and recent messages

## Architecture

### Data Structure

```python
# Global message log (single source of truth)
"global_messages:{user_id}" -> [
    {
        "id": "msg_uuid",
        "timestamp": 1234567890,
        "role": "user|assistant",
        "content": "message content",
        "channel_id": "channel"
    }
]

# Analysis pointer (tracks processing state)
"last_analysis:{user_id}" -> {
    "last_analysis_timestamp": 1234567890,
    "last_message_index": 50,
    "analyzed_message_count": 10
}

# Topic knowledge base (extracted insights)
"topics:{user_id}" -> {
    "dogs": {
        "summary": "User interested in Golden Retriever",
        "key_details": ["family-friendly", "budget $2000"],
        "last_discussed": 1234567890,
        "importance": 8,
        "related_topics": ["pets", "training"]
    }
}
```

### Workflow

1. **Pre-processing**: Load recent messages (last 30) + relevant topic knowledge
2. **Conversation**: Natural LLM responses with context awareness
3. **Post-processing**: Save message exchange to global log
4. **Analysis**: LLM calls `analyze_conversation()` when topic concludes
5. **Topic Extraction**: Async processing extracts topics and updates knowledge base

### Tools

- **`analyze_conversation()`**: LLM-triggered topic analysis and knowledge extraction

### Intents

- **`TopicAnalysisIntent`**: Processes unanalyzed messages for topic extraction

## Benefits

- ✅ **Simple**: Global log eliminates conversation ID complexity
- ✅ **Topic Continuity**: Can resume conversations about specific topics
- ✅ **Recent Context**: Always has last 30 messages regardless of topic shifts
- ✅ **LLM-Safe**: Async processing via intents (compliant with docs 25/26)
- ✅ **Scalable**: Works for homelab, can grow with enhanced topic matching

## Implementation Details

### Lifecycle Functions

- **`load_conversation_context()`**: Pre-processor for context loading
- **`save_message_to_log()`**: Post-processor for message storage

### Intent Handlers

- **`process_topic_analysis_intent()`**: Handles topic extraction and knowledge base updates

### Key Helper Functions

- **`_get_unanalyzed_messages()`**: Gets messages since last analysis
- **`_update_analysis_pointer()`**: Updates processing checkpoint
- **`_analyze_topics_with_llm()`**: LLM-based topic extraction (mock implementation)
- **`_update_topic_knowledge_base()`**: Updates topic knowledge with analysis results

## Configuration

```python
ROLE_CONFIG = {
    "name": "conversation",
    "version": "5.0.0",
    "llm_type": "DEFAULT",
    "fast_reply": True,
    "memory_enabled": True,
    "lifecycle": {
        "pre_processing": {"enabled": True, "functions": ["load_conversation_context"]},
        "post_processing": {"enabled": True, "functions": ["save_message_to_log"]},
    }
}
```

## Future Enhancements

- **Semantic Topic Matching**: Use embeddings for better topic similarity
- **Memory Importance Ranking**: Enhanced LLM-based importance assessment
- **Topic Relationships**: Build connections between related topics
- **Graph Database**: Migrate from Redis to proper graph database for complex relationships

## Testing

Comprehensive test suite covers:

- Role configuration and lifecycle
- Intent validation and processing
- Helper function behavior
- Message storage and retrieval
- Analysis pointer management
- Topic knowledge base updates

Run tests: `python -m pytest tests/unit/test_conversation_role.py -v`
