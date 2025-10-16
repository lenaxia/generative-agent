# Router-Driven Context Selection for Household Assistant

**Document ID:** 33
**Created:** 2025-10-15
**Status:** Implementation Design
**Priority:** High
**Context:** Household Assistant Enhancement with Memory and Location Context

## Executive Summary

This document outlines the implementation of intelligent context selection for the Universal Agent System, transforming it into a household assistant with memory and location awareness. The design uses enum-based context types with specific implementations and leverages the existing router role to make surgical context gathering decisions, minimizing latency while maximizing contextual intelligence.

## Problem Statement

### Current Limitations

- System lacks memory of previous interactions
- No location awareness for contextual responses
- No proactive household assistant behaviors
- Simple request/response model without situational understanding

### Requirements

- Add memory and location context without breaking existing functionality
- Maintain LLM-safe single event loop architecture
- Minimize latency impact for simple commands
- Enable household assistant behaviors (proactive, contextual, memory-aware)
- Use existing MQTT broker infrastructure

## Solution Overview

### Core Innovation: Router-Driven Context Selection with Enum Types

The **router role decides what context is required** using predefined enum types, enabling surgical context gathering with zero overhead for simple commands.

```mermaid
graph TD
    A[User Input] --> B[Enhanced Router]
    B --> C[Context Collector]
    C --> D[Role Execution]
    D --> E[Response]
    E --> F[Async Memory Assessment]
    F --> G[Store if Important]
```

## Detailed Design

### 1. Context Type Enumeration

#### Enum-Based Context Types

```python
# common/context_types.py
class ContextType(Enum):
    LOCATION = "location"        # User's current room (MQTT push)
    RECENT_MEMORY = "recent_memory"  # Last 3 interactions (Redis pull on-demand)
    ENVIRONMENT = "environment"  # Weather, temperature (API pull with caching)
    PRESENCE = "presence"       # Who else is home (MQTT push)
    SCHEDULE = "schedule"       # Upcoming events (API pull on-demand)

class ContextCollector:
    """Enum-based context collector with specific implementations"""

    def __init__(self):
        # MQTT state (push contexts - real-time updates)
        self.current_locations = {}    # user_id -> location
        self.current_presence = {}     # user_id -> home/away

        # Cached data (interval contexts - periodic updates)
        self.weather_cache = {}
        self.last_weather_update = 0

        # MQTT client for Home Assistant integration
        self.mqtt_client = None

    async def initialize(self):
        """Setup MQTT subscriptions for push contexts"""
        self.mqtt_client = await self._setup_mqtt_client()

        # Subscribe to Home Assistant person states
        await self.mqtt_client.subscribe(
            "homeassistant/person/+/state",
            self._handle_location_update
        )

    async def gather_context(self, user_id: str, context_types: List[str]) -> dict:
        """Gather specific enum-based contexts"""
        context = {}

        for context_type in context_types:
            if context_type == ContextType.LOCATION.value:
                context["location"] = self._get_location(user_id)

            elif context_type == ContextType.RECENT_MEMORY.value:
                context["recent_memory"] = await self._get_recent_memory(user_id)

            elif context_type == ContextType.ENVIRONMENT.value:
                context["environment"] = await self._get_environment()

            elif context_type == ContextType.PRESENCE.value:
                context["presence"] = self._get_presence(user_id)

            elif context_type == ContextType.SCHEDULE.value:
                context["schedule"] = await self._get_schedule(user_id)

        return context

    # PUSH CONTEXT: Location (0ms - already in memory from MQTT)
    def _get_location(self, user_id: str) -> str:
        """Get current location from MQTT state"""
        return self.current_locations.get(user_id)

    async def _handle_location_update(self, topic: str, payload: dict):
        """Handle MQTT location updates from Home Assistant"""
        # Extract person from topic: homeassistant/person/mike/state
        person = topic.split("/")[2]
        location = payload.get("state")

        self.current_locations[person] = location
        logger.debug(f"Location update: {person} -> {location}")

    # PULL ON-DEMAND: Memory (~15ms - Redis search when needed)
    async def _get_recent_memory(self, user_id: str) -> List[str]:
        """Get recent memories via live Redis search"""
        keys_result = redis_get_keys(f"memory:{user_id}:*")
        if not keys_result.get("success"):
            return []

        recent_keys = sorted(keys_result["keys"])[-3:]
        memories = []

        for key in recent_keys:
            memory_result = redis_read(key)
            if memory_result.get("success"):
                memories.append(memory_result["value"])

        return memories

    # PULL INTERVAL: Weather (0ms - cached, updated every 30 minutes)
    async def _get_environment(self) -> dict:
        """Get weather from interval cache"""

        # Check if cache is stale
        if time.time() - self.last_weather_update > 1800:  # 30 minutes
            await self._update_weather_cache()

        return self.weather_cache

    async def _update_weather_cache(self):
        """Update weather cache via background task"""
        try:
            weather_data = await self._call_weather_api()
            self.weather_cache = {
                "temperature": weather_data.get("temperature"),
                "condition": weather_data.get("condition"),
                "time_of_day": self._get_time_of_day()
            }
            self.last_weather_update = time.time()

        except Exception as e:
            logger.warning(f"Weather update failed: {e}")

    # PUSH CONTEXT: Presence (0ms - already in memory from MQTT)
    def _get_presence(self, user_id: str) -> List[str]:
        """Get who else is home from MQTT state"""
        others_home = []
        for person, location in self.current_locations.items():
            if person != user_id and location == "home":
                others_home.append(person)
        return others_home

    # PULL ON-DEMAND: Schedule (~20ms - API call when needed)
    async def _get_schedule(self, user_id: str) -> List[dict]:
        """Get upcoming calendar events"""
        # Placeholder for calendar API integration
        try:
            events = await self._call_calendar_api(user_id)
            return events[:3]  # Next 3 events
        except Exception as e:
            logger.warning(f"Schedule fetch failed: {e}")
            return []
```

### 2. Enhanced Router Role with BNF Grammar

#### Router Implementation

```python
# Enhanced roles/core_router.py

ROLE_CONFIG = {
    "name": "router",
    "version": "2.0.0",
    "description": "Request routing with intelligent context selection",
    "llm_type": "WEAK",
    "fast_reply": True,
    "context_aware": True
}

ROUTER_PROMPT_WITH_BNF = """
Analyze this user request and determine routing and context requirements.

USER REQUEST: "{user_input}"

AVAILABLE CONTEXT TYPES: location, recent_memory, environment, presence, schedule

AVAILABLE ROLES: timer, weather, smart_home, planning, router

Output must follow this BNF grammar:

<routing_response> ::= "{" <route_field> "," <confidence_field> "," <parameters_field> "," <context_field> "}"

<route_field> ::= '"route": "' <role_name> '"'
<confidence_field> ::= '"confidence": ' <float_0_to_1>
<parameters_field> ::= '"parameters": {' <parameter_list> '}'
<context_field> ::= '"context_requirements": [' <context_list> ']'

<role_name> ::= "timer" | "weather" | "smart_home" | "planning" | "router"
<float_0_to_1> ::= "0." <digit> <digit>
<parameter_list> ::= <parameter> | <parameter> "," <parameter_list> | ""
<parameter> ::= '"' <param_name> '": "' <param_value> '"'
<context_list> ::= <context_type> | <context_type> "," <context_list> | ""
<context_type> ::= '"location"' | '"recent_memory"' | '"environment"' | '"presence"' | '"schedule"'

CONTEXT SELECTION GUIDELINES:
- location: For device control, room-specific actions
- recent_memory: For "usual", "like before", "remember" queries
- environment: For outdoor activities, comfort decisions
- presence: For whole-house actions, privacy considerations
- schedule: For time-sensitive responses, planning

EXAMPLES:
"Turn on the lights" → {{"route": "smart_home", "confidence": 0.95, "parameters": {{"action": "turn_on", "device": "lights"}}, "context_requirements": ["location"]}}

"Set a timer for 5 minutes" → {{"route": "timer", "confidence": 0.98, "parameters": {{"duration": "5m", "action": "set"}}, "context_requirements": []}}

"What did I do earlier?" → {{"route": "planning", "confidence": 0.90, "parameters": {{"query": "recent_activities"}}, "context_requirements": ["recent_memory"]}}

"Play my usual music" → {{"route": "smart_home", "confidence": 0.92, "parameters": {{"action": "play_music", "type": "preferred"}}, "context_requirements": ["location", "recent_memory"]}}
"""

@tool
def route_with_context_selection(user_input: str) -> dict:
    """Route request and determine required context using BNF grammar"""

    prompt = ROUTER_PROMPT_WITH_BNF.format(user_input=user_input)
    response = call_llm(prompt)
    return parse_bnf_router_response(response)
```

### 3. Post-Workflow Memory Assessment

#### Async Memory Assessment System

```python
# supervisor/memory_assessor.py

class MemoryAssessor:
    """Post-workflow memory importance assessment and storage"""

    def __init__(self):
        self.assessment_llm = self._create_memory_assessment_agent()

    def _create_memory_assessment_agent(self):
        """Create specialized LLM for memory assessment"""
        return UniversalAgent(
            role="memory_assessor",
            llm_type="WEAK",  # Fast, cheap model for assessment
            system_prompt="""Assess interaction importance for memory storage.

Rate interactions 0.0-1.0 based on:
- Personal information shared (names, relationships, preferences)
- Future commitments or plans mentioned
- Device preferences or settings changes
- Significant events or conversations
- Learning opportunities for future interactions

Output JSON only: {"importance": 0.0-1.0}

EXAMPLES:
"Turn on lights" → {"importance": 0.1}
"I met my neighbor Bob today" → {"importance": 0.8}
"Set timer for 5 minutes" → {"importance": 0.2}
"I prefer jazz music in the evening" → {"importance": 0.7}
"Remind me to call mom tomorrow" → {"importance": 0.6}"""
        )

    async def assess_and_store_if_important(self, user_id: str, prompt: str,
                                          response: str, location: str = None,
                                          workflow_id: str = None):
        """Assess interaction importance and store if worthwhile (completely async)"""

        try:
            # Create assessment prompt
            assessment_prompt = f"""
            USER INPUT: "{prompt}"
            SYSTEM RESPONSE: "{response}"
            LOCATION: {location or "unknown"}

            Rate the importance of storing this interaction.
            """

            # Async LLM call for assessment
            assessment_result = await self.assessment_llm.execute(assessment_prompt)
            assessment_data = json.loads(assessment_result)
            importance = assessment_data.get("importance", 0.0)

            # Store if above threshold
            if importance > 0.3:
                await self._store_memory(
                    user_id=user_id,
                    prompt=prompt,
                    response=response,
                    location=location,
                    importance=importance
                )

                logger.info(f"Stored memory for {user_id} (importance: {importance:.2f})")
            else:
                logger.debug(f"Skipped storing low-importance interaction (importance: {importance:.2f})")

        except Exception as e:
            logger.warning(f"Memory assessment failed for {user_id}: {e}")

    async def _store_memory(self, user_id: str, prompt: str, response: str,
                          location: str, importance: float):
        """Store memory with importance-based TTL"""

        memory_key = f"memory:{user_id}:{int(time.time())}"
        memory_content = f"Q: {prompt}\nA: {response}"

        if location:
            memory_content = f"[{location}] {memory_content}"

        # TTL based on importance (more important = longer retention)
        base_ttl = 2592000  # 30 days
        importance_multiplier = min(importance * 2, 1.0)  # Up to 2x for very important
        ttl = int(base_ttl * importance_multiplier)

        redis_write(memory_key, memory_content, ttl=ttl)

        logger.debug(f"Stored memory with {ttl/86400:.1f} day TTL")
```

### 4. WorkflowEngine Integration

#### Single-Path Context-Aware Processing

```python
# Enhanced supervisor/workflow_engine.py

class ContextAwareWorkflowEngine(WorkflowEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_collector = ContextCollector()
        self.memory_assessor = MemoryAssessor()

    async def initialize_context_systems(self):
        """Initialize context collection and memory assessment"""
        await self.context_collector.initialize()

    async def handle_request_with_context(self, request: RequestMetadata) -> str:
        """Single-path request handling with optional context enhancement"""

        # Step 1: Router determines role AND context requirements
        routing_result = await self._route_with_context_selection(request.prompt)

        # Step 2: Gather context if required (fast, specific enum methods)
        context = {}
        if routing_result.get("context_requirements"):
            context = await self.context_collector.gather_context(
                user_id=request.metadata.get("user_id"),
                context_types=routing_result["context_requirements"]
            )

        # Step 3: Enhance prompt if context exists
        enhanced_prompt = request.prompt
        if context:
            enhanced_prompt = self._add_context_to_prompt(request.prompt, context)

        # Step 4: Execute role (same path regardless of context)
        result = await self.universal_agent.execute(
            prompt=enhanced_prompt,
            role=routing_result["route"],
            parameters=routing_result.get("parameters", {}),
            context=request.metadata
        )

        # Step 5: Async memory assessment (non-blocking post-processing)
        if request.metadata.get("user_id"):
            asyncio.create_task(
                self.memory_assessor.assess_and_store_if_important(
                    user_id=request.metadata["user_id"],
                    prompt=request.prompt,
                    response=result,
                    location=context.get("location"),
                    workflow_id=request.metadata.get("workflow_id")
                )
            )

        return result

    def _add_context_to_prompt(self, base_prompt: str, context: dict) -> str:
        """Add context to prompt in structured format"""

        if not context:
            return base_prompt

        context_parts = []

        if context.get("location"):
            context_parts.append(f"Location: {context['location']}")

        if context.get("recent_memory"):
            recent = context["recent_memory"][-1] if context["recent_memory"] else ""
            if recent:
                context_parts.append(f"Recent: {recent[:50]}...")

        if context.get("environment"):
            env = context["environment"]
            env_parts = []
            if env.get("temperature"):
                env_parts.append(f"{env['temperature']}°F")
            if env.get("condition"):
                env_parts.append(env["condition"])
            if env_parts:
                context_parts.append(f"Environment: {', '.join(env_parts)}")

        if context.get("presence"):
            others = context["presence"]
            if others:
                context_parts.append(f"Also home: {', '.join(others)}")

        if context.get("schedule"):
            events = context["schedule"]
            if events:
                next_event = events[0]
                context_parts.append(f"Next: {next_event.get('title')} at {next_event.get('time')}")

        if context_parts:
            return f"{base_prompt}\n\nContext: {' | '.join(context_parts)}"

        return base_prompt
```

### 5. Home Assistant MQTT Integration

#### Location Tracking via MQTT

```python
# Enhanced common/channel_handlers/home_assistant_handler.py

class MQTTLocationTracker:
    """Track Home Assistant person states via MQTT"""

    def __init__(self, mqtt_broker_host: str, mqtt_broker_port: int = 1883):
        self.mqtt_host = mqtt_broker_host
        self.mqtt_port = mqtt_broker_port
        self.mqtt_client = None
        self.context_collector = None  # Injected by supervisor

    async def initialize(self):
        """Setup MQTT client and subscriptions"""

        import aiomqtt

        self.mqtt_client = aiomqtt.Client(
            hostname=self.mqtt_host,
            port=self.mqtt_port
        )

        await self.mqtt_client.connect()

        # Subscribe to person state changes
        await self.mqtt_client.subscribe("homeassistant/person/+/state")

        # Start message processing loop
        asyncio.create_task(self._process_mqtt_messages())

    async def _process_mqtt_messages(self):
        """Process incoming MQTT messages"""

        async for message in self.mqtt_client.messages:
            try:
                topic = message.topic.value
                payload = json.loads(message.payload.decode())

                if topic.startswith("homeassistant/person/"):
                    await self._handle_person_state_change(topic, payload)

            except Exception as e:
                logger.warning(f"MQTT message processing failed: {e}")

    async def _handle_person_state_change(self, topic: str, payload: dict):
        """Handle person state changes from Home Assistant"""

        # Extract person name from topic
        person = topic.split("/")[2]
        new_state = payload.get("state")

        # Update context collector if available
        if self.context_collector:
            self.context_collector.current_locations[person] = new_state

        logger.info(f"Person state update: {person} -> {new_state}")
```

## Implementation Plan

### Phase 1: Context Foundation (Week 1)

**Files Created:**

- [`common/context_types.py`](common/context_types.py) - Context enum and collector
- [`supervisor/memory_assessor.py`](supervisor/memory_assessor.py) - Async memory assessment

**Tasks:**

1. Implement ContextType enum and ContextCollector class
2. Create memory assessment system with WEAK LLM
3. Add basic MQTT client setup
4. Test individual context gathering methods

### Phase 2: Enhanced Router (Week 2)

**Files Modified:**

- [`roles/core_router.py`](roles/core_router.py) - Add BNF grammar and context selection

**Tasks:**

1. Update router role with BNF grammar for context requirements
2. Add context selection guidelines to router prompt
3. Test router context decision accuracy
4. Validate BNF parsing reliability

### Phase 3: WorkflowEngine Integration (Week 3)

**Files Modified:**

- [`supervisor/workflow_engine.py`](supervisor/workflow_engine.py) - Context-aware request handling

**Tasks:**

1. Integrate ContextCollector with WorkflowEngine
2. Implement single-path context enhancement
3. Add async memory assessment to request flow
4. Test end-to-end context-aware processing

### Phase 4: MQTT and External Integration (Week 4)

**Files Modified:**

- [`common/channel_handlers/home_assistant_handler.py`](common/channel_handlers/home_assistant_handler.py) - MQTT integration

**Tasks:**

1. Implement MQTT location tracking
2. Add weather API integration with caching
3. Add calendar API integration (optional)
4. Performance testing and optimization

## Performance Analysis

### Latency Impact by Context Type

- **No context**: 0ms overhead (same execution path)
- **Location only**: 0ms (MQTT push, already in memory)
- **Recent memory**: ~15ms (Redis key scan)
- **Environment**: 0ms (cached weather data)
- **Multiple contexts**: ~15ms (parallel gathering, dominated by memory search)

### Memory Usage

- **MQTT state**: ~1KB for location/presence data
- **Weather cache**: ~5KB updated every 30 minutes
- **Memory storage**: ~1KB per stored interaction
- **Total overhead**: ~10KB base + ~30MB per user over 30 days

### Resource Requirements

- **MQTT broker**: Existing (already available)
- **Redis**: Existing (already in use)
- **Background tasks**: 1 weather update task every 30 minutes
- **Network**: Minimal MQTT traffic for location updates

## Configuration

```yaml
# config.yaml additions
household_assistant:
  context_awareness:
    enabled: true

  context_types:
    location:
      enabled: true
      source: "mqtt"
      mqtt_topic: "homeassistant/person/+/state"

    recent_memory:
      enabled: true
      source: "redis"
      max_memories: 3

    environment:
      enabled: true
      source: "api"
      cache_minutes: 30
      weather_api_key: "${WEATHER_API_KEY}"

    presence:
      enabled: true
      source: "mqtt"

    schedule:
      enabled: false # Optional calendar integration

  memory_assessment:
    enabled: true
    model: "WEAK"
    importance_threshold: 0.3
    async_processing: true

  mqtt:
    broker_host: "homeassistant.local"
    broker_port: 1883
    username: "${MQTT_USERNAME}"
    password: "${MQTT_PASSWORD}"
```

## Expected Outcomes

### User Experience Improvements

- **Location-aware responses**: "Turned on bedroom lights at 30% for nighttime"
- **Memory-enhanced interactions**: "Playing your usual jazz playlist"
- **Contextual information**: "72°F and sunny - perfect for your outdoor meeting at 2 PM"
- **Presence-aware actions**: "Turned on all lights since everyone is home"

### Performance Characteristics

- **Simple commands**: No latency impact
- **Context-enhanced commands**: ~15ms additional latency
- **Memory assessment**: 0ms (async post-processing)
- **Overall system**: <1% latency increase for 80% intelligence improvement

This refined design provides intelligent household assistant capabilities while maintaining LLM-safe simplicity and leveraging your existing MQTT infrastructure.
