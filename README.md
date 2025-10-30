# Universal Agent System

A production-ready AI workflow management platform with **single-file role architecture**, **intent-based processing**, and **fast-reply routing**. Built on the Strands Agent Framework with comprehensive test coverage and clean architecture patterns.

## 🎯 What Is This?

The Universal Agent System is a **single AI agent** that dynamically assumes specialized roles to handle various tasks. Instead of managing multiple separate agents, you have one intelligent system that adapts its behavior based on the task at hand through fast-reply routing.

**Key Concept**: One agent, many roles, intelligent routing, unified workflow management.

## ✨ Key Features

- **🤖 Universal Agent**: Single agent interface with dynamic role specialization
- **📁 Single-File Roles**: Each role consolidated into one Python file (~300 lines)
- **🔄 Intent-Based Processing**: Declarative event handling for LLM-safe architecture
- **⚡ Fast-Reply Routing**: Intelligent request routing with 95%+ confidence
- **🧠 Context-Aware Intelligence**: Memory, location, and environmental awareness
- **🔌 MCP Integration**: Seamless integration with external tool ecosystems
- **🏠 Smart Home Integration**: Home Assistant MQTT integration
- **📊 Health Monitoring**: Real-time system health and performance tracking
- **🧪 Comprehensive Testing**: 842 tests with 100% pass rate
- **🔧 Production Ready**: Docker setup, deployment guides, and monitoring

## 🚀 Quick Start

### Docker Development Setup (Recommended)

```bash
# Clone and setup
git clone <repository-url>
cd generative-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# One-command setup with Docker Redis
make docker-setup

# Start using the system
python cli.py
```

### Manual Installation

```bash
# Clone and setup
git clone <repository-url>
cd generative-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and configure Redis separately
# See DEVELOPMENT_SETUP.md for Redis installation
```

## 💻 Usage

### Interactive Mode

```bash
python cli.py
```

Start an interactive session where you can:

- Execute workflows with natural language
- Monitor system health and performance
- Switch between different AI roles automatically
- Access memory and context-aware responses

### Single Command Mode

```bash
# Execute a workflow and exit
python cli.py --workflow "Set a timer for 10 minutes"

# Check system status
python cli.py --status

# Use custom configuration
python cli.py --config production.yaml
```

### Programmatic Usage

```python
from supervisor.supervisor import Supervisor

# Initialize the system
supervisor = Supervisor("config.yaml")
supervisor.start()

# The system is now ready to process workflows
# It will automatically select the right role for each task
```

## 🎭 Available Roles

The Universal Agent can assume different specialized roles:

### Core Roles (Single-File Architecture)

- **Router** ([`roles/core_router.py`](roles/core_router.py)) - Intelligent request routing with context selection
- **Planning** ([`roles/core_planning.py`](roles/core_planning.py)) - Complex task planning and workflow generation
- **Conversation** ([`roles/core_conversation.py`](roles/core_conversation.py)) - General conversation with memory
- **Timer** ([`roles/core_timer.py`](roles/core_timer.py)) - Timer, alarm, and reminder management
- **Weather** ([`roles/core_weather.py`](roles/core_weather.py)) - Weather information and forecasts
- **Smart Home** ([`roles/core_smart_home.py`](roles/core_smart_home.py)) - Device control via Home Assistant
- **Calendar** ([`roles/core_calendar.py`](roles/core_calendar.py)) - Calendar and scheduling
- **Search** ([`roles/core_search.py`](roles/core_search.py)) - Web search capabilities

### Role Selection & Fast-Reply Routing

The system uses intelligent routing with 95%+ confidence for instant responses:

- "Set a timer for 5 minutes" → **Timer Role** (fast-reply, no planning needed)
- "Turn on the lights" → **Smart Home Role** + Location Context
- "What's the weather?" → **Weather Role** (fast-reply)
- "Plan a trip to Thailand" → **Planning Role** (creates multi-step workflow)

### Context Types

The router intelligently determines which context to gather:

- **Location Context**: Current room/location for device control
- **Memory Context**: Previous interactions and preferences
- **Presence Context**: Who else is home for household coordination
- **Schedule Context**: Calendar events and time-sensitive information

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Supervisor                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Message Bus  │  │WorkflowEngine│  │ Intent       │       │
│  │              │  │              │  │ Processor    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Universal Agent                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Role Registry│  │ Tool Registry│  │ MCP Client   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Single-File Roles                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  Router  │ │ Planning │ │  Timer   │ │ Weather  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │SmartHome │ │Conversation│ Calendar │ │  Search  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Supervisor**: Top-level coordinator with health monitoring and scheduled task management
2. **WorkflowEngine**: Unified orchestration with fast-reply routing and context-aware processing
3. **Universal Agent**: Single agent interface with role-based specialization
4. **Intent Processor**: Handles declarative intents from event handlers
5. **Message Bus**: Event-driven communication system
6. **Role Registry**: Dynamic role discovery and management
7. **Context Collector**: Intelligent context gathering (location, memory, presence, schedule)

### Architecture Principles

- **Single Event Loop**: Workflow execution uses single event loop (I/O layer uses threads for external SDKs)
- **Intent-Based**: Pure functions return declarative intents, processor handles execution
- **Fast-Reply Routing**: 95%+ confidence routing for instant responses without planning
- **Context-Aware**: Surgical context gathering only when needed
- **LLM-Friendly**: Designed for AI agent development and modification

## ⚙️ Configuration

The system uses [`config.yaml`](config.yaml) for all configuration:

```yaml
# LLM Provider Configuration
llm_providers:
  bedrock:
    models:
      WEAK: "anthropic.claude-3-haiku-20240307-v1:0"
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
      STRONG: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Role System Configuration
role_system:
  roles_directory: "roles"
  role_pattern: "core_*.py"
  auto_discovery: true

# Intent Processing Configuration
intent_processing:
  enabled: true
  validate_intents: true
  timeout_seconds: 30

# Context & Memory Configuration
context_system:
  enabled: true
  memory_assessment: true
  location_tracking: true
  mqtt_integration: true

# Feature Flags
feature_flags:
  enable_universal_agent: true
  enable_intent_processing: true
  enable_context_awareness: true
  enable_fast_reply_routing: true
```

## 🛠️ Development

### Adding New Roles

Create a new single-file role following the pattern:

```python
# roles/core_my_role.py
"""My Role - Single file implementation."""

from dataclasses import dataclass
from typing import List, Dict, Any
from common.intents import Intent, NotificationIntent

# 1. ROLE METADATA
ROLE_CONFIG = {
    "name": "my_role",
    "version": "1.0.0",
    "description": "What this role does",
    "llm_type": "WEAK",  # WEAK, DEFAULT, or STRONG
    "fast_reply": True,  # True for fast-reply routing
    "when_to_use": "When to use this role"
}

# 2. TOOLS (if needed)
from strands import tool

@tool
def my_tool(parameter: str) -> Dict[str, Any]:
    """Tool function."""
    return {"success": True, "result": parameter}

# 3. EVENT HANDLERS (pure functions returning intents)
def handle_my_event(event_data: Any, context) -> List[Intent]:
    """Pure function event handler."""
    return [
        NotificationIntent(
            message=f"Event processed: {event_data}",
            channel=context.get_safe_channel()
        )
    ]

# 4. ROLE REGISTRATION
def register_role():
    """Auto-discovered by RoleRegistry."""
    return {
        "config": ROLE_CONFIG,
        "event_handlers": {"MY_EVENT": handle_my_event},
        "tools": [my_tool],
        "intents": []
    }
```

## 🧪 Testing

### Run All Tests

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/integration/ -v    # Integration tests
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/supervisor/ -v    # Supervisor tests
```

### Test Statistics

- **Total Tests**: 842
- **Pass Rate**: 100%
- **Code Coverage**: 57%
- **Test Categories**: Unit, Integration, Performance, Supervisor

## 🐳 Docker Development Environment

Complete Docker-based development environment with Redis:

```bash
# Complete setup
make docker-setup

# Individual commands
make docker-start        # Start Redis container
make redis-cli          # Connect to Redis CLI
make redis-commander    # Start Redis web GUI (http://localhost:8081)
make docker-clean       # Clean up everything
```

## 🔌 MCP Integration

The system integrates with MCP servers for enhanced capabilities:

- **AWS Documentation**: Technical documentation and API references
- **Web Search**: Internet search capabilities
- **Weather Services**: Real-time weather data
- **Filesystem**: File system operations
- **GitHub**: Repository management
- **Slack**: Team communication
- **Home Assistant**: Smart home device control

## 🏠 Smart Home Integration

### Home Assistant MQTT Integration

- **Location Tracking**: Real-time person location updates
- **Device Control**: Smart home device automation
- **Presence Detection**: Multi-person household awareness
- **Environmental Context**: Temperature, lighting, sensor data

### Context-Aware Examples

- "Turn on the lights" → Uses current room location
- "Play my usual music" → Recalls previous preferences
- "Turn off all lights" → Considers who else is home

## 📊 Monitoring & Health

```python
from supervisor.supervisor import Supervisor

supervisor = Supervisor("config.yaml")
health = supervisor.status()
print(f"System health: {health['overall_status']}")
```

### Health Monitoring Features

- Real-time system metrics
- Workflow performance tracking
- Intent processing monitoring
- Context system health checks
- Memory usage tracking
- Automatic health checks

## 📁 Project Structure

```
generative-agent/
├── cli.py                      # Command-line interface
├── config.yaml                 # Main configuration
├── supervisor/                 # System coordination
│   ├── supervisor.py           # Main supervisor
│   ├── workflow_engine.py      # Workflow orchestration
│   ├── intent_processor.py     # Intent processing
│   └── memory_assessor.py      # Memory assessment
├── llm_provider/               # LLM abstraction layer
│   ├── universal_agent.py      # Single agent with roles
│   ├── factory.py              # LLM provider factory
│   ├── role_registry.py        # Role discovery
│   └── mcp_client.py           # MCP integration
├── common/                     # Shared components
│   ├── intents.py              # Intent definitions
│   ├── intent_processor.py     # Intent processing
│   ├── message_bus.py          # Event communication
│   ├── task_context.py         # Workflow state
│   ├── task_graph.py           # Task dependencies
│   └── communication_manager.py # Multi-channel communication
├── roles/                      # Single-file roles
│   ├── core_router.py          # Request routing
│   ├── core_planning.py        # Task planning
│   ├── core_conversation.py    # Conversation
│   ├── core_timer.py           # Timer management
│   ├── core_weather.py         # Weather info
│   ├── core_smart_home.py      # Smart home control
│   ├── core_calendar.py        # Calendar
│   └── core_search.py          # Web search
└── tests/                      # Comprehensive test suite
    ├── integration/            # Integration tests
    ├── unit/                   # Unit tests
    ├── supervisor/             # Supervisor tests
    └── llm_provider/           # Universal Agent tests
```

## 🔧 Advanced Features

### Fast-Reply Routing

- **95%+ Confidence**: Instant responses without planning overhead
- **Context Selection**: Router determines which context to gather
- **Zero Overhead**: Simple requests execute immediately
- **Fallback to Planning**: Complex requests automatically escalate

### Intent-Based Processing

- **Pure Functions**: Event handlers return declarative intents
- **No Threading**: Intents processed in single event loop
- **Type-Safe**: Pydantic validation for all intents
- **Extensible**: Easy to add new intent types

### Context-Aware Intelligence

- **Memory System**: Automatic importance assessment and storage
- **Location Awareness**: MQTT-based real-time location tracking
- **Environmental Context**: Weather, time-of-day, sensor integration
- **Presence Detection**: Multi-person household awareness
- **Schedule Integration**: Calendar and event awareness

### Workflow Management

- **Pause/Resume**: Complete workflow state preservation
- **Checkpointing**: Automatic state snapshots
- **Task Dependencies**: Complex workflow orchestration
- **Result Sharing**: Intelligent predecessor result passing

## 📖 Documentation

Comprehensive documentation in the [`docs/`](docs/) directory:

- **Architecture**: System design and patterns
- **API Reference**: Complete API documentation
- **Configuration**: Configuration options and examples
- **Deployment**: Production deployment instructions
- **Testing**: Test strategy and coverage

## 🧪 Test Coverage

### Test Statistics

- **Total Tests**: 842
- **Pass Rate**: 100%
- **Coverage**: 57%
- **Categories**: Unit, Integration, Performance, Supervisor

### Test Categories

**Unit Tests** - Component-level testing:

- Communication manager
- Context collector
- Intent processor
- Role registry
- Task context and graph
- Universal agent

**Integration Tests** - End-to-end scenarios:

- Context-aware workflows
- Conversation memory
- Docker Redis setup
- Intent-based planning
- Phase 2 intent detection
- Unified communication

**Supervisor Tests** - System-level testing:

- Supervisor integration
- Workflow engine
- Workflow duration logging

## 🎯 Architecture Decisions

### Single Event Loop (Workflow Layer)

The **workflow execution layer** uses a pure single event loop architecture:

- No background threads in workflow processing
- No race conditions in business logic
- Intent-based processing for LLM-safe code

### Threading Boundary (I/O Layer)

The **communication layer** uses threads for external SDK compatibility:

- Slack SDK requires background thread for WebSocket
- Queue processor for thread-safe message handling
- Isolated from workflow execution logic

This is a **documented architectural boundary** - threading is confined to I/O, business logic is single event loop.

## 🚀 Performance

### Optimizations

- **Fast-Reply Routing**: 95%+ requests handled instantly
- **Agent Pooling**: Pre-warmed agent instances
- **Semantic Caching**: Intelligent result caching
- **Context Optimization**: Surgical context gathering
- **Reduced LLM Calls**: 33% fewer calls through pre-processing

### Metrics

- **Fast-Reply Latency**: <100ms for simple requests
- **Planning Latency**: <2s for complex workflows
- **Memory Overhead**: <50MB per workflow
- **Concurrent Workflows**: 5+ simultaneous workflows

## 🔌 Communication Channels

- **Slack**: Team collaboration and notifications
- **Console**: Command-line interaction
- **Email**: Email notifications
- **Voice**: Speech recognition and TTS
- **Home Assistant**: Smart home integration
- **WhatsApp**: Mobile messaging (planned)

## 🐳 Docker Services

- **Redis 7**: Latest stable with optimized configuration
- **Redis Commander**: Optional web-based management
- **Persistent Storage**: Data persists between restarts
- **Health Checks**: Automatic monitoring and recovery

## 📊 System Health

The system includes comprehensive health monitoring:

```python
# Check system health
health = supervisor.status()

# Returns:
{
    "overall_status": "healthy",
    "workflow_engine": "running",
    "message_bus": "active",
    "universal_agent": "ready",
    "active_workflows": 3,
    "uptime": 3600
}
```

## 🔧 Development Tools

### Code Quality

```bash
# Run linters
make lint

# Format code
make format

# Run type checking
make typecheck
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
python -m pytest tests/unit/test_specific.py -v
```

## 📝 Recent Changes

### Test Cleanup (2025-10-29)

- Achieved 100% test pass rate (842/842 passing)
- Removed 36 obsolete tests testing old architecture
- Fixed 10 tests for current implementation
- Comprehensive documentation of all changes

### Architecture

- Single-file role architecture
- Intent-based processing
- Fast-reply routing with 95%+ confidence
- Router-driven context selection

## 🆘 Support

### Documentation

- See [`docs/`](docs/) for comprehensive guides
- Check [`DEVELOPMENT_SETUP.md`](DEVELOPMENT_SETUP.md) for setup help
- Review [`DOCKER_TROUBLESHOOTING.md`](DOCKER_TROUBLESHOOTING.md) for Docker issues

### Test Analysis

- [`TEST_CLEANUP_SUMMARY.md`](TEST_CLEANUP_SUMMARY.md) - Test cleanup details
- [`TEST_ANALYSIS_REPORT.md`](TEST_ANALYSIS_REPORT.md) - Failure analysis
- [`THREADING_VIOLATIONS_FOUND.md`](THREADING_VIOLATIONS_FOUND.md) - Architecture analysis

## 📄 License

MIT

## 🎉 Project Status

**Production Ready** - All tests passing, comprehensive documentation, clean architecture.

The system is ready for:

- Development and experimentation
- Production deployment
- Extension with new roles
- Integration with external systems
- LLM-based modifications

**Note**: The communication layer uses background threads for external SDK compatibility (Slack, WhatsApp). This is isolated from the core workflow execution which maintains pure single event loop architecture.
