# Configuration Guide

## Overview

The StrandsAgent Universal Agent system uses a centralized configuration approach with a single `config.yaml` file that supports environment variable substitution, feature flags, and role-based optimization.

## Main Configuration File

### Structure

The main configuration file `config.yaml` is organized into logical sections:

```yaml
# System-wide logging configuration
logging:
  log_level: INFO
  log_file: logs/system.log
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_file_size: 10MB
  backup_count: 5

# LLM provider configurations with semantic types
llm_providers:
  # Default model for balanced tasks
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
    max_tokens: 4096
    timeout: 30

  # Strong model for complex reasoning
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: us.amazon.nova-premier-v1:0
    temperature: 0.1
    max_tokens: 8192
    timeout: 60

  # Weak model for simple tasks
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: us.amazon.nova-lite-v1:0
    temperature: 0.5
    max_tokens: 2048
    timeout: 15

# Universal Agent configuration
universal_agent:
  framework: strands
  max_concurrent_tasks: 5
  checkpoint_interval: 300
  retry_attempts: 3
  retry_delay: 1.0

  # Role-based LLM optimization
  role_optimization:
    planning: STRONG # Complex reasoning and task planning
    search: WEAK # Simple search queries
    weather: WEAK # Basic weather lookups
    summarizer: DEFAULT # Balanced text processing
    slack: DEFAULT # Conversational interactions

# MCP server configuration
mcp:
  config_file: config/mcp_config.yaml
  enabled: true
  timeout: 30
  retry_attempts: 3

# Feature flags for system capabilities
features:
  universal_agent_enabled: true
  mcp_integration_enabled: true
  task_scheduling_enabled: true
  checkpoint_persistence: true
  metrics_collection: true
  health_monitoring: true

# Development and debugging settings
development:
  debug_mode: false
  test_mode: false
  mock_external_services: false
  verbose_logging: false
  performance_profiling: false

# Legacy agent configuration (backward compatibility)
agents:
  planning_agent:
    enabled: true
    model_type: strong
  search_agent:
    enabled: true
    model_type: weak
  weather_agent:
    enabled: true
    model_type: weak
  summarizer_agent:
    enabled: true
    model_type: default
  slack_agent:
    enabled: true
    model_type: default
```

## Configuration Sections

### Logging Configuration

Controls system-wide logging behavior:

```yaml
logging:
  log_level: INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: logs/system.log # Log file path
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_file_size: 10MB # Maximum log file size before rotation
  backup_count: 5 # Number of backup log files to keep
  console_output: true # Enable console logging
  structured_logging: false # Enable JSON structured logging
```

**Log Levels:**

- `DEBUG`: Detailed debugging information
- `INFO`: General system information (recommended for production)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical system failures

### LLM Provider Configuration

Defines available LLM models with semantic types:

```yaml
llm_providers:
  # Configuration name (can be any identifier)
  default:
    llm_class: DEFAULT # Semantic type: WEAK, DEFAULT, STRONG
    provider_type: bedrock # Provider: bedrock, openai, anthropic
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3 # Creativity level (0.0-1.0)
    max_tokens: 4096 # Maximum response tokens
    timeout: 30 # Request timeout in seconds

    # Provider-specific parameters
    additional_params:
      top_p: 0.9
      frequency_penalty: 0.0
      presence_penalty: 0.0
```

**Semantic Types:**

- `WEAK`: Fast, cost-effective models for simple tasks
- `DEFAULT`: Balanced models for general-purpose tasks
- `STRONG`: Powerful models for complex reasoning and analysis

**Supported Providers:**

- `bedrock`: AWS Bedrock models
- `openai`: OpenAI models
- `anthropic`: Anthropic Claude models

### Universal Agent Configuration

Controls Universal Agent behavior and optimization:

```yaml
universal_agent:
  framework: strands # Framework type (strands only)
  max_concurrent_tasks: 5 # Maximum parallel task execution
  checkpoint_interval: 300 # Checkpoint creation interval (seconds)
  retry_attempts: 3 # Maximum retry attempts for failed tasks
  retry_delay: 1.0 # Delay between retries (seconds)
  task_timeout: 600 # Maximum task execution time (seconds)

  # Role-based model optimization
  role_optimization:
    planning: STRONG # Use STRONG models for planning tasks
    search: WEAK # Use WEAK models for search tasks
    weather: WEAK # Use WEAK models for weather tasks
    summarizer: DEFAULT # Use DEFAULT models for summarization
    slack: DEFAULT # Use DEFAULT models for Slack tasks

  # Tool configuration per role
  role_tools:
    planning:
      - create_task_plan
      - validate_dependencies
      - optimize_execution_order
    search:
      - web_search
      - search_with_filters
    weather:
      - get_weather
      - get_forecast
    summarizer:
      - summarize_text
      - extract_key_points
    slack:
      - send_message
      - get_channel_history
```

### MCP Server Configuration

Controls Model Context Protocol server integration:

```yaml
mcp:
  config_file: config/mcp_config.yaml # Separate MCP configuration file
  enabled: true # Enable/disable MCP integration
  timeout: 30 # MCP server connection timeout
  retry_attempts: 3 # Connection retry attempts
  discovery_interval: 60 # Server discovery interval (seconds)

  # Server-specific settings
  servers:
    aws_docs:
      enabled: true
      priority: high
    web_search:
      enabled: true
      priority: medium
    weather:
      enabled: true
      priority: low
```

### Feature Flags

Enable or disable system capabilities:

```yaml
features:
  universal_agent_enabled: true # Enable Universal Agent system
  mcp_integration_enabled: true # Enable MCP server integration
  task_scheduling_enabled: true # Enable task scheduling and queuing
  checkpoint_persistence: true # Enable workflow pause/resume
  metrics_collection: true # Enable metrics and monitoring
  health_monitoring: true # Enable health checks
  security_enabled: false # Enable authentication/authorization
  rate_limiting_enabled: false # Enable API rate limiting
  caching_enabled: false # Enable response caching
```

### Development Settings

Configuration for development and debugging:

```yaml
development:
  debug_mode: false # Enable debug logging and features
  test_mode: false # Enable test mode with mocked services
  mock_external_services: false # Mock external API calls
  verbose_logging: false # Enable verbose logging
  performance_profiling: false # Enable performance profiling
  auto_reload: false # Auto-reload configuration changes

  # Test-specific settings
  test_settings:
    fast_execution: true # Skip delays in test mode
    mock_llm_responses: false # Use mock LLM responses
    disable_external_calls: true # Disable external API calls
```

## Environment Variable Substitution

The system supports environment variable substitution using `${VAR:default}` syntax:

### Basic Substitution

```yaml
llm_providers:
  default:
    model_id: ${BEDROCK_MODEL_ID:us.amazon.nova-pro-v1:0}
    temperature: ${TEMPERATURE:0.3}
    timeout: ${TIMEOUT:30}
```

### Advanced Substitution

```yaml
logging:
  log_level: ${LOG_LEVEL:INFO}
  log_file: ${LOG_DIR:logs}/${LOG_FILE:system.log}

universal_agent:
  max_concurrent_tasks: ${MAX_TASKS:5}
  checkpoint_interval: ${CHECKPOINT_INTERVAL:300}
```

### Environment File Support

Create a `.env` file for environment-specific settings:

```bash
# .env file
BEDROCK_MODEL_ID=us.amazon.nova-premier-v1:0
TEMPERATURE=0.2
LOG_LEVEL=DEBUG
MAX_TASKS=10
CHECKPOINT_INTERVAL=180

# MCP settings
MCP_ENABLED=true
MCP_TIMEOUT=45

# Feature flags
SECURITY_ENABLED=true
RATE_LIMITING_ENABLED=true
```

## Configuration Validation

### Automatic Validation

The system automatically validates configuration on startup:

```python
from config.config_manager import ConfigManager

# Load and validate configuration
config_manager = ConfigManager("config.yaml")
config = config_manager.get_config()

# Validation errors are reported with specific details
if not config_manager.is_valid():
    errors = config_manager.get_validation_errors()
    for error in errors:
        print(f"Configuration error: {error}")
```

### Validation Rules

**Required Fields:**

- `llm_providers.default` must be defined
- `universal_agent.framework` must be "strands"
- `logging.log_level` must be valid level

**Value Constraints:**

- `temperature` must be between 0.0 and 1.0
- `max_concurrent_tasks` must be positive integer
- `timeout` values must be positive numbers
- `retry_attempts` must be non-negative integer

### Custom Validation

Add custom validation rules:

```python
from config.config_manager import ConfigManager

class CustomConfigManager(ConfigManager):
    def validate_custom_rules(self, config):
        errors = []

        # Custom validation logic
        if config.get('universal_agent', {}).get('max_concurrent_tasks', 0) > 20:
            errors.append("max_concurrent_tasks should not exceed 20 for stability")

        return errors
```

## Environment-Specific Configuration

### Development Configuration

```yaml
# config/development.yaml
logging:
  log_level: DEBUG
  console_output: true

development:
  debug_mode: true
  verbose_logging: true
  mock_external_services: true

features:
  performance_profiling: true
  auto_reload: true
```

### Production Configuration

```yaml
# config/production.yaml
logging:
  log_level: INFO
  log_file: /var/log/strands-agent/system.log
  structured_logging: true

universal_agent:
  max_concurrent_tasks: 10
  checkpoint_interval: 180

features:
  security_enabled: true
  rate_limiting_enabled: true
  metrics_collection: true
  health_monitoring: true

development:
  debug_mode: false
  test_mode: false
```

### Testing Configuration

```yaml
# config/testing.yaml
logging:
  log_level: WARNING
  console_output: false

development:
  test_mode: true
  mock_external_services: true
  fast_execution: true

features:
  mcp_integration_enabled: false
  metrics_collection: false
```

## Configuration Management

### Loading Configuration

```python
from config.config_manager import ConfigManager

# Load default configuration
config_manager = ConfigManager()

# Load specific configuration file
config_manager = ConfigManager("config/production.yaml")

# Load with environment variable override
config_manager = ConfigManager(
    config_file="${CONFIG_FILE:config.yaml}"
)
```

### Runtime Configuration Updates

```python
# Update configuration at runtime
config_manager.update_config({
    'universal_agent': {
        'max_concurrent_tasks': 8
    }
})

# Reload configuration from file
config_manager.reload()

# Get specific configuration section
llm_config = config_manager.get_section('llm_providers')
```

### Configuration Monitoring

```python
# Monitor configuration changes
def on_config_change(old_config, new_config):
    print("Configuration updated")
    # Handle configuration changes

config_manager.add_change_listener(on_config_change)
```

## Security Configuration

### Authentication Settings

```yaml
security:
  authentication:
    enabled: true
    method: token # token, oauth, basic
    token_expiry: 3600 # Token expiry in seconds

  authorization:
    enabled: true
    default_role: user # Default role for authenticated users

  # API security
  api:
    cors_enabled: true
    cors_origins: ["http://localhost:3000"]
    rate_limiting:
      enabled: true
      requests_per_minute: 100
      burst_size: 20
```

### Secrets Management

```yaml
secrets:
  provider: environment # environment, vault, aws_secrets

  # Environment variable mapping
  environment_mapping:
    openai_api_key: OPENAI_API_KEY
    slack_token: SLACK_BOT_TOKEN
    weather_api_key: WEATHER_API_KEY

  # AWS Secrets Manager (if using aws_secrets provider)
  aws_secrets:
    region: us-west-2
    secret_names:
      - strands-agent/api-keys
      - strands-agent/database-credentials
```

## Performance Tuning

### Concurrency Settings

```yaml
performance:
  # WorkflowEngine settings
  workflow_engine:
    max_concurrent_tasks: 5 # Balance throughput vs resources
    task_queue_size: 100 # Maximum queued tasks
    worker_pool_size: 10 # Background worker threads

  # Universal Agent settings
  universal_agent:
    model_cache_size: 50 # Cached model instances
    conversation_cache_size: 100 # Cached conversation contexts
    tool_timeout: 30 # Tool execution timeout

  # Memory management
  memory:
    max_conversation_history: 1000 # Maximum conversation messages
    checkpoint_compression: true # Compress checkpoint data
    garbage_collection_interval: 300 # GC interval in seconds
```

### Optimization Settings

```yaml
optimization:
  # Model selection optimization
  model_selection:
    auto_optimization: true # Automatically optimize model selection
    performance_tracking: true # Track model performance
    cost_optimization: true # Optimize for cost efficiency

  # Caching configuration
  caching:
    enabled: true
    ttl: 3600 # Cache TTL in seconds
    max_size: 1000 # Maximum cached items

  # Batch processing
  batching:
    enabled: true
    batch_size: 10 # Tasks per batch
    batch_timeout: 5 # Batch collection timeout
```

## Monitoring Configuration

### Metrics Collection

```yaml
monitoring:
  metrics:
    enabled: true
    collection_interval: 60 # Metrics collection interval
    retention_days: 30 # Metrics retention period

    # Metric categories
    categories:
      system: true # System resource metrics
      workflow: true # Workflow execution metrics
      agent: true # Universal Agent metrics
      mcp: true # MCP server metrics

  # Health checks
  health_checks:
    enabled: true
    check_interval: 30 # Health check interval
    timeout: 10 # Health check timeout

    # Component health checks
    components:
      workflow_engine: true
      universal_agent: true
      mcp_servers: true
      database: true
```

### Alerting Configuration

```yaml
alerting:
  enabled: true

  # Alert channels
  channels:
    email:
      enabled: true
      smtp_server: smtp.example.com
      recipients: ["admin@example.com"]

    slack:
      enabled: true
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#alerts"

  # Alert rules
  rules:
    high_error_rate:
      condition: "error_rate > 0.05"
      severity: critical
      cooldown: 300

    high_memory_usage:
      condition: "memory_usage > 0.8"
      severity: warning
      cooldown: 600
```

## MCP Server Configuration

### MCP Configuration File

The `config/mcp_config.yaml` file defines MCP server connections:

```yaml
# MCP Server Configuration
servers:
  aws_docs:
    command: uvx
    args: ["awslabs.aws-documentation-mcp-server@latest"]
    env:
      AWS_REGION: us-west-2
    enabled: true
    priority: high
    timeout: 30

  web_search:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-web-search"]
    env:
      SEARCH_API_KEY: ${SEARCH_API_KEY}
    enabled: true
    priority: medium
    timeout: 20

  weather:
    command: python
    args: ["-m", "weather_mcp_server"]
    env:
      WEATHER_API_KEY: ${WEATHER_API_KEY}
    enabled: true
    priority: low
    timeout: 15

  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    enabled: true
    priority: medium
    timeout: 10

  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
    enabled: false
    priority: low
    timeout: 25

  slack:
    command: python
    args: ["-m", "slack_mcp_server"]
    env:
      SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN}
    enabled: true
    priority: medium
    timeout: 20
```

### MCP Server Management

```python
from llm_provider.mcp_client import MCPClient

# Initialize MCP client with configuration
mcp_client = MCPClient("config/mcp_config.yaml")

# Start all enabled servers
mcp_client.start_servers()

# Check server status
status = mcp_client.get_server_status("aws_docs")
print(f"AWS Docs server: {status}")

# Restart failed server
mcp_client.restart_server("web_search")
```

## Configuration Best Practices

### Security Best Practices

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Rotate credentials**: Regularly rotate API keys and tokens
3. **Principle of least privilege**: Grant minimal required permissions
4. **Audit configuration**: Log all configuration changes

```yaml
# Good: Use environment variables
secrets:
  openai_api_key: ${OPENAI_API_KEY}
  slack_token: ${SLACK_BOT_TOKEN}

# Bad: Hard-coded secrets
secrets:
  openai_api_key: "sk-1234567890abcdef"  # Never do this!
```

### Performance Best Practices

1. **Right-size concurrency**: Balance throughput with resource usage
2. **Optimize model selection**: Use appropriate semantic types
3. **Configure timeouts**: Prevent hanging operations
4. **Enable caching**: Cache frequently accessed data

```yaml
# Optimized for high throughput
universal_agent:
  max_concurrent_tasks: 10
  role_optimization:
    search: WEAK # Use fast models for simple searches
    planning: STRONG # Use powerful models for complex planning
```

### Monitoring Best Practices

1. **Enable comprehensive metrics**: Monitor all system components
2. **Set appropriate alerts**: Alert on critical issues only
3. **Configure retention**: Balance storage with historical data needs
4. **Use structured logging**: Enable JSON logging for better parsing

## Troubleshooting Configuration

### Common Configuration Issues

#### Invalid YAML Syntax

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

#### Missing Environment Variables

```bash
# Check required environment variables
python -c "
import os
required_vars = ['OPENAI_API_KEY', 'SLACK_BOT_TOKEN']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'Missing environment variables: {missing}')
else:
    print('All required environment variables are set')
"
```

#### Configuration Validation Errors

```python
from config.config_manager import ConfigManager

config_manager = ConfigManager("config.yaml")
if not config_manager.is_valid():
    for error in config_manager.get_validation_errors():
        print(f"Validation error: {error}")
```

### Configuration Testing

Test configuration before deployment:

```python
# Test configuration loading
def test_configuration():
    try:
        config_manager = ConfigManager("config.yaml")
        config = config_manager.get_config()

        # Test LLM factory creation
        from llm_provider.factory import LLMFactory
        llm_factory = LLMFactory(config['llm_providers'])

        # Test model creation
        model = llm_factory.create_strands_model(LLMType.DEFAULT)

        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    test_configuration()
```

## Migration from Legacy Configuration

### Automatic Migration

The system automatically migrates legacy agent configurations:

```yaml
# Legacy format (still supported)
agents:
  planning_agent:
    enabled: true
    model_type: strong
    temperature: 0.1

# Automatically converted to:
universal_agent:
  role_optimization:
    planning: STRONG
llm_providers:
  strong:
    temperature: 0.1
```

### Manual Migration Steps

1. **Backup existing configuration**:

   ```bash
   cp config.yaml config.yaml.backup
   ```

2. **Update configuration format**:

   ```bash
   python scripts/migrate_config.py config.yaml
   ```

3. **Validate new configuration**:

   ```bash
   python -m config.config_manager --validate config.yaml
   ```

4. **Test system startup**:
   ```bash
   python -m supervisor.supervisor --config config.yaml --test
   ```

## Configuration Schema

### Complete Schema Reference

```yaml
# Root configuration schema
logging: # Logging configuration (optional)
  log_level: str # Log level (default: INFO)
  log_file: str # Log file path (default: logs/system.log)
  log_format: str # Log format string
  max_file_size: str # Max file size before rotation
  backup_count: int # Number of backup files

llm_providers: # LLM provider configurations (required)
  <provider_name>: # Provider configuration name
    llm_class: str # WEAK, DEFAULT, STRONG (required)
    provider_type: str # bedrock, openai, anthropic (required)
    model_id: str # Model identifier (required)
    temperature: float # Temperature 0.0-1.0 (default: 0.3)
    max_tokens: int # Maximum tokens (default: 4096)
    timeout: int # Timeout in seconds (default: 30)
    additional_params: dict # Provider-specific parameters (optional)

universal_agent: # Universal Agent configuration (optional)
  framework: str # Framework type (default: strands)
  max_concurrent_tasks: int # Max parallel tasks (default: 5)
  checkpoint_interval: int # Checkpoint interval seconds (default: 300)
  retry_attempts: int # Max retry attempts (default: 3)
  retry_delay: float # Retry delay seconds (default: 1.0)
  role_optimization: dict # Role to LLM type mapping (optional)
  role_tools: dict # Role to tools mapping (optional)

mcp: # MCP configuration (optional)
  config_file: str # MCP config file path (default: config/mcp_config.yaml)
  enabled: bool # Enable MCP integration (default: true)
  timeout: int # Connection timeout (default: 30)
  retry_attempts: int # Connection retries (default: 3)

features: # Feature flags (optional)
  universal_agent_enabled: bool # Enable Universal Agent (default: true)
  mcp_integration_enabled: bool # Enable MCP integration (default: true)
  task_scheduling_enabled: bool # Enable task scheduling (default: true)
  checkpoint_persistence: bool # Enable checkpointing (default: true)
  metrics_collection: bool # Enable metrics (default: true)
  health_monitoring: bool # Enable health checks (default: true)

development: # Development settings (optional)
  debug_mode: bool # Enable debug mode (default: false)
  test_mode: bool # Enable test mode (default: false)
  mock_external_services: bool # Mock external services (default: false)
  verbose_logging: bool # Enable verbose logging (default: false)

agents: # Legacy agent configuration (optional, for backward compatibility)
  <agent_name>: # Legacy agent configuration
    enabled: bool # Enable agent (default: true)
    model_type: str # Model type (weak, default, strong)
```
