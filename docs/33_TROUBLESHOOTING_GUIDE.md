# Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered when using the StrandsAgent Universal Agent system.

## Common Issues and Solutions

### System Startup Issues

#### Issue: System fails to start with configuration errors

**Symptoms:**

- Configuration validation errors on startup
- Missing required configuration sections
- Invalid configuration values

**Solutions:**

1. **Validate configuration syntax:**

   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **Check required sections:**

   ```python
   from config.config_manager import ConfigManager

   config_manager = ConfigManager("config.yaml")
   if not config_manager.is_valid():
       for error in config_manager.get_validation_errors():
           print(f"Error: {error}")
   ```

3. **Verify environment variables:**
   ```bash
   # Check if required environment variables are set
   echo "OPENAI_API_KEY: ${OPENAI_API_KEY:-NOT_SET}"
   echo "SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN:-NOT_SET}"
   ```

#### Issue: LLM Factory creation fails

**Symptoms:**

- "No valid LLM configuration found" error
- Provider-specific authentication errors
- Model ID not found errors

**Solutions:**

1. **Check LLM provider configuration:**

   ```yaml
   llm_providers:
     default:
       llm_class: DEFAULT
       provider_type: bedrock
       model_id: us.amazon.nova-pro-v1:0
   ```

2. **Verify provider credentials:**

   ```bash
   # For Bedrock
   aws sts get-caller-identity

   # For OpenAI
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

3. **Test model availability:**

   ```python
   from llm_provider.factory import LLMFactory, LLMType

   factory = LLMFactory(configs)
   try:
       model = factory.create_strands_model(LLMType.DEFAULT)
       print("✅ Model creation successful")
   except Exception as e:
       print(f"❌ Model creation failed: {e}")
   ```

### WorkflowEngine Issues

#### Issue: Workflows fail to start

**Symptoms:**

- "Failed to create task plan" errors
- Universal Agent initialization failures
- Task delegation errors

**Solutions:**

1. **Check Universal Agent status:**

   ```python
   status = workflow_engine.get_universal_agent_status()
   print(f"Universal Agent enabled: {status['universal_agent_enabled']}")
   print(f"Framework: {status['framework']}")
   ```

2. **Verify role configuration:**

   ```yaml
   universal_agent:
     role_optimization:
       planning: STRONG
       search: WEAK
       # Ensure all required roles are configured
   ```

3. **Check task planning:**

   ```python
   # Test task planning directly
   from llm_provider.planning_tools import create_task_plan

   plan = create_task_plan("Test instruction", ["search", "summarizer"])
   print(f"Task plan: {plan}")
   ```

#### Issue: Tasks hang or timeout

**Symptoms:**

- Tasks remain in RUNNING state indefinitely
- Timeout errors in logs
- High memory usage

**Solutions:**

1. **Check task timeouts:**

   ```yaml
   universal_agent:
     task_timeout: 600 # Increase timeout for complex tasks
   ```

2. **Monitor resource usage:**

   ```python
   import psutil

   # Check memory usage
   memory = psutil.virtual_memory()
   print(f"Memory usage: {memory.percent}%")

   # Check CPU usage
   cpu = psutil.cpu_percent(interval=1)
   print(f"CPU usage: {cpu}%")
   ```

3. **Review task complexity:**
   ```python
   # Use appropriate model types for task complexity
   # WEAK for simple tasks, STRONG for complex reasoning
   agent = universal_agent.assume_role("search", llm_type=LLMType.WEAK)
   ```

### Universal Agent Issues

#### Issue: Role switching fails

**Symptoms:**

- "Unknown role" errors
- Role assumption failures
- Tool availability errors

**Solutions:**

1. **Verify supported roles:**

   ```python
   supported_roles = ["planning", "search", "weather", "summarizer", "slack"]
   print(f"Supported roles: {supported_roles}")
   ```

2. **Check role configuration:**

   ```yaml
   universal_agent:
     role_optimization:
       planning: STRONG
       search: WEAK
       weather: WEAK
       summarizer: DEFAULT
       slack: DEFAULT
   ```

3. **Test role assumption:**
   ```python
   try:
       agent = universal_agent.assume_role("planning")
       print("✅ Role assumption successful")
   except Exception as e:
       print(f"❌ Role assumption failed: {e}")
   ```

#### Issue: Tool execution failures

**Symptoms:**

- "Tool not found" errors
- Tool execution timeouts
- External service connection failures

**Solutions:**

1. **Check tool registration:**

   ```python
   from llm_provider.tool_registry import ToolRegistry

   registry = ToolRegistry()
   tools = registry.get_available_tools()
   print(f"Available tools: {tools}")
   ```

2. **Test tool execution:**

   ```python
   from llm_provider.search_tools import web_search

   try:
       result = web_search("test query")
       print(f"✅ Tool execution successful: {result}")
   except Exception as e:
       print(f"❌ Tool execution failed: {e}")
   ```

3. **Check external service connectivity:**

   ```bash
   # Test internet connectivity
   curl -I https://www.google.com

   # Test specific APIs
   curl -H "Authorization: Bearer $API_KEY" https://api.service.com/health
   ```

### MCP Integration Issues

#### Issue: MCP servers fail to start

**Symptoms:**

- "MCP server connection failed" errors
- "MCP dependencies not available" warnings
- Tool unavailability from MCP servers

**Solutions:**

1. **Check MCP configuration:**

   ```yaml
   mcp:
     config_file: config/mcp_config.yaml
     enabled: true
     timeout: 30
   ```

2. **Verify MCP server commands:**

   ```bash
   # Test MCP server command manually
   uvx awslabs.aws-documentation-mcp-server@latest
   ```

3. **Check MCP client status:**

   ```python
   from llm_provider.mcp_client import MCPClient

   client = MCPClient("config/mcp_config.yaml")
   status = client.get_all_server_status()
   for server, info in status.items():
       print(f"{server}: {info['status']}")
   ```

4. **Install missing MCP dependencies:**
   ```bash
   # Install required MCP tools
   pip install mcp
   npm install -g @modelcontextprotocol/server-web-search
   ```

### Performance Issues

#### Issue: Slow workflow execution

**Symptoms:**

- Long response times
- High latency between tasks
- Resource exhaustion

**Solutions:**

1. **Optimize concurrency settings:**

   ```yaml
   universal_agent:
     max_concurrent_tasks: 10 # Increase for better throughput
   ```

2. **Use appropriate model types:**

   ```yaml
   universal_agent:
     role_optimization:
       search: WEAK # Use fast models for simple tasks
       planning: STRONG # Use powerful models only when needed
   ```

3. **Enable performance monitoring:**

   ```yaml
   development:
     performance_profiling: true

   features:
     metrics_collection: true
   ```

4. **Check system resources:**

   ```python
   import psutil

   # Monitor system resources
   print(f"CPU: {psutil.cpu_percent()}%")
   print(f"Memory: {psutil.virtual_memory().percent}%")
   print(f"Disk: {psutil.disk_usage('/').percent}%")
   ```

#### Issue: High memory usage

**Symptoms:**

- Memory usage continuously increasing
- Out of memory errors
- System slowdown over time

**Solutions:**

1. **Configure conversation history limits:**

   ```yaml
   universal_agent:
     max_conversation_history: 500 # Limit conversation history
   ```

2. **Enable checkpoint compression:**

   ```yaml
   features:
     checkpoint_compression: true
   ```

3. **Monitor memory usage:**

   ```python
   # Add memory monitoring to workflows
   import psutil

   def monitor_memory():
       memory = psutil.virtual_memory()
       if memory.percent > 80:
           print(f"⚠️ High memory usage: {memory.percent}%")
   ```

### State Management Issues

#### Issue: Checkpoint creation/restoration fails

**Symptoms:**

- "Failed to create checkpoint" errors
- Workflow resume failures
- State corruption errors

**Solutions:**

1. **Check checkpoint permissions:**

   ```bash
   # Ensure checkpoint directory is writable
   mkdir -p checkpoints
   chmod 755 checkpoints
   ```

2. **Validate checkpoint data:**

   ```python
   import json

   # Test checkpoint serialization
   checkpoint = workflow_engine.pause_workflow(workflow_id)

   # Verify checkpoint can be serialized
   json_data = json.dumps(checkpoint)
   restored = json.loads(json_data)
   ```

3. **Enable checkpoint debugging:**
   ```yaml
   development:
     debug_mode: true
     verbose_logging: true
   ```

#### Issue: TaskGraph execution errors

**Symptoms:**

- Dependency resolution failures
- Circular dependency errors
- Task execution order issues

**Solutions:**

1. **Validate task dependencies:**

   ```python
   from common.task_graph import TaskGraph

   # Check for circular dependencies
   task_graph = TaskGraph()
   # Add tasks and dependencies

   if task_graph.has_cycles():
       print("❌ Circular dependency detected")
   else:
       print("✅ Task graph is valid")
   ```

2. **Debug task execution order:**
   ```python
   # Get execution order
   ready_tasks = task_graph.get_ready_tasks()
   print(f"Ready tasks: {[task.task_id for task in ready_tasks]}")
   ```

### Message Bus Issues

#### Issue: Event delivery failures

**Symptoms:**

- Events not received by subscribers
- Message bus connection errors
- Event processing delays

**Solutions:**

1. **Check message bus status:**

   ```python
   from common.message_bus import MessageBus

   message_bus = MessageBus()
   status = message_bus.get_status()
   print(f"Message bus status: {status}")
   ```

2. **Test event publishing:**

   ```python
   # Test event publishing and subscription
   def test_handler(event):
       print(f"Received event: {event}")

   message_bus.subscribe("TEST_EVENT", test_handler)
   message_bus.publish("TEST_EVENT", {"test": "data"})
   ```

3. **Monitor event queue:**
   ```python
   # Check event queue size
   queue_size = message_bus.get_queue_size()
   if queue_size > 100:
       print(f"⚠️ Large event queue: {queue_size}")
   ```

## Debugging Techniques

### Enable Debug Mode

```yaml
development:
  debug_mode: true
  verbose_logging: true

logging:
  log_level: DEBUG
```

### Trace Workflow Execution

```python
# Add execution tracing
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Trace workflow steps
workflow_id = workflow_engine.start_workflow("test instruction")
logger.debug(f"Started workflow: {workflow_id}")

# Monitor workflow progress
while True:
    status = workflow_engine.get_workflow_status(workflow_id)
    logger.debug(f"Workflow {workflow_id} status: {status}")

    if status['state'] in ['COMPLETED', 'FAILED']:
        break

    time.sleep(1)
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile workflow execution
profiler = cProfile.Profile()
profiler.enable()

# Execute workflow
workflow_id = workflow_engine.start_workflow("complex instruction")

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions by cumulative time
```

## Health Monitoring

### System Health Checks

```python
# Check overall system health
def check_system_health():
    health = {
        "workflow_engine": check_workflow_engine_health(),
        "universal_agent": check_universal_agent_health(),
        "mcp_servers": check_mcp_servers_health(),
        "message_bus": check_message_bus_health()
    }

    overall_status = "healthy"
    for component, status in health.items():
        if status != "healthy":
            overall_status = "degraded"
            print(f"⚠️ {component}: {status}")

    return overall_status

def check_workflow_engine_health():
    try:
        metrics = workflow_engine.get_workflow_metrics()
        if metrics['active_workflows'] > 100:
            return "overloaded"
        return "healthy"
    except Exception:
        return "unhealthy"
```

### Component-Specific Health Checks

```python
# WorkflowEngine health
def check_workflow_engine_health():
    try:
        # Test basic functionality
        test_id = workflow_engine._generate_workflow_id()
        metrics = workflow_engine.get_workflow_metrics()

        # Check for concerning metrics
        if metrics['failed_workflows'] > 10:
            return "degraded"

        return "healthy"
    except Exception as e:
        return f"unhealthy: {e}"

# Universal Agent health
def check_universal_agent_health():
    try:
        status = workflow_engine.get_universal_agent_status()
        if not status['universal_agent_enabled']:
            return "disabled"

        # Test role assumption
        agent = universal_agent.assume_role("search", llm_type=LLMType.WEAK)
        return "healthy"
    except Exception as e:
        return f"unhealthy: {e}"
```

## Error Codes and Messages

### WorkflowEngine Errors

| Error Code | Message                    | Solution                                  |
| ---------- | -------------------------- | ----------------------------------------- |
| WE001      | Workflow creation failed   | Check Universal Agent configuration       |
| WE002      | Task execution timeout     | Increase task timeout or optimize task    |
| WE003      | Checkpoint creation failed | Check filesystem permissions              |
| WE004      | Invalid workflow ID        | Verify workflow ID format                 |
| WE005      | Concurrency limit exceeded | Reduce concurrent tasks or increase limit |

### Universal Agent Errors

| Error Code | Message                 | Solution                                                          |
| ---------- | ----------------------- | ----------------------------------------------------------------- |
| UA001      | Unknown role requested  | Use supported role (planning, search, weather, summarizer, slack) |
| UA002      | Model creation failed   | Check LLM provider configuration                                  |
| UA003      | Tool execution failed   | Verify tool availability and parameters                           |
| UA004      | Role assumption timeout | Check model availability and network                              |
| UA005      | Invalid LLM type        | Use WEAK, DEFAULT, or STRONG                                      |

### MCP Integration Errors

| Error Code | Message                   | Solution                                     |
| ---------- | ------------------------- | -------------------------------------------- |
| MCP001     | Server connection failed  | Check MCP server command and arguments       |
| MCP002     | Tool not available        | Verify MCP server is running and tool exists |
| MCP003     | Authentication failed     | Check MCP server credentials                 |
| MCP004     | Server startup timeout    | Increase MCP timeout or check server health  |
| MCP005     | Protocol version mismatch | Update MCP server to compatible version      |

## Diagnostic Commands

### System Diagnostics

```bash
# Check system status
python -c "
from supervisor.supervisor import Supervisor
supervisor = Supervisor('config.yaml')
status = supervisor.status()
print('System Status:', status['status'])
print('Components:', list(status.keys()))
"

# Test workflow execution
python -c "
from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory
from common.message_bus import MessageBus

# Test basic workflow
message_bus = MessageBus()
llm_factory = LLMFactory(configs)
workflow_engine = WorkflowEngine(llm_factory, message_bus)

workflow_id = workflow_engine.start_workflow('Test workflow')
print(f'Test workflow started: {workflow_id}')
"
```

### Component Diagnostics

```bash
# Test Universal Agent
python -c "
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory

llm_factory = LLMFactory(configs)
universal_agent = UniversalAgent(llm_factory)

# Test each role
roles = ['planning', 'search', 'weather', 'summarizer', 'slack']
for role in roles:
    try:
        agent = universal_agent.assume_role(role)
        print(f'✅ {role}: OK')
    except Exception as e:
        print(f'❌ {role}: {e}')
"

# Test MCP servers
python -c "
from llm_provider.mcp_client import MCPClient

client = MCPClient('config/mcp_config.yaml')
servers = client.get_all_server_status()

for name, status in servers.items():
    print(f'{name}: {status[\"status\"]}')
"
```

## Log Analysis

### Important Log Patterns

**Successful Operations:**

```
INFO - WorkflowEngine - Workflow wf_12345 started successfully
INFO - UniversalAgent - Assumed role 'planning' with STRONG model
INFO - TaskContext - Checkpoint created for workflow wf_12345
```

**Warning Patterns:**

```
WARNING - MCPClient - Server 'aws_docs' connection timeout, retrying...
WARNING - WorkflowEngine - High queue size: 50 tasks pending
WARNING - UniversalAgent - Model response truncated due to token limit
```

**Error Patterns:**

```
ERROR - WorkflowEngine - Task execution failed: task_123
ERROR - UniversalAgent - Role assumption failed: unknown role 'invalid_role'
ERROR - MCPClient - Server 'web_search' startup failed after 3 attempts
```

### Log Analysis Scripts

```bash
# Find error patterns
grep -E "ERROR|CRITICAL" logs/system.log | tail -20

# Analyze workflow performance
grep "Workflow.*completed" logs/system.log | \
  awk '{print $NF}' | \
  sort -n | \
  tail -10

# Check MCP server issues
grep "MCP" logs/system.log | grep -E "ERROR|WARNING"
```

## Performance Optimization

### Identify Performance Bottlenecks

```python
import time
import logging

# Add performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.timings = {}

    def start_timer(self, operation):
        self.timings[operation] = time.time()

    def end_timer(self, operation):
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            logging.info(f"Operation {operation} took {duration:.2f}s")
            return duration
        return None

# Use in workflow execution
monitor = PerformanceMonitor()

monitor.start_timer("workflow_execution")
workflow_id = workflow_engine.start_workflow(instruction)
monitor.end_timer("workflow_execution")
```

### Optimization Strategies

1. **Model Selection Optimization:**

   ```yaml
   # Use faster models for simple tasks
   universal_agent:
     role_optimization:
       search: WEAK # Fast model for searches
       weather: WEAK # Fast model for weather
       planning: STRONG # Powerful model only for complex planning
   ```

2. **Concurrency Optimization:**

   ```yaml
   # Optimize based on system resources
   universal_agent:
     max_concurrent_tasks: 8 # Increase for better throughput
   ```

3. **Caching Configuration:**

   ```yaml
   features:
     caching_enabled: true

   caching:
     ttl: 3600 # Cache responses for 1 hour
     max_size: 1000 # Cache up to 1000 responses
   ```

## Recovery Procedures

### Workflow Recovery

```python
# Recover failed workflows
def recover_failed_workflows():
    metrics = workflow_engine.get_workflow_metrics()
    failed_workflows = metrics.get('failed_workflows', [])

    for workflow_id in failed_workflows:
        try:
            # Attempt to restart workflow
            status = workflow_engine.get_workflow_status(workflow_id)
            if status['state'] == 'FAILED':
                # Get last checkpoint
                checkpoint = workflow_engine.get_last_checkpoint(workflow_id)
                if checkpoint:
                    workflow_engine.resume_workflow(workflow_id, checkpoint)
                    print(f"✅ Recovered workflow {workflow_id}")
        except Exception as e:
            print(f"❌ Failed to recover workflow {workflow_id}: {e}")
```

### System Recovery

```python
# Complete system recovery procedure
def system_recovery():
    print("🔄 Starting system recovery...")

    # 1. Stop all active workflows
    active_workflows = workflow_engine.get_active_workflows()
    for workflow_id in active_workflows:
        try:
            checkpoint = workflow_engine.pause_workflow(workflow_id)
            print(f"✅ Paused workflow {workflow_id}")
        except Exception as e:
            print(f"❌ Failed to pause workflow {workflow_id}: {e}")

    # 2. Restart MCP servers
    mcp_client.restart_all_servers()

    # 3. Clear caches
    workflow_engine.clear_caches()

    # 4. Resume workflows
    for workflow_id, checkpoint in saved_checkpoints.items():
        try:
            workflow_engine.resume_workflow(workflow_id, checkpoint)
            print(f"✅ Resumed workflow {workflow_id}")
        except Exception as e:
            print(f"❌ Failed to resume workflow {workflow_id}: {e}")

    print("✅ System recovery completed")
```

## Support and Escalation

### Collecting Diagnostic Information

```bash
#!/bin/bash
# diagnostic_collection.sh

echo "=== System Information ==="
python --version
pip list | grep -E "(strands|pydantic)"

echo "=== Configuration Status ==="
python -c "
from config.config_manager import ConfigManager
cm = ConfigManager('config.yaml')
print('Valid:', cm.is_valid())
if not cm.is_valid():
    for error in cm.get_validation_errors():
        print('Error:', error)
"

echo "=== System Health ==="
python -c "
from supervisor.supervisor import Supervisor
supervisor = Supervisor('config.yaml')
status = supervisor.status()
print('Status:', status.get('status', 'unknown'))
"

echo "=== Recent Errors ==="
tail -50 logs/system.log | grep -E "ERROR|CRITICAL"

echo "=== Resource Usage ==="
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\".\").percent}%')
"
```

### Creating Support Tickets

When creating support tickets, include:

1. **System Information:**

   - Python version
   - StrandsAgent version
   - Operating system
   - Hardware specifications

2. **Configuration:**

   - Sanitized configuration file (remove secrets)
   - Environment variable list (names only)
   - Feature flags status

3. **Error Information:**

   - Complete error messages
   - Stack traces
   - Recent log entries
   - Steps to reproduce

4. **Diagnostic Output:**
   - System health check results
   - Component status information
   - Performance metrics
   - Resource usage data

### Emergency Procedures

#### Complete System Reset

```bash
# Emergency system reset procedure
echo "🚨 Emergency system reset initiated"

# 1. Stop all processes
pkill -f "supervisor"
pkill -f "workflow_engine"

# 2. Clear temporary data
rm -rf /tmp/strands_*
rm -rf checkpoints/*

# 3. Reset configuration to defaults
cp config/default.yaml config.yaml

# 4. Restart system
python -m supervisor.supervisor --config config.yaml

echo "✅ Emergency reset completed"
```

#### Data Recovery

```bash
# Recover from backup
echo "🔄 Starting data recovery"

# Restore configuration
cp config.yaml.backup config.yaml

# Restore checkpoints
cp -r checkpoints.backup/* checkpoints/

# Restore logs
cp logs.backup/* logs/

echo "✅ Data recovery completed"
```
