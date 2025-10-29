# Workflow Duration Logging

This document describes the workflow duration logging system that tracks and logs execution times for workflows completed in both CLI and Slack interfaces.

## Overview

The workflow duration logging system provides comprehensive tracking of workflow execution times, enabling performance monitoring, analytics, and optimization insights. It automatically logs workflow durations when workflows complete successfully in either the CLI or Slack interfaces.

## Features

- **Automatic Duration Tracking**: Seamlessly tracks workflow execution times without manual intervention
- **Multi-Interface Support**: Works with both CLI and Slack workflow completions
- **Comprehensive Metrics**: Captures workflow type, source, user information, and performance data
- **Structured Logging**: Uses JSONL format for easy parsing and analysis
- **Performance Analytics**: Provides summary statistics and performance insights
- **Configurable Storage**: Customizable log file location and rotation settings
- **Error Tracking**: Logs both successful and failed workflow executions

## Architecture

### Core Components

1. **WorkflowDurationLogger**: Main logging class that manages duration tracking
2. **WorkflowDurationMetrics**: Data structure for storing workflow metrics
3. **CLI Integration**: Automatic tracking in `cli.py` for command-line workflows
4. **Slack Integration**: Automatic tracking in `slack.py` for Slack bot workflows
5. **Configuration**: YAML-based configuration for logging settings

### Data Flow

```
Workflow Start → Duration Logger → Track Start Time
     ↓
Workflow Execution → Monitor Progress
     ↓
Workflow Complete → Duration Logger → Calculate Duration → Log to File
```

## Configuration

The workflow duration logging is configured in `config.yaml`:

```yaml
# Workflow Duration Logging Configuration
workflow_duration_logging:
  enabled: true
  log_file: "logs/workflow_durations.jsonl"
  max_log_file_size_mb: 100
  enable_console_logging: true

  # Performance monitoring settings
  performance_monitoring:
    track_cli_workflows: true
    track_slack_workflows: true
    track_api_workflows: true

  # Analytics and reporting
  analytics:
    enable_performance_summary: true
    summary_interval_hours: 24
    retention_days: 30

  # Log rotation settings
  rotation:
    enabled: true
    backup_count: 5
    compress_backups: true
```

## Usage

### Automatic Tracking

Duration logging is automatically enabled when workflows complete in:

- **CLI Interface**: Both single workflow execution and interactive mode
- **Slack Interface**: Bot mentions, direct messages, and slash commands

No manual intervention is required - the system automatically tracks workflow start and completion times.

### Manual Integration

For custom integrations, use the duration logger directly:

```python
from supervisor.workflow_duration_logger import get_duration_logger, WorkflowSource, WorkflowType

# Get the global logger instance
duration_logger = get_duration_logger()

# Start tracking a workflow
metrics = duration_logger.start_workflow_tracking(
    workflow_id="custom_workflow_123",
    source=WorkflowSource.API,
    workflow_type=WorkflowType.COMPLEX_WORKFLOW,
    instruction="Custom workflow instruction",
    user_id="user123"  # Optional
)

# Complete tracking when workflow finishes
completed_metrics = duration_logger.complete_workflow_tracking(
    workflow_id="custom_workflow_123",
    success=True,
    role="weather",  # Optional
    confidence=0.95,  # Optional
    task_count=3  # Optional
)
```

## Log Format

Workflow duration logs are stored in JSONL format with the following structure:

```json
{
  "workflow_id": "wf_abc123",
  "source": "CLI",
  "workflow_type": "COMPLEX_WORKFLOW",
  "start_time": 1696123456.789,
  "end_time": 1696123459.234,
  "duration_seconds": 2.445,
  "instruction": "Get weather for Seattle",
  "role": "weather",
  "confidence": 0.92,
  "task_count": 2,
  "success": true,
  "error_message": null,
  "user_id": null,
  "channel_id": null,
  "timestamp": "2023-10-01T12:34:56.789000"
}
```

### Field Descriptions

- **workflow_id**: Unique identifier for the workflow
- **source**: Source interface (CLI, SLACK, API)
- **workflow_type**: Type of workflow (FAST_REPLY, COMPLEX_WORKFLOW, UNKNOWN)
- **start_time**: Unix timestamp when workflow started
- **end_time**: Unix timestamp when workflow completed
- **duration_seconds**: Total execution time in seconds
- **instruction**: Original user instruction/prompt
- **role**: Role used for execution (for fast-reply workflows)
- **confidence**: Routing confidence score (0.0-1.0)
- **task_count**: Number of tasks in the workflow
- **success**: Whether the workflow completed successfully
- **error_message**: Error message if workflow failed
- **user_id**: User identifier (for Slack workflows)
- **channel_id**: Channel identifier (for Slack workflows)
- **timestamp**: ISO 8601 formatted timestamp

## Analytics and Reporting

### Performance Summary

Get performance statistics for a time period:

```python
from supervisor.workflow_duration_logger import get_duration_logger

logger = get_duration_logger()

# Get summary for last 24 hours
summary = logger.get_performance_summary(hours=24)

print(f"Total workflows: {summary['total_workflows']}")
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average duration: {summary['average_duration_seconds']:.2f}s")
print(f"Workflows by source: {summary['workflows_by_source']}")
```

### Recent Metrics

Retrieve recent workflow metrics:

```python
# Get last 100 workflow entries
recent_metrics = logger.get_recent_metrics(limit=100)

for metric in recent_metrics:
    print(f"Workflow {metric['workflow_id']}: {metric['duration_seconds']:.2f}s")
```

## Workflow Types

The system automatically categorizes workflows into types:

### FAST_REPLY

- Single-step workflows that use fast-path routing
- Typically complete in under 3 seconds
- Use WEAK model for quick responses
- Identified by workflow ID prefix `fr_`

### COMPLEX_WORKFLOW

- Multi-step workflows that require planning
- May involve multiple tasks and dependencies
- Use DEFAULT or STRONG models
- Identified by workflow ID prefix `wf_`

### UNKNOWN

- Workflows where type cannot be determined
- Fallback category for edge cases

## Integration Examples

### CLI Workflow Tracking

The CLI interface automatically tracks workflows in both single execution and interactive modes:

```bash
# Single workflow - automatically tracked
python cli.py --workflow "Get weather for Seattle"

# Interactive mode - each workflow automatically tracked
python cli.py
➤ What's the weather in New York?
```

### Slack Workflow Tracking

Slack workflows are automatically tracked for:

```
# Bot mentions
@mybot What's the weather like today?

# Direct messages
What's the weather in London?

# Slash commands
/ai Tell me about the weather
```

## Performance Monitoring

### Key Metrics

Monitor these key performance indicators:

1. **Average Duration**: Overall workflow execution time
2. **Success Rate**: Percentage of workflows that complete successfully
3. **Throughput**: Number of workflows per hour/day
4. **Error Rate**: Percentage of failed workflows
5. **Source Distribution**: Breakdown by CLI vs Slack usage
6. **Type Distribution**: Fast-reply vs complex workflow ratio

### Optimization Insights

Use duration logs to identify:

- **Slow Workflows**: Workflows taking longer than expected
- **Error Patterns**: Common failure modes and their causes
- **Usage Patterns**: Peak usage times and interface preferences
- **Performance Trends**: Changes in execution times over time

## API Reference

### WorkflowDurationLogger

Main class for duration tracking:

```python
class WorkflowDurationLogger:
    def __init__(self, log_file_path: str, enable_console_logging: bool = True, max_log_file_size_mb: int = 100)
    def start_workflow_tracking(self, workflow_id: str, source: WorkflowSource, ...) -> WorkflowDurationMetrics
    def complete_workflow_tracking(self, workflow_id: str, success: bool = True, ...) -> Optional[WorkflowDurationMetrics]
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]
```

### Global Functions

```python
def get_duration_logger() -> WorkflowDurationLogger
def initialize_duration_logger(log_file_path: str, ...) -> WorkflowDurationLogger
```

### Enums

```python
class WorkflowSource(str, Enum):
    CLI = "CLI"
    SLACK = "SLACK"
    API = "API"
    UNKNOWN = "UNKNOWN"

class WorkflowType(str, Enum):
    FAST_REPLY = "FAST_REPLY"
    COMPLEX_WORKFLOW = "COMPLEX_WORKFLOW"
    UNKNOWN = "UNKNOWN"
```

## Testing

Run the comprehensive test suite:

```bash
# Run workflow duration logger tests
python -m pytest tests/supervisor/test_workflow_duration_logger.py -v

# Quick functionality test
python -c "
from supervisor.workflow_duration_logger import WorkflowDurationLogger, WorkflowSource, WorkflowType
import tempfile, os
temp_dir = tempfile.mkdtemp()
log_file = os.path.join(temp_dir, 'test.jsonl')
logger = WorkflowDurationLogger(log_file, enable_console_logging=False)
metrics = logger.start_workflow_tracking('test_123', WorkflowSource.CLI, WorkflowType.FAST_REPLY, 'test instruction')
completed = logger.complete_workflow_tracking('test_123', success=True, role='weather')
print('✅ Workflow duration logger working correctly!')
os.remove(log_file); os.rmdir(temp_dir)
"
```

## Troubleshooting

### Common Issues

1. **Log File Not Created**

   - Check directory permissions for log file path
   - Ensure parent directories exist
   - Verify configuration is correct

2. **Missing Duration Entries**

   - Check that workflows are completing successfully
   - Verify logger is initialized before workflow execution
   - Check for exceptions in workflow completion

3. **Performance Impact**
   - Duration logging has minimal overhead (< 1ms per workflow)
   - Disable console logging in production for better performance
   - Use log rotation to prevent large file sizes

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("supervisor.workflow_duration_logger").setLevel(logging.DEBUG)
```

## Best Practices

1. **Configuration**

   - Set appropriate log file size limits
   - Enable log rotation for production
   - Configure retention policies

2. **Monitoring**

   - Regularly review performance summaries
   - Set up alerts for high error rates
   - Monitor average duration trends

3. **Analysis**

   - Use external tools for advanced analytics
   - Export data for visualization
   - Correlate with system metrics

4. **Maintenance**
   - Clean up old log files regularly
   - Monitor disk space usage
   - Archive historical data as needed

## Related Documentation

- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
