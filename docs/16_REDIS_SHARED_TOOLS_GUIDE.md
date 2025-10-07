# Redis Shared Tools Guide

This guide provides comprehensive documentation for the Redis shared tools that enable roles to utilize Redis for caching, data storage, and inter-role communication.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Key Management](#key-management)
- [Best Practices](#best-practices)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

The Redis shared tools provide a comprehensive interface for roles to interact with Redis, offering both synchronous and asynchronous operations with automatic key prefixing to prevent collisions between different roles/agents.

### Key Benefits
- **Automatic Key Prefixing**: Prevents key collisions between different roles
- **TTL Support**: Optional time-to-live for all write operations
- **Sync/Async Operations**: Both blocking and fire-and-forget operations
- **Configuration Flexibility**: Support for both config.yaml and environment variables
- **Comprehensive Error Handling**: Graceful degradation when Redis is unavailable
- **Full Test Coverage**: Extensive unit tests with mocking support

## Features

### Core Operations
- **Synchronous Operations**: `redis_write()`, `redis_read()`, `redis_get_keys()`
- **Asynchronous Operations**: `redis_write_async()`, `redis_read_async()`
- **Utility Functions**: `redis_delete()`, `redis_exists()`, `redis_health_check()`
- **Role Management**: `redis_clear_role_data()`

### Advanced Features
- **Automatic Role Detection**: Keys are automatically prefixed with the calling role name
- **JSON Serialization**: Automatic serialization/deserialization of complex data types
- **TTL Management**: Optional time-to-live for cached data
- **Connection Pooling**: Efficient Redis connection management
- **Health Monitoring**: Built-in health check functionality

## Configuration

### Config.yaml Configuration

Add Redis configuration to your `config.yaml`:

```yaml
# Redis Configuration
redis:
  enabled: true
  connection:
    host: "localhost"
    port: 6379
    db: 0
    # NOTE: Password must be set via REDIS_PASSWORD environment variable for security
    socket_connect_timeout: 5
    socket_timeout: 5
    retry_on_timeout: true
    decode_responses: true
  
  # Connection pooling
  pool:
    max_connections: 10
    retry_on_timeout: true
  
  # Key management
  key_management:
    auto_prefix: true                 # Automatically prefix keys with role name
    default_ttl: 3600                # Default TTL in seconds (1 hour)
    max_ttl: 86400                   # Maximum allowed TTL (24 hours)
  
  # Environment variable overrides (these take precedence)
  # REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
  use_env_overrides: true
```

### Environment Variables

Environment variables take precedence over config.yaml settings. **For security reasons, Redis password must always be set via environment variable:**

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=your_secure_password  # REQUIRED for authenticated Redis
```

**Security Note:** Never store Redis passwords in configuration files. Always use the `REDIS_PASSWORD` environment variable.

### Dependencies

Add to your `requirements.txt`:

```
redis>=5.0.0
aioredis>=2.0.0
PyYAML>=6.0.0
```

## API Reference

### Synchronous Operations

#### `redis_write(key: str, value: Union[str, int, float, dict, list], ttl: Optional[int] = None) -> Dict[str, Any]`

Write data to Redis synchronously.

**Parameters:**
- `key`: Redis key (will be prefixed with role name)
- `value`: Value to store (strings, numbers, dicts, and lists supported)
- `ttl`: Time to live in seconds (optional)

**Returns:**
- Dict containing operation result with `success`, `key`, `prefixed_key`, `ttl`, and `message` fields

**Example:**
```python
result = redis_write("user_session", {"user_id": 123, "login_time": "2024-01-01"}, ttl=3600)
if result["success"]:
    print(f"Data written to {result['prefixed_key']}")
```

#### `redis_read(key: str) -> Dict[str, Any]`

Read data from Redis synchronously.

**Parameters:**
- `key`: Redis key (will be prefixed with role name)

**Returns:**
- Dict containing `success`, `key`, `prefixed_key`, `value`, `ttl`, and `message` fields

**Example:**
```python
result = redis_read("user_session")
if result["success"]:
    user_data = result["value"]
    print(f"Retrieved: {user_data}")
```

#### `redis_get_keys(pattern: str = "*") -> Dict[str, Any]`

Get all keys for the current role matching a pattern.

**Parameters:**
- `pattern`: Key pattern to match (applied after role prefix)

**Returns:**
- Dict containing `success`, `pattern`, `role`, `keys`, `count`, and `message` fields

**Example:**
```python
keys_result = redis_get_keys("session:*")
for key in keys_result["keys"]:
    data = redis_read(key)
```

### Asynchronous Operations

#### `redis_write_async(key: str, value: Union[str, int, float, dict, list], ttl: Optional[int] = None) -> Dict[str, Any]`

Write data to Redis asynchronously (fire and forget).

**Parameters:**
- Same as `redis_write()`

**Returns:**
- Dict indicating the async operation was started

**Example:**
```python
result = redis_write_async("log_entry", {"timestamp": "2024-01-01", "event": "user_login"})
print("Async write started")  # Returns immediately
```

#### `redis_read_async(key: str, callback: Optional[callable] = None) -> Dict[str, Any]`

Read data from Redis asynchronously with optional callback.

**Parameters:**
- `key`: Redis key (will be prefixed with role name)
- `callback`: Optional callback function to handle the result

**Returns:**
- Dict indicating the async operation was started

**Example:**
```python
def handle_result(result):
    if result["success"]:
        print(f"Got data: {result['value']}")

redis_read_async("user_session", callback=handle_result)
```

### Utility Functions

#### `redis_delete(key: str) -> Dict[str, Any]`

Delete a key from Redis.

#### `redis_exists(key: str) -> Dict[str, Any]`

Check if a key exists in Redis.

#### `redis_health_check() -> Dict[str, Any]`

Check Redis connection health and get server information.

#### `redis_clear_role_data(role: Optional[str] = None) -> Dict[str, Any]`

Clear all data for a specific role (use with caution).

## Usage Examples

### Basic Usage in Role Lifecycle

```python
# roles/my_role/lifecycle.py
from roles.shared_tools.redis_tools import redis_write, redis_read

async def cache_user_data(instruction: str, context: TaskContext, parameters: Dict) -> Dict[str, Any]:
    user_id = parameters.get("user_id")
    
    # Try to read from cache first
    cache_result = redis_read(f"user:{user_id}")
    if cache_result["success"]:
        return {"user_data": cache_result["value"]}
    
    # Fetch from API if not cached
    user_data = await fetch_user_from_api(user_id)
    
    # Cache for 1 hour
    redis_write(f"user:{user_id}", user_data, ttl=3600)
    
    return {"user_data": user_data}
```

### Session Management

```python
# Store user session
session_data = {
    "user_id": 123,
    "login_time": "2024-01-01T10:00:00Z",
    "permissions": ["read", "write"]
}
redis_write("session:abc123", session_data, ttl=1800)  # 30 minutes

# Retrieve session
session = redis_read("session:abc123")
if session["success"]:
    user_permissions = session["value"]["permissions"]
```

### Async Logging

```python
# Fire-and-forget logging
log_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "level": "INFO",
    "message": "User performed action",
    "user_id": 123
}
redis_write_async("logs:info", log_entry, ttl=86400)  # 24 hours
```

### Cross-Role Communication

```python
# Role A writes data
redis_write("shared:config", {"feature_enabled": True}, ttl=3600)

# Role B reads data (different role, different prefix)
# This would look for "role_b:shared:config", not "role_a:shared:config"
config = redis_read("shared:config")
```

## Key Management

### Automatic Prefixing

All keys are automatically prefixed with the calling role name to prevent collisions:

- Role "weather" writing key "forecast" → Redis key "weather:forecast"
- Role "timer" writing key "forecast" → Redis key "timer:forecast"

### Key Patterns

Use patterns with `redis_get_keys()` to find related keys:

```python
# Get all session keys
sessions = redis_get_keys("session:*")

# Get all cache keys
cache_keys = redis_get_keys("cache:*")

# Get all keys (for current role only)
all_keys = redis_get_keys("*")
```

### TTL Management

- Default TTL can be configured in config.yaml
- TTL is optional for all write operations
- Use `redis_read()` to check remaining TTL
- TTL of -1 means no expiration

## Best Practices

### 1. Use Appropriate TTLs

```python
# Short-lived session data
redis_write("session:token", token_data, ttl=1800)  # 30 minutes

# Medium-term cache
redis_write("cache:user_profile", profile, ttl=3600)  # 1 hour

# Long-term configuration
redis_write("config:settings", settings, ttl=86400)  # 24 hours
```

### 2. Handle Errors Gracefully

```python
result = redis_read("important_data")
if not result["success"]:
    logger.warning(f"Redis read failed: {result.get('error', 'Unknown error')}")
    # Fallback to database or default values
    data = fetch_from_database()
else:
    data = result["value"]
```

### 3. Use Async Operations for Non-Critical Data

```python
# Critical data - use sync
user_session = redis_read("session:current")

# Logging/metrics - use async
redis_write_async("metrics:page_view", view_data)
```

### 4. Organize Keys with Namespaces

```python
# Good: organized with namespaces
redis_write("user:profile:123", profile_data)
redis_write("user:preferences:123", prefs_data)
redis_write("cache:api:weather", weather_data)

# Avoid: flat key structure
redis_write("user_profile_123", profile_data)
```

### 5. Monitor Redis Health

```python
health = redis_health_check()
if not health["success"]:
    logger.error("Redis is unavailable, using fallback storage")
    use_fallback_storage()
```

## Testing

### Unit Testing with Mocks

```python
from unittest.mock import patch, Mock
from roles.shared_tools.redis_tools import redis_write, redis_read

def test_redis_operations():
    with patch('roles.shared_tools.redis_tools._get_redis_client') as mock_client:
        mock_redis = Mock()
        mock_client.return_value = mock_redis
        
        # Test write
        mock_redis.set.return_value = True
        result = redis_write("test_key", "test_value")
        assert result["success"] is True
        
        # Test read
        mock_redis.get.return_value = "test_value"
        mock_redis.ttl.return_value = 3600
        result = redis_read("test_key")
        assert result["success"] is True
        assert result["value"] == "test_value"
```

### Integration Testing

```python
def test_redis_integration():
    # Skip if Redis not available
    health = redis_health_check()
    if not health["success"]:
        pytest.skip("Redis not available")
    
    # Test full workflow
    test_data = {"test": True}
    
    # Write
    write_result = redis_write("integration_test", test_data, ttl=60)
    assert write_result["success"]
    
    # Read
    read_result = redis_read("integration_test")
    assert read_result["success"]
    assert read_result["value"] == test_data
    
    # Cleanup
    redis_delete("integration_test")
```

### Running Tests

```bash
# Run Redis tools tests
python -m pytest tests/unit/test_redis_tools.py -v

# Run with timeout to prevent hangs
python -m pytest tests/unit/test_redis_tools.py -v --timeout=30
```

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed

**Error:** `Redis connection failed: Connection refused`

**Solutions:**
- Ensure Redis server is running: `redis-server`
- Check host/port configuration
- Verify network connectivity
- Check firewall settings

#### 2. Import Error

**Error:** `ImportError: Redis not available`

**Solutions:**
- Install Redis dependencies: `pip install redis>=5.0.0 aioredis>=2.0.0`
- Verify installation: `python -c "import redis; print('Redis available')"`

#### 3. Key Not Found

**Error:** `Key not found` in read operations

**Debugging:**
- Check if key exists: `redis_exists("your_key")`
- List all keys for role: `redis_get_keys("*")`
- Verify TTL hasn't expired
- Check role name prefixing

#### 4. Async Operations Not Working

**Issues:**
- Async writes appear to succeed but data not stored
- Callbacks not called for async reads

**Solutions:**
- Check Redis server logs for errors
- Verify aioredis installation
- Use sync operations for critical data

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.getLogger('roles.shared_tools.redis_tools').setLevel(logging.DEBUG)
```

#### Check Redis Server Status

```python
health = redis_health_check()
print(f"Redis connected: {health['connected']}")
print(f"Redis version: {health.get('redis_version', 'Unknown')}")
```

#### Monitor Key Usage

```python
# Get all keys for current role
all_keys = redis_get_keys("*")
print(f"Total keys for role: {all_keys['count']}")
for key in all_keys["keys"]:
    print(f"  {key}")
```

#### Test Configuration Loading

```python
from roles.shared_tools.redis_tools import _get_redis_config
config = _get_redis_config()
print(f"Redis config: {config}")
```

## Performance Considerations

### Connection Management
- Redis connections are pooled and reused
- Connections are created lazily on first use
- Failed connections are retried automatically

### Memory Usage
- Use appropriate TTLs to prevent memory bloat
- Monitor Redis memory usage: `redis_health_check()`
- Consider data compression for large values

### Network Latency
- Use async operations for non-critical data
- Batch operations when possible
- Consider Redis pipelining for multiple operations

## Security Considerations

### Authentication
- Use Redis AUTH if password is configured
- Store passwords in environment variables, not config files

### Network Security
- Use Redis over secure networks only
- Consider Redis over TLS for production
- Restrict Redis access with firewall rules

### Data Privacy
- Be mindful of sensitive data in Redis
- Use appropriate TTLs for sensitive information
- Consider encryption for highly sensitive data

## Migration and Deployment

### Development to Production
1. Update Redis configuration in config.yaml
2. Set production environment variables
3. Test Redis connectivity in production environment
4. Monitor Redis performance and memory usage

### Scaling Considerations
- Consider Redis Cluster for high availability
- Monitor connection pool usage
- Plan for Redis memory requirements

---

For more information, see:
- [Redis Documentation](https://redis.io/documentation)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [Async Redis Client](https://aioredis.readthedocs.io/)