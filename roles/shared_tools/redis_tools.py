"""Redis tools for generative agent roles.

Provides Redis operations with automatic key prefixing by role/agent to avoid collisions.
Supports both synchronous and asynchronous operations with TTL support.
"""

import asyncio
import inspect
import logging
import os
import threading
from typing import Any, Optional, Union

try:
    # Handle Python 3.12 compatibility issue with aioredis TimeoutError
    import sys

    import redis

    if sys.version_info >= (3, 12):
        # For Python 3.12+, we need to handle the TimeoutError conflict
        try:
            import aioredis
        except TypeError as e:
            if "duplicate base class TimeoutError" in str(e):
                # Skip aioredis for now in Python 3.12 due to compatibility issue
                aioredis = None
            else:
                raise
    else:
        import aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)

# Global Redis connections
_redis_client = None
_async_redis_client = None
_redis_lock = threading.Lock()


def _get_calling_role() -> str:
    """Automatically detect the calling role/agent from the call stack.

    Returns:
        Role name extracted from the call stack or 'unknown' if not detectable
    """
    frame = inspect.currentframe()
    try:
        # Walk up the call stack to find role context
        while frame:
            frame = frame.f_back
            if not frame:
                break

            # Try different detection strategies
            role = _detect_role_from_filename(frame.f_code.co_filename)
            if role:
                return role

            role = _detect_role_from_local_vars(frame.f_locals)
            if role:
                return role

    finally:
        del frame

    return "unknown"


def _detect_role_from_filename(filename: str) -> str:
    """Detect role from filename path."""
    if "/roles/" not in filename:
        return ""

    parts = filename.split("/roles/")
    if len(parts) <= 1:
        return ""

    role_part = parts[1].split("/")[0]
    if role_part and role_part != "shared_tools":
        return role_part

    return ""


def _detect_role_from_local_vars(local_vars: dict) -> str:
    """Detect role from local variables in frame."""
    for var_name in ["role_name", "role", "agent_name", "agent"]:
        if var_name not in local_vars:
            continue

        value = local_vars[var_name]
        if isinstance(value, str) and value:
            return value
        elif hasattr(value, "name") and isinstance(value.name, str):
            return value.name

    return ""


def _get_redis_config() -> dict[str, Any]:
    """Get Redis configuration from config.yaml with environment variable overrides."""
    # Default configuration
    default_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "retry_on_timeout": True,
    }

    # Try to load from config.yaml
    try:
        import yaml

        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
                redis_config = config.get("redis", {})
                if redis_config.get("enabled", True):
                    connection_config = redis_config.get("connection", {})
                    default_config.update(connection_config)
    except Exception as e:
        logger.warning(f"Failed to load Redis config from config.yaml: {e}")

    # Environment variable overrides (always take precedence)
    env_overrides = {
        "host": os.getenv("REDIS_HOST"),
        "port": os.getenv("REDIS_PORT"),
        "db": os.getenv("REDIS_DB"),
        "password": os.getenv("REDIS_PASSWORD"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            if key in ["port", "db"]:
                default_config[key] = int(value)
            else:
                default_config[key] = value

    return default_config


def _get_redis_client():
    """Get or create synchronous Redis client."""
    global _redis_client

    if not REDIS_AVAILABLE:
        raise ImportError("Redis not available. Install with: pip install redis>=5.0.0")

    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                config = _get_redis_config()
                _redis_client = redis.Redis(**config)

                # Test connection
                try:
                    _redis_client.ping()
                    logger.info("Redis connection established")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}")
                    _redis_client = None
                    raise

    return _redis_client


async def _get_async_redis_client():
    """Get or create asynchronous Redis client."""
    global _async_redis_client

    if not REDIS_AVAILABLE or aioredis is None:
        raise ImportError(
            "Redis not available or aioredis has compatibility issues. Install with: pip install aioredis>=2.0.0"
        )

    if _async_redis_client is None:
        config = _get_redis_config()
        # Remove sync-specific options
        config.pop("socket_connect_timeout", None)
        config.pop("socket_timeout", None)
        config.pop("retry_on_timeout", None)

        _async_redis_client = aioredis.from_url(
            f"redis://{config['host']}:{config['port']}/{config['db']}",
            password=config.get("password"),
            decode_responses=config["decode_responses"],
        )

        # Test connection
        try:
            await _async_redis_client.ping()
            logger.info("Async Redis connection established")
        except Exception as e:
            logger.warning(f"Async Redis connection failed: {e}")
            _async_redis_client = None
            raise

    return _async_redis_client


def _prefix_key(key: str, role: Optional[str] = None) -> str:
    """Add role prefix to Redis key to avoid collisions.

    Args:
        key: Original key
        role: Role name (auto-detected if not provided)

    Returns:
        Prefixed key in format "role:key"
    """
    if role is None:
        role = _get_calling_role()
    return f"{role}:{key}"


def _unprefix_key(prefixed_key: str, role: Optional[str] = None) -> str:
    """Remove role prefix from Redis key.

    Args:
        prefixed_key: Key with role prefix
        role: Role name (auto-detected if not provided)

    Returns:
        Original key without prefix
    """
    if role is None:
        role = _get_calling_role()
    prefix = f"{role}:"
    if prefixed_key.startswith(prefix):
        return prefixed_key[len(prefix) :]
    return prefixed_key


# Synchronous Redis Operations


def redis_write(
    key: str, value: Union[str, int, float, dict, list], ttl: Optional[int] = None
) -> dict[str, Any]:
    """Synchronously write data to Redis with optional TTL.

    Args:
        key: Redis key (will be prefixed with role name)
        value: Value to store (strings, numbers, dicts, and lists supported)
        ttl: Time to live in seconds (optional)

    Returns:
        Dict containing operation result

    Example:
        redis_write("user_session", {"user_id": 123, "login_time": "2024-01-01"}, ttl=3600)
    """
    try:
        client = _get_redis_client()
        prefixed_key = _prefix_key(key)

        # Handle different value types
        if isinstance(value, (dict, list)):
            import json

            serialized_value = json.dumps(value)
        else:
            serialized_value = str(value)

        # Set value with optional TTL
        if ttl:
            result = client.setex(prefixed_key, ttl, serialized_value)
        else:
            result = client.set(prefixed_key, serialized_value)

        logger.debug(f"Redis write: {prefixed_key} = {serialized_value[:100]}...")

        return {
            "success": bool(result),
            "key": key,
            "prefixed_key": prefixed_key,
            "ttl": ttl,
            "message": "Data written successfully",
        }

    except Exception as e:
        logger.error(f"Redis write error: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "message": "Failed to write data to Redis",
        }


def redis_read(key: str) -> dict[str, Any]:
    """Synchronously read data from Redis.

    Args:
        key: Redis key (will be prefixed with role name)

    Returns:
        Dict containing the value and metadata

    Example:
        result = redis_read("user_session")
        if result["success"]:
            user_data = result["value"]
    """
    try:
        client = _get_redis_client()
        prefixed_key = _prefix_key(key)

        value = client.get(prefixed_key)

        if value is None:
            return {
                "success": False,
                "key": key,
                "prefixed_key": prefixed_key,
                "value": None,
                "message": "Key not found",
            }

        # Try to deserialize JSON
        try:
            import json

            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value

        # Get TTL if available
        ttl = client.ttl(prefixed_key)
        ttl = ttl if ttl > 0 else None

        logger.debug(f"Redis read: {prefixed_key} = {str(parsed_value)[:100]}...")

        return {
            "success": True,
            "key": key,
            "prefixed_key": prefixed_key,
            "value": parsed_value,
            "ttl": ttl,
            "message": "Data read successfully",
        }

    except Exception as e:
        logger.error(f"Redis read error: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "message": "Failed to read data from Redis",
        }


def redis_get_keys(pattern: str = "*") -> dict[str, Any]:
    """Get all keys for the current role matching a pattern.

    Args:
        pattern: Key pattern to match (applied after role prefix)

    Returns:
        Dict containing list of keys (without role prefix)

    Example:
        keys = redis_get_keys("session:*")
        for _key in keys["keys"]:
            data = redis_read(key)
    """
    try:
        client = _get_redis_client()
        role = _get_calling_role()

        # Create pattern with role prefix
        prefixed_pattern = _prefix_key(pattern, role)

        # Get matching keys
        prefixed_keys = client.keys(prefixed_pattern)

        # Remove role prefix from results
        keys = [_unprefix_key(k, role) for k in prefixed_keys]

        logger.debug(
            f"Redis get_keys: found {len(keys)} keys for pattern {prefixed_pattern}"
        )

        return {
            "success": True,
            "pattern": pattern,
            "role": role,
            "keys": keys,
            "count": len(keys),
            "message": f"Found {len(keys)} keys",
        }

    except Exception as e:
        logger.error(f"Redis get_keys error: {e}")
        return {
            "success": False,
            "pattern": pattern,
            "error": str(e),
            "keys": [],
            "count": 0,
            "message": "Failed to get keys from Redis",
        }


# Asynchronous Redis Operations (Fire and Forget)


def redis_write_async(
    key: str, value: Union[str, int, float, dict, list], ttl: Optional[int] = None
) -> dict[str, Any]:
    """Asynchronously write data to Redis (fire and forget).

    Returns immediately without waiting for completion.

    Args:
        key: Redis key (will be prefixed with role name)
        value: Value to store
        ttl: Time to live in seconds (optional)

    Returns:
        Dict indicating the async operation was started

    Example:
        redis_write_async("log_entry", {"timestamp": "2024-01-01", "event": "user_login"})
    """
    try:
        _start_async_write_operation(key, value, ttl)
        return _create_write_success_response(key, ttl)
    except Exception as e:
        logger.error(f"Failed to start async Redis write: {e}")
        return _create_write_error_response(key, str(e))


def _start_async_write_operation(
    key: str, value: Union[str, int, float, dict, list], ttl: Optional[int]
):
    """Start the async write operation."""

    async def _do_write():
        try:
            await _perform_redis_write(key, value, ttl)
            logger.debug(f"Async Redis write completed: {_prefix_key(key)}")
        except Exception as e:
            logger.error(f"Async Redis write error: {e}")

    _execute_async_operation(_do_write)


async def _perform_redis_write(
    key: str, value: Union[str, int, float, dict, list], ttl: Optional[int]
):
    """Perform the actual Redis write operation."""
    client = await _get_async_redis_client()
    prefixed_key = _prefix_key(key)
    serialized_value = _serialize_value(value)

    if ttl:
        await client.setex(prefixed_key, ttl, serialized_value)
    else:
        await client.set(prefixed_key, serialized_value)


def _serialize_value(value: Union[str, int, float, dict, list]) -> str:
    """Serialize value for Redis storage."""
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value)
    else:
        return str(value)


def _create_write_success_response(key: str, ttl: Optional[int]) -> dict:
    """Create success response for write operation."""
    return {
        "success": True,
        "key": key,
        "prefixed_key": _prefix_key(key),
        "ttl": ttl,
        "async": True,
        "message": "Async write operation started",
    }


def _create_write_error_response(key: str, error: str) -> dict:
    """Create error response for write operation."""
    return {
        "success": False,
        "key": key,
        "error": error,
        "async": True,
        "message": "Failed to start async write operation",
    }


def redis_read_async(key: str, callback: Optional[callable] = None) -> dict[str, Any]:
    """Asynchronously read data from Redis (fire and forget).

    Returns immediately. Use callback to handle the result.

    Args:
        key: Redis key (will be prefixed with role name)
        callback: Optional callback function to handle the result

    Returns:
        Dict indicating the async operation was started

    Example:
        def handle_result(result):
            if result["success"]:
                print(f"Got data: {result['value']}")

        redis_read_async("user_session", callback=handle_result)
    """
    try:
        _start_async_read_operation(key, callback)
        return _create_async_success_response(key, callback)
    except Exception as e:
        logger.error(f"Failed to start async Redis read: {e}")
        return _create_async_error_response(key, str(e))


def _start_async_read_operation(key: str, callback: Optional[callable]):
    """Start the async read operation."""

    async def _do_read():
        try:
            result = await _perform_redis_read(key)
            logger.debug(f"Async Redis read completed: {result.get('prefixed_key')}")

            if callback:
                callback(result)
        except Exception as e:
            logger.error(f"Async Redis read error: {e}")
            if callback:
                callback(_create_read_error_result(key, str(e)))

    _execute_async_operation(_do_read)


async def _perform_redis_read(key: str) -> dict:
    """Perform the actual Redis read operation."""
    client = await _get_async_redis_client()
    prefixed_key = _prefix_key(key)
    value = await client.get(prefixed_key)

    if value is None:
        return _create_not_found_result(key, prefixed_key)

    parsed_value = _parse_redis_value(value)
    ttl = await _get_ttl_if_available(client, prefixed_key)

    return _create_read_success_result(key, prefixed_key, parsed_value, ttl)


def _parse_redis_value(value):
    """Parse Redis value, attempting JSON deserialization."""
    try:
        import json

        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


async def _get_ttl_if_available(client, prefixed_key):
    """Get TTL for key if available."""
    ttl = await client.ttl(prefixed_key)
    return ttl if ttl > 0 else None


def _execute_async_operation(async_func):
    """Execute async operation in appropriate event loop context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(async_func())
        else:
            asyncio.run(async_func())
    except RuntimeError:
        _run_in_thread(async_func)


def _run_in_thread(async_func):
    """Run async function in a separate thread."""
    import threading

    thread = threading.Thread(target=lambda: asyncio.run(async_func()))
    thread.daemon = True
    thread.start()


def _create_not_found_result(key: str, prefixed_key: str) -> dict:
    """Create result for key not found."""
    return {
        "success": False,
        "key": key,
        "prefixed_key": prefixed_key,
        "value": None,
        "message": "Key not found",
    }


def _create_read_success_result(key: str, prefixed_key: str, parsed_value, ttl) -> dict:
    """Create successful read result."""
    return {
        "success": True,
        "key": key,
        "prefixed_key": prefixed_key,
        "value": parsed_value,
        "ttl": ttl,
        "message": "Data read successfully",
    }


def _create_read_error_result(key: str, error: str) -> dict:
    """Create error result for read operation."""
    return {
        "success": False,
        "key": key,
        "error": error,
        "message": "Failed to read data from Redis",
    }


def _create_async_success_response(key: str, callback: Optional[callable]) -> dict:
    """Create success response for async operation start."""
    return {
        "success": True,
        "key": key,
        "prefixed_key": _prefix_key(key),
        "async": True,
        "callback": callback is not None,
        "message": "Async read operation started",
    }


def _create_async_error_response(key: str, error: str) -> dict:
    """Create error response for async operation start."""
    return {
        "success": False,
        "key": key,
        "error": error,
        "async": True,
        "message": "Failed to start async read operation",
    }


# Utility Functions


def redis_delete(key: str) -> dict[str, Any]:
    """Delete a key from Redis.

    Args:
        key: Redis key to delete (will be prefixed with role name)

    Returns:
        Dict containing operation result
    """
    try:
        client = _get_redis_client()
        prefixed_key = _prefix_key(key)

        result = client.delete(prefixed_key)

        logger.debug(f"Redis delete: {prefixed_key} (deleted: {bool(result)})")

        return {
            "success": True,
            "key": key,
            "prefixed_key": prefixed_key,
            "deleted": bool(result),
            "message": "Delete operation completed",
        }

    except Exception as e:
        logger.error(f"Redis delete error: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "message": "Failed to delete key from Redis",
        }


def redis_exists(key: str) -> dict[str, Any]:
    """Check if a key exists in Redis.

    Args:
        key: Redis key to check (will be prefixed with role name)

    Returns:
        Dict containing existence check result
    """
    try:
        client = _get_redis_client()
        prefixed_key = _prefix_key(key)

        exists = bool(client.exists(prefixed_key))

        logger.debug(f"Redis exists: {prefixed_key} = {exists}")

        return {
            "success": True,
            "key": key,
            "prefixed_key": prefixed_key,
            "exists": exists,
            "message": f"Key {'exists' if exists else 'does not exist'}",
        }

    except Exception as e:
        logger.error(f"Redis exists error: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "message": "Failed to check key existence in Redis",
        }


def redis_clear_role_data(role: Optional[str] = None) -> dict[str, Any]:
    """Clear all data for a specific role (use with caution).

    Args:
        role: Role name (auto-detected if not provided)

    Returns:
        Dict containing operation result
    """
    try:
        client = _get_redis_client()
        if role is None:
            role = _get_calling_role()

        # Get all keys for this role
        pattern = f"{role}:*"
        keys = client.keys(pattern)

        if keys:
            deleted = client.delete(*keys)
        else:
            deleted = 0

        logger.info(f"Redis clear role data: {role} (deleted {deleted} keys)")

        return {
            "success": True,
            "role": role,
            "deleted_count": deleted,
            "message": f"Cleared {deleted} keys for role {role}",
        }

    except Exception as e:
        logger.error(f"Redis clear role data error: {e}")
        return {
            "success": False,
            "role": role or "unknown",
            "error": str(e),
            "message": "Failed to clear role data from Redis",
        }


# Health Check


def redis_health_check() -> dict[str, Any]:
    """Check Redis connection health.

    Returns:
        Dict containing health status
    """
    try:
        client = _get_redis_client()

        # Test basic operations
        test_key = _prefix_key("health_check")
        client.set(test_key, "test", ex=10)  # 10 second TTL
        client.get(test_key)
        client.delete(test_key)

        info = client.info()

        return {
            "success": True,
            "connected": True,
            "redis_version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "message": "Redis is healthy",
        }

    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "success": False,
            "connected": False,
            "error": str(e),
            "message": "Redis health check failed",
        }
