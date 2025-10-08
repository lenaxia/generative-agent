"""
Redis tools for generative agent roles.

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
    import aioredis
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global Redis connections
_redis_client = None
_async_redis_client = None
_redis_lock = threading.Lock()


def _get_calling_role() -> str:
    """
    Automatically detect the calling role/agent from the call stack.

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

            # Look for _role indicators in the call stack
            filename = frame.f_code.co_filename

            # Check if we're in a role directory
            if "/roles/" in filename:
                parts = filename.split("/roles/")
                if len(parts) > 1:
                    role_part = parts[1].split("/")[0]
                    if role_part and role_part != "shared_tools":
                        return role_part

            # Check local variables for role context
            local_vars = frame.f_locals
            for var_name in ["role_name", "role", "agent_name", "agent"]:
                if var_name in local_vars:
                    value = local_vars[var_name]
                    if isinstance(value, str) and value:
                        return value
                    elif hasattr(value, "name") and isinstance(value.name, str):
                        return value.name

    finally:
        del frame

    return "unknown"


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

    if not REDIS_AVAILABLE:
        raise ImportError(
            "Redis not available. Install with: pip install aioredis>=2.0.0"
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
    """
    Add role prefix to Redis key to avoid collisions.

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
    """
    Remove role prefix from Redis key.

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
    """
    Synchronously write data to Redis with optional TTL.

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
    """
    Synchronously read data from Redis.

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
    """
    Get all keys for the current role matching a pattern.

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
    """
    Asynchronously write data to Redis (fire and forget).
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

    def _async_write():
        """Internal async write function."""

        async def _do_write():
            try:
                client = await _get_async_redis_client()
                prefixed_key = _prefix_key(key)

                # Handle different value types
                if isinstance(value, (dict, list)):
                    import json

                    serialized_value = json.dumps(value)
                else:
                    serialized_value = str(value)

                # Set value with optional TTL
                if ttl:
                    await client.setex(prefixed_key, ttl, serialized_value)
                else:
                    await client.set(prefixed_key, serialized_value)

                logger.debug(f"Async Redis write completed: {prefixed_key}")

            except Exception as e:
                logger.error(f"Async Redis write error: {e}")

        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, schedule the task
                asyncio.create_task(_do_write())
            else:
                # If no event loop is running, run it
                asyncio.run(_do_write())
        except RuntimeError:
            # Fallback: run in a new thread
            import threading

            thread = threading.Thread(target=lambda: asyncio.run(_do_write()))
            thread.daemon = True
            thread.start()

    try:
        _async_write()

        return {
            "success": True,
            "key": key,
            "prefixed_key": _prefix_key(key),
            "ttl": ttl,
            "async": True,
            "message": "Async write operation started",
        }

    except Exception as e:
        logger.error(f"Failed to start async Redis write: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "async": True,
            "message": "Failed to start async write operation",
        }


def redis_read_async(key: str, callback: Optional[callable] = None) -> dict[str, Any]:
    """
    Asynchronously read data from Redis (fire and forget).
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

    def _async_read():
        """Internal async read function."""

        async def _do_read():
            try:
                client = await _get_async_redis_client()
                prefixed_key = _prefix_key(key)

                value = await client.get(prefixed_key)

                if value is None:
                    result = {
                        "success": False,
                        "key": key,
                        "prefixed_key": prefixed_key,
                        "value": None,
                        "message": "Key not found",
                    }
                else:
                    # Try to deserialize JSON
                    try:
                        import json

                        parsed_value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_value = value

                    # Get TTL if available
                    ttl = await client.ttl(prefixed_key)
                    ttl = ttl if ttl > 0 else None

                    result = {
                        "success": True,
                        "key": key,
                        "prefixed_key": prefixed_key,
                        "value": parsed_value,
                        "ttl": ttl,
                        "message": "Data read successfully",
                    }

                logger.debug(f"Async Redis read completed: {prefixed_key}")

                # Call callback if provided
                if callback:
                    callback(result)

            except Exception as e:
                logger.error(f"Async Redis read error: {e}")
                if callback:
                    callback(
                        {
                            "success": False,
                            "key": key,
                            "error": str(e),
                            "message": "Failed to read data from Redis",
                        }
                    )

        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, schedule the task
                asyncio.create_task(_do_read())
            else:
                # If no event loop is running, run it
                asyncio.run(_do_read())
        except RuntimeError:
            # Fallback: run in a new thread
            import threading

            thread = threading.Thread(target=lambda: asyncio.run(_do_read()))
            thread.daemon = True
            thread.start()

    try:
        _async_read()

        return {
            "success": True,
            "key": key,
            "prefixed_key": _prefix_key(key),
            "async": True,
            "callback": callback is not None,
            "message": "Async read operation started",
        }

    except Exception as e:
        logger.error(f"Failed to start async Redis read: {e}")
        return {
            "success": False,
            "key": key,
            "error": str(e),
            "async": True,
            "message": "Failed to start async read operation",
        }


# Utility Functions


def redis_delete(key: str) -> dict[str, Any]:
    """
    Delete a key from Redis.

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
    """
    Check if a key exists in Redis.

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
    """
    Clear all data for a specific role (use with caution).

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
    """
    Check Redis connection health.

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
