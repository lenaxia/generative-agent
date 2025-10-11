"""Timer Role Lifecycle Functions

Pre-processing and post-processing functions for the hybrid timer role.
Handles parameter extraction, validation, timer operations, and confirmation formatting.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class TimerManager:
    """Central timer management system with Redis persistence.

    Handles timer CRUD operations, expiry tracking, and Redis-based persistence
    for the comprehensive timer system.
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        """Initialize TimerManager with Redis connection.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,  # We'll handle encoding manually for consistency
        )
        self.max_duration = 365 * 24 * 3600  # 1 year maximum

    async def create_timer(
        self,
        timer_type: str,
        duration_seconds: Optional[int] = None,
        alarm_time: Optional[datetime] = None,
        name: str = "",
        label: str = "",
        custom_message: str = "",
        user_id: str = "",
        channel_id: str = "",
        request_context: Optional[dict] = None,
        notification_config: Optional[dict] = None,
        actions: Optional[list[dict]] = None,
        recurring: Optional[dict] = None,
        timezone: str = "UTC",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new timer with Redis persistence.

        Args:
            timer_type: Type of timer (countdown, alarm, recurring)
            duration_seconds: Duration for countdown timers
            alarm_time: Target time for alarm timers
            name: Timer name
            label: Optional timer label
            custom_message: Custom expiry message
            user_id: User who created the timer
            channel_id: Channel for notifications
            request_context: Original request context
            notification_config: Notification configuration
            actions: Actions to execute on expiry
            recurring: Recurring timer configuration
            timezone: Timer timezone
            metadata: Additional metadata for the timer (notification preferences, etc.)

        Returns:
            Timer ID

        Raises:
            ValueError: For invalid timer parameters
        """
        # Validate inputs
        self._validate_timer_inputs(
            timer_type, duration_seconds, alarm_time, name, user_id
        )

        # Generate timer ID
        timer_id = f"timer_{uuid.uuid4().hex}"

        # Calculate expiry time
        current_time = int(time.time())
        if timer_type == "countdown":
            expires_at = current_time + duration_seconds
        elif timer_type == "alarm":
            expires_at = int(alarm_time.timestamp())
        else:  # recurring
            expires_at = (
                current_time + duration_seconds
                if duration_seconds
                else int(alarm_time.timestamp())
            )

        # Build timer object
        timer_data = {
            "id": timer_id,
            "type": timer_type,
            "name": name,
            "label": label,
            "custom_message": custom_message or f"{name} expired!",
            "created_at": current_time,
            "expires_at": expires_at,
            "duration_seconds": duration_seconds,
            "timezone": timezone,
            "status": "active",
            "snooze_count": 0,
            "completion_rate": 1.0,
            "user_id": user_id,
            "channel_id": channel_id,
            "request_context": request_context or {},
            "notification_config": notification_config or {},
            "actions": actions or [],
            "recurring": recurring,
            "metadata": metadata or {},
        }

        # Store in Redis
        await self._store_timer_in_redis(timer_data)

        logger.info(f"Created timer {timer_id} for user {user_id}")
        return timer_id

    async def get_timer(self, timer_id: str) -> Optional[dict]:
        """Retrieve timer by ID.

        Args:
            timer_id: Timer identifier

        Returns:
            Timer data or None if not found
        """
        try:

            def _hgetall():
                return self.redis.hgetall(f"timer:data:{timer_id}")

            timer_data = await asyncio.get_event_loop().run_in_executor(None, _hgetall)

            if not timer_data:
                return None

            # Decode and parse timer data
            decoded_timer = {}
            for key, value in timer_data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value

                # Parse JSON fields
                if key_str in [
                    "request_context",
                    "notification_config",
                    "actions",
                    "recurring",
                    "metadata",
                ]:
                    try:
                        decoded_timer[key_str] = (
                            json.loads(value_str) if value_str else {}
                        )
                    except json.JSONDecodeError:
                        decoded_timer[key_str] = {}
                elif key_str in [
                    "created_at",
                    "expires_at",
                    "duration_seconds",
                    "snooze_count",
                ]:
                    decoded_timer[key_str] = int(value_str) if value_str else 0
                elif key_str == "completion_rate":
                    decoded_timer[key_str] = float(value_str) if value_str else 1.0
                else:
                    decoded_timer[key_str] = value_str

            return decoded_timer

        except Exception as e:
            logger.error(f"Failed to get timer {timer_id}: {e}")
            return None

    async def cancel_timer(self, timer_id: str) -> bool:
        """Cancel an active timer.

        Args:
            timer_id: Timer identifier

        Returns:
            True if cancelled successfully, False if not found
        """
        try:
            # Check if timer exists
            timer = await self.get_timer(timer_id)
            if not timer:
                return False

            # Update status to cancelled
            def _hset_status():
                return self.redis.hset(f"timer:data:{timer_id}", "status", "cancelled")

            await asyncio.get_event_loop().run_in_executor(None, _hset_status)

            # Remove from active queue
            def _zrem_timer():
                return self.redis.zrem("timer:active_queue", timer_id)

            await asyncio.get_event_loop().run_in_executor(None, _zrem_timer)

            # Remove from user and channel sets
            if timer.get("user_id"):

                def _srem_user():
                    return self.redis.srem(f"timer:user:{timer['user_id']}", timer_id)

                await asyncio.get_event_loop().run_in_executor(None, _srem_user)

            if timer.get("channel_id"):

                def _srem_channel():
                    return self.redis.srem(
                        f"timer:channel:{timer['channel_id']}", timer_id
                    )

                await asyncio.get_event_loop().run_in_executor(None, _srem_channel)

            logger.info(f"Cancelled timer {timer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel timer {timer_id}: {e}")
            return False

    async def list_timers(
        self,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        status: str = "active",
    ) -> list[dict]:
        """List timers by user or channel.

        Args:
            user_id: Filter by user ID
            channel_id: Filter by channel ID
            status: Filter by status (default: active)

        Returns:
            List of timer data
        """
        try:
            timer_ids = set()

            # Get timer IDs from appropriate sets
            if user_id:

                def _smembers_user():
                    return self.redis.smembers(f"timer:user:{user_id}")

                user_timer_ids = await asyncio.get_event_loop().run_in_executor(
                    None, _smembers_user
                )
                timer_ids.update(
                    tid.decode() if isinstance(tid, bytes) else tid
                    for tid in user_timer_ids
                )

            if channel_id:

                def _smembers_channel():
                    return self.redis.smembers(f"timer:channel:{channel_id}")

                channel_timer_ids = await asyncio.get_event_loop().run_in_executor(
                    None, _smembers_channel
                )
                timer_ids.update(
                    tid.decode() if isinstance(tid, bytes) else tid
                    for tid in channel_timer_ids
                )

            # If no filters specified, get all active timers
            if not user_id and not channel_id:

                def _zrange_active():
                    return self.redis.zrange("timer:active_queue", 0, -1)

                active_timer_ids = await asyncio.get_event_loop().run_in_executor(
                    None, _zrange_active
                )
                timer_ids.update(
                    tid.decode() if isinstance(tid, bytes) else tid
                    for tid in active_timer_ids
                )

            # Fetch timer data
            timers = []
            for timer_id in timer_ids:
                timer = await self.get_timer(timer_id)
                if timer and (not status or timer.get("status") == status):
                    timers.append(timer)

            return timers

        except Exception as e:
            logger.error(f"Failed to list timers: {e}")
            return []

    async def get_expiring_timers(self, current_time: int) -> list[dict]:
        """Get timers that are expiring (expired or about to expire).

        Args:
            current_time: Current timestamp

        Returns:
            List of expiring timer data
        """
        try:
            # Get expired timer IDs from sorted set
            def _zrangebyscore():
                return self.redis.zrangebyscore("timer:active_queue", 0, current_time)

            expired_timer_ids = await asyncio.get_event_loop().run_in_executor(
                None, _zrangebyscore
            )

            # Fetch timer data
            expiring_timers = []
            for timer_id in expired_timer_ids:
                timer_id_str = (
                    timer_id.decode() if isinstance(timer_id, bytes) else timer_id
                )
                timer = await self.get_timer(timer_id_str)
                if timer and timer.get("status") == "active":
                    expiring_timers.append(timer)

            return expiring_timers

        except Exception as e:
            logger.error(f"Failed to get expiring timers: {e}")
            return []

    async def update_timer_status(self, timer_id: str, status: str) -> bool:
        """Update timer status.

        Args:
            timer_id: Timer identifier
            status: New status

        Returns:
            True if updated successfully
        """
        try:
            # Check if timer exists
            timer = await self.get_timer(timer_id)
            if not timer:
                return False

            # Update status
            def _hset_status():
                return self.redis.hset(f"timer:data:{timer_id}", "status", status)

            await asyncio.get_event_loop().run_in_executor(None, _hset_status)

            # If completed or cancelled, remove from active queue
            if status in ["completed", "cancelled"]:

                def _zrem_timer():
                    return self.redis.zrem("timer:active_queue", timer_id)

                await asyncio.get_event_loop().run_in_executor(None, _zrem_timer)

            logger.info(f"Updated timer {timer_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update timer {timer_id} status: {e}")
            return False

    async def snooze_timer(self, timer_id: str, snooze_seconds: int) -> bool:
        """Snooze a timer by extending its expiry time.

        Args:
            timer_id: Timer identifier
            snooze_seconds: Seconds to snooze

        Returns:
            True if snoozed successfully
        """
        try:
            timer = await self.get_timer(timer_id)
            if not timer or timer.get("status") != "active":
                return False

            # Calculate new expiry time
            new_expires_at = timer["expires_at"] + snooze_seconds
            new_snooze_count = timer.get("snooze_count", 0) + 1

            # Update timer data
            def _hset_expires():
                return self.redis.hset(
                    f"timer:data:{timer_id}", "expires_at", new_expires_at
                )

            def _hset_snooze():
                return self.redis.hset(
                    f"timer:data:{timer_id}", "snooze_count", new_snooze_count
                )

            def _zadd_snooze():
                return self.redis.zadd("timer:active_queue", {timer_id: new_expires_at})

            await asyncio.get_event_loop().run_in_executor(None, _hset_expires)
            await asyncio.get_event_loop().run_in_executor(None, _hset_snooze)
            await asyncio.get_event_loop().run_in_executor(None, _zadd_snooze)

            logger.info(f"Snoozed timer {timer_id} for {snooze_seconds} seconds")
            return True

        except Exception as e:
            logger.error(f"Failed to snooze timer {timer_id}: {e}")
            return False

    async def create_recurring_instance(self, timer: dict) -> Optional[str]:
        """Create next instance of a recurring timer.

        Args:
            timer: Original recurring timer data

        Returns:
            New timer ID or None if no more instances needed
        """
        try:
            recurring_config = timer.get("recurring", {})
            if not recurring_config:
                return None

            # Check if we should create another instance
            max_occurrences = recurring_config.get("max_occurrences")
            end_date = recurring_config.get("end_date")
            current_time = int(time.time())

            if end_date and current_time > end_date:
                return None

            # Calculate next occurrence time based on pattern
            pattern = recurring_config.get("pattern", "daily")
            next_expires_at = self._calculate_next_occurrence(
                timer["expires_at"], pattern
            )

            if not next_expires_at:
                return None

            # Create new timer instance
            new_timer_data = timer.copy()
            new_timer_id = f"timer_{uuid.uuid4().hex}"
            new_timer_data.update(
                {
                    "id": new_timer_id,
                    "created_at": current_time,
                    "expires_at": next_expires_at,
                    "status": "active",
                    "snooze_count": 0,
                }
            )

            # Store new timer
            await self._store_timer_in_redis(new_timer_data)

            logger.info(f"Created recurring timer instance {new_timer_id}")
            return new_timer_id

        except Exception as e:
            logger.error(f"Failed to create recurring instance: {e}")
            return None

    def _validate_timer_inputs(
        self,
        timer_type: str,
        duration_seconds: Optional[int],
        alarm_time: Optional[datetime],
        name: str,
        user_id: str,
    ):
        """Validate timer input parameters."""
        valid_types = ["countdown", "alarm", "recurring"]
        if timer_type not in valid_types:
            raise ValueError(
                f"Invalid timer type '{timer_type}'. Must be one of: {valid_types}"
            )

        if timer_type == "countdown":
            if not duration_seconds or duration_seconds <= 0:
                raise ValueError("Duration must be positive for countdown timers")
            if duration_seconds > self.max_duration:
                raise ValueError("Duration exceeds maximum allowed (1 year)")

        if not name.strip():
            raise ValueError("Timer name cannot be empty")

        if not user_id.strip():
            raise ValueError("User ID cannot be empty")

    async def _store_timer_in_redis(self, timer_data: dict):
        """Store timer data in Redis with proper indexing."""
        timer_id = timer_data["id"]

        # Store timer data as hash
        timer_hash = {}
        for key, value in timer_data.items():
            if isinstance(value, (dict, list)):
                timer_hash[key] = json.dumps(value)
            else:
                timer_hash[key] = str(value)

        # Use a wrapper function for hset with mapping
        def _hset_with_mapping():
            return self.redis.hset(f"timer:data:{timer_id}", mapping=timer_hash)

        await asyncio.get_event_loop().run_in_executor(None, _hset_with_mapping)

        # Add to active queue (sorted by expiry time)
        def _zadd_timer():
            return self.redis.zadd(
                "timer:active_queue", {timer_id: timer_data["expires_at"]}
            )

        await asyncio.get_event_loop().run_in_executor(None, _zadd_timer)

        # Add to user index
        if timer_data.get("user_id"):

            def _sadd_user():
                return self.redis.sadd(f"timer:user:{timer_data['user_id']}", timer_id)

            await asyncio.get_event_loop().run_in_executor(None, _sadd_user)

        # Add to channel index
        if timer_data.get("channel_id"):

            def _sadd_channel():
                return self.redis.sadd(
                    f"timer:channel:{timer_data['channel_id']}", timer_id
                )

            await asyncio.get_event_loop().run_in_executor(None, _sadd_channel)

    def _calculate_next_occurrence(
        self, current_expires_at: int, pattern: str
    ) -> Optional[int]:
        """Calculate next occurrence time for recurring timers."""
        if pattern == "daily":
            return current_expires_at + (24 * 3600)
        elif pattern == "weekly":
            return current_expires_at + (7 * 24 * 3600)
        elif pattern == "monthly":
            # Approximate monthly recurrence (30 days)
            return current_expires_at + (30 * 24 * 3600)
        else:
            # For now, default to daily for unknown patterns
            return current_expires_at + (24 * 3600)


# Global timer manager instance
_timer_manager = None


def get_timer_manager() -> TimerManager:
    """Get global timer manager instance."""
    global _timer_manager
    if _timer_manager is None:
        try:
            _timer_manager = TimerManager()
        except Exception as e:
            logger.error(f"Timer manager not available: {e}")
            return None
    return _timer_manager


async def parse_timer_parameters(
    instruction: str,
    context: Optional[Any] = None,
    parameters: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Parse and normalize timer parameters from user input.

    Args:
        instruction: The original user instruction
        context: TaskContext (unused but required by Universal Agent)
        parameters: Dict containing extracted parameters from routing
        **kwargs: Additional keyword arguments including request metadata

    Returns:
        Dict containing parsed and normalized parameters
    """
    try:
        # Extract parameters from the routing result
        if not parameters:
            parameters = {}

        action = parameters.get("action", "")
        duration = parameters.get("duration", "")
        time = parameters.get("time", "")
        label = parameters.get("label", "")
        timer_id = parameters.get("timer_id", "")

        parsed_data = {
            "action_requested": action,
            "timer_duration": None,
            "alarm_time": None,
            "timer_label": label,
            "timer_id": timer_id,
            "duration_seconds": None,
            "alarm_datetime": None,
        }

        # Parse duration for timer operations
        if duration and action == "set":
            duration_seconds = _parse_duration_to_seconds(duration)
            if duration_seconds:
                parsed_data["timer_duration"] = duration
                parsed_data["duration_seconds"] = duration_seconds
                # Calculate expiry time
                expiry_time = datetime.now() + timedelta(seconds=duration_seconds)
                parsed_data["expiry_time"] = expiry_time.strftime("%H:%M:%S")

        # Parse time for alarm operations
        if time and action == "alarm_set":
            alarm_datetime = _parse_alarm_time(time)
            if alarm_datetime:
                parsed_data["alarm_time"] = time
                parsed_data["alarm_datetime"] = alarm_datetime

        # EXECUTE TIMER OPERATIONS IN PRE-PROCESSING
        timer_manager = get_timer_manager()
        execution_result = {}

        try:
            if action == "set" and parsed_data.get("duration_seconds"):
                # Extract context information for proper notification routing
                user_id = "system"
                channel_id = "default"

                if context:
                    # Extract user and channel from TaskContext
                    if hasattr(context, "user_id") and context.user_id:
                        user_id = context.user_id
                    if hasattr(context, "channel_id") and context.channel_id:
                        channel_id = context.channel_id

                    logger.info(
                        f"Timer context from TaskContext: user_id={user_id}, channel_id={channel_id}"
                    )

                # Ensure proper channel format - don't modify if already properly formatted
                if (
                    channel_id
                    and channel_id != "default"
                    and not channel_id.startswith(("slack:", "console", "email", "sms"))
                ):
                    channel_id = f"slack:{channel_id}"

                # Execute timer creation
                timer_id = await timer_manager.create_timer(
                    timer_type="countdown",
                    duration_seconds=parsed_data["duration_seconds"],
                    name=parsed_data.get("timer_label")
                    or f"{parsed_data.get('timer_duration')} timer",
                    label=parsed_data.get("timer_label", ""),
                    user_id=user_id,
                    channel_id=channel_id,
                )
                execution_result = {
                    "success": True,
                    "timer_id": timer_id,
                    "message": f"Timer set for {parsed_data.get('timer_duration')}. Timer ID: {timer_id}",
                }

            elif action == "list":
                # Execute timer listing
                user_id = context.get("user_id") if context else None
                timers = await timer_manager.list_timers(user_id=user_id)
                execution_result = {
                    "success": True,
                    "timers": timers,
                    "count": len(timers),
                    "message": f"Found {len(timers)} active timers",
                }

            elif action == "cancel" and parsed_data.get("timer_id"):
                # Execute timer cancellation
                success = await timer_manager.cancel_timer(parsed_data["timer_id"])
                execution_result = {
                    "success": success,
                    "message": f"Timer {parsed_data['timer_id']} {'cancelled' if success else 'not found'}",
                }

            parsed_data["execution_result"] = execution_result

        except Exception as e:
            logger.error(f"Timer operation failed: {e}")
            parsed_data["execution_result"] = {
                "success": False,
                "error": str(e),
                "message": f"Timer operation failed: {str(e)}",
            }

        logger.info(f"Parsed timer parameters with execution: {parsed_data}")
        return parsed_data

    except Exception as e:
        logger.error(f"Failed to parse timer parameters: {e}")
        return {
            "action_requested": action if "action" in locals() else "",
            "error": f"Parameter parsing failed: {str(e)}",
        }


async def validate_timer_request(
    instruction: str,
    context: Optional[Any] = None,
    parameters: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Validate timer request parameters.

    Args:
        instruction: The original user instruction
        context: TaskContext (unused but required by Universal Agent)
        parameters: Dict containing extracted parameters from routing
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing validation results
    """
    try:
        # Extract parameters from the routing result
        if not parameters:
            parameters = {}

        action = parameters.get("action", "")
        duration = parameters.get("duration", "")
        time = parameters.get("time", "")

        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Validate action
        valid_actions = ["set", "cancel", "list", "alarm_set", "alarm_cancel"]
        if action not in valid_actions:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Invalid action '{action}'. Must be one of: {valid_actions}"
            )

        # Validate duration for timer set
        if action == "set":
            if not duration:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Duration is required for timer set operations"
                )
            elif not _parse_duration_to_seconds(duration):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid duration format '{duration}'. Use formats like '5m', '1h30m', '120'"
                )

        # Validate time for alarm set
        if action == "alarm_set":
            if not time:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Time is required for alarm set operations"
                )
            elif not _parse_alarm_time(time):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid time format '{time}'. Use formats like '14:30', '2:30 PM'"
                )

        logger.info(f"Timer validation result: {validation_result}")
        return {"validation_result": validation_result}

    except Exception as e:
        logger.error(f"Timer validation failed: {e}")
        return {
            "validation_result": {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }
        }


async def format_timer_confirmation(
    llm_response: str,
    context: Optional[Any] = None,
    pre_data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Format the LLM response for clear timer confirmation.

    Args:
        llm_response: Raw response from LLM
        context: TaskContext (unused but required by Universal Agent)
        pre_data: Pre-processing data (unused but required by Universal Agent)
        **kwargs: Additional keyword arguments

    Returns:
        Formatted confirmation message
    """
    try:
        # Clean up the response and ensure it's user-friendly
        formatted_response = llm_response.strip()

        return formatted_response

    except Exception as e:
        logger.error(f"Failed to format timer confirmation: {e}")
        return f"Timer operation completed. (Formatting error: {str(e)})"


async def schedule_notification(
    llm_response: str,
    context: Optional[Any] = None,
    pre_data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Schedule system notification for timer/alarm (placeholder implementation).

    Args:
        llm_response: LLM response containing timer details
        context: TaskContext (unused but required by Universal Agent)
        pre_data: Pre-processing data (unused but required by Universal Agent)
        **kwargs: Additional keyword arguments

    Returns:
        Original LLM response (post-processors should return string)
    """
    try:
        # Placeholder implementation - would integrate with system notifications
        logger.info("Timer notification scheduling requested (not implemented)")
        # Post-processors should return the (potentially modified) LLM response
        return llm_response

    except Exception as e:
        logger.error(f"Failed to schedule notification: {e}")
        return llm_response


async def audit_timer_action(
    llm_response: str,
    context: Optional[Any] = None,
    pre_data: Optional[dict] = None,
    **kwargs,
) -> str:
    """Audit log timer action for tracking and compliance.

    Args:
        llm_response: LLM response containing timer details
        context: TaskContext (unused but required by Universal Agent)
        pre_data: Pre-processing data (unused but required by Universal Agent)
        **kwargs: Additional keyword arguments

    Returns:
        Original LLM response (post-processors should return string)
    """
    try:
        # Log timer action for audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "timer_operation",
            "response": llm_response[:200],  # Truncate for logging
            "user": "system",  # Would be actual user in real implementation
        }

        logger.info(f"Timer audit entry: {audit_entry}")
        # Post-processors should return the (potentially modified) LLM response
        return llm_response

    except Exception as e:
        logger.error(f"Failed to audit timer action: {e}")
        return llm_response


def _parse_duration_to_seconds(duration_str: str) -> Optional[int]:
    """Parse duration string to seconds.

    Args:
        duration_str: Duration in formats like "5m", "1h30m", "120"

    Returns:
        Duration in seconds or None if invalid
    """
    try:
        duration_str = duration_str.lower().strip()

        # Handle plain number (assume seconds)
        if duration_str.isdigit():
            return int(duration_str)

        # Parse complex duration (e.g., "1h30m", "5m", "2h")
        total_seconds = 0

        # Extract hours
        hours_match = re.search(r"(\d+)h", duration_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600

        # Extract minutes
        minutes_match = re.search(r"(\d+)m", duration_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60

        # Extract seconds
        seconds_match = re.search(r"(\d+)s", duration_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))

        return total_seconds if total_seconds > 0 else None

    except Exception as e:
        logger.error(f"Failed to parse duration '{duration_str}': {e}")
        return None


def _parse_alarm_time(time_str: str) -> Optional[datetime]:
    """Parse alarm time string to datetime.

    Args:
        time_str: Time in formats like "14:30", "2:30 PM", "2024-12-25 09:00"

    Returns:
        Parsed datetime or None if invalid
    """
    try:
        time_str = time_str.strip()

        # Try different time formats
        time_formats = [
            "%H:%M",  # 14:30
            "%I:%M %p",  # 2:30 PM
            "%Y-%m-%d %H:%M",  # 2024-12-25 09:00
            "%m/%d/%Y %H:%M",  # 12/25/2024 09:00
        ]

        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt)

                # If no date specified, assume today
                if fmt in ["%H:%M", "%I:%M %p"]:
                    today = datetime.now().date()
                    parsed_time = datetime.combine(today, parsed_time.time())

                return parsed_time

            except ValueError:
                continue

        return None

    except Exception as e:
        logger.error(f"Failed to parse alarm time '{time_str}': {e}")


# Event handler functions for dynamic event-driven architecture
async def handle_timer_expiry_action(
    event_data: dict[str, Any],
    llm,  # EventHandlerLLM utility
    workflow_engine,
    communication_manager,
    context: dict[str, Any],
):
    """Handle timer expiry by parsing action and creating workflows.

    This function is called automatically when TIMER_EXPIRED events are published.
    It uses LLM to parse the original timer request with current context and
    creates appropriate workflows or notifications.

    Args:
        event_data: Timer expiry event data
        llm: EventHandlerLLM utility for easy LLM access
        workflow_engine: WorkflowEngine instance for creating workflows
        communication_manager: CommunicationManager for notifications
        context: Execution context (same as llm.get_context())
    """
    try:
        timer_id = event_data.get("timer_id")
        original_request = llm.get_original_request()
        user_id = llm.get_user_id()
        channel = llm.get_channel()
        room = llm.get_context("device_context.room")

        logger.info(
            f"Processing timer expiry action for timer {timer_id}: {original_request}"
        )

        # Simple LLM call to determine action type
        decision_prompt = f"""
        For timer request '{original_request}' in room '{room}', should I:
        A) Create a workflow (for actions like turning on lights, controlling devices)
        B) Send a notification (for simple reminders)

        Consider:
        - If the request involves controlling devices, choose A
        - If the request is just a reminder or notification, choose B
        - If uncertain, choose B for safety

        Answer: A or B
        """

        action_decision = await llm.invoke(decision_prompt, model_type="WEAK")
        is_workflow = "A" in action_decision or "workflow" in action_decision.lower()

        if not is_workflow:
            # Simple notification
            from common.communication_manager import ChannelType, MessageFormat

            await communication_manager.send_notification(
                message=f"Timer reminder: {original_request}",
                channels=[ChannelType.SLACK],
                recipient=channel,
                message_format=MessageFormat.PLAIN_TEXT,
                metadata=context,
            )
            logger.info(f"Sent timer notification: {original_request}")
        else:
            # Parse action and create workflow instruction
            parsing_prompt = f"""
            Parse this timer action into a clear workflow instruction:

            Timer Action: "{original_request}"
            Room: {room}
            User: {user_id}
            Current Time: {datetime.now().isoformat()}

            Convert this into a specific, actionable workflow instruction.
            For example: "turn on the lights" → "turn on bedroom lights to 50% brightness"

            Return only the workflow instruction, nothing else.
            """

            workflow_instruction = await llm.invoke(parsing_prompt, model_type="WEAK")

            if workflow_instruction and workflow_instruction.strip():
                # Create workflow with parsed instruction
                workflow_id = await workflow_engine.start_workflow(
                    instruction=workflow_instruction.strip(), context=context
                )
                logger.info(f"Created workflow {workflow_id}: {workflow_instruction}")
            else:
                # Fallback to notification if parsing failed
                from common.communication_manager import ChannelType, MessageFormat

                await communication_manager.send_notification(
                    message=f"Timer reminder: {original_request}",
                    channels=[ChannelType.SLACK],
                    recipient=channel,
                    message_format=MessageFormat.PLAIN_TEXT,
                    metadata=context,
                )
                logger.info(f"Fallback notification sent: {original_request}")

    except Exception as e:
        logger.error(f"Error in timer expiry handler: {e}")
        # Always fallback to basic notification
        from common.communication_manager import ChannelType, MessageFormat

        await communication_manager.send_notification(
            message=f"Timer reminder: {llm.get_original_request()}",
            channels=[ChannelType.SLACK],
            recipient=llm.get_channel(),
            message_format=MessageFormat.PLAIN_TEXT,
            metadata=context,
        )


async def handle_location_based_timer_update(
    event_data: dict[str, Any],
    llm,  # EventHandlerLLM utility
    workflow_engine,
    communication_manager,
    context: dict[str, Any],
):
    """Handle user location changes that affect pending timers.

    Args:
        event_data: Location change event data
        llm: EventHandlerLLM utility for easy LLM access
        workflow_engine: WorkflowEngine instance
        communication_manager: CommunicationManager instance
        context: Execution context
    """
    try:
        logger.info(f"Handling location update for timers: {event_data}")

        # This is a placeholder for future location-based timer functionality
        # Could check if any pending timers need location updates
        # For now, just log the event

        user_id = event_data.get("user_id", "unknown")
        new_location = event_data.get("location", {})

        logger.info(f"User {user_id} location changed to: {new_location}")
        # Future: Update location-dependent timers

    except Exception as e:
        logger.error(f"Error handling location-based timer update: {e}")
        return None
