# Redis Commands for Timer System

## Timer Storage Structure

The timer system uses two Redis data structures:

1. **Sorted Set**: `timer:active_queue` - Tracks all active timers with expiry timestamps
2. **Hash Keys**: `timer:timer:data:{timer_id}` - Stores individual timer metadata

**Note**: Timer data keys have double `timer:` prefix because:
- First `timer:` = Auto-detected role prefix from `redis_tools.py`
- Second `timer:` = Explicit key prefix in timer code
- Active queue uses direct client access (no auto-prefix)

---

## Quick Reference Commands

### View All Active Timers (IDs + Expiry Times)
```bash
redis-cli -h localhost
> ZRANGE timer:active_queue 0 -1 WITHSCORES
```

Output format: `timer_id` `expiry_timestamp`
```
1) "timer_79c07b2a"
2) "1734746094.763"  # Unix timestamp
3) "timer_abc123"
4) "1734746150.123"
```

### Count Active Timers
```bash
> ZCARD timer:active_queue
```

### View Active Timers (Not Expired)
```bash
# Get current timestamp first
> TIME

# Get timers expiring AFTER current time
> ZRANGEBYSCORE timer:active_queue 1734746000 +inf WITHSCORES
```

### View Expired Timers (Should Be Empty)
```bash
> ZRANGEBYSCORE timer:active_queue 0 1734746000 WITHSCORES
```

### Get Specific Timer Data
```bash
> GET timer:timer:data:timer_79c07b2a
```

Output: JSON string with timer metadata
```json
{
  "id": "timer_79c07b2a",
  "duration": "1m",
  "duration_seconds": 60,
  "label": "1 minute timer",
  "created_at": 1734746034.763,
  "expires_at": 1734746094.763,
  "status": "active",
  "user_id": "api_user",
  "channel_id": "console",
  "deferred_workflow": "",
  "event_context": {...}
}
```

---

## Monitoring Commands

### Watch Timers in Real-Time
```bash
# Monitor all Redis commands (verbose)
> MONITOR

# Or watch specific patterns
> PSUBSCRIBE timer:*
```

### Check Timer Expiry in Real-Time
```bash
# Run in loop to watch expiries
while true; do
  redis-cli -h localhost ZRANGEBYSCORE timer:active_queue 0 $(date +%s) WITHSCORES
  sleep 1
done
```

---

## Useful Inspection Commands

### List All Timer Keys
```bash
> KEYS timer:timer:data:*
```

Output:
```
1) "timer:timer:data:timer_79c07b2a"
2) "timer:timer:data:timer_abc123"
3) "timer:timer:data:timer_xyz789"
```

### Get All Timer Data at Once
```bash
# Get all timer IDs from active queue
> ZRANGE timer:active_queue 0 -1

# Then get each timer's data
> GET timer:timer:data:timer_79c07b2a
> GET timer:timer:data:timer_abc123
```

### Check Timer TTL (Time To Live)
```bash
> TTL timer:timer:data:timer_79c07b2a
```

Output: Seconds until key expires (returns -1 if no TTL, -2 if doesn't exist)

---

## Manual Timer Management

### Manually Remove a Timer
```bash
# Remove from active queue
> ZREM timer:active_queue timer_79c07b2a

# Delete timer data
> DEL timer:timer:data:timer_79c07b2a
```

### Clear All Timers
```bash
# Clear active queue
> DEL timer:active_queue

# Delete all timer data keys
> EVAL "return redis.call('del', unpack(redis.call('keys', 'timer:timer:data:*')))" 0
```

---

## Debugging Commands

### Check if Redis is Working
```bash
> PING
PONG
```

### View Recent Operations
```bash
> INFO stats
```

### Check Memory Usage
```bash
> MEMORY USAGE timer:active_queue
> MEMORY USAGE timer:timer:data:timer_79c07b2a
```

---

## Production Monitoring Script

Save as `monitor_timers.sh`:

```bash
#!/bin/bash

echo "=== Timer System Status ==="
echo

# Count active timers
ACTIVE=$(redis-cli -h localhost ZCARD timer:active_queue)
echo "Active Timers: $ACTIVE"
echo

# Show all timers with expiry times
echo "Timer Queue:"
redis-cli -h localhost ZRANGE timer:active_queue 0 -1 WITHSCORES | \
  while read timer_id; read expiry; do
    if [ -n "$timer_id" ]; then
      current=$(date +%s)
      remaining=$((expiry - current))
      echo "  $timer_id expires in ${remaining}s"
    fi
  done
echo

# Show expired timers that should be cleaned
CURRENT=$(date +%s)
echo "Expired (should be empty):"
redis-cli -h localhost ZRANGEBYSCORE timer:active_queue 0 $CURRENT
```

Usage:
```bash
chmod +x monitor_timers.sh
./monitor_timers.sh
```

---

## Common Issues

### Timer Created But Not Showing Up
```bash
# Check if key exists (should have double timer: prefix)
> EXISTS timer:timer:data:timer_79c07b2a
> GET timer:timer:data:timer_79c07b2a

# Check if in active queue (single timer: prefix)
> ZSCORE timer:active_queue timer_79c07b2a
```

### Timer Not Expiring
```bash
# Check current time vs expiry time
> TIME
> ZSCORE timer:active_queue timer_79c07b2a

# Check if heartbeat is running
> KEYS heartbeat:*
```

### Redis Connection Issues
```bash
# Check Redis is running
docker ps | grep redis

# Test connection
redis-cli -h localhost ping

# Check logs
docker logs <redis_container_id>
```

---

## Testing Timer Flow

### Create and Track a Timer End-to-End

```bash
# Terminal 1: Monitor Redis operations
redis-cli -h localhost
> MONITOR

# Terminal 2: Create timer via CLI
python3 cli.py
> set a timer for 30 seconds

# Terminal 3: Watch timer progress
watch -n 1 'redis-cli -h localhost ZRANGE timer:active_queue 0 -1 WITHSCORES'

# After 30 seconds, verify timer expired
redis-cli -h localhost ZRANGE timer:active_queue 0 -1
# Should be empty
```

---

## Quick Cheat Sheet

| Task | Command |
|------|---------|
| List all timers | `ZRANGE timer:active_queue 0 -1 WITHSCORES` |
| Count timers | `ZCARD timer:active_queue` |
| Get timer data | `GET timer:timer:data:{timer_id}` |
| Remove timer | `ZREM timer:active_queue {timer_id}` |
| Clear all | `DEL timer:active_queue` |
| List timer keys | `KEYS timer:timer:data:*` |
| Check TTL | `TTL timer:timer:data:{timer_id}` |

---

## Notes

- Timer IDs are auto-generated: `timer_{8_hex_chars}`
- Expiry times are Unix timestamps (seconds since epoch)
- Timer data has TTL = duration + 60 seconds (for cleanup)
- Active queue score = expiry timestamp (for efficient range queries)
- System checks for expired timers every 5 seconds via heartbeat
