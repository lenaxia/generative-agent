# Docker Development Environment

This directory contains Docker configuration files for the Universal Agent System development environment.

## Files

- **`redis.conf`**: Redis configuration optimized for development
- **`docker compose.yml`**: Main Docker Compose configuration (in project root)

## Quick Start

```bash
# Start the development environment
make docker-setup

# Or manually:
docker compose up -d redis
```

## Redis Configuration

The Redis container is configured with development-friendly settings:

### Performance Settings

- **Memory Limit**: 256MB with LRU eviction policy
- **Persistence**: Both RDB snapshots and AOF logging enabled
- **Connection Pool**: Up to 50 connections
- **Timeout Settings**: Optimized for local development

### Security Settings

- **Protected Mode**: Disabled (development only)
- **Password**: None required (development only)
- **Bind Address**: All interfaces (0.0.0.0)

### Persistence Settings

- **RDB Snapshots**:
  - Save after 900 seconds if at least 1 key changed
  - Save after 300 seconds if at least 10 keys changed
  - Save after 60 seconds if at least 10000 keys changed
- **AOF Logging**: Enabled with `everysec` fsync policy
- **Data Directory**: `/data` (persisted in Docker volume)

## Development Workflow

### Starting Redis

```bash
# Start Redis container
make docker-start

# Verify Redis is running
docker compose ps
```

### Connecting to Redis

```bash
# Using Docker Compose
make redis-cli

# Using local Redis CLI (if installed)
redis-cli -h localhost -p 6379
```

### Monitoring Redis

```bash
# View logs
make docker-logs

# Start Redis Commander (Web GUI)
make redis-commander
# Visit http://localhost:8081 (admin/admin)
```

### Application Integration

The Universal Agent System automatically detects and uses Docker Redis:

```python
# Redis tools work automatically
from roles.shared_tools.redis_tools import redis_health_check
health = redis_health_check()
print(health)  # Should show connected: true
```

### Testing

```bash
# Run Docker-specific tests
make docker-test

# Run all tests with Docker Redis
make test
```

## Troubleshooting

### Redis Won't Start

```bash
# Check if port 6379 is in use
lsof -i :6379

# Stop local Redis if running
sudo systemctl stop redis-server  # Linux
brew services stop redis          # macOS

# Clean and restart Docker
make docker-clean
make docker-start
```

### Connection Issues

```bash
# Test Redis connectivity
docker compose exec redis redis-cli ping

# Check Redis configuration
docker compose exec redis redis-cli config get "*"

# View detailed logs
docker compose logs redis -f
```

### Performance Issues

```bash
# Monitor Redis performance
docker compose exec redis redis-cli --latency

# Check memory usage
docker compose exec redis redis-cli info memory

# Monitor slow queries
docker compose exec redis redis-cli slowlog get 10
```

## Configuration Customization

To modify Redis settings, edit `docker/redis.conf` and restart the container:

```bash
# After editing redis.conf
docker compose restart redis
```

Common customizations:

- **Memory limit**: Change `maxmemory 256mb`
- **Persistence**: Modify `save` directives
- **Logging**: Adjust `loglevel`
- **Security**: Add `requirepass` for password protection

## Production Considerations

This configuration is optimized for development. For production:

1. **Enable password authentication**
2. **Restrict bind address**
3. **Enable protected mode**
4. **Configure proper memory limits**
5. **Set up Redis clustering if needed**
6. **Configure SSL/TLS encryption**

See the main documentation for production deployment guidelines.
