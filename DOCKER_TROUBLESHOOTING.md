# Docker Setup Troubleshooting Guide

This guide helps resolve common Docker setup issues with the Universal Agent System.

## Quick Diagnosis

Run the Docker environment check:

```bash
make docker-check
```

This will show you the status of Docker, Docker daemon, and Docker Compose.

## Common Issues and Solutions

### 1. Docker Compose Version Issues

**Error**: `URLSchemeUnknown: Not supported URL scheme http+docker`

**Cause**: Older docker-compose versions (1.29.x) may have compatibility issues.

**Solutions**:

#### Option A: Use Docker Compose V2 (Recommended)

```bash
# Install Docker Compose V2
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker compose version
```

#### Option B: Update docker-compose V1

```bash
# Remove old version
sudo apt-get remove docker-compose

# Install latest version
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

#### Option C: Use Docker Desktop

Install Docker Desktop which includes the latest Docker Compose.

### 2. Docker Daemon Not Running

**Error**: `Docker daemon not running`

**Solutions**:

#### Linux (systemd)

```bash
# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

#### Linux (non-systemd)

```bash
# Start Docker daemon
sudo service docker start
```

#### macOS/Windows

Start Docker Desktop application.

### 3. Permission Issues

**Error**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker

# Test Docker access
docker run hello-world
```

### 4. Port Conflicts

**Error**: `Port 6379 already in use`

**Solutions**:

#### Check what's using the port

```bash
# Linux/macOS
lsof -i :6379

# Or
netstat -tulpn | grep 6379
```

#### Stop local Redis

```bash
# Linux (systemd)
sudo systemctl stop redis-server

# Linux (service)
sudo service redis-server stop

# macOS (Homebrew)
brew services stop redis
```

#### Use different port

Edit `docker-compose.yml` to use a different port:

```yaml
services:
  redis:
    ports:
      - "6380:6379" # Use port 6380 instead
```

### 5. Container Startup Issues

**Check container logs**:

```bash
make docker-logs

# Or directly:
docker logs generative-agent-redis
```

**Common log errors**:

#### Permission denied on /data

```bash
# Fix data directory permissions
docker-compose down
docker volume rm generative-agent_redis_data
make docker-start
```

#### Memory issues

```bash
# Check available memory
free -h

# Reduce Redis memory limit in docker/redis.conf
# Change: maxmemory 256mb
# To:     maxmemory 128mb
```

### 6. Network Issues

**Error**: `network generative-agent-network not found`

**Solution**:

```bash
# Clean up and restart
make docker-clean
make docker-start
```

### 7. Application Can't Connect to Redis

**Check Redis connectivity**:

```bash
# Test Redis connection
make redis-cli
# In Redis CLI, run: ping
# Should return: PONG
```

**Check application configuration**:

```bash
# Verify Redis health from application
python -c "
from roles.shared_tools.redis_tools import redis_health_check
import json
print(json.dumps(redis_health_check(), indent=2))
"
```

## Manual Docker Commands

If Makefile commands fail, try these manual commands:

### Start Redis manually

```bash
# Using Docker Compose V2
docker compose up -d redis

# Using Docker Compose V1
docker-compose up -d redis

# Using Docker directly
docker run -d \
  --name generative-agent-redis \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine redis-server --appendonly yes
```

### Check container status

```bash
# List running containers
docker ps

# Check specific container
docker ps -f name=generative-agent-redis

# View container logs
docker logs generative-agent-redis

# Execute commands in container
docker exec -it generative-agent-redis redis-cli
```

### Clean up manually

```bash
# Stop containers
docker stop generative-agent-redis generative-agent-redis-commander

# Remove containers
docker rm generative-agent-redis generative-agent-redis-commander

# Remove volumes
docker volume rm generative-agent_redis_data

# Remove networks
docker network rm generative-agent-network
```

## Alternative Setup Methods

### 1. Local Redis Installation

If Docker continues to have issues, install Redis locally:

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS (Homebrew)

```bash
brew install redis
brew services start redis
```

#### Windows

Download and install Redis from the official website or use WSL.

### 2. Cloud Redis Services

For production or if local setup fails:

- **Redis Cloud**: https://redis.com/redis-enterprise-cloud/
- **AWS ElastiCache**: https://aws.amazon.com/elasticache/
- **Google Cloud Memorystore**: https://cloud.google.com/memorystore
- **Azure Cache for Redis**: https://azure.microsoft.com/en-us/services/cache/

Update `config.yaml` with the cloud Redis connection details.

## Getting Help

1. **Check Docker version compatibility**:

   ```bash
   docker --version
   docker-compose --version  # or docker compose version
   ```

2. **Run comprehensive check**:

   ```bash
   make docker-check
   ```

3. **Test basic Docker functionality**:

   ```bash
   docker run hello-world
   ```

4. **Check system resources**:

   ```bash
   free -h  # Memory
   df -h    # Disk space
   ```

5. **Review Docker daemon logs**:

   ```bash
   # Linux (systemd)
   journalctl -u docker.service

   # Linux (older systems)
   sudo tail -f /var/log/docker.log
   ```

If issues persist, please provide the output of `make docker-check` and any error messages when seeking help.
