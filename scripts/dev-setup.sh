#!/bin/bash

# Development setup script for Universal Agent System
# This script helps set up the development environment with Docker Redis

set -e

echo "🚀 Setting up Universal Agent System development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    echo "   Try: sudo systemctl start docker  # Linux"
    echo "   Or start Docker Desktop application"
    exit 1
fi

# Check if Docker Compose V2 is installed
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose V2 is not installed. Please install Docker Compose V2."
    echo "   Run: sudo apt-get install docker-compose-plugin"
    exit 1
fi

echo "✅ Using Docker Compose V2"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "📦 Starting Redis container..."
docker compose up -d redis

echo "⏳ Waiting for Redis to be ready..."
timeout=30
counter=0
while ! docker compose exec redis redis-cli ping > /dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Redis failed to start within $timeout seconds"
        exit 1
    fi
    echo "Waiting for Redis... ($counter/$timeout)"
    sleep 1
    counter=$((counter + 1))
done

echo "✅ Redis is ready!"

# Test Redis connection from host
echo "🔍 Testing Redis connection..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
        echo "✅ Redis connection test successful"
    else
        echo "⚠️  Redis CLI test failed, but container is running"
    fi
else
    echo "ℹ️  Redis CLI not installed locally, skipping connection test"
fi

# Install Python dependencies if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "📚 Installing Python dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  No virtual environment detected. Please activate your virtual environment and run:"
    echo "   pip install -r requirements.txt"
    echo "   pip install -r requirements-dev.txt"
fi

# Run Redis health check using the application
echo "🏥 Running application Redis health check..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    python -c "
from roles.shared_tools.redis_tools import redis_health_check
import json
result = redis_health_check()
print('Redis Health Check:', json.dumps(result, indent=2))
if result.get('success'):
    print('✅ Application Redis connection successful')
else:
    print('❌ Application Redis connection failed')
    exit(1)
" || echo "⚠️  Health check failed - you may need to install dependencies first"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate your virtual environment (if not already active)"
echo "   2. Install dependencies: pip install -r requirements.txt requirements-dev.txt"
echo "   3. Run the application: python cli.py"
echo "   4. Run tests: make test"
echo ""
echo "🔧 Useful commands:"
echo "   • Start Redis: docker compose up -d redis"
echo "   • Stop Redis: docker compose down"
echo "   • View Redis logs: docker compose logs redis"
echo "   • Redis CLI: docker compose exec redis redis-cli"
echo "   • Redis Commander (GUI): docker compose --profile tools up -d redis-commander"
echo "     Then visit: http://localhost:8081 (admin/admin)"
echo ""
