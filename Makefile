.PHONY: help install-dev setup-pre-commit format lint test test-unit test-integration test-llm security-scan clean build check-all docker-setup docker-start docker-stop docker-logs docker-test

# Default target
help:
	@echo "StrandsAgent Universal Agent System - Development Commands"
	@echo "========================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install-dev        Install development dependencies"
	@echo "  setup-pre-commit   Install and setup pre-commit hooks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  format            Format code with black and isort"
	@echo "  auto-fix-safe     Automatically fix safe issues (unused imports, formatting)"
	@echo "  auto-fix          Automatically fix all auto-fixable issues (includes variables)"
	@echo "  lint              Run all linters (flake8, mypy, pylint, bandit, yamllint)"
	@echo "  check-all         Run format check, lint, and all tests"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test              Run all tests with coverage"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-llm          Run LLM provider tests only"
	@echo ""
	@echo "Security Commands:"
	@echo "  security-scan     Run security scans (bandit, safety, pip-audit)"
	@echo ""
	@echo "Build Commands:"
	@echo "  build             Build package for distribution"
	@echo "  clean             Clean build artifacts and cache files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-setup      Set up development environment with Docker Redis"
	@echo "  docker-start      Start Docker Redis container"
	@echo "  docker-stop       Stop Docker containers"
	@echo "  docker-logs       View Docker container logs"
	@echo "  docker-test       Run tests with Docker Redis"
	@echo "  redis-cli         Connect to Redis CLI in Docker container"
	@echo ""
	@echo "Usage: make <command>"

# Setup Commands
install-dev:
	@echo "📦 Installing development dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ Development dependencies installed"

setup-pre-commit:
	@echo "🔧 Setting up pre-commit hooks..."
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

# Code Quality Commands
format:
	@echo "🎨 Formatting code..."
	black .
	isort .
	@echo "✅ Code formatted"

auto-fix:
	@echo "🔧 Running automatic fixes..."
	@echo "  → Removing unused imports and variables..."
	autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive . --exclude=venv,agents/deprecated,.benchmarks,.roo
	@echo "  → Fixing import order..."
	isort .
	@echo "  → Formatting code..."
	black .
	@echo "  → Upgrading Python syntax..."
	find . -name "*.py" -not -path "./venv/*" -not -path "./agents/deprecated/*" -not -path "./.benchmarks/*" -not -path "./.roo/*" -exec pyupgrade --py39-plus {} \; 2>/dev/null || true
	@echo "✅ Automatic fixes completed"

auto-fix-safe:
	@echo "🔧 Running safe automatic fixes..."
	@echo "  → Removing unused imports..."
	autoflake --remove-all-unused-imports --in-place --recursive . --exclude=venv,agents/deprecated,.benchmarks,.roo
	@echo "  → Fixing import order..."
	isort .
	@echo "  → Formatting code..."
	black .
	@echo "✅ Safe automatic fixes completed"

lint:
	@echo "🔍 Running linters..."
	@echo "  → Running black (format check)..."
	black --check --diff .
	@echo "  → Running isort (import check)..."
	isort --check-only --diff .
	@echo "  → Running flake8 (style guide)..."
	flake8 .
	@echo "  → Running mypy (type checking)..."
	mypy .
	@echo "  → Running pylint (code analysis)..."
	pylint llm_provider/ supervisor/ common/ config/
	@echo "  → Running bandit (security)..."
	bandit -r . --exclude tests/,agents/deprecated/
	@echo "  → Running yamllint (YAML linting)..."
	yamllint .
	@echo "✅ All linters passed"

# Testing Commands
test:
	@echo "🧪 Running all tests with coverage..."
	timeout 1800 python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --timeout=300
	@echo "✅ All tests completed"

test-unit:
	@echo "🧪 Running unit tests..."
	timeout 600 python -m pytest tests/unit/ -v --timeout=300
	@echo "✅ Unit tests completed"

test-integration:
	@echo "🧪 Running integration tests..."
	timeout 900 python -m pytest tests/integration/ -v --timeout=300
	@echo "✅ Integration tests completed"

test-llm:
	@echo "🧪 Running LLM provider tests..."
	timeout 900 python -m pytest tests/llm_provider/ -v --timeout=300
	@echo "✅ LLM provider tests completed"

# Security Commands
security-scan:
	@echo "🔒 Running security scans..."
	@echo "  → Running bandit (security vulnerabilities)..."
	bandit -r . -f json -o bandit-report.json --exclude tests/,agents/deprecated/
	@echo "  → Running safety (known vulnerabilities)..."
	safety check --json --output safety-report.json || true
	@echo "  → Running pip-audit (dependency vulnerabilities)..."
	pip-audit --requirement requirements.txt --format=json --output=pip-audit-report.json || true
	@echo "✅ Security scans completed"
	@echo "📊 Reports generated: bandit-report.json, safety-report.json, pip-audit-report.json"

# Build Commands
build:
	@echo "🏗️  Building package..."
	python -m pip install --upgrade build twine
	python -m build
	twine check dist/*
	@echo "✅ Package built successfully"

clean:
	@echo "🧹 Cleaning build artifacts and cache files..."
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -f bandit-report.json
	rm -f safety-report.json
	rm -f pip-audit-report.json
	@echo "✅ Cleanup completed"

# Comprehensive Check
check-all: format lint test
	@echo "🎉 All checks passed! Ready for commit."

# Development Workflow Helpers
dev-setup: install-dev setup-pre-commit
	@echo "🚀 Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make check-all' to verify everything works"
	@echo "  2. Start developing!"
	@echo "  3. Use 'make format' before committing"
	@echo "  4. Use 'make test' to run tests"

quick-check:
	@echo "⚡ Running quick checks..."
	black --check .
	isort --check-only .
	flake8 .
	timeout 300 python -m pytest tests/unit/ -x --timeout=60
	@echo "✅ Quick checks passed"

# Performance Testing
benchmark:
	@echo "📊 Running performance benchmarks..."
	timeout 1200 python -m pytest tests/ -k "benchmark" --benchmark-json=benchmark.json --timeout=300
	@echo "✅ Benchmarks completed"

# Documentation
docs-check:
	@echo "📚 Checking documentation..."
	pydocstyle llm_provider/ supervisor/ common/ config/ --convention=google
	@echo "✅ Documentation check completed"

# Git Helpers
pre-push: check-all
	@echo "🚀 Pre-push checks completed successfully!"

# System Health Check
health-check:
	@echo "🏥 Running system health check..."
	@timeout 300 python -c "\
	print('🧪 System Health Check'); \
	print('=' * 30); \
	from llm_provider.role_registry import RoleRegistry; \
	registry = RoleRegistry('roles'); \
	print('✅ RoleRegistry initialized'); \
	from llm_provider.universal_agent import UniversalAgent; \
	from llm_provider.factory import LLMFactory; \
	from unittest.mock import Mock; \
	llm_factory = Mock(); \
	agent = UniversalAgent(llm_factory, registry); \
	print('✅ UniversalAgent initialized'); \
	print('🎉 All systems operational!');"
	@echo "✅ Health check completed"

# Version Information
version:
	@echo "StrandsAgent Universal Agent System"
	@echo "=================================="
	@python --version
	@pip --version
	@echo "Project version: 1.0.0"

# Docker Commands
docker-setup:
	@echo "🐳 Setting up development environment with Docker Redis..."
	@./scripts/dev-setup.sh
	@echo "✅ Docker development environment ready!"

docker-start:
	@echo "🐳 Starting Docker Redis container..."
	docker-compose up -d redis
	@echo "⏳ Waiting for Redis to be ready..."
	@timeout 30 bash -c 'until docker-compose exec redis redis-cli ping > /dev/null 2>&1; do sleep 1; done'
	@echo "✅ Redis container is ready!"

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Docker containers stopped"

docker-logs:
	@echo "📋 Docker container logs:"
	docker-compose logs redis

docker-test: docker-start
	@echo "🧪 Running tests with Docker Redis..."
	timeout 900 python -m pytest tests/integration/test_docker_redis_setup.py -v --timeout=300
	@echo "✅ Docker tests completed"

redis-cli:
	@echo "🔧 Connecting to Redis CLI..."
	docker-compose exec redis redis-cli

redis-commander:
	@echo "🎛️  Starting Redis Commander (GUI)..."
	docker-compose --profile tools up -d redis-commander
	@echo "✅ Redis Commander available at http://localhost:8081 (admin/admin)"

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker cleanup completed"

# Enhanced health check with Redis
health-check-full: docker-start
	@echo "🏥 Running comprehensive system health check..."
	@timeout 300 python -c "\
	print('🧪 System Health Check'); \
	print('=' * 30); \
	from roles.shared_tools.redis_tools import redis_health_check; \
	health = redis_health_check(); \
	print('Redis Health:', health); \
	from llm_provider.role_registry import RoleRegistry; \
	registry = RoleRegistry('roles'); \
	print('✅ RoleRegistry initialized'); \
	from llm_provider.universal_agent import UniversalAgent; \
	from llm_provider.factory import LLMFactory; \
	from unittest.mock import Mock; \
	llm_factory = Mock(); \
	agent = UniversalAgent(llm_factory, registry); \
	print('✅ UniversalAgent initialized'); \
	print('🎉 All systems operational!');"
	@echo "✅ Comprehensive health check completed"
