# Development Setup Guide

This guide will help you set up the development environment for the StrandsAgent Universal Agent System with all the linting, testing, and CI/CD workflows.

## Quick Start

### Option 1: With Docker Redis (Recommended)

```bash
# 1. Setup development environment with Docker Redis
make docker-setup

# 2. Verify everything works
make docker-test

# 3. Start developing!
python cli.py
```

### Option 2: Traditional Setup

```bash
# 1. Install development dependencies
make install-dev

# 2. Setup pre-commit hooks
make setup-pre-commit

# 3. Verify everything works
make check-all
```

## Detailed Setup Instructions

### 1. Prerequisites

- Python 3.9+ installed
- Git installed
- Virtual environment activated
- **Docker and Docker Compose** (for Redis development environment)

### 2. Install Development Dependencies

```bash
# Install all development tools
pip install -r requirements-dev.txt
```

This installs:

- **Code Formatting**: `black`, `isort`
- **Linting**: `flake8`, `mypy`, `pylint`, `bandit`, `yamllint`
- **Testing**: `pytest`, `coverage`, `pytest-benchmark`
- **Security**: `safety`, `bandit`
- **Development**: `pre-commit`

### 3. Setup Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Test the hooks (optional)
pre-commit run --all-files
```

### 4. Development Workflow

#### Code Formatting

```bash
# Format code automatically
make format

# Check formatting without changes
black --check .
isort --check-only .
```

#### Linting

```bash
# Run all linters
make lint

# Run individual linters
flake8 .                    # Style guide
mypy .                      # Type checking
pylint llm_provider/        # Code analysis
bandit -r .                 # Security scan
yamllint .                  # YAML linting
```

#### Testing

```bash
# Run all tests with coverage
make test

# Run specific test suites
make test-unit              # Unit tests only
make test-integration       # Integration tests only
make test-llm              # LLM provider tests only

# Run with specific options
pytest tests/unit/ -v --timeout=300
```

#### Security Scanning

```bash
# Run comprehensive security scan
make security-scan

# Individual security tools
bandit -r . --exclude tests/
safety check
pip-audit --requirement requirements.txt
```

### 5. IDE Configuration

#### VSCode

The repository includes VSCode settings in `.vscode/settings.json` that:

- Configures Python interpreter and formatting
- Enables all linters
- Sets up testing integration
- Configures file associations and exclusions

Recommended extensions are listed in the settings file.

#### Other IDEs

Configure your IDE to use:

- **Formatter**: Black (line length: 88)
- **Import sorter**: isort (black profile)
- **Linters**: flake8, mypy, pylint
- **Test runner**: pytest

### 6. Git Workflow

#### Pre-commit Checks

The pre-commit hooks automatically run:

- Code formatting (black, isort)
- Linting (flake8, mypy, pylint)
- Security scanning (bandit)
- YAML validation (yamllint)

#### Manual Checks

```bash
# Before committing
make format                 # Format code
make lint                   # Check linting
make test-unit             # Run quick tests

# Before pushing
make check-all             # Comprehensive check
```

### 7. CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

#### On Every Push/PR:

- **Code Quality**: Format checking, linting, type checking
- **Testing**: Unit, integration, and LLM provider tests across Python 3.9-3.11
- **Security**: Vulnerability scanning with Trivy, Bandit, Safety
- **Build**: Package building and validation

#### On Main Branch:

- **Performance**: Benchmark tracking
- **Health Check**: System integration verification

### 8. Configuration Files

| File                      | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| `.flake8`                 | Style guide enforcement configuration                   |
| `pyproject.toml`          | Tool configuration (black, isort, mypy, pylint, pytest) |
| `yamllint.yml`            | YAML linting rules                                      |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration                          |
| `requirements-dev.txt`    | Development dependencies                                |
| `.vscode/settings.json`   | VSCode IDE configuration                                |
| `Makefile`                | Development task automation                             |

### 9. Common Commands

```bash
# Setup
make dev-setup              # Complete development setup
make install-dev            # Install dependencies only

# Development
make format                 # Format code
make lint                   # Run linters
make test                   # Run tests
make check-all             # Format + lint + test

# Quick checks
make quick-check           # Fast validation
make health-check          # System health verification

# Maintenance
make clean                 # Clean build artifacts
make security-scan         # Security analysis
make build                 # Build package
```

### 10. Troubleshooting

#### Common Issues

**Pre-commit hooks failing:**

```bash
# Update hooks
pre-commit autoupdate

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

**Linting errors:**

```bash
# Auto-fix formatting issues
make format

# Check specific linter output
flake8 . --show-source
mypy . --show-error-codes
```

**Test failures:**

```bash
# Run tests with more verbose output
pytest tests/ -v -s --tb=long

# Run specific failing test
pytest tests/unit/test_specific.py::test_function -v
```

**Import errors:**

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use the Makefile which sets it automatically
make test
```

#### Performance Issues

**Slow linting:**

- Use `make quick-check` for faster validation
- Run specific linters: `flake8 .` instead of `make lint`

**Slow tests:**

- Use `make test-unit` for faster feedback
- Run specific test files: `pytest tests/unit/test_specific.py`

### 11. Contributing Guidelines

1. **Before starting work:**

   - Run `make dev-setup` to ensure proper environment
   - Create a feature branch from `main`

2. **During development:**

   - Use `make format` before committing
   - Run `make quick-check` frequently
   - Write tests for new functionality

3. **Before submitting PR:**

   - Run `make check-all` to ensure all checks pass
   - Update documentation if needed
   - Fill out the PR template completely

4. **PR Review Process:**
   - All CI checks must pass
   - Code review approval required
   - Security and performance impact assessed

### 12. Advanced Configuration

#### Custom Linting Rules

Edit configuration files to customize rules:

- `.flake8`: Style guide rules
- `pyproject.toml`: Tool-specific settings
- `yamllint.yml`: YAML validation rules

#### Performance Tuning

```bash
# Run benchmarks
make benchmark

# Profile specific functions
python -m cProfile -o profile.stats script.py
```

#### Security Configuration

```bash
# Custom bandit configuration
bandit -r . -f json -o custom-security-report.json

# Safety with custom database
safety check --db custom-safety-db.json
```

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report via GitHub Issues
- **Development Questions**: Check existing issues or create new ones

## Next Steps

After setup:

1. Explore the codebase: `make health-check`
2. Run the test suite: `make test`
3. Try making a small change and commit it
4. Review the CI/CD pipeline results

## Docker Development Environment

The project includes a Docker-based development environment that provides Redis out of the box.

### Docker Setup

#### Prerequisites

- Docker installed and running
- Docker Compose installed

#### Quick Setup

```bash
# Complete setup with one command
make docker-setup
```

#### Manual Setup

```bash
# Start Redis container
make docker-start

# Verify Redis is running
make docker-logs

# Connect to Redis CLI
make redis-cli

# Start Redis Commander (GUI) - optional
make redis-commander
# Visit http://localhost:8081 (admin/admin)
```

#### Docker Commands

| Command                | Description                            |
| ---------------------- | -------------------------------------- |
| `make docker-setup`    | Complete development environment setup |
| `make docker-start`    | Start Redis container                  |
| `make docker-stop`     | Stop all containers                    |
| `make docker-logs`     | View container logs                    |
| `make docker-test`     | Run Docker integration tests           |
| `make redis-cli`       | Connect to Redis CLI                   |
| `make redis-commander` | Start Redis GUI (port 8081)            |
| `make docker-clean`    | Clean Docker resources                 |

#### Configuration

The Docker setup includes:

- **Redis 7**: Latest stable Redis with optimized development configuration
- **Persistent Storage**: Data persists between container restarts
- **Health Checks**: Automatic health monitoring
- **Redis Commander**: Optional web-based Redis GUI

#### Redis Configuration

The Redis container uses a custom configuration (`docker/redis.conf`) optimized for development:

- **Memory Limit**: 256MB with LRU eviction
- **Persistence**: Both RDB snapshots and AOF logging
- **Performance**: Optimized for development workloads
- **Security**: Development-friendly settings (no password required)

#### Environment Variables

The application automatically detects Docker Redis when available:

- `REDIS_HOST=localhost` (default)
- `REDIS_PORT=6379` (default)
- `REDIS_DB=0` (default)

#### Troubleshooting Docker Setup

**Redis connection issues:**

```bash
# Check if Redis container is running
docker compose ps

# Check Redis logs
make docker-logs

# Test Redis connection
make redis-cli
# Then run: ping
```

**Port conflicts:**

```bash
# If port 6379 is in use, stop local Redis
sudo systemctl stop redis-server  # Linux
brew services stop redis          # macOS

# Or modify docker-compose.yml to use different port
```

**Container startup issues:**

```bash
# Clean and restart
make docker-clean
make docker-start
```

### Integration with Application

The Universal Agent System automatically uses Docker Redis when available:

```python
# Redis tools automatically connect to Docker Redis
from roles.shared_tools.redis_tools import redis_write, redis_read

# Write data
result = redis_write("test_key", {"message": "Hello Docker Redis!"})

# Read data
data = redis_read("test_key")
```

The TimerManager also works seamlessly with Docker Redis:

```python
from roles.timer.lifecycle import TimerManager

# Automatically connects to Docker Redis on localhost:6379
timer_manager = TimerManager()
```

Happy coding! 🚀
