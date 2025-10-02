# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the StrandsAgent Universal Agent system in production environments.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4 GB RAM
- 2 CPU cores
- 10 GB disk space

**Recommended Requirements:**
- Python 3.11 or higher
- 16 GB RAM
- 8 CPU cores
- 50 GB disk space
- SSD storage for better I/O performance

### Dependencies

**Core Dependencies:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# Install Python dependencies
pip install -r requirements.txt
```

**Optional Dependencies:**
```bash
# For MCP server integration
npm install -g @modelcontextprotocol/server-web-search
npm install -g @modelcontextprotocol/server-filesystem

# For enhanced monitoring
pip install prometheus-client grafana-api

# For production WSGI server
pip install gunicorn uvicorn
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/strands-universal-agent.git
cd strands-universal-agent
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration Setup

```bash
# Copy example configuration
cp config/config.example.yaml config.yaml

# Create necessary directories
mkdir -p logs checkpoints data

# Set proper permissions
chmod 755 logs checkpoints data
```

## Configuration for Production

### 1. Environment Variables

Create a `.env` file for production settings:

```bash
# .env
# LLM Provider Settings
BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# External Service API Keys
WEATHER_API_KEY=your_weather_api_key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SEARCH_API_KEY=your_search_api_key

# System Settings
LOG_LEVEL=INFO
MAX_CONCURRENT_TASKS=10
CHECKPOINT_INTERVAL=180

# Security Settings
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Database Settings (if using external state persistence)
DATABASE_URL=postgresql://user:pass@localhost:5432/strands_agent

# Monitoring Settings
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
```

### 2. Production Configuration

```yaml
# config/production.yaml
logging:
  log_level: ${LOG_LEVEL:INFO}
  log_file: /var/log/strands-agent/system.log
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_file_size: 100MB
  backup_count: 10
  structured_logging: true

llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: ${BEDROCK_MODEL_ID:us.amazon.nova-pro-v1:0}
    temperature: 0.3
    max_tokens: 4096
    timeout: 60
  
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: ${BEDROCK_STRONG_MODEL:us.amazon.nova-premier-v1:0}
    temperature: 0.1
    max_tokens: 8192
    timeout: 120
  
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: ${BEDROCK_WEAK_MODEL:us.amazon.nova-lite-v1:0}
    temperature: 0.5
    max_tokens: 2048
    timeout: 30

universal_agent:
  framework: strands
  max_concurrent_tasks: ${MAX_CONCURRENT_TASKS:10}
  checkpoint_interval: ${CHECKPOINT_INTERVAL:180}
  retry_attempts: 3
  retry_delay: 2.0
  task_timeout: 1800
  
  role_optimization:
    planning: STRONG
    search: WEAK
    weather: WEAK
    summarizer: DEFAULT
    slack: DEFAULT

mcp:
  config_file: config/mcp_production.yaml
  enabled: true
  timeout: 45
  retry_attempts: 3

features:
  universal_agent_enabled: true
  mcp_integration_enabled: true
  task_scheduling_enabled: true
  checkpoint_persistence: true
  metrics_collection: ${METRICS_ENABLED:true}
  health_monitoring: ${HEALTH_CHECK_ENABLED:true}
  security_enabled: true
  rate_limiting_enabled: true

security:
  authentication:
    enabled: true
    method: token
    token_expiry: 3600
  
  api:
    cors_enabled: true
    cors_origins: ["https://your-frontend.com"]
    rate_limiting:
      requests_per_minute: 1000
      burst_size: 100

monitoring:
  metrics:
    enabled: true
    collection_interval: 30
    retention_days: 90
  
  health_checks:
    enabled: true
    check_interval: 15
    timeout: 5
  
  alerting:
    enabled: true
    channels:
      slack:
        webhook_url: ${SLACK_WEBHOOK_URL}
        channel: "#alerts"

development:
  debug_mode: false
  test_mode: false
  mock_external_services: false
```

## Deployment Methods

### 1. Docker Deployment

#### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for MCP servers
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MCP servers
RUN npm install -g @modelcontextprotocol/server-web-search
RUN npm install -g @modelcontextprotocol/server-filesystem

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs checkpoints data

# Set permissions
RUN chmod 755 logs checkpoints data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from supervisor.supervisor import Supervisor; s = Supervisor('config.yaml'); print('healthy' if s.status()['status'] == 'running' else exit(1))"

# Run application
CMD ["python", "-m", "supervisor.supervisor", "--config", "config.yaml"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  strands-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT_TASKS=10
      - BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from supervisor.supervisor import Supervisor; s = Supervisor('config.yaml'); exit(0 if s.status()['status'] == 'running' else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  # Optional: PostgreSQL for state persistence
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: strands_agent
      POSTGRES_USER: strands_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### 2. Kubernetes Deployment

#### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agent
  labels:
    app: strands-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: strands-agent
  template:
    metadata:
      labels:
        app: strands-agent
    spec:
      containers:
      - name: strands-agent
        image: your-registry/strands-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: MAX_CONCURRENT_TASKS
          value: "10"
        envFrom:
        - secretRef:
            name: strands-agent-secrets
        - configMapRef:
            name: strands-agent-config
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: checkpoints
          mountPath: /app/checkpoints
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: logs
        emptyDir: {}
      - name: checkpoints
        persistentVolumeClaim:
          claimName: strands-agent-checkpoints
      - name: config
        configMap:
          name: strands-agent-config
```

#### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: strands-agent-service
spec:
  selector:
    app: strands-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: strands-agent-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - strands-agent.your-domain.com
    secretName: strands-agent-tls
  rules:
  - host: strands-agent.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: strands-agent-service
            port:
              number: 80
```

### 3. Systemd Service

For traditional Linux server deployment:

```ini
# /etc/systemd/system/strands-agent.service
[Unit]
Description=StrandsAgent Universal Agent System
After=network.target

[Service]
Type=simple
User=strands-agent
Group=strands-agent
WorkingDirectory=/opt/strands-agent
Environment=PATH=/opt/strands-agent/venv/bin
ExecStart=/opt/strands-agent/venv/bin/python -m supervisor.supervisor --config config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/strands-agent/logs /opt/strands-agent/checkpoints

[Install]
WantedBy=multi-user.target
```

**Installation:**
```bash
# Create service user
sudo useradd -r -s /bin/false strands-agent

# Install service
sudo cp strands-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable strands-agent
sudo systemctl start strands-agent

# Check status
sudo systemctl status strands-agent
```

## Production Configuration

### 1. Security Hardening

**File Permissions:**
```bash
# Set secure file permissions
chmod 600 config.yaml .env
chmod 755 logs checkpoints data
chown -R strands-agent:strands-agent /opt/strands-agent
```

**Firewall Configuration:**
```bash
# Configure firewall (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # Application port
sudo ufw enable
```

**SSL/TLS Configuration:**
```yaml
# config/production.yaml
security:
  tls:
    enabled: true
    cert_file: /etc/ssl/certs/strands-agent.crt
    key_file: /etc/ssl/private/strands-agent.key
    ca_file: /etc/ssl/certs/ca-bundle.crt
```

### 2. Database Setup

**PostgreSQL Setup:**
```sql
-- Create database and user
CREATE DATABASE strands_agent;
CREATE USER strands_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE strands_agent TO strands_user;

-- Create tables for state persistence
\c strands_agent;

CREATE TABLE workflows (
    id VARCHAR(50) PRIMARY KEY,
    state VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checkpoint_data JSONB
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_workflows_state ON workflows(state);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
```

**Database Configuration:**
```yaml
# config/production.yaml
database:
  enabled: true
  url: ${DATABASE_URL:postgresql://strands_user:password@localhost:5432/strands_agent}
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
```

### 3. Caching Setup

**Redis Configuration:**
```yaml
# config/production.yaml
caching:
  enabled: true
  backend: redis
  redis:
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}
    password: ${REDIS_PASSWORD}
    db: 0
    max_connections: 50
  
  # Cache settings
  default_ttl: 3600
  max_memory: 1GB
  eviction_policy: allkeys-lru
```

## Monitoring and Observability

### 1. Health Checks

**Health Check Endpoint:**
```python
# supervisor/health_check.py
from flask import Flask, jsonify
import psutil
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check system resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Check application components
        workflow_engine_health = check_workflow_engine()
        universal_agent_health = check_universal_agent()
        mcp_health = check_mcp_servers()
        
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "memory_percent": memory.percent,
                "cpu_percent": cpu,
                "disk_percent": disk.percent
            },
            "components": {
                "workflow_engine": workflow_engine_health,
                "universal_agent": universal_agent_health,
                "mcp_servers": mcp_health
            }
        }
        
        # Determine overall health
        if any(comp != "healthy" for comp in health_data["components"].values()):
            health_data["status"] = "degraded"
        
        if memory.percent > 90 or cpu > 90 or disk.percent > 90:
            health_data["status"] = "critical"
        
        status_code = 200 if health_data["status"] == "healthy" else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 503

@app.route('/ready')
def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Quick checks for service readiness
        from supervisor.supervisor import Supervisor
        supervisor = Supervisor('config.yaml')
        status = supervisor.status()
        
        if status['status'] == 'running':
            return jsonify({"status": "ready"}), 200
        else:
            return jsonify({"status": "not_ready"}), 503
            
    except Exception as e:
        return jsonify({"status": "not_ready", "error": str(e)}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
```

### 2. Metrics Collection

**Prometheus Integration:**
```python
# supervisor/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
workflow_counter = Counter('workflows_total', 'Total number of workflows', ['status'])
workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution time')
active_workflows = Gauge('active_workflows', 'Number of active workflows')
task_counter = Counter('tasks_total', 'Total number of tasks', ['role', 'status'])

class MetricsCollector:
    def __init__(self, port=8002):
        self.port = port
        start_http_server(port)
    
    def record_workflow_started(self, workflow_id):
        active_workflows.inc()
    
    def record_workflow_completed(self, workflow_id, duration, status):
        workflow_counter.labels(status=status).inc()
        workflow_duration.observe(duration)
        active_workflows.dec()
    
    def record_task_completed(self, task_role, status):
        task_counter.labels(role=task_role, status=status).inc()
```

### 3. Logging Configuration

**Structured Logging:**
```python
# supervisor/logging_config.py
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'workflow_id'):
            log_entry['workflow_id'] = record.workflow_id
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        return json.dumps(log_entry)

# Configure structured logging
def setup_structured_logging():
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
```

## Load Balancing and Scaling

### 1. Horizontal Scaling

**Load Balancer Configuration (Nginx):**
```nginx
# /etc/nginx/sites-available/strands-agent
upstream strands_agent_backend {
    least_conn;
    server 10.0.1.10:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name strands-agent.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name strands-agent.your-domain.com;
    
    ssl_certificate /etc/ssl/certs/strands-agent.crt;
    ssl_certificate_key /etc/ssl/private/strands-agent.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://strands_agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    location /health {
        proxy_pass http://strands_agent_backend/health;
        access_log off;
    }
}
```

### 2. Auto-Scaling

**Kubernetes HPA:**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: strands-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: strands-agent
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: active_workflows
      target:
        type: AverageValue
        averageValue: "10"
```

## Backup and Recovery

### 1. Backup Strategy

**Automated Backup Script:**
```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/strands-agent"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$DATE"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup configuration
cp -r config/ "$BACKUP_PATH/config/"

# Backup checkpoints
cp -r checkpoints/ "$BACKUP_PATH/checkpoints/"

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_PATH/logs/" \;

# Backup database (if using PostgreSQL)
if [ "$DATABASE_ENABLED" = "true" ]; then
    pg_dump $DATABASE_URL > "$BACKUP_PATH/database.sql"
fi

# Create archive
tar -czf "$BACKUP_DIR/strands-agent-backup-$DATE.tar.gz" -C "$BACKUP_PATH" .

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_PATH"
```

### 2. Recovery Procedures

**Disaster Recovery Script:**
```bash
#!/bin/bash
# scripts/recovery.sh

BACKUP_FILE="$1"
RECOVERY_DIR="/opt/strands-agent"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "Starting recovery from $BACKUP_FILE"

# Stop service
sudo systemctl stop strands-agent

# Create recovery directory
mkdir -p "$RECOVERY_DIR/recovery"
cd "$RECOVERY_DIR/recovery"

# Extract backup
tar -xzf "$BACKUP_FILE"

# Restore configuration
cp -r config/* "$RECOVERY_DIR/config/"

# Restore checkpoints
cp -r checkpoints/* "$RECOVERY_DIR/checkpoints/"

# Restore database (if applicable)
if [ -f "database.sql" ]; then
    psql $DATABASE_URL < database.sql
fi

# Set permissions
chown -R strands-agent:strands-agent "$RECOVERY_DIR"
chmod 600 "$RECOVERY_DIR/config.yaml"

# Start service
sudo systemctl start strands-agent

# Verify recovery
sleep 10
if sudo systemctl is-active --quiet strands-agent; then
    echo "✅ Recovery completed successfully"
else
    echo "❌ Recovery failed - check logs"
    sudo systemctl status strands-agent
fi
```

## Deployment Checklist

### Pre-Deployment

- [ ] Configuration files validated
- [ ] Environment variables set
- [ ] Dependencies installed
- [ ] Database schema created
- [ ] SSL certificates configured
- [ ] Firewall rules configured
- [ ] Backup procedures tested
- [ ] Monitoring configured
- [ ] Health checks working

### Deployment

- [ ] Application deployed
- [ ] Services started
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] SSL certificates active
- [ ] Monitoring active
- [ ] Alerts configured

### Post-Deployment

- [ ] End-to-end testing completed
- [ ] Performance testing passed
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Runbooks updated
- [ ] Backup verified
- [ ] Monitoring dashboards configured

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check system health and metrics
- Review error logs
- Verify backup completion
- Monitor resource usage

**Weekly:**
- Update dependencies
- Review performance metrics
- Clean up old logs and checkpoints
- Test backup recovery procedures

**Monthly:**
- Security updates
- Performance optimization review
- Capacity planning review
- Documentation updates

### Maintenance Scripts

```bash
# scripts/maintenance.sh
#!/bin/bash

echo "Starting maintenance tasks..."

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clean old checkpoints
find checkpoints/ -name "*.json" -mtime +7 -delete

# Update dependencies
pip install --upgrade -r requirements.txt

# Run health checks
python -c "
from supervisor.supervisor import Supervisor
supervisor = Supervisor('config.yaml')
health = supervisor.get_system_health()
print(f'System health: {health}')
"

# Restart service if needed
if [ "$RESTART_REQUIRED" = "true" ]; then
    sudo systemctl restart strands-agent
fi

echo "Maintenance completed"