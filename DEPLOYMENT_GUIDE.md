# AI Object Counting Application - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AI Object Counting Application in various environments, from development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Development Deployment](#development-deployment)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Database Setup](#database-setup)
7. [Frontend Deployment](#frontend-deployment)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Node.js**: 16 or higher (for frontend)
- **Memory**: Minimum 4GB RAM (8GB recommended for AI models)
- **Storage**: Minimum 10GB free space
- **Database**: MySQL 8.0+ or PostgreSQL 12+ (for production)

### Required Software

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
```

## Environment Configuration

### 1. Environment Variables

Copy the example configuration file:

```bash
cp environment_config.example .env
```

Edit `.env` with your specific configuration:

```bash
# Application Settings
OBJ_DETECT_ENV=production
OBJ_DETECT_API_HOST=0.0.0.0
OBJ_DETECT_API_PORT=5000

# Database Configuration
DATABASE_TYPE=mysql
MYSQL_HOST=localhost
MYSQL_USER=obj_detect_user
MYSQL_PASSWORD=your_secure_password
MYSQL_DATABASE=obj_detect_db

# Security
SECRET_KEY=your-production-secret-key-here

# File Storage
MEDIA_DIRECTORY=media
MAX_FILE_SIZE=10485760

# AI Configuration
AI_DEVICE=cpu
```

### 2. Required Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OBJ_DETECT_ENV` | Environment (development/testing/production) | Yes | development |
| `SECRET_KEY` | Flask secret key | Yes (production) | dev-secret-key |
| `DATABASE_TYPE` | Database type (sqlite/mysql/postgresql) | Yes | sqlite |
| `MYSQL_HOST` | MySQL host | Yes (if MySQL) | localhost |
| `MYSQL_USER` | MySQL username | Yes (if MySQL) | - |
| `MYSQL_PASSWORD` | MySQL password | Yes (if MySQL) | - |
| `MYSQL_DATABASE` | MySQL database name | Yes (if MySQL) | - |

## Development Deployment

### Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ai-object-counting-app
   pip install -r requirements.txt
   ```

2. **Start development server**:
   ```bash
   python start_development.py
   ```

3. **Start frontend** (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**:
   - Backend API: http://localhost:5000
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:5000/apidocs

### Development Features

- **Auto-reload**: Code changes automatically restart the server
- **Debug mode**: Detailed error messages and logging
- **SQLite database**: No external database required
- **Local media storage**: Images stored in `dev_media/` directory

## Production Deployment

### 1. System Preparation

```bash
# Create application user
sudo useradd -m -s /bin/bash ai-counter
sudo mkdir -p /opt/ai-object-counter
sudo chown ai-counter:ai-counter /opt/ai-object-counter

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3-pip nginx
```

### 2. Application Deployment

```bash
# Switch to application user
sudo su - ai-counter

# Clone application
cd /opt/ai-object-counter
git clone <repository-url> .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp environment_config.example .env
# Edit .env with production values
```

### 3. Database Setup

#### MySQL Setup

```bash
# Install MySQL
sudo apt-get install mysql-server

# Create database and user
sudo mysql -u root -p
```

```sql
CREATE DATABASE obj_detect_db;
CREATE USER 'obj_detect_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON obj_detect_db.* TO 'obj_detect_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
```

```sql
CREATE DATABASE obj_detect_db;
CREATE USER obj_detect_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE obj_detect_db TO obj_detect_user;
\q
```

### 4. Start Production Server

```bash
# Start with production script
python start_production.py
```

### 5. Process Management (Systemd)

Create systemd service file:

```bash
sudo nano /etc/systemd/system/ai-object-counter.service
```

```ini
[Unit]
Description=AI Object Counting Application
After=network.target mysql.service

[Service]
Type=simple
User=ai-counter
WorkingDirectory=/opt/ai-object-counter
Environment=PATH=/opt/ai-object-counter/venv/bin
ExecStart=/opt/ai-object-counter/venv/bin/python start_production.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-object-counter
sudo systemctl start ai-object-counter
sudo systemctl status ai-object-counter
```

## Docker Deployment

### 1. Build and Run

```bash
# Build Docker image
docker build -t ai-object-counter .

# Run with Docker Compose
docker-compose up -d
```

### 2. Docker Compose Services

- **ai-object-counter**: Main application
- **mysql**: Database server
- **frontend**: React frontend
- **nginx**: Reverse proxy (optional)

### 3. Environment Configuration

Create `.env` file for Docker Compose:

```bash
# Database
MYSQL_ROOT_PASSWORD=root_password_123
MYSQL_DATABASE=obj_detect_db
MYSQL_USER=obj_detect_user
MYSQL_PASSWORD=secure_password_123

# Application
SECRET_KEY=your-production-secret-key-here
OBJ_DETECT_ENV=production
```

## Database Setup

### 1. Initialize Database

```bash
# Run database initialization
python -c "
from src.database_config import db_config
db_config.create_tables()
print('Database tables created successfully')
"
```

### 2. Create Default Object Types

```bash
# Create default object types
python -c "
from src.storage import database, ObjectType
from src.config import config

# Create default object types
default_types = [
    'car', 'person', 'bike', 'dog', 'cat',
    'bird', 'boat', 'bus', 'truck', 'motorcycle'
]

for obj_type in default_types:
    existing = database.get(ObjectType, name=obj_type)
    if not existing:
        new_type = ObjectType(name=obj_type, description=f'Object type for {obj_type}')
        new_type.save()
        print(f'Created object type: {obj_type}')

print('Default object types created')
"
```

## Frontend Deployment

### 1. Development

```bash
cd frontend
npm install
npm run dev
```

### 2. Production Build

```bash
cd frontend
npm install
npm run build
```

### 3. Serve with Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Logging

### 1. Log Files

- **Application logs**: `logs/app.log`
- **Error logs**: `logs/error.log`
- **Access logs**: `logs/access.log`

### 2. Health Monitoring

```bash
# Check application health
curl http://localhost:5000/health

# Check performance metrics
curl http://localhost:5000/api/performance/metrics
```

### 3. Log Rotation

```bash
# Setup logrotate
sudo nano /etc/logrotate.d/ai-object-counter
```

```
/opt/ai-object-counter/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ai-counter ai-counter
    postrotate
        systemctl reload ai-object-counter
    endscript
}
```

## Troubleshooting

### Common Issues

1. **AI Models Not Loading**
   ```bash
   # Check model directory permissions
   ls -la models/
   
   # Download models manually
   python -c "from src.pipeline.pipeline import pipeline; pipeline._load_models()"
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   python -c "from src.database_config import db_config; print('Database connected')"
   ```

3. **File Upload Issues**
   ```bash
   # Check media directory permissions
   ls -la media/
   chmod 755 media/
   ```

4. **Memory Issues**
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch size in configuration
   export MAX_BATCH_SIZE=5
   ```

### Performance Optimization

1. **Database Optimization**
   - Use connection pooling
   - Add database indexes
   - Regular database maintenance

2. **AI Model Optimization**
   - Use GPU if available
   - Implement model caching
   - Optimize batch processing

3. **File Storage Optimization**
   - Use external storage (S3, GCS)
   - Implement file compression
   - Regular cleanup of old files

### Security Considerations

1. **Environment Variables**
   - Never commit `.env` files
   - Use strong secret keys
   - Rotate credentials regularly

2. **Database Security**
   - Use strong passwords
   - Limit database access
   - Enable SSL connections

3. **File Upload Security**
   - Validate file types
   - Scan for malware
   - Limit file sizes

## Support

For additional support:

1. Check the logs: `tail -f logs/app.log`
2. Review the API documentation: http://localhost:5000/apidocs
3. Check system resources: `htop`, `df -h`
4. Verify configuration: `python -c "from src.config import config; print(config.__dict__)"`

## Conclusion

This deployment guide provides comprehensive instructions for deploying the AI Object Counting Application in various environments. Follow the appropriate section based on your deployment needs and environment requirements.








