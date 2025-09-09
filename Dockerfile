# AI Object Counting Application - Docker Configuration
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OBJ_DETECT_ENV=development

# Set work directory
WORKDIR /app

# Install system dependencies (OpenCV + MySQL client + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY run_tests.py .
COPY create_db.sql .
COPY start_development.py .
COPY environment_config.example .env.example

# Create necessary directories
RUN mkdir -p media logs models

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start command
CMD ["python", "start_development.py"]
