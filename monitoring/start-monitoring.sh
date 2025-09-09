#!/bin/bash

# Start Monitoring Stack for AI Object Counting Application
# This script starts Prometheus and Grafana for monitoring the application

echo "🚀 Starting AI Object Counter Monitoring Stack..."
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if the backend is running
echo "🔍 Checking if backend is running..."
if ! curl -s http://localhost:5000/health > /dev/null; then
    echo "⚠️  Backend is not running on localhost:5000"
    echo "   Please start the backend first with: python start_development.py"
    echo "   Or start it in the background and continue..."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the monitoring stack
echo "📊 Starting Prometheus and Grafana..."
cd "$(dirname "$0")"
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."

if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is running at http://localhost:9090"
else
    echo "❌ Prometheus failed to start"
fi

if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ Grafana is running at http://localhost:3001"
    echo "   Default credentials: admin / admin123"
else
    echo "❌ Grafana failed to start"
fi

echo ""
echo "🎉 Monitoring stack started!"
echo "=========================="
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana:    http://localhost:3001 (admin/admin123)"
echo "🔧 Backend:    http://localhost:5000"
echo ""
echo "📋 Useful commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop stack:    docker-compose down"
echo "   Restart:       docker-compose restart"
echo ""
echo "📊 Dashboard: The AI Object Counter dashboard should be available in Grafana"
echo "   Navigate to: Dashboards > AI Object Counter > AI Object Counter - Performance Dashboard"
