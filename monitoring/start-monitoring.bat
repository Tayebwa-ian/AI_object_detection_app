@echo off
REM Start Monitoring Stack for AI Object Counting Application
REM This script starts Prometheus and Grafana for monitoring the application

echo 🚀 Starting AI Object Counter Monitoring Stack...
echo ================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if the backend is running
echo 🔍 Checking if backend is running...
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Backend is not running on localhost:5000
    echo    Please start the backend first with: python start_development.py
    echo    Or start it in the background and continue...
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
)

REM Start the monitoring stack
echo 📊 Starting Prometheus and Grafana...
cd /d "%~dp0"
docker-compose up -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo 🔍 Checking service status...

curl -s http://localhost:9090/-/healthy >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus is running at http://localhost:9090
) else (
    echo ❌ Prometheus failed to start
)

curl -s http://localhost:3001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Grafana is running at http://localhost:3001
    echo    Default credentials: admin / admin123
) else (
    echo ❌ Grafana failed to start
)

echo.
echo 🎉 Monitoring stack started!
echo ==========================
echo 📊 Prometheus: http://localhost:9090
echo 📈 Grafana:    http://localhost:3001 (admin/admin123)
echo 🔧 Backend:    http://localhost:5000
echo.
echo 📋 Useful commands:
echo    View logs:     docker-compose logs -f
echo    Stop stack:    docker-compose down
echo    Restart:       docker-compose restart
echo.
echo 📊 Dashboard: The AI Object Counter dashboard should be available in Grafana
echo    Navigate to: Dashboards ^> AI Object Counter ^> AI Object Counter - Performance Dashboard

pause
