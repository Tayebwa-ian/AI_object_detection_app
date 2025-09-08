@echo off
echo 🚀 Setting up Frontend...
echo.

echo 📦 Installing dependencies...
call npm install

if %errorlevel% neq 0 (
    echo ❌ npm install failed. Trying with --force...
    call npm install --force
)

echo.
echo ✅ Frontend setup complete!
echo.
echo 🎯 Next steps:
echo 1. Start backend: cd ../backend && py app.py
echo 2. Start frontend: npm run dev
echo 3. Open: http://localhost:3000
echo.
pause



