# Getting Started with AI Object Counting Application

Welcome! This guide will help you get up and running with the AI Object Counting Application quickly.

## 🚀 Quick Start (Choose One)

### Option 1: Super Easy Start
```bash
python start_app.py
```
This script will check everything and start the server for you!

### Option 2: Docker (Recommended)
```bash
docker compose up --build
```
Then visit: http://localhost:5000

### Option 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python start_development.py

# In another terminal, start frontend (optional)
cd frontend
npm install
npm run dev
```

## 🧪 Test Everything Works

Run the simple test to make sure everything is working:
```bash
python tests/test_basic.py
```

Or run all tests:
```bash
python run_tests.py
```

## 📚 What You Can Do

### 1. Count Objects in Images
- Upload an image via the web interface at http://localhost:3000
- Or use the API: `POST /api/count`

### 2. Batch Processing
- Process multiple images at once: `POST /api/batch/process`

### 3. Few-Shot Learning
- Register custom object types: `POST /api/fewshot/register`
- Count with custom types: `POST /api/fewshot/count`

### 4. Monitor Performance
- Check system health: `GET /api/performance/health`
- View metrics: `GET /api/performance/metrics`

## 🔧 Configuration

Copy and edit the environment file:
```bash
cp environment_config.example .env
# Edit .env with your settings
```

## 📖 API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:5000/apidocs
- API Health: http://localhost:5000/api/performance/health

## 🆘 Need Help?

1. Check the logs: `tail -f logs/app.log`
2. Run tests: `python run_tests.py`
3. Check the main README.md for detailed documentation

## 🎯 Key Endpoints

- **Count objects**: `POST /api/count`
- **Auto-detect**: `POST /api/count-all`
- **Batch process**: `POST /api/batch/process`
- **Few-shot register**: `POST /api/fewshot/register`
- **Few-shot count**: `POST /api/fewshot/count`
- **Get results**: `GET /api/results`
- **System health**: `GET /api/performance/health`

## 🐳 Docker Commands

```bash
# Start everything
docker compose up --build

# Start with frontend
docker compose --profile frontend up --build

# Start with monitoring
docker compose --profile monitoring up --build

# Stop everything
docker compose down

# View logs
docker compose logs -f ai-object-counter
```

## 🎉 You're Ready!

The application is now ready to use. Start with the simple test, then explore the API documentation to see all available features.

Happy object counting! 🎯
