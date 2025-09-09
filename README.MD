# AI Object Counting Application

A full-stack AI-powered object counting application that uses advanced computer vision models (SAM + Transformers) to detect and count objects in images. The application supports single image processing, batch processing, few-shot learning, and comprehensive monitoring.

## 🚀 Features

- **AI-Powered Object Detection**: Uses SAM (Segment Anything Model) and ResNet for accurate object detection
- **Few-Shot Learning**: Register custom object types with minimal training examples
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Monitoring**: Performance metrics and system health monitoring
- **User Feedback System**: Submit corrections to improve model accuracy
- **RESTful API**: Comprehensive API with Swagger documentation
- **Modern Frontend**: React/TypeScript frontend with Tailwind CSS
- **Docker Support**: Easy deployment with Docker Compose
- **Multiple Database Support**: SQLite (development) and MySQL/PostgreSQL (production)

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 16 or higher (for frontend development)
- **Docker & Docker Compose**: For containerized deployment
- **Memory**: Minimum 4GB RAM (8GB recommended for AI models)
- **Storage**: Minimum 10GB free space

## 🏃‍♂️ Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-object-counting-app
   ```

2. **Start all services**:
   ```bash
   docker compose up --build
   ```

3. **Access the application**:
   - Backend API: http://localhost:5000
   - Frontend UI: http://localhost:3000
   - API Documentation: http://localhost:5000/apidocs

### Option 2: Local Development

1. **Setup Backend**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start development server
   python start_development.py
   ```

2. **Setup Frontend** (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Access the application**:
   - Backend API: http://localhost:5000
   - Frontend UI: http://localhost:3000

## 📚 API Documentation

### Core Endpoints

#### Object Detection
- `POST /api/count` - Count objects in a single image
- `POST /api/count-all` - Auto-detect all objects in an image
- `POST /api/batch/process` - Process multiple images in batch

#### Object Types Management
- `GET /api/object-types` - Get all available object types
- `POST /api/object-types` - Create new object type
- `GET /api/object/<id>` - Get specific object type
- `PUT /api/object/<id>` - Update object type
- `DELETE /api/object/<id>` - Delete object type

#### Results Management
- `GET /api/results` - Get all prediction results (paginated)
- `GET /api/results/<id>` - Get specific result details
- `PUT /api/correct/<id>` - Submit correction for prediction
- `DELETE /api/results/<id>` - Delete result

#### Few-Shot Learning
- `POST /api/fewshot/register` - Register new object type with few-shot learning
- `POST /api/fewshot/count` - Count objects using few-shot classification
- `GET /api/fewshot/object-types` - List all few-shot object types
- `GET /api/fewshot/object-types/<name>` - Get specific few-shot object type
- `DELETE /api/fewshot/object-types/<name>` - Delete few-shot object type
- `GET /api/fewshot/predictions` - Get few-shot prediction history

#### Performance Monitoring
- `GET /api/performance/metrics` - Get current performance metrics
- `GET /api/performance/object-types` - Get statistics by object type
- `GET /api/performance/database` - Get database statistics
- `GET /api/performance/health` - Get comprehensive system health
- `POST /api/performance/reset` - Reset performance statistics

### Example Usage

#### Count Objects in an Image
```bash
curl -X POST http://localhost:5000/api/count \
  -F "image=@your_image.jpg" \
  -F "object_type=person" \
  -F "description=Count people in this image"
```

#### Register Custom Object Type (Few-Shot Learning)
```bash
curl -X POST http://localhost:5000/api/fewshot/register \
  -F "object_name=my_custom_object" \
  -F "description=Custom object for counting" \
  -F "support_images=@image1.jpg" \
  -F "support_images=@image2.jpg" \
  -F "support_images=@image3.jpg"
```

#### Batch Processing
```bash
curl -X POST http://localhost:5000/api/batch/process \
  -F "images[]=@image1.jpg" \
  -F "images[]=@image2.jpg" \
  -F "images[]=@image3.jpg" \
  -F "object_type=car" \
  -F "description=Batch processing cars"
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_api/ -v
python -m pytest tests/test_storage/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Run Quick Test
```bash
# Quick test with minimal images
python quick_test.py
```

### Run Complete Workflow Test
```bash
# Complete test with AI generation and few-shot learning
python test_complete_workflow.py
```

## 🐳 Docker Configuration

### Docker Compose Services

- **ai-object-counter**: Main Flask application
- **mysql**: MySQL database server
- **frontend**: React frontend (optional)

### Environment Variables

Create a `.env` file for Docker Compose:

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

### Docker Commands

```bash
# Build and start services
docker compose up --build

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild specific service
docker compose up --build ai-object-counter
```

## 📁 Project Structure

```
ai-object-counting-app/
├── src/                          # Backend source code
│   ├── api/                      # API routes and views
│   │   ├── views/                # API endpoint implementations
│   │   ├── serializers/          # Data serialization
│   │   └── utils/                # API utilities
│   ├── pipeline/                 # AI processing pipeline
│   │   ├── pipeline.py           # Main AI pipeline
│   │   ├── fewshot_service.py    # Few-shot learning service
│   │   └── postprocess.py        # Post-processing utilities
│   ├── storage/                  # Database models and storage
│   │   ├── base_model.py         # Base database model
│   │   ├── inputs.py             # Input model
│   │   ├── outputs.py            # Output model
│   │   ├── object_types.py       # Object type model
│   │   └── fewshot_models.py     # Few-shot learning models
│   ├── monitoring/               # Performance monitoring
│   │   ├── metrics.py            # Metrics collection
│   │   └── decorators.py         # Monitoring decorators
│   ├── app.py                    # Flask application
│   └── config.py                 # Configuration management
├── frontend/                     # React frontend
│   ├── src/                      # Frontend source code
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   ├── services/             # API services
│   │   └── styles/               # CSS styles
│   ├── package.json              # Frontend dependencies
│   └── Dockerfile                # Frontend Docker configuration
├── tests/                        # Test suite
│   ├── test_api/                 # API tests
│   ├── test_storage/             # Database tests
│   └── test_complete_workflow.py # Integration tests
├── docs/                         # Documentation
│   ├── app_architecture.md       # Application architecture
│   ├── DB_architecture.md        # Database architecture
│   └── fewshot_learning_integration.md # Few-shot learning docs
├── monitoring/                   # Monitoring configuration
│   ├── prometheus.yml            # Prometheus configuration
│   └── grafana/                  # Grafana dashboards
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Backend Docker configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Copy `environment_config.example` to `.env` and configure:

```bash
# Application Settings
OBJ_DETECT_ENV=development
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_TYPE=sqlite  # or mysql, postgresql
MYSQL_HOST=localhost
MYSQL_USER=obj_detect_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=obj_detect_db

# AI Configuration
AI_DEVICE=cpu  # or cuda, mps
MODEL_DIRECTORY=models

# File Storage
MEDIA_DIRECTORY=media
MAX_FILE_SIZE=10485760  # 10MB

# Performance
MAX_BATCH_SIZE=10
PROCESSING_TIMEOUT=120
```

### Database Setup

#### SQLite (Development)
No setup required - automatically created.

#### MySQL (Production)
```sql
CREATE DATABASE obj_detect_db;
CREATE USER 'obj_detect_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON obj_detect_db.* TO 'obj_detect_user'@'localhost';
FLUSH PRIVILEGES;
```

## 📊 Monitoring

### Performance Metrics
- Request processing times
- Success/failure rates
- Object type statistics
- Database performance
- System health scores

### Health Checks
- `GET /api/performance/health` - Comprehensive system health
- `GET /metrics` - Prometheus metrics endpoint

### Logging
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`
- Development logs: `logs/dev_app.log`

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
   ```bash
   cp environment_config.example .env
   # Edit .env with production values
   ```

2. **Database Setup**:
   ```bash
   # For MySQL
   mysql -u root -p < create_db.sql
   ```

3. **Start Production Server**:
   ```bash
   python start_production.py
   ```

### Docker Production Deployment

```bash
# Build and deploy
docker compose -f docker-compose.yml up -d

# Check status
docker compose ps

# View logs
docker compose logs -f ai-object-counter
```

## 🔍 Troubleshooting

### Common Issues

1. **AI Models Not Loading**:
   ```bash
   # Check model directory
   ls -la models/
   
   # Download models manually
   python -c "from src.pipeline.pipeline import pipeline; pipeline._load_models()"
   ```

2. **Database Connection Issues**:
   ```bash
   # Test database connection
   python -c "from src.storage import database; print('Database connected')"
   ```

3. **File Upload Issues**:
   ```bash
   # Check media directory permissions
   ls -la media/
   chmod 755 media/
   ```

4. **Memory Issues**:
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch size
   export MAX_BATCH_SIZE=5
   ```

### Performance Optimization

1. **Database Optimization**:
   - Use connection pooling
   - Add database indexes
   - Regular maintenance

2. **AI Model Optimization**:
   - Use GPU if available
   - Implement model caching
   - Optimize batch processing

3. **File Storage Optimization**:
   - Use external storage (S3, GCS)
   - Implement file compression
   - Regular cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python -m pytest tests/`
6. Commit changes: `git commit -m "Add feature"`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the logs: `tail -f logs/app.log`
2. Review API documentation: http://localhost:5000/apidocs
3. Check system resources: `htop`, `df -h`
4. Verify configuration: `python -c "from src.config import config; print(config.__dict__)"`

## 🎯 Roadmap

- [ ] GPU acceleration support
- [ ] Real-time video processing
- [ ] Advanced analytics dashboard
- [ ] Multi-tenant support
- [ ] API rate limiting
- [ ] Advanced caching strategies
- [ ] Mobile app integration

---

**Happy Object Counting! 🎉**
