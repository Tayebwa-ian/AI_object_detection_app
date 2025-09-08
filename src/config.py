#!/usr/bin/python3
"""
Configuration Management for AI Object Counting Application
"""
import os
import logging
from pathlib import Path
from typing import Optional, List

class Config:
    """Base configuration class"""
    
    # Application Settings
    ENV = os.getenv('OBJ_DETECT_ENV', 'development')
    HOST = os.getenv('OBJ_DETECT_API_HOST', '0.0.0.0')
    PORT = int(os.getenv('OBJ_DETECT_API_PORT', '5000'))
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')
    DATABASE_PATH = os.getenv('OBJ_DETECT_MYSQL_DB', 'obj_detect.db')
    
    # MySQL Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'obj_detect_user')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'obj_detect_db')
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'obj_detect_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE', 'obj_detect_db')
    
    # File Storage Configuration
    MEDIA_DIRECTORY = os.getenv('MEDIA_DIRECTORY', 'media')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10485760'))  # 10MB
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,bmp,gif').split(',')
    
    # AI Model Configuration
    AI_DEVICE = os.getenv('AI_DEVICE', 'cpu')
    MODEL_DIRECTORY = os.getenv('MODEL_DIRECTORY', 'models')
    SAM_MODEL_TYPE = os.getenv('SAM_MODEL_TYPE', 'vit_b')
    SAM_CHECKPOINT_URL = os.getenv('SAM_CHECKPOINT_URL', 
        'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')
    TRANSFORMERS_MODEL = os.getenv('TRANSFORMERS_MODEL', 'google/vit-base-patch16-224')
    # Performance tuning (safe defaults for CPU)
    FAST_MODE = os.getenv('FAST_MODE', 'True').lower() == 'true'
    MAX_IMAGE_DIM = int(os.getenv('MAX_IMAGE_DIM', '1024'))  # downscale long edge
    SAM_POINTS_PER_SIDE = int(os.getenv('SAM_POINTS_PER_SIDE', '8'))
    SAM_PRED_IOU_THRESH = float(os.getenv('SAM_PRED_IOU_THRESH', '0.8'))
    SAM_STABILITY_SCORE_THRESH = float(os.getenv('SAM_STABILITY_SCORE_THRESH', '0.9'))
    SAM_MIN_MASK_REGION_AREA = int(os.getenv('SAM_MIN_MASK_REGION_AREA', '2000'))
    TOP_SEGMENTS = int(os.getenv('TOP_SEGMENTS', '15'))
    
    # Performance Configuration
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '10'))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))
    PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT', '120'))
    BATCH_PROCESSING_TIMEOUT = int(os.getenv('BATCH_PROCESSING_TIMEOUT', '300'))
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # Monitoring Configuration
    ENABLE_PERFORMANCE_MONITORING = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'True').lower() == 'true'
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
    
    # Frontend Configuration
    FRONTEND_BUILD_DIR = os.getenv('FRONTEND_BUILD_DIR', 'frontend/dist')
    FRONTEND_DEV_PORT = int(os.getenv('FRONTEND_DEV_PORT', '3000'))
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL based on configuration"""
        if cls.DATABASE_TYPE == 'mysql':
            return f"mysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
        elif cls.DATABASE_TYPE == 'postgresql':
            return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DATABASE}"
        else:  # sqlite
            return f"sqlite:///{cls.DATABASE_PATH}"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MEDIA_DIRECTORY,
            cls.MODEL_DIRECTORY,
            os.path.dirname(cls.LOG_FILE) if cls.LOG_FILE else 'logs'
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(cls.LOG_FILE) if cls.LOG_FILE else 'logs'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler() if cls.DEBUG else logging.NullHandler()
            ]
        )
        
        # Set specific loggers
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    DATABASE_PATH = 'dev_obj_detect.db'
    MEDIA_DIRECTORY = 'dev_media'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = False
    DATABASE_PATH = 'test_obj_detect.db'
    MEDIA_DIRECTORY = 'test_media'
    LOG_LEVEL = 'WARNING'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    
    def __init__(self):
        super().__init__()
        
        # Production-specific settings
        if not self.SECRET_KEY or self.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("SECRET_KEY must be set in production environment")
        
        # Use external database in production
        if self.DATABASE_TYPE == 'sqlite':
            raise ValueError("SQLite should not be used in production. Use MySQL or PostgreSQL.")

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('OBJ_DETECT_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global configuration instance
config = get_config()

# Setup directories and logging
config.setup_directories()
config.setup_logging()
