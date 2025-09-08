#!/usr/bin/python3
"""
Production Startup Script for AI Object Counting Application
"""
import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_environment():
    """Setup production environment"""
    # Set production environment
    os.environ['OBJ_DETECT_ENV'] = 'production'
    
    # Ensure required environment variables are set
    required_vars = [
        'SECRET_KEY',
        'DATABASE_TYPE'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before starting the application.")
        sys.exit(1)
    
    # Validate database configuration
    database_type = os.getenv('DATABASE_TYPE', 'sqlite')
    if database_type == 'sqlite':
        print("⚠️  Warning: Using SQLite in production is not recommended.")
        print("Consider using MySQL or PostgreSQL for better performance and reliability.")
    
    print("✅ Environment validation passed")

def setup_directories():
    """Create necessary directories"""
    directories = [
        'media',
        'logs',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def setup_logging():
    """Setup production logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"✅ Logging configured: {log_level} level, file: {log_file}")

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import flask
        import sqlalchemy
        import torch
        import transformers
        import segment_anything
        print("✅ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install all requirements: pip install -r requirements.txt")
        return False

def start_application():
    """Start the Flask application"""
    try:
        from src.app import app
        from src.config import config
        
        print("🚀 Starting AI Object Counting Application...")
        print(f"   Environment: {config.ENV}")
        print(f"   Host: {config.HOST}")
        print(f"   Port: {config.PORT}")
        print(f"   Database: {config.DATABASE_TYPE}")
        print(f"   Debug: {config.DEBUG}")
        
        # Start the application
        app.run(
            host=config.HOST,
            port=config.PORT,
            threaded=True,
            debug=config.DEBUG
        )
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🔧 AI Object Counting Application - Production Startup")
    print("=" * 60)
    
    # Setup steps
    setup_environment()
    setup_directories()
    setup_logging()
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🎉 All startup checks passed!")
    print("=" * 60)
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()








