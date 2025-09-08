#!/usr/bin/python3
"""
Development Startup Script for AI Object Counting Application
"""
import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def setup_development_environment():
    """Setup environment variables for development."""
    os.environ.setdefault('OBJ_DETECT_ENV', 'development')
    os.environ.setdefault('OBJ_DETECT_MYSQL_DB', 'dev_obj_detect.db')
    os.environ.setdefault('FLASK_DEBUG', 'True')
    os.environ.setdefault('LOG_LEVEL', 'DEBUG')
    print("Development environment configured")


def setup_directories():
    """Create necessary directories for development."""
    for directory in ['dev_media', 'logs', 'models']:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def setup_logging():
    """Setup development logging."""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/dev_app.log'),
            logging.StreamHandler()
        ]
    )
    print("Development logging configured")


def start_development_server():
    """Start the Flask development server."""
    try:
        from src.app import app
        from src.config import config

        print("Starting AI Object Counting (Development Mode)...")
        print(f"  Environment: {config.ENV}")
        print(f"  Host: {config.HOST}")
        print(f"  Port: {config.PORT}")
        print(f"  Database: {config.DATABASE_TYPE} ({config.DATABASE_PATH})")
        print(f"  Debug: {config.DEBUG}")
        print(f"  Media Directory: {config.MEDIA_DIRECTORY}")

        print("\nAvailable Endpoints:")
        print("  - GET  /health")
        print("  - GET  /api/object-types")
        print("  - POST /api/count")
        print("  - POST /api/count-all")
        print("  - POST /api/batch/process")
        print("  - GET  /api/results")
        print("  - GET  /api/performance/*")

        app.run(
            host=config.HOST,
            port=config.PORT,
            threaded=True,
            debug=config.DEBUG
        )
    except Exception as e:
        logging.error(f"Failed to start development server: {e}")
        print(f"Failed to start development server: {e}")
        sys.exit(1)


def main():
    print("AI Object Counting Application - Development Startup")
    print("=" * 60)
    setup_development_environment()
    setup_directories()
    setup_logging()
    print("Development environment ready!\n")
    start_development_server()


if __name__ == "__main__":
    main()






