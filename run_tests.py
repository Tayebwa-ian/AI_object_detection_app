#!/usr/bin/python3
"""
Test Runner Script for AI Object Counting Application with Coverage (unittest)
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def setup_test_environment():
    """Setup environment variables for testing."""
    os.environ.setdefault('OBJ_DETECT_ENV', 'testing')
    os.environ.setdefault('OBJ_DETECT_MYSQL_DB', 'test_obj_detect.db')
    os.environ.setdefault('FLASK_DEBUG', 'False')
    os.environ.setdefault('LOG_LEVEL', 'DEBUG')
    print("Test environment configured")


def setup_directories():
    """Create necessary directories for testing."""
    for directory in ['test_media', 'logs', 'models']:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def setup_logging():
    """Setup test logging."""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/test_app.log'),
            logging.StreamHandler()
        ]
    )
    print("Test logging configured")


def run_tests_with_coverage():
    """Run unittest test suite with coverage and display reports (lines + branches)."""
    print("Running unittest suite with coverage (lines + branches)...\n")
    try:
        # Run unittest suite with coverage (branch coverage included)
        result = subprocess.run(
            [
                "coverage", "run", "--branch", "--source=src", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"
            ],
            check=False
        )

        # Generate terminal report with missing lines highlighted
        subprocess.run(["coverage", "report", "-m", "--show-missing"])

        sys.exit(result.returncode)

    except FileNotFoundError:
        logging.error("coverage is not installed. Run `pip install coverage`.")
        print("Error: coverage is not installed. Run `pip install coverage`.")
        sys.exit(1)



def main():
    print("AI Object Counting Application - Test Runner with Coverage (unittest)")
    print("=" * 70)
    setup_test_environment()
    setup_directories()
    setup_logging()
    print("Test environment ready!\n")
    run_tests_with_coverage()


if __name__ == "__main__":
    main()
