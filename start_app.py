#!/usr/bin/env python3
"""
Simple Startup Script for AI Object Counting Application

This script helps your friend get started quickly with the application.
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 AI Object Counting Application")
    print("=" * 60)
    print("Welcome! This script will help you get started.")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("⚠️  Warning: Python 3.9+ recommended")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    """Check if dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    try:
        import flask
        import torch
        import transformers
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def run_simple_test():
    """Run simple test to verify everything works"""
    print("\n🧪 Running simple test...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_simple.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Simple test passed!")
            return True
        else:
            print("❌ Simple test failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def start_development_server():
    """Start the development server"""
    print("\n🚀 Starting development server...")
    print("The server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start the development server
        subprocess.run([sys.executable, "start_development.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to version 3.9 or higher")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first")
        return 1
    
    # Run simple test
    if not run_simple_test():
        print("\n❌ Simple test failed. Please check the errors above.")
        return 1
    
    print("\n🎉 Everything looks good!")
    print("Your AI Object Counting Application is ready to use!")
    
    # Ask if user wants to start the server
    print("\n" + "=" * 60)
    response = input("Would you like to start the development server now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        start_development_server()
    else:
        print("\n👋 You can start the server later by running:")
        print("   python start_development.py")
        print("\nOr use Docker:")
        print("   docker compose up --build")
    
    return 0

if __name__ == '__main__':
    exit(main())
