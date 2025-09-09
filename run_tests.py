#!/usr/bin/env python3
"""
Test Runner for AI Object Counting Application

This script runs all tests in the project and provides a comprehensive test report.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_environment():
    """Check if the environment is set up correctly"""
    print("🔍 Checking Environment Setup")
    print("-" * 40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("⚠️  Warning: Python 3.9+ recommended")
    
    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('tests'):
        print("❌ Error: Please run this script from the project root directory")
        return False
    
    # Check if requirements are installed
    try:
        import flask
        import pytest
        print("✅ Required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def run_unit_tests():
    """Run unit tests"""
    print("\n🚀 Running Unit Tests")
    print("=" * 60)
    
    # Run pytest on the tests directory
    success = run_command(
        "python -m pytest tests/ -v --tb=short",
        "Unit Tests (pytest)"
    )
    
    return success

def run_comprehensive_api_tests():
    """Run comprehensive API tests"""
    print("\n🚀 Running Comprehensive API Tests")
    print("=" * 60)
    
    # Run the comprehensive test suite
    success = run_command(
        "python tests/test_comprehensive_api.py",
        "Comprehensive API Test Suite"
    )
    
    return success

def run_quick_tests():
    """Run quick tests"""
    print("\n🚀 Running Quick Tests")
    print("=" * 60)
    
    # Run quick test if it exists
    if os.path.exists('quick_test.py'):
        success = run_command(
            "python quick_test.py",
            "Quick Test Suite"
        )
    else:
        print("⚠️  quick_test.py not found, skipping")
        success = True
    
    return success

def run_workflow_tests():
    """Run workflow tests"""
    print("\n🚀 Running Workflow Tests")
    print("=" * 60)
    
    # Run complete workflow test if it exists
    if os.path.exists('test_complete_workflow.py'):
        success = run_command(
            "python test_complete_workflow.py",
            "Complete Workflow Test"
        )
    else:
        print("⚠️  test_complete_workflow.py not found, skipping")
        success = True
    
    return success

def run_coverage_tests():
    """Run tests with coverage"""
    print("\n🚀 Running Coverage Tests")
    print("=" * 60)
    
    # Check if coverage is available
    try:
        import coverage
        success = run_command(
            "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term",
            "Coverage Analysis"
        )
        
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    except ImportError:
        print("⚠️  Coverage not available, install with: pip install coverage")
        success = True
    
    return success

def generate_test_report(results):
    """Generate a test report"""
    print("\n" + "=" * 60)
    print("📊 TEST REPORT SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 40)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The application is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed.")
        print("Check the output above for details.")
        return False

def main():
    """Main test runner function"""
    print("🧪 AI Object Counting Application - Test Runner")
    print("=" * 60)
    print("This script will run all available tests for the application.")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return 1
    
    # Run all tests
    results = {}
    
    # Unit tests
    results['Unit Tests'] = run_unit_tests()
    
    # Comprehensive API tests
    results['Comprehensive API Tests'] = run_comprehensive_api_tests()
    
    # Quick tests
    results['Quick Tests'] = run_quick_tests()
    
    # Workflow tests
    results['Workflow Tests'] = run_workflow_tests()
    
    # Coverage tests
    results['Coverage Tests'] = run_coverage_tests()
    
    # Generate report
    success = generate_test_report(results)
    
    print("\n" + "=" * 60)
    print("🏁 Test Runner Complete")
    print("=" * 60)
    
    if success:
        print("✅ All tests passed successfully!")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1

if __name__ == '__main__':
    exit(main())
