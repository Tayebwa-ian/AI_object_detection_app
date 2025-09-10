#!/usr/bin/env python3
"""
Basic Test Suite for AI Object Counting Application

This is a very simple test that checks basic functionality without complex imports.
"""

import unittest
import sys
import os

def test_basic_functionality():
    """Test basic functionality without complex imports"""
    print("Running Basic Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check if required files exist
    total_tests += 1
    print("Test 1: Checking required files...")
    required_files = [
        'src/app.py',
        'src/config.py',
        'requirements.txt',
        'README.md',
        'docker-compose.yml',
        'Dockerfile'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"X Missing file: {file_path}")
            all_files_exist = False
        else:
            print(f"OK Found: {file_path}")
    
    if all_files_exist:
        print("OK All required files exist")
        tests_passed += 1
    else:
        print("X Some required files are missing")
    
    # Test 2: Check if directories exist
    total_tests += 1
    print("\nTest 2: Checking required directories...")
    required_dirs = [
        'src',
        'src/api',
        'src/pipeline',
        'src/storage',
        'tests',
        'frontend'
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            print(f"X Missing directory: {dir_path}")
            all_dirs_exist = False
        else:
            print(f"OK Found directory: {dir_path}")
    
    if all_dirs_exist:
        print("OK All required directories exist")
        tests_passed += 1
    else:
        print("X Some required directories are missing")
    
    # Test 3: Check if Python files are valid
    total_tests += 1
    print("\nTest 3: Checking Python file syntax...")
    python_files = [
        'src/app.py',
        'src/config.py',
        'start_development.py',
        'start_production.py'
    ]
    
    syntax_ok = True
    for py_file in python_files:
        if os.path.exists(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
                print(f"OK Syntax OK: {py_file}")
            except SyntaxError as e:
                print(f"X Syntax error in {py_file}: {e}")
                syntax_ok = False
            except Exception as e:
                print(f"! Could not check {py_file}: {e}")
        else:
            print(f"! File not found: {py_file}")
    
    if syntax_ok:
        print("OK All Python files have valid syntax")
        tests_passed += 1
    else:
        print("X Some Python files have syntax errors")
    
    # Test 4: Check if requirements.txt has content
    total_tests += 1
    print("\nTest 4: Checking requirements.txt...")
    if os.path.exists('requirements.txt'):
        try:
            with open('requirements.txt', 'r') as f:
                content = f.read().strip()
                if content and len(content) > 10:
                    print("OK requirements.txt has content")
                    tests_passed += 1
                else:
                    print("X requirements.txt is empty or too short")
        except Exception as e:
            print(f"X Could not read requirements.txt: {e}")
    else:
        print("X requirements.txt not found")
    
    # Test 5: Check if README has content
    total_tests += 1
    print("\nTest 5: Checking README.md...")
    if os.path.exists('README.md'):
        try:
            with open('README.md', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and len(content) > 100:
                    print("OK README.md has substantial content")
                    tests_passed += 1
                else:
                    print("X README.md is empty or too short")
        except Exception as e:
            print(f"X Could not read README.md: {e}")
    else:
        print("X README.md not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\nSUCCESS: All basic tests passed!")
        print("OK Your application structure looks good!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the app: python start_development.py")
        print("3. Or use Docker: docker compose up --build")
        return True
    else:
        print(f"\nWARNING: {total_tests - tests_passed} test(s) failed.")
        print("Please check the issues above.")
        return False

if __name__ == '__main__':
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
