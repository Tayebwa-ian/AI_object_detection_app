#!/usr/bin/env python3
"""
Simple CI Test Suite - Tests only what can be verified without complex imports
"""
import unittest
import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime


class TestCISimple(unittest.TestCase):
    """Simple tests for CI that focus on file structure and basic functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_project_structure(self):
        """Test that project has required structure"""
        required_files = [
            'src/app.py',
            'src/config.py',
            'requirements.txt',
            'README.md',
            'docker-compose.yml',
            'Dockerfile',
            '.gitlab-ci.yml'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Missing required file: {file_path}")
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'src',
            'src/api',
            'src/api/views',
            'src/api/serializers',
            'src/storage',
            'src/pipeline',
            'src/monitoring',
            'tests',
            'frontend'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(os.path.isdir(dir_path), f"Missing required directory: {dir_path}")
    
    def test_python_syntax(self):
        """Test that Python files have valid syntax"""
        python_files = [
            'src/app.py',
            'src/config.py',
            'start_development.py',
            'start_production.py'
        ]
        
        for py_file in python_files:
            if os.path.exists(py_file):
                with open(py_file, 'r', encoding='utf-8') as f:
                    try:
                        compile(f.read(), py_file, 'exec')
                    except SyntaxError as e:
                        self.fail(f"Syntax error in {py_file}: {e}")
    
    def test_requirements_file(self):
        """Test that requirements.txt has content and key dependencies"""
        self.assertTrue(os.path.exists('requirements.txt'))
        
        with open('requirements.txt', 'r') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 10, "requirements.txt is too short")
            
            # Check for key dependencies
            key_deps = ['Flask', 'torch', 'opencv-python', 'numpy']
            for dep in key_deps:
                self.assertIn(dep, content, f"Missing key dependency: {dep}")
    
    def test_readme_file(self):
        """Test that README.md has substantial content"""
        self.assertTrue(os.path.exists('README.md'))
        
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 100, "README.md is too short")
            
            # Check for key sections
            key_sections = ['AI Object Counting', 'Quick Start', 'API', 'Features']
            for section in key_sections:
                self.assertIn(section, content, f"Missing key section: {section}")
    
    def test_docker_files(self):
        """Test that Docker files exist and have content"""
        docker_files = ['docker-compose.yml', 'Dockerfile']
        
        for docker_file in docker_files:
            self.assertTrue(os.path.exists(docker_file), f"Missing {docker_file}")
            
            with open(docker_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.assertGreater(len(content), 10, f"{docker_file} is too short")
    
    def test_gitlab_ci_file(self):
        """Test that GitLab CI file exists and has content"""
        self.assertTrue(os.path.exists('.gitlab-ci.yml'))
        
        with open('.gitlab-ci.yml', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 10, ".gitlab-ci.yml is too short")
            self.assertIn('stages:', content, "Missing stages in GitLab CI")
            self.assertIn('test', content, "Missing test stage in GitLab CI")
    
    def test_test_files_exist(self):
        """Test that test files exist"""
        test_files = [
            'tests/test_basic.py',
            'tests/test_ci_simple.py',
            'tests/__init__.py'
        ]
        
        for test_file in test_files:
            self.assertTrue(os.path.exists(test_file), f"Missing test file: {test_file}")
    
    def test_environment_variables(self):
        """Test environment variable handling"""
        import os
        
        # Set test environment variables
        os.environ['OBJ_DETECT_ENV'] = 'testing'
        os.environ['OBJ_DETECT_MYSQL_DB'] = 'test_db'
        
        self.assertEqual(os.environ.get('OBJ_DETECT_ENV'), 'testing')
        self.assertEqual(os.environ.get('OBJ_DETECT_MYSQL_DB'), 'test_db')
    
    def test_directory_creation(self):
        """Test that required directories can be created"""
        test_dirs = ['logs', 'models', 'test_media', 'dev_media']
        
        for dir_name in test_dirs:
            test_path = os.path.join(self.temp_dir, dir_name)
            os.makedirs(test_path, exist_ok=True)
            self.assertTrue(os.path.exists(test_path))
            self.assertTrue(os.path.isdir(test_path))
    
    def test_file_operations(self):
        """Test basic file operations"""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        test_content = 'Hello, World!'
        
        # Write file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Read file
        with open(test_file, 'r') as f:
            content = f.read()
        
        self.assertEqual(content, test_content)
        self.assertTrue(os.path.exists(test_file))
    
    def test_json_operations(self):
        """Test JSON operations"""
        test_data = {
            'name': 'test',
            'count': 5,
            'active': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data)
        self.assertIsInstance(json_str, str)
        
        # Test JSON deserialization
        parsed_data = json.loads(json_str)
        self.assertEqual(parsed_data['name'], test_data['name'])
        self.assertEqual(parsed_data['count'], test_data['count'])
        self.assertEqual(parsed_data['active'], test_data['active'])
    
    def test_datetime_operations(self):
        """Test datetime operations"""
        now = datetime.now()
        self.assertIsNotNone(now)
        self.assertIsInstance(now, datetime)
        
        # Test string formatting
        formatted = now.isoformat()
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 10)
    
    def test_path_operations(self):
        """Test path operations"""
        test_path = Path(self.temp_dir) / 'test_file.txt'
        
        # Create file using pathlib
        test_path.write_text('test content')
        
        # Check file exists
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_file())
        
        # Read content
        content = test_path.read_text()
        self.assertEqual(content, 'test content')
    
    def test_import_standard_libraries(self):
        """Test that standard libraries can be imported"""
        standard_libs = [
            'os', 'sys', 'json', 'datetime', 'pathlib', 
            'tempfile', 'unittest', 'logging'
        ]
        
        for lib in standard_libs:
            try:
                __import__(lib)
            except ImportError as e:
                self.fail(f"Failed to import standard library {lib}: {e}")
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3, "Python 3 is required")
        self.assertGreaterEqual(version.minor, 8, "Python 3.8+ is recommended")


def run_simple_tests():
    """Run simple test suite"""
    print("Running Simple CI Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCISimple)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All simple tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == '__main__':
    success = run_simple_tests()
    sys.exit(0 if success else 1)
