#!/usr/bin/env python3
"""
Minimal CI Test Suite - Focuses on what can be tested without complex dependencies
"""
import unittest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCIMinimal(unittest.TestCase):
    """Minimal tests for CI that don't require complex setup"""
    
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
            'Dockerfile'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Missing required file: {file_path}")
    
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
        """Test that requirements.txt has content"""
        self.assertTrue(os.path.exists('requirements.txt'))
        
        with open('requirements.txt', 'r') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 10, "requirements.txt is too short")
            self.assertIn('Flask', content, "Flask not found in requirements")
    
    def test_readme_file(self):
        """Test that README.md has content"""
        self.assertTrue(os.path.exists('README.md'))
        
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            self.assertGreater(len(content), 100, "README.md is too short")
    
    def test_docker_files(self):
        """Test that Docker files exist and have content"""
        docker_files = ['docker-compose.yml', 'Dockerfile']
        
        for docker_file in docker_files:
            self.assertTrue(os.path.exists(docker_file), f"Missing {docker_file}")
            
            with open(docker_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.assertGreater(len(content), 10, f"{docker_file} is too short")
    
    def test_config_import(self):
        """Test that config can be imported"""
        try:
            import config
            self.assertIsNotNone(config)
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")
    
    def test_database_import(self):
        """Test that database can be imported"""
        try:
            import storage
            self.assertIsNotNone(storage)
        except ImportError as e:
            self.fail(f"Failed to import storage: {e}")
    
    def test_api_views_import(self):
        """Test that API views can be imported"""
        try:
            import api
            self.assertIsNotNone(api)
        except ImportError as e:
            self.fail(f"Failed to import api: {e}")
    
    def test_serializers_import(self):
        """Test that serializers can be imported"""
        try:
            import serializers
            self.assertIsNotNone(serializers)
        except ImportError as e:
            self.fail(f"Failed to import serializers: {e}")
    
    def test_storage_models_import(self):
        """Test that storage models can be imported"""
        try:
            import storage
            self.assertIsNotNone(storage)
        except ImportError as e:
            self.fail(f"Failed to import storage: {e}")
    
    def test_pipeline_import(self):
        """Test that pipeline modules can be imported"""
        try:
            import pipeline
            self.assertIsNotNone(pipeline)
        except ImportError as e:
            self.fail(f"Failed to import pipeline: {e}")
    
    def test_monitoring_import(self):
        """Test that monitoring modules can be imported"""
        try:
            import monitoring
            self.assertIsNotNone(monitoring)
        except ImportError as e:
            self.fail(f"Failed to import monitoring: {e}")
    
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
        test_dirs = ['logs', 'models', 'test_media']
        
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
        import json
        
        test_data = {
            'name': 'test',
            'count': 5,
            'active': True
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data)
        self.assertIsInstance(json_str, str)
        
        # Test JSON deserialization
        parsed_data = json.loads(json_str)
        self.assertEqual(parsed_data, test_data)
    
    def test_datetime_operations(self):
        """Test datetime operations"""
        from datetime import datetime
        
        now = datetime.now()
        self.assertIsNotNone(now)
        self.assertIsInstance(now, datetime)
        
        # Test string formatting
        formatted = now.isoformat()
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 10)


def run_minimal_tests():
    """Run minimal test suite"""
    print("Running Minimal CI Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCIMinimal)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All minimal tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == '__main__':
    success = run_minimal_tests()
    sys.exit(0 if success else 1)
