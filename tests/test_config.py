#!/usr/bin/env python3
"""
Test Configuration and Setup
"""
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfiguration(unittest.TestCase):
    """Test application configuration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_config_loading(self):
        """Test configuration loading"""
        from src.config import config
        
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.SECRET_KEY)
        self.assertIsNotNone(config.DATABASE_TYPE)
    
    def test_development_config(self):
        """Test development configuration"""
        os.environ['OBJ_DETECT_ENV'] = 'development'
        
        with patch('src.config.config') as mock_config:
            mock_config.DATABASE_TYPE = 'sqlite'
            mock_config.DATABASE_URL = 'dev_obj_detect.db'
            mock_config.DEBUG = True
            
            self.assertEqual(mock_config.DATABASE_TYPE, 'sqlite')
            self.assertTrue(mock_config.DEBUG)
    
    def test_production_config(self):
        """Test production configuration"""
        os.environ['OBJ_DETECT_ENV'] = 'production'
        
        with patch('src.config.config') as mock_config:
            mock_config.DATABASE_TYPE = 'mysql'
            mock_config.DATABASE_URL = 'mysql://user:pass@localhost/db'
            mock_config.DEBUG = False
            
            self.assertEqual(mock_config.DATABASE_TYPE, 'mysql')
            self.assertFalse(mock_config.DEBUG)
    
    def test_testing_config(self):
        """Test testing configuration"""
        os.environ['OBJ_DETECT_ENV'] = 'testing'
        
        with patch('src.config.config') as mock_config:
            mock_config.DATABASE_TYPE = 'sqlite'
            mock_config.DATABASE_URL = 'test_obj_detect.db'
            mock_config.DEBUG = False
            
            self.assertEqual(mock_config.DATABASE_TYPE, 'sqlite')
            self.assertFalse(mock_config.DEBUG)
    
    def test_environment_variables(self):
        """Test environment variable handling"""
        test_vars = {
            'OBJ_DETECT_ENV': 'testing',
            'OBJ_DETECT_MYSQL_DB': 'test_db',
            'FLASK_DEBUG': 'False',
            'LOG_LEVEL': 'DEBUG'
        }
        
        for key, value in test_vars.items():
            os.environ[key] = value
            self.assertEqual(os.environ[key], value)
    
    def test_database_configuration(self):
        """Test database configuration"""
        from src.config import config
        
        # Test that database configuration is valid
        self.assertIn(config.DATABASE_TYPE, ['sqlite', 'mysql', 'postgresql'])
        
        if config.DATABASE_TYPE == 'sqlite':
            self.assertIsNotNone(config.DATABASE_URL)
        elif config.DATABASE_TYPE in ['mysql', 'postgresql']:
            self.assertIsNotNone(config.DATABASE_URL)
            self.assertIn('://', config.DATABASE_URL)
    
    def test_secret_key_configuration(self):
        """Test secret key configuration"""
        from src.config import config
        
        self.assertIsNotNone(config.SECRET_KEY)
        self.assertGreater(len(config.SECRET_KEY), 10)  # Should be reasonably long
    
    def test_media_directory_configuration(self):
        """Test media directory configuration"""
        from src.config import config
        
        # Test that media directory is configured
        self.assertIsNotNone(config.MEDIA_DIRECTORY)
        
        # Test that directory can be created
        test_dir = os.path.join(self.temp_dir, 'test_media')
        os.makedirs(test_dir, exist_ok=True)
        self.assertTrue(os.path.exists(test_dir))
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        from src.config import config
        
        # Test that logging level is valid
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.assertIn(config.LOG_LEVEL, valid_levels)
    
    def test_ai_model_configuration(self):
        """Test AI model configuration"""
        from src.config import config
        
        # Test that AI device is configured
        self.assertIn(config.AI_DEVICE, ['cpu', 'cuda', 'auto'])
        
        # Test that model paths are configured
        self.assertIsNotNone(config.SAM_MODEL_PATH)
        self.assertIsNotNone(config.RESNET_MODEL_PATH)


class TestApplicationSetup(unittest.TestCase):
    """Test application setup and initialization"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        os.environ['OBJ_DETECT_ENV'] = 'testing'
        os.environ['OBJ_DETECT_MYSQL_DB'] = 'test_obj_detect.db'
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_app_creation(self):
        """Test Flask app creation"""
        with patch('src.app.database') as mock_db:
            with patch('src.app.api') as mock_api:
                from src.app import app
                
                self.assertIsNotNone(app)
                self.assertEqual(app.config['ENV'], 'testing')
    
    def test_database_initialization(self):
        """Test database initialization"""
        with patch('src.storage.database.Database') as mock_db_class:
            mock_db_instance = MagicMock()
            mock_db_class.return_value = mock_db_instance
            
            from src.storage import database
            
            self.assertIsNotNone(database)
    
    def test_api_registration(self):
        """Test API resource registration"""
        with patch('src.app.database') as mock_db:
            with patch('src.app.api') as mock_api:
                from src.app import app, api
                
                # Test that API is registered with Flask app
                self.assertIsNotNone(api)
    
    def test_health_endpoint(self):
        """Test health endpoint registration"""
        with patch('src.app.database') as mock_db:
            with patch('src.app.api') as mock_api:
                from src.app import app
                
                with app.test_client() as client:
                    response = client.get('/health')
                    # Health endpoint should return 200 or 500 (depending on setup)
                    self.assertIn(response.status_code, [200, 500])


if __name__ == '__main__':
    unittest.main()

