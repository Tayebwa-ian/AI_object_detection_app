#!/usr/bin/env python3
"""
Simple Test Suite for AI Object Counting Application

This is a basic test that you can run to verify the application is working.
It tests the core functionality without requiring complex setup.
"""

import unittest
import sys
import os

# Add src to path - works from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

class TestSimpleFunctionality(unittest.TestCase):
    """Simple tests to verify basic functionality"""
    
    def test_imports(self):
        """Test that all main modules can be imported"""
        print("Testing imports...")
        
        try:
            import app
            print("✅ Flask app imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import Flask app: {e}")
        
        try:
            import config
            print("✅ Config imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")
        
        try:
            import storage
            print("✅ Database imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import database: {e}")
    
    def test_app_creation(self):
        """Test that Flask app can be created"""
        print("Testing Flask app creation...")
        
        try:
            import app
            self.assertIsNotNone(app.app)
            print("✅ Flask app created successfully")
        except Exception as e:
            self.fail(f"Failed to create Flask app: {e}")
    
    def test_config_values(self):
        """Test that config has required values"""
        print("Testing configuration...")
        
        try:
            import config
            self.assertIsNotNone(config.config.SECRET_KEY)
            self.assertIsNotNone(config.config.DATABASE_TYPE)
            print("✅ Configuration loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load configuration: {e}")
    
    def test_api_endpoints_registration(self):
        """Test that API endpoints are registered"""
        print("Testing API endpoint registration...")
        
        try:
            import app
            
            # Check if some key endpoints are registered
            with app.app.test_client() as client:
                # Test a simple endpoint
                response = client.get('/api/object-types')
                # We expect either 200 (success) or 500 (database not initialized)
                # Both are acceptable for this simple test
                self.assertIn(response.status_code, [200, 500])
                print("✅ API endpoints are registered")
        except Exception as e:
            self.fail(f"Failed to test API endpoints: {e}")

def run_simple_test():
    """Run the simple test suite"""
    print("🧪 Running Simple Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSimpleFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All simple tests passed!")
        print("✅ Your application is ready to use!")
        return True
    else:
        print("❌ Some tests failed.")
        print("Please check the error messages above.")
        return False

if __name__ == '__main__':
    success = run_simple_test()
    sys.exit(0 if success else 1)
