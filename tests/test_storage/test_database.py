#!/usr/bin/env python3
"""
Test Database Operations
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.storage.engine.database import Database
from src.storage.base_model import BaseModel


class TestDatabase(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test database
        self.db = Database(self.db_path)
        self.db.init_db()
    
    def tearDown(self):
        """Clean up test database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertIsNotNone(self.db)
        self.assertTrue(os.path.exists(self.db_path))
    
    def test_create_tables(self):
        """Test table creation"""
        # Tables should be created during init_db
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check that our main tables exist
        expected_tables = ['inputs', 'object_types', 'outputs', 'fewshot_object_types', 'fewshot_predictions']
        for table in expected_tables:
            self.assertIn(table, tables)
    
    def test_save_operation(self):
        """Test save operation"""
        # Mock a model instance
        mock_model = MagicMock()
        mock_model.__tablename__ = 'test_table'
        mock_model.to_dict.return_value = {'id': 'test_id', 'name': 'test'}
        
        # Mock the insert operation
        with patch.object(self.db.connection, 'execute') as mock_execute:
            with patch.object(self.db.connection, 'commit') as mock_commit:
                result = self.db.save(mock_model)
                
                mock_execute.assert_called_once()
                mock_commit.assert_called_once()
                self.assertIsNotNone(result)
    
    def test_get_operation(self):
        """Test get operation"""
        # Mock the query operation
        with patch.object(self.db.connection, 'execute') as mock_execute:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ('test_id', 'test_name')
            mock_execute.return_value = mock_cursor
            
            result = self.db.get('test_table', 'test_id')
            
            mock_execute.assert_called_once()
            self.assertIsNotNone(result)
    
    def test_all_operation(self):
        """Test get all operation"""
        # Mock the query operation
        with patch.object(self.db.connection, 'execute') as mock_execute:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('test_id1', 'test_name1'), ('test_id2', 'test_name2')]
            mock_execute.return_value = mock_cursor
            
            result = self.db.all('test_table')
            
            mock_execute.assert_called_once()
            self.assertEqual(len(result), 2)
    
    def test_delete_operation(self):
        """Test delete operation"""
        # Mock a model instance
        mock_model = MagicMock()
        mock_model.__tablename__ = 'test_table'
        mock_model.id = 'test_id'
        
        # Mock the delete operation
        with patch.object(self.db.connection, 'execute') as mock_execute:
            with patch.object(self.db.connection, 'commit') as mock_commit:
                self.db.delete(mock_model)
                
                mock_execute.assert_called_once()
                mock_commit.assert_called_once()
    
    def test_update_operation(self):
        """Test update operation"""
        # Mock a model instance
        mock_model = MagicMock()
        mock_model.__tablename__ = 'test_table'
        mock_model.id = 'test_id'
        mock_model.to_dict.return_value = {'id': 'test_id', 'name': 'updated'}
        
        # Mock the update operation
        with patch.object(self.db.connection, 'execute') as mock_execute:
            with patch.object(self.db.connection, 'commit') as mock_commit:
                result = self.db.update(mock_model)
                
                mock_execute.assert_called_once()
                mock_commit.assert_called_once()
                self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

