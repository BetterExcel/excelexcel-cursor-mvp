"""
Test suite for workbook operations
"""
import unittest
import pandas as pd
from app.services.workbook import (
    new_workbook,
    get_sheet,
    set_sheet,
    list_sheets,
    ensure_sheet,
    set_cell_by_a1
)


class TestWorkbook(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.workbook = new_workbook()
    
    def test_new_workbook_creation(self):
        """Test that a new workbook is created with default sheet."""
        self.assertIsInstance(self.workbook, dict)
        self.assertTrue(len(self.workbook) > 0)
        self.assertIn('Sheet1', list_sheets(self.workbook))
    
    def test_list_sheets(self):
        """Test listing sheets in workbook."""
        sheets = list_sheets(self.workbook)
        self.assertIsInstance(sheets, list)
        self.assertIn('Sheet1', sheets)
    
    def test_get_set_sheet(self):
        """Test getting and setting sheet data."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Set sheet data
        set_sheet(self.workbook, 'Sheet1', test_data)
        
        # Get sheet data
        retrieved_data = get_sheet(self.workbook, 'Sheet1')
        
        # Verify data integrity
        pd.testing.assert_frame_equal(test_data, retrieved_data)
    
    def test_ensure_sheet(self):
        """Test ensuring a sheet exists."""
        sheet_name = 'TestSheet'
        ensure_sheet(self.workbook, sheet_name, rows=10, cols=5)
        
        # Check sheet was created
        self.assertIn(sheet_name, list_sheets(self.workbook))
        
        # Check sheet has correct dimensions
        sheet_data = get_sheet(self.workbook, sheet_name)
        self.assertEqual(len(sheet_data), 10)
        self.assertEqual(len(sheet_data.columns), 5)
    
    def test_set_cell_by_a1(self):
        """Test setting cell value by A1 notation."""
        sheet_data = get_sheet(self.workbook, 'Sheet1')
        
        # Set cell value
        set_cell_by_a1(sheet_data, 'A1', 'test_value')
        
        # Verify cell was set
        self.assertEqual(sheet_data.at[0, 'A'], 'test_value')


if __name__ == '__main__':
    unittest.main()
