"""
Integration tests for the complete application
"""
import unittest
import pandas as pd
from app.services.workbook import new_workbook, get_sheet, set_sheet
from app.ui.formula import evaluate_formula


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.workbook = new_workbook()
        self.test_data = pd.DataFrame({
            'Sales': [100, 200, 300, 400, 500],
            'Costs': [50, 100, 150, 200, 250],
            'Profit': [None, None, None, None, None]
        })
        set_sheet(self.workbook, 'Sheet1', self.test_data)
    
    def test_end_to_end_calculation(self):
        """Test complete workflow: data input -> formula -> result."""
        # Get the sheet
        sheet = get_sheet(self.workbook, 'Sheet1')
        
        # Apply formula to calculate profit
        for i in range(len(sheet)):
            profit = evaluate_formula('=A1-B1', sheet, current_row=i)
            sheet.at[i, 'Profit'] = profit
        
        # Verify calculations
        expected_profits = [50, 100, 150, 200, 250]
        for i, expected in enumerate(expected_profits):
            self.assertEqual(sheet.at[i, 'Profit'], expected)
    
    def test_aggregation_formulas(self):
        """Test aggregation formulas on the data."""
        sheet = get_sheet(self.workbook, 'Sheet1')
        
        # Test total sales
        total_sales = evaluate_formula('=SUM(A1:A5)', sheet)
        self.assertEqual(total_sales, 1500)
        
        # Test average costs
        avg_costs = evaluate_formula('=AVERAGE(B1:B5)', sheet)
        self.assertEqual(avg_costs, 150)
    
    def test_data_manipulation_workflow(self):
        """Test complete data manipulation workflow."""
        # Start with empty sheet
        empty_data = pd.DataFrame({
            'A': [None] * 5,
            'B': [None] * 5,
            'C': [None] * 5
        })
        set_sheet(self.workbook, 'TestSheet', empty_data)
        
        # Add some data
        sheet = get_sheet(self.workbook, 'TestSheet')
        sheet.loc[0] = [10, 20, None]
        sheet.loc[1] = [30, 40, None]
        
        # Apply formula
        for i in range(2):
            if pd.notna(sheet.at[i, 'A']) and pd.notna(sheet.at[i, 'B']):
                result = evaluate_formula('=A1+B1', sheet, current_row=i)
                sheet.at[i, 'C'] = result
        
        # Verify results
        self.assertEqual(sheet.at[0, 'C'], 30)
        self.assertEqual(sheet.at[1, 'C'], 70)


if __name__ == '__main__':
    unittest.main()
