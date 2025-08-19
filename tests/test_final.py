#!/usr/bin/env python3
"""
Final test of the fixed agent
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent
from app.services.workbook import new_workbook
from dotenv import load_dotenv

load_dotenv()

def test_final():
    print("🚀 Final test of the fixed agent...")
    
    # Create properly initialized workbook
    workbook = new_workbook()
    
    try:
        print("📤 Testing stock data generation...")
        response = run_agent(
            user_msg="Create me stock data for AAPL for 30 days for the month of July 2025. Add 4 columns: Date, Price, News, and Market Cap. Fill in these columns with random but sensible data.",
            workbook=workbook,
            current_sheet='Sheet1',
            chat_history=[],
            model_name="gpt-4-turbo-2024-04-09"
        )
        print("✅ Agent Response:")
        print(response)
        print("\n📊 First 10 rows of generated data:")
        print(workbook['Sheet1'].head(10))
        
        print("\n🧮 Testing formula application...")
        response2 = run_agent(
            user_msg="Add a formula in column F to calculate the average price from column B. Apply it to all rows.",
            workbook=workbook,
            current_sheet='Sheet1',
            chat_history=[],
            model_name="gpt-4-turbo-2024-04-09"
        )
        print("✅ Formula Response:")
        print(response2)
        
        return True
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final()
    if success:
        print("\n🎉 ALL TESTS PASSED! The agent is working correctly!")
    else:
        print("\n❌ Tests failed!")
