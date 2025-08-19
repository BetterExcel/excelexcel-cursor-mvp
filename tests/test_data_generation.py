#!/usr/bin/env python3
"""
Test script to check if agent is calling tools properly
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def test_data_generation():
    print("🔍 Testing data generation request...")
    
    # Create properly initialized workbook
    from app.services.workbook import new_workbook
    workbook = new_workbook()
    
    try:
        print("📤 Sending data generation request...")
        response = run_agent(
            user_msg="Create me stock data for AAPL for 5 days. Add 4 columns: Date, Price, News, and Market Cap. Fill in these columns with random but sensible data.",
            workbook=workbook,
            current_sheet='Sheet1',
            chat_history=[],
            model_name="gpt-4-turbo-2024-04-09"
        )
        print("✅ Agent Response:")
        print(response)
        print("\n📊 Workbook after operation:")
        print(workbook['Sheet1'])
        return True
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_generation()
