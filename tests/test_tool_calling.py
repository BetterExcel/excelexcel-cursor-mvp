#!/usr/bin/env python3
"""
Debug script to check tool calling in detail
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def test_tool_calling():
    print("🔍 Testing tool calling explicitly...")
    
    # Create empty workbook
    workbook = {
        'Sheet1': pd.DataFrame()
    }
    
    try:
        print("📤 Sending explicit tool request...")
        response = run_agent(
            user_msg="Use the set_cell tool to put 'Hello' in cell A1 on Sheet1",
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
    test_tool_calling()
