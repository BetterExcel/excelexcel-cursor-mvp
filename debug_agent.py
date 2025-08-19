#!/usr/bin/env python3
"""
Debug script to test agent functionality
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def test_agent():
    print("🔍 Testing agent functionality...")
    
    # Create simple test workbook
    workbook = {
        'Sheet1': pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
    }
    
    try:
        print("📤 Sending message to agent...")
        response = run_agent(
            user_msg="Hello, what data do you see in my sheet?",
            workbook=workbook,
            current_sheet='Sheet1',
            chat_history=[],
            model_name="gpt-4-turbo-2024-04-09"
        )
        print("✅ Agent Response:")
        print(response)
        return True
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent()
    if success:
        print("\n✅ Agent test passed!")
    else:
        print("\n❌ Agent test failed!")
