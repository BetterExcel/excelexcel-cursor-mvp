#!/usr/bin/env python3
"""
Debug the agent function calling
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from app.agent.agent import run_agent, TOOLS
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

def debug_agent():
    print("🔍 Debugging agent...")
    
    print(f"📋 Number of tools available: {len(TOOLS)}")
    print("🔧 Available tools:")
    for tool in TOOLS:
        func_name = tool["function"]["name"]
        func_desc = tool["function"]["description"]
        print(f"  • {func_name}: {func_desc}")
    
    # Create properly initialized workbook
    from app.services.workbook import new_workbook
    workbook = new_workbook()
    
    print("\n📤 Testing with simple set_cell request...")
    
    try:
        response = run_agent(
            user_msg="Use the set_cell tool to put 'Test Value' in cell A1",
            workbook=workbook,
            current_sheet='Sheet1',
            chat_history=[],
            model_name="gpt-4-turbo-2024-04-09"
        )
        print("✅ Agent Response:")
        print(response)
        print("\n📊 Workbook after:")
        print(workbook['Sheet1'])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_agent()
