#!/usr/bin/env python3
"""
Minimal test of OpenAI function calling
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

def test_function_calling():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
    
    # Simple tool for testing
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function that just returns a message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to return"}
                    },
                    "required": ["message"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When asked to test a function, use the test_function tool."},
        {"role": "user", "content": "Use the test_function tool to send the message 'Hello World'"}
    ]
    
    try:
        print("🔍 Testing OpenAI function calling...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        print("✅ Response received:")
        message = response.choices[0].message
        print(f"Content: {message.content}")
        print(f"Tool calls: {message.tool_calls}")
        
        if message.tool_calls:
            print("✅ Tool calling is working!")
            for tc in message.tool_calls:
                print(f"Tool: {tc.function.name}")
                print(f"Args: {tc.function.arguments}")
        else:
            print("❌ No tool calls made")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_function_calling()
