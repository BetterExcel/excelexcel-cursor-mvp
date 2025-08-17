#!/usr/bin/env python3
"""
List available models from OpenAI API
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🔍 Fetching list of available models...")

try:
    models = client.models.list()
    
    print("📋 Available models:")
    print("=" * 50)
    
    # Filter and sort models
    model_names = []
    for model in models.data:
        model_names.append(model.id)
    
    model_names.sort()
    
    # Look for o3, gpt-4, and other relevant models
    categories = {
        "o3 models": [],
        "GPT-4 models": [],
        "GPT-3.5 models": [],
        "Other models": []
    }
    
    for name in model_names:
        if "o3" in name.lower():
            categories["o3 models"].append(name)
        elif "gpt-4" in name.lower():
            categories["GPT-4 models"].append(name)
        elif "gpt-3.5" in name.lower():
            categories["GPT-3.5 models"].append(name)
        else:
            categories["Other models"].append(name)
    
    for category, models in categories.items():
        if models:
            print(f"\n🎯 {category}:")
            for model in models:
                print(f"  • {model}")
    
    # Specifically look for any model with "o3" in the name
    o3_models = [m for m in model_names if "o3" in m.lower()]
    if o3_models:
        print(f"\n🎉 Found {len(o3_models)} o3-related models!")
        for model in o3_models:
            print(f"  🚀 {model}")
    else:
        print("\n❌ No o3 models found in your available models")
        print("💡 You may need to:")
        print("   1. Join the o3 waitlist")
        print("   2. Upgrade your API tier")
        print("   3. Wait for broader availability")

except Exception as e:
    print(f"❌ Error fetching models: {e}")

print("\n" + "=" * 50)
