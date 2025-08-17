#!/usr/bin/env python3
"""
Simple model test
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models_to_test = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ {model} works!")
        break
    except Exception as e:
        print(f"❌ {model} failed: {e}")
