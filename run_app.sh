#!/bin/bash

# Excel-Cursor Application Startup Script
echo "🚀 Starting Excel-Cursor AI Spreadsheet Application..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create it from .env.example"
    echo "The app may not work without proper OpenAI API key configuration."
fi

# Activate virtual environment and start app
source .venv/bin/activate
echo "✅ Virtual environment activated"

echo "🌐 Starting Streamlit application..."
echo "📊 Application will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"

streamlit run streamlit_app_enhanced.py --server.port 8501
