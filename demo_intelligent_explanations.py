#!/usr/bin/env python3
"""
Demo: Intelligent Explanation System with LangChain + LangGraph

This script demonstrates the REAL intelligent explanation system that uses
LangChain to generate contextual, intelligent explanations of spreadsheet changes.
"""

import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.explanation.intelligent_workflow import IntelligentExplanationWorkflow
from app.explanation.local_llm import get_local_llm, check_local_llm_availability
from app.explanation import ExplanationWorkflow

def create_sample_data():
    """Create sample DataFrames for testing."""
    # Before: Empty DataFrame
    before_df = pd.DataFrame()
    
    # After: DataFrame with animal names
    after_df = pd.DataFrame({
        'A': ['Lion', 'Tiger', 'Elephant', 'Giraffe', 'Zebra'],
        'B': ['Penguin', 'Dolphin', 'Shark', 'Whale', 'Octopus'],
        'C': ['Eagle', 'Hawk', 'Owl', 'Falcon', 'Vulture'],
        'D': ['Snake', 'Lizard', 'Turtle', 'Crocodile', 'Alligator'],
        'E': ['Butterfly', 'Bee', 'Ant', 'Spider', 'Scorpion']
    })
    
    return before_df, after_df

def demo_llm_availability():
    """Demo: Check local LLM availability."""
    print("🔍 **Local LLM Availability Check**")
    print("=" * 50)
    
    llm_info = check_local_llm_availability()
    
    print(f"Provider Type: {llm_info['provider_type']}")
    print(f"Model Name: {llm_info['model_name']}")
    print(f"Is Available: {llm_info['is_available']}")
    print()
    
    print("Available Providers:")
    for provider, available in llm_info['available_providers'].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {provider.title()}: {status}")
    
    print()
    return llm_info['is_available']

def demo_intelligent_workflow():
    """Demo: Intelligent workflow with LangChain + LangGraph."""
    print("🤖 **Intelligent Explanation Workflow Demo**")
    print("=" * 50)
    
    # Get local LLM
    local_llm = get_local_llm()
    
    if local_llm:
        print("✅ Local LLM found! Using intelligent workflow.")
        print(f"LLM Type: {type(local_llm).__name__}")
        
        # Create intelligent workflow
        intelligent_workflow = IntelligentExplanationWorkflow(local_llm)
        
        # Test data
        before_df, after_df = create_sample_data()
        
        # Generate intelligent explanation
        print("\n📊 Generating intelligent explanation...")
        explanation = intelligent_workflow.generate_intelligent_explanation(
            operation_type="data_creation",
            before_df=before_df,
            after_df=after_df,
            operation_context={
                'user_request': 'I want the name of 50 different animals in 10 rows and 5 columns',
                'operation_type': 'data_creation',
                'timestamp': '2025-01-27T10:00:00'
            }
        )
        
        print("\n🎯 **Intelligent Explanation Generated:**")
        print("-" * 40)
        print(explanation)
        
    else:
        print("❌ No local LLM available. Using fallback mode.")
        print("💡 Install Ollama or other local LLM for intelligent explanations.")

def demo_fallback_workflow():
    """Demo: Fallback workflow without LLM."""
    print("\n🔄 **Fallback Workflow Demo (No LLM)**")
    print("=" * 50)
    
    # Create basic workflow
    basic_workflow = ExplanationWorkflow()
    
    # Test data
    before_df, after_df = create_sample_data()
    
    # Generate basic explanation
    print("📊 Generating basic explanation...")
    explanation = basic_workflow.generate_explanation(
        operation_type="data_creation",
        before_df=before_df,
        after_df=after_df,
        operation_context={
            'user_request': 'I want the name of 50 different animals in 10 rows and 5 columns',
            'operation_type': 'data_creation',
            'timestamp': '2025-01-27T10:00:00'
        }
    )
    
    print("\n📋 **Basic Explanation Generated:**")
    print("-" * 40)
    print(explanation)

def demo_comparison():
    """Demo: Compare intelligent vs basic explanations."""
    print("\n⚖️ **Intelligent vs Basic Explanation Comparison**")
    print("=" * 60)
    
    # Test data
    before_df, after_df = create_sample_data()
    
    # Get local LLM
    local_llm = get_local_llm()
    
    if local_llm:
        print("🔍 **Intelligent Explanation (with LangChain):**")
        print("-" * 50)
        
        intelligent_workflow = IntelligentExplanationWorkflow(local_llm)
        intelligent_explanation = intelligent_workflow.generate_intelligent_explanation(
            operation_type="data_creation",
            before_df=before_df,
            after_df=after_df,
            operation_context={
                'user_request': 'I want the name of 50 different animals in 10 rows and 5 columns',
                'operation_type': 'data_creation',
                'timestamp': '2025-01-27T10:00:00'
            }
        )
        
        print(intelligent_explanation)
        
        print("\n" + "="*60)
        print("🔍 **Basic Explanation (template-based):**")
        print("-" * 50)
        
        basic_workflow = ExplanationWorkflow()
        basic_explanation = basic_workflow.generate_explanation(
            operation_type="data_creation",
            before_df=before_df,
            after_df=after_df,
            operation_context={
                'user_request': 'I want the name of 50 different animals in 10 rows and 5 columns',
                'operation_type': 'data_creation',
                'timestamp': '2025-01-27T10:00:00'
            }
        )
        
        print(basic_explanation)
        
        print("\n" + "="*60)
        print("💡 **Key Differences:**")
        print("• Intelligent: Uses LangChain to analyze data and generate contextual insights")
        print("• Basic: Uses predefined templates with simple data counting")
        print("• Intelligent: Provides meaningful analysis and suggestions")
        print("• Basic: Provides structural information only")
        
    else:
        print("❌ No local LLM available for comparison.")
        print("💡 Install Ollama to see the difference!")

def main():
    """Main demo function."""
    print("🚀 **Intelligent Explanation System Demo**")
    print("=" * 60)
    print("This demo shows the REAL LangChain + LangGraph integration")
    print("for generating intelligent spreadsheet explanations.")
    print()
    
    # Check LLM availability
    llm_available = demo_llm_availability()
    
    if llm_available:
        # Demo intelligent workflow
        demo_intelligent_workflow()
        
        # Demo comparison
        demo_comparison()
    else:
        # Demo fallback workflow
        demo_fallback_workflow()
        
        print("\n💡 **To Enable Intelligent Explanations:**")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Run: ollama pull llama3.2:3b")
        print("3. Start Ollama service")
        print("4. Restart this demo")
    
    print("\n🎉 **Demo Complete!**")
    print("The intelligent system is now integrated into your Streamlit app.")
    print("Try it out with: streamlit run streamlit_app_enhanced.py")

if __name__ == "__main__":
    main()
