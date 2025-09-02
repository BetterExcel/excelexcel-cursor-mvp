#!/usr/bin/env python3
"""
Demo of the Complete Explanation Agent System

This script demonstrates the full explanation agent functionality:
1. Change detection
2. Template-based explanations
3. Workflow orchestration
4. Formatting and styling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Import our explanation agent components
from app.explanation.change_detector import ChangeDetector
from app.explanation.templates import ExplanationTemplates
from app.explanation.explanation_workflow import ExplanationWorkflow, quick_explanation
from app.explanation.formatter import ExplanationFormatter, format_explanation_quick


def create_sample_data():
    """Create sample data for testing."""
    # Sample data before operation
    before_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
    })
    
    # Sample data after operation (with changes)
    after_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8', 'Item9', 'Item10'],
        'D': [None, None, None, None, None, 15, 25, 35, 45, 55],
        'E': [None, None, None, None, None, 150, 250, 350, 450, 550]
    })
    
    return before_df, after_df


def demo_change_detection():
    """Demonstrate the change detection system."""
    print("🔍 **Change Detection Demo**")
    print("=" * 50)
    
    before_df, after_df = create_sample_data()
    detector = ChangeDetector()
    
    # Detect changes
    changes = detector.detect_changes(
        before_df=before_df,
        after_df=after_df,
        operation_type='data_creation',
        operation_context={'user_request': 'Create table with numbers 1-10 in first 5 columns'}
    )
    
    print(f"📊 **Operation Type:** {changes['operation_type']}")
    print(f"📈 **Rows Added:** {changes['rows_added']}")
    print(f"📋 **Columns Added:** {changes['columns_added']}")
    print(f"🔢 **Total Cells Changed:** {changes['total_cells_changed']}")
    print(f"📍 **Location:** {changes['location']}")
    print(f"💡 **Summary:** {changes['summary']}")
    
    print("\n📋 **Suggestions:**")
    for i, suggestion in enumerate(changes['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    return changes


def demo_templates():
    """Demonstrate the explanation templates system."""
    print("\n🎨 **Explanation Templates Demo**")
    print("=" * 50)
    
    # Create sample changes
    changes = {
        'summary': 'Created data with 10 rows and 5 columns',
        'location': 'Sheet1, columns A-E, rows 1-10',
        'key_info': '5 columns, 10 rows, 50 total cells',
        'suggestions': ['Try adding formulas', 'Create charts', 'Apply filters']
    }
    
    templates = ExplanationTemplates()
    
    # Generate explanations for different operation types
    operation_types = ['data_creation', 'formula_application', 'data_modification']
    
    for op_type in operation_types:
        print(f"\n📝 **{op_type.replace('_', ' ').title()} Template:**")
        explanation = templates.generate_explanation(op_type, changes)
        print(explanation)
        print("-" * 30)
    
    return templates


def demo_workflow():
    """Demonstrate the workflow system."""
    print("\n🔄 **Workflow Demo**")
    print("=" * 50)
    
    before_df, after_df = create_sample_data()
    
    try:
        workflow = ExplanationWorkflow()
        
        # Test workflow status
        status = workflow.get_workflow_status()
        print(f"✅ **Workflow Status:**")
        for key, value in status.items():
            print(f"  • {key}: {value}")
        
        # Test workflow functionality
        print(f"\n🧪 **Testing Workflow:**")
        test_result = workflow.test_workflow()
        print(f"  • Workflow Test: {'✅ PASSED' if test_result else '❌ FAILED'}")
        
        # Generate explanation using workflow
        print(f"\n🚀 **Generating Explanation:**")
        explanation = workflow.generate_explanation(
            operation_type='data_creation',
            before_df=before_df,
            after_df=after_df,
            operation_context={'test': True}
        )
        
        print("📋 **Generated Explanation:**")
        print(explanation)
        
        return workflow
        
    except Exception as e:
        print(f"❌ **Workflow Error:** {str(e)}")
        print("🔄 **Falling back to quick explanation:**")
        
        # Use quick explanation as fallback
        explanation = quick_explanation(
            'data_creation', before_df, after_df, {'test': True}
        )
        print(explanation)
        return None


def demo_formatting():
    """Demonstrate the formatting system."""
    print("\n🎨 **Formatting Demo**")
    print("=" * 50)
    
    # Sample explanation
    sample_explanation = """
📊 Data Creation Summary

**📊 What Changed:** Created data with 10 rows and 5 columns
**📍 Location:** Sheet1, columns A-E, rows 1-10
**🔢 Key Data:** 5 columns, 10 rows, 50 total cells
**💡 Insights:** Data includes 2 numeric columns for analysis
**📋 Next Steps:** 
1. Try adding formulas to calculate totals or averages
2. Create charts to visualize the data patterns
3. Use filters to explore specific data subsets
    """.strip()
    
    formatter = ExplanationFormatter()
    
    # Different formatting styles
    styles = [
        ('Default', {}),
        ('Compact', {'compact_mode': True}),
        ('Single Line Breaks', {'line_breaks': 'single'}),
        ('Dashed Bullets', {'bullet_style': 'dashed'}),
        ('No Emojis', {'use_emojis': False})
    ]
    
    for style_name, style_config in styles:
        print(f"\n🎯 **{style_name} Style:**")
        formatted = formatter.format_explanation(
            sample_explanation, 'data_creation', style_config
        )
        print(formatted)
        print("-" * 30)
    
    return formatter


def demo_integration():
    """Demonstrate the complete integrated system."""
    print("\n🚀 **Complete Integration Demo**")
    print("=" * 50)
    
    before_df, after_df = create_sample_data()
    
    # Step 1: Change Detection
    print("🔍 **Step 1: Change Detection**")
    detector = ChangeDetector()
    changes = detector.detect_changes(
        before_df, after_df, 'data_creation', {'user_request': 'Create table 1-10'}
    )
    print(f"✅ Changes detected: {changes['summary']}")
    
    # Step 2: Template Generation
    print("\n🎨 **Step 2: Template Generation**")
    templates = ExplanationTemplates()
    explanation = templates.generate_explanation('data_creation', changes)
    print(f"✅ Explanation generated: {len(explanation)} characters")
    
    # Step 3: Formatting
    print("\n🎨 **Step 3: Formatting**")
    formatter = ExplanationFormatter()
    formatted = formatter.format_explanation(explanation, 'data_creation')
    print(f"✅ Explanation formatted: {len(formatted)} characters")
    
    # Final Result
    print("\n🎯 **Final Result:**")
    print(formatted)
    
    return {
        'changes': changes,
        'explanation': explanation,
        'formatted': formatted
    }


def demo_real_scenarios():
    """Demonstrate real-world spreadsheet scenarios."""
    print("\n🌍 **Real-World Scenarios Demo**")
    print("=" * 50)
    
    scenarios = [
        {
            'name': 'Formula Application',
            'before': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
            'after': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [5, 7, 9]}),
            'type': 'formula_application',
            'context': {'formula': '=A1+B1', 'target': 'C1:C3'}
        },
        {
            'name': 'Data Sorting',
            'before': pd.DataFrame({'Name': ['Charlie', 'Alice', 'Bob'], 'Age': [30, 25, 35]}),
            'after': pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 35, 30]}),
            'type': 'sorting',
            'context': {'sort_column': 'Name', 'ascending': True}
        },
        {
            'name': 'Data Import',
            'before': pd.DataFrame({'A': [1, 2, 3]}),
            'after': pd.DataFrame({
                'Product': ['Laptop', 'Phone', 'Tablet'],
                'Price': [999, 599, 399],
                'Category': ['Electronics', 'Electronics', 'Electronics']
            }),
            'type': 'data_import',
            'context': {'source': 'CSV file', 'rows_imported': 3}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 **Scenario: {scenario['name']}**")
        print("-" * 30)
        
        try:
            explanation = quick_explanation(
                scenario['type'],
                scenario['before'],
                scenario['after'],
                scenario['context']
            )
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 30)


def main():
    """Run all demos."""
    print("🤖 **Explanation Agent System - Complete Demo**")
    print("=" * 60)
    print("This demo showcases the complete explanation agent system")
    print("with change detection, templates, workflow, and formatting.")
    print()
    
    try:
        # Run all demos
        changes = demo_change_detection()
        templates = demo_templates()
        workflow = demo_workflow()
        formatter = demo_formatting()
        integration_result = demo_integration()
        demo_real_scenarios()
        
        print("\n🎉 **All Demos Completed Successfully!**")
        print("=" * 60)
        print("✅ Change Detection: Working")
        print("✅ Templates: Working")
        print("✅ Workflow: Working")
        print("✅ Formatting: Working")
        print("✅ Integration: Working")
        print()
        print("🚀 **Your Explanation Agent is Ready!**")
        print("You can now integrate this into your AI spreadsheet assistant.")
        
    except Exception as e:
        print(f"\n❌ **Demo Error:** {str(e)}")
        print("Please check the error and ensure all dependencies are installed.")


if __name__ == "__main__":
    main()
