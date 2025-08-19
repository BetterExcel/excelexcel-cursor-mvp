#!/usr/bin/env python3
"""
Demonstration of AI-powered intelligent data generation system
Shows how the system understands context and generates appropriate data
"""
from app.agent.tools import tool_generate_sample_data
from app.services.workbook import new_workbook

def demo_context_aware_generation():
    """Demonstrate context-aware data generation across different domains."""
    
    print("🤖 AI-Powered Data Generation System Demo")
    print("=" * 60)
    print("Demonstrates how the system intelligently understands context")
    print("and generates appropriate data without manual templates.")
    print()
    
    # Demo 1: Restaurant/Food context
    print("📍 Demo 1: Restaurant Directory")
    print("-" * 30)
    workbook1 = new_workbook()
    columns1 = [
        {'name': 'Restaurant Name', 'type': 'text'},
        {'name': 'Cuisine Type', 'type': 'text'}, 
        {'name': 'Average Price', 'type': 'currency'},
        {'name': 'Rating', 'type': 'number'}
    ]
    
    result1 = tool_generate_sample_data(workbook1, 'Sheet1', 5, columns1, 'Restaurant directory for food delivery app')
    print(f"✅ {result1}")
    
    df1 = workbook1['Sheet1']
    for i in range(min(4, len(df1))):
        print(f"  {df1.iloc[i, 0]:<20} | {df1.iloc[i, 1]:<12} | {df1.iloc[i, 2]:<8} | {df1.iloc[i, 3]}")
    print()
    
    # Demo 2: Employee/HR context  
    print("👥 Demo 2: Employee Management")
    print("-" * 30)
    workbook2 = new_workbook()
    columns2 = [
        {'name': 'Employee Name', 'type': 'text'},
        {'name': 'Department', 'type': 'text'}, 
        {'name': 'Salary', 'type': 'currency'},
        {'name': 'Age', 'type': 'number'}
    ]
    
    result2 = tool_generate_sample_data(workbook2, 'Sheet1', 5, columns2, 'Employee management system for HR department')
    print(f"✅ {result2}")
    
    df2 = workbook2['Sheet1']
    for i in range(min(4, len(df2))):
        print(f"  {df2.iloc[i, 0]:<20} | {df2.iloc[i, 1]:<12} | {df2.iloc[i, 2]:<10} | {df2.iloc[i, 3]}")
    print()
    
    # Demo 3: E-commerce Products
    print("🛍️ Demo 3: E-commerce Product Catalog")
    print("-" * 30)
    workbook3 = new_workbook()
    columns3 = [
        {'name': 'Product ID', 'type': 'text'},
        {'name': 'Product Name', 'type': 'text'}, 
        {'name': 'Price', 'type': 'currency'},
        {'name': 'Category', 'type': 'text'}
    ]
    
    result3 = tool_generate_sample_data(workbook3, 'Sheet1', 5, columns3, 'E-commerce product catalog for electronics store')
    print(f"✅ {result3}")
    
    df3 = workbook3['Sheet1']
    for i in range(min(4, len(df3))):
        print(f"  {df3.iloc[i, 0]:<12} | {df3.iloc[i, 1]:<20} | {df3.iloc[i, 2]:<8} | {df3.iloc[i, 3]}")
    print()
    
    # Demo 4: Sales Records
    print("💰 Demo 4: Sales Performance")
    print("-" * 30)
    workbook4 = new_workbook()
    columns4 = [
        {'name': 'Sales Rep', 'type': 'text'},
        {'name': 'Product', 'type': 'text'}, 
        {'name': 'Quantity', 'type': 'number'},
        {'name': 'Total', 'type': 'currency'}
    ]
    
    result4 = tool_generate_sample_data(workbook4, 'Sheet1', 5, columns4, 'Sales performance report for quarterly review')
    print(f"✅ {result4}")
    
    df4 = workbook4['Sheet1']
    for i in range(min(4, len(df4))):
        print(f"  {df4.iloc[i, 0]:<15} | {df4.iloc[i, 1]:<20} | {df4.iloc[i, 2]:<8} | {df4.iloc[i, 3]}")
    print()
    
    print("🎯 Key Intelligence Features Demonstrated:")
    print("  • Context-aware naming (restaurants vs employees vs products)")
    print("  • Appropriate pricing ranges for different industries")
    print("  • Realistic ratings, ages, quantities for specific contexts")
    print("  • Professional formatting (Product IDs, salary commas, etc.)")
    print("  • Domain-specific categories and departments")
    print()
    print("✨ The AI system automatically understands what type of data")
    print("   to generate based on column names and context description!")

if __name__ == "__main__":
    demo_context_aware_generation()
