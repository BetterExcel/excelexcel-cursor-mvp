#!/usr/bin/env python3
"""
Test script to demonstrate the improved product data generation
"""
from app.agent.tools import tool_generate_sample_data
from app.services.workbook import new_workbook

def test_product_data_generation():
    """Test the improved product data generation with realistic data."""
    print("🧪 Testing Product Data Generation")
    print("=" * 50)
    
    # Create workbook
    workbook = new_workbook()
    
    # Define product columns as in the prompt
    columns = [
        {"name": "Product ID", "type": "text"},
        {"name": "Product Name", "type": "text"},
        {"name": "Product Price", "type": "currency"},
        {"name": "Product Description", "type": "text"}
    ]
    
    # Generate 20 products
    result = tool_generate_sample_data(
        workbook, 
        "Sheet1", 
        20, 
        columns, 
        "Product catalog with diverse electronics and accessories"
    )
    
    print(f"✅ Generation Result: {result}")
    print()
    
    # Display generated data
    df = workbook["Sheet1"]
    print("📊 Generated Product Data:")
    print("-" * 100)
    
    # Print header
    headers = [df.iloc[0, j] for j in range(4)]
    print(f"{'Row':<4} | {'Product ID':<12} | {'Product Name':<25} | {'Price':<10} | {'Description':<30}")
    print("-" * 100)
    
    # Print first 15 data rows (skip header)
    for i in range(1, min(16, len(df))):
        product_id = str(df.iloc[i, 0])[:12]
        product_name = str(df.iloc[i, 1])[:25] 
        price = str(df.iloc[i, 2])[:10]
        description = str(df.iloc[i, 3])[:30]
        print(f"{i:<4} | {product_id:<12} | {product_name:<25} | {price:<10} | {description:<30}")
    
    print("-" * 100)
    
    # Analyze data quality
    print("\n🔍 Data Quality Analysis:")
    
    # Check for uniqueness in product IDs
    product_ids = [str(df.iloc[i, 0]) for i in range(1, len(df))]
    unique_ids = len(set(product_ids))
    print(f"• Product ID Uniqueness: {unique_ids}/{len(product_ids)} unique IDs")
    
    # Check for variety in product names
    product_names = [str(df.iloc[i, 1]) for i in range(1, len(df))]
    unique_names = len(set(product_names))
    print(f"• Product Name Variety: {unique_names}/{len(product_names)} unique names")
    
    # Check price range
    prices = []
    for i in range(1, len(df)):
        price_str = str(df.iloc[i, 2])
        if price_str.startswith('$'):
            price_val = float(price_str[1:])
            prices.append(price_val)
    
    if prices:
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        print(f"• Price Range: ${min_price:.2f} - ${max_price:.2f} (avg: ${avg_price:.2f})")
    
    # Check description variety
    descriptions = [str(df.iloc[i, 3]) for i in range(1, len(df))]
    unique_descriptions = len(set(descriptions))
    print(f"• Description Variety: {unique_descriptions}/{len(descriptions)} unique descriptions")
    
    print(f"\n✅ Test completed! Data quality is {'EXCELLENT' if unique_ids > 15 and unique_names > 15 else 'GOOD'}")

if __name__ == "__main__":
    test_product_data_generation()
