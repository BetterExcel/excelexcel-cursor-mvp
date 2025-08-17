#!/usr/bin/env python3
"""
Data Directory Management Utility
Helps clean up and manage files in the data directory
"""

import os
import glob
from datetime import datetime
import pandas as pd

def list_data_files():
    """List all data files with their details."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ Data directory does not exist")
        return
    
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("📂 No CSV files found in data directory")
        return
    
    print(f"📊 Found {len(files)} CSV files in data directory:")
    print("-" * 80)
    
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        try:
            df = pd.read_csv(file_path)
            rows, cols = df.shape
            print(f"📄 {filename}")
            print(f"   Size: {size:,} bytes | Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Data: {rows:,} rows × {cols} columns")
            if cols > 0:
                print(f"   Columns: {', '.join(list(df.columns)[:5])}{' ...' if len(df.columns) > 5 else ''}")
            print()
        except Exception as e:
            print(f"📄 {filename}")
            print(f"   Size: {size:,} bytes | Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   ⚠️  Error reading file: {str(e)}")
            print()

def clean_duplicates():
    """Remove duplicate files with similar names, keeping the latest."""
    data_dir = "data"
    files = glob.glob(os.path.join(data_dir, "*_imported_*.csv"))
    
    if not files:
        print("✅ No import duplicates found")
        return
    
    # Group files by base name
    groups = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract base name (everything before _imported_)
        if "_imported_" in filename:
            base_name = filename.split("_imported_")[0]
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file_path)
    
    removed_count = 0
    for base_name, file_list in groups.items():
        if len(file_list) > 1:
            # Sort by modification time, keep the latest
            file_list.sort(key=lambda f: os.path.getmtime(f))
            latest_file = file_list[-1]
            
            print(f"🗂️  Found {len(file_list)} versions of '{base_name}':")
            for i, file_path in enumerate(file_list):
                filename = os.path.basename(file_path)
                modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                status = "KEEPING (latest)" if file_path == latest_file else "removing"
                print(f"   {i+1}. {filename} ({modified.strftime('%Y-%m-%d %H:%M:%S')}) - {status}")
            
            # Remove all but the latest
            for file_path in file_list[:-1]:
                os.remove(file_path)
                removed_count += 1
            print()
    
    if removed_count > 0:
        print(f"✅ Removed {removed_count} duplicate files")
    else:
        print("✅ No duplicates to remove")

def show_file_content(filename):
    """Show the first few rows of a data file."""
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {filename}")
        return
    
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Content preview of {filename}:")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))
        
        if len(df) > 10:
            print(f"\n... and {len(df) - 10:,} more rows")
            
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("📂 Data Directory Management Utility")
        print("Usage:")
        print("  python data_manager.py list           - List all data files")
        print("  python data_manager.py clean          - Remove duplicate files")
        print("  python data_manager.py show <filename> - Show file content")
        print()
        list_data_files()
    
    elif sys.argv[1] == "list":
        list_data_files()
    
    elif sys.argv[1] == "clean":
        clean_duplicates()
    
    elif sys.argv[1] == "show" and len(sys.argv) > 2:
        show_file_content(sys.argv[2])
    
    else:
        print("❌ Invalid command. Use 'list', 'clean', or 'show <filename>'")
