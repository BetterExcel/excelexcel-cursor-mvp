from typing import Dict, Any, List
import pandas as pd
import random
import datetime
import os
from dotenv import load_dotenv
from app.services.workbook import (
    get_sheet, set_sheet, ensure_sheet, set_cell_by_a1,
)
from app.ui.formula import evaluate_formula

# Load environment variables from .env file
load_dotenv()


# Tool implementations operate IN-PLACE on the workbook dict

def tool_set_cell(workbook: Dict[str, pd.DataFrame], sheet: str, cell: str, value: str) -> str:
    df = get_sheet(workbook, sheet)
    set_cell_by_a1(df, cell, value)
    set_sheet(workbook, sheet, df)
    return f"Set {sheet}!{cell} to {value!r}."


def tool_get_cell(workbook: Dict[str, pd.DataFrame], sheet: str, cell: str) -> str:
    df = get_sheet(workbook, sheet)
    from app.ui.formula import get_cell_value
    val = get_cell_value(df, cell)
    return f"{sheet}!{cell} = {val}"


def tool_apply_formula(workbook: Dict[str, pd.DataFrame], sheet: str, cell: str, formula: str, by_column: bool=False) -> str:
    df = get_sheet(workbook, sheet)
    if by_column:
        import re
        col = ''.join([c for c in cell if c.isalpha()])
        for r in range(len(df)):
            val = evaluate_formula(formula, df, current_row=r)
            # if a matching column label exists, write by label; else fallback to A1 index
            try:
                df[col] = df.get(col, df.get(col, None))
            except Exception:
                pass
            set_cell_by_a1(df, f"{col}{r+1}", val)
    else:
        val = evaluate_formula(formula, df)
        set_cell_by_a1(df, cell, val)
    set_sheet(workbook, sheet, df)
    return f"Applied {formula!r} to {sheet}!{cell}{' (entire column)' if by_column else ''}."


def tool_sort(workbook: Dict[str, pd.DataFrame], sheet: str, column: str, ascending: bool=True) -> str:
    df = get_sheet(workbook, sheet)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in sheet '{sheet}'.")
    df = df.sort_values(by=column, ascending=ascending, kind="mergesort").reset_index(drop=True)
    set_sheet(workbook, sheet, df)
    return f"Sorted '{sheet}' by '{column}' {'ascending' if ascending else 'descending'}."


def tool_filter_equals(workbook: Dict[str, pd.DataFrame], sheet: str, column: str, value: str) -> str:
    df = get_sheet(workbook, sheet)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    filtered = df[df[column].astype(str) == str(value)]
    # Do not overwrite; return preview (first 5 rows)
    preview = filtered.head().to_markdown(index=False)
    return f"Filtered preview (first 5 rows):\n{preview}"


def tool_add_sheet(workbook: Dict[str, pd.DataFrame], name: str, rows: int=20, cols: int=5) -> str:
    ensure_sheet(workbook, name, rows, cols)
    return f"Added sheet '{name}' with {rows} rows x {cols} cols."


def tool_make_chart(workbook: Dict[str, pd.DataFrame], sheet: str, x: str, ys: List[str], kind: str="line") -> str:
    import app.charts as charts
    df = get_sheet(workbook, sheet)
    fig = charts.quick_plot(df, x, ys, kind)
    # We cannot return the figure object through the LLM; just confirm success.
    return f"Chart created for {sheet}: x={x}, y={ys}, kind={kind}. (Shown in UI if requested.)"


def tool_export(workbook: Dict[str, pd.DataFrame], sheet: str, fmt: str="csv") -> str:
    df = get_sheet(workbook, sheet)
    if fmt.lower() == "csv":
        return df.to_csv(index=False)
    elif fmt.lower() == "markdown":
        return df.to_markdown(index=False)
    else:
        raise ValueError("Unsupported export fmt. Use 'csv' or 'markdown'.")


def tool_create_csv_file(workbook: Dict[str, pd.DataFrame], sheet: str, filename: str, data: List[List] = None) -> str:
    """Create a new CSV file with given data or current sheet data and save it to data directory"""
    import os
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    if data:
        # Create DataFrame from provided data
        df = pd.DataFrame(data)
        # If data has headers, use first row as column names
        if len(data) > 1:
            df.columns = data[0]
            df = df.iloc[1:].reset_index(drop=True)
    else:
        # Use current sheet data
        df = get_sheet(workbook, sheet)
    
    # Ensure filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Save to data directory
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    
    # Also update the workbook with this data if it's a new sheet
    set_sheet(workbook, sheet, df)
    
    return f"Created CSV file '{filepath}' with {len(df)} rows and {len(df.columns)} columns. Data also available in sheet '{sheet}'."


def tool_save_current_sheet(workbook: Dict[str, pd.DataFrame], sheet: str, filename: str = None) -> str:
    """Save the current sheet as a CSV file in the data directory"""
    import os
    from datetime import datetime
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sheet}_{timestamp}.csv"
    
    # Ensure filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Get current sheet data
    df = get_sheet(workbook, sheet)
    
    # Save to data directory
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    
    return f"Saved sheet '{sheet}' as '{filepath}' with {len(df)} rows and {len(df.columns)} columns."


def tool_generate_sample_data(workbook: Dict[str, pd.DataFrame], sheet: str, rows: int, columns: List[Dict], context: str) -> str:
    """Generate sample data using AI-powered intelligent data generation."""
    import random
    import datetime
    from datetime import timedelta
    
    # Ensure the sheet exists
    ensure_sheet(workbook, sheet, rows + 1, len(columns))  # +1 for header
    df = get_sheet(workbook, sheet)
    
    # Clear existing data
    df.iloc[:, :] = ""
    
    # Try AI-powered generation first, fallback to intelligent templates
    try:
        data = _generate_ai_powered_data(columns, rows, context)
    except Exception as e:
        print(f"AI generation failed ({e}), using intelligent template generation")
        data = _generate_intelligent_template_data(columns, rows, context)
    
    # Set headers
    for i, col_spec in enumerate(columns):
        if i < len(df.columns):
            df.iloc[0, i] = col_spec["name"]
    
    # Fill data rows
    for row_idx in range(1, min(rows + 1, len(df))):
        for col_idx, col_spec in enumerate(columns):
            if col_idx < len(df.columns):
                col_name = col_spec["name"]
                if col_name in data and (row_idx - 1) < len(data[col_name]):
                    df.iloc[row_idx, col_idx] = data[col_name][row_idx - 1]
    
    col_names = ", ".join([col["name"] for col in columns])
    return f"Generated {rows} rows of sample data for {context} with columns: {col_names}. Data includes realistic values based on the context."


def _generate_ai_powered_data(columns: List[Dict], rows: int, context: str) -> Dict[str, List]:
    """Generate data using OpenAI API for maximum intelligence and variety."""
    import openai
    import json
    import os
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not available")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Create smart prompt for AI data generation
    column_descriptions = []
    for col in columns:
        column_descriptions.append(f"- {col['name']} ({col['type']})")
    
    prompt = f"""Generate {rows} rows of realistic sample data for: {context}

Columns needed:
{chr(10).join(column_descriptions)}

Requirements:
1. Generate DIVERSE, REALISTIC data that would make sense in a professional spreadsheet
2. For text fields, create varied, meaningful content (no repetitive patterns)
3. For currency/numbers, use appropriate ranges for the context
4. For dates, use realistic date ranges
5. Make sure each row is unique and interesting
6. Consider the context "{context}" when generating content
7. Use professional naming conventions and realistic business data

CRITICAL: Return ONLY a JSON object with this exact structure:
{{
  "{columns[0]['name']}": ["value1", "value2", "value3", ...],
  "{columns[1]['name']}": ["value1", "value2", "value3", ...],
  ...
}}

Use the EXACT column names provided. Each array should have exactly {rows} values. Return ONLY the JSON object, no explanations or markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data generator for professional spreadsheets. Generate realistic, diverse sample data. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.8,  # Higher creativity for more varied data
            timeout=30
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Clean up response text - remove any markdown code blocks
            clean_response = response_text
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(clean_response)
            
            # Validate structure
            for col in columns:
                col_name = col["name"]
                if col_name not in data:
                    raise ValueError(f"Missing column: {col_name}")
                if len(data[col_name]) != rows:
                    raise ValueError(f"Wrong number of rows for {col_name}: expected {rows}, got {len(data[col_name])}")
            
            return data
            
        except json.JSONDecodeError as e:
            # Log the raw response for debugging
            print(f"DEBUG: AI returned invalid JSON. Raw response: {response_text[:500]}...")
            raise ValueError(f"AI returned invalid JSON: {e}")
            
    except Exception as e:
        # Log more detailed error information
        print(f"DEBUG: AI generation error details: {e}")
        raise ValueError(f"AI data generation failed: {e}")


def _generate_intelligent_template_data(columns: List[Dict], rows: int, context: str) -> Dict[str, List]:
    """Generate data using intelligent template system with context awareness."""
    import random
    from datetime import datetime, timedelta
    
    data = {}
    context_lower = context.lower()
    
    for col_spec in columns:
        col_name = col_spec["name"]
        col_type = col_spec["type"]
        col_name_lower = col_name.lower()
        
        data[col_name] = []
        
        # Intelligent data generation based on context and column names
        if col_type == "date":
            start_date = datetime.date.today() - timedelta(days=rows-1)
            for i in range(rows):
                data[col_name].append((start_date + timedelta(days=i)).strftime("%Y-%m-%d"))
                
        elif col_type == "number":
            if "rating" in col_name_lower or "score" in col_name_lower:
                # Ratings (1-5 scale)
                for i in range(rows):
                    data[col_name].append(round(random.uniform(3.5, 5.0), 1))
            elif "age" in col_name_lower:
                for i in range(rows):
                    data[col_name].append(random.randint(22, 65))
            elif "quantity" in col_name_lower or "qty" in col_name_lower:
                for i in range(rows):
                    data[col_name].append(random.randint(1, 50))
            elif "stock" in context_lower or "price" in col_name_lower:
                base_price = random.uniform(150, 200)
                prices = [base_price]
                for i in range(1, rows):
                    change = random.uniform(-0.05, 0.05)
                    new_price = prices[-1] * (1 + change)
                    prices.append(round(new_price, 2))
                data[col_name] = prices
            else:
                for i in range(rows):
                    data[col_name].append(round(random.uniform(10, 1000), 2))
                    
        elif col_type == "currency":
            if "restaurant" in context_lower or "food" in context_lower:
                # Restaurant pricing
                for i in range(rows):
                    price = round(random.uniform(12, 45), 2)
                    data[col_name].append(f"${price:.2f}")
            elif "salary" in col_name_lower:
                for i in range(rows):
                    salary = random.randint(45000, 120000)
                    salary = round(salary / 1000) * 1000
                    data[col_name].append(f"${salary:,}")
            elif "product" in context_lower:
                for i in range(rows):
                    price = random.uniform(25, 800)
                    if price < 100:
                        price = round(price / 5) * 5
                    else:
                        price = round(price / 10) * 10
                    data[col_name].append(f"${price:.2f}")
            else:
                for i in range(rows):
                    data[col_name].append(f"${random.uniform(20, 200):.2f}")
                
        elif col_type == "text":
            # Context-aware text generation
            if ("restaurant" in context_lower or "food" in context_lower) and "name" in col_name_lower:
                restaurant_styles = ["Bistro", "Café", "Grill", "Kitchen", "House", "Corner", "Garden", "Palace", "Tavern", "Diner"]
                restaurant_adjectives = ["Golden", "Silver", "Blue", "Green", "Sunset", "Ocean", "Mountain", "Urban", "Classic", "Modern", "Royal", "Fresh"]
                for i in range(rows):
                    adj = random.choice(restaurant_adjectives)
                    style = random.choice(restaurant_styles)
                    data[col_name].append(f"{adj} {style}")
            elif "cuisine" in col_name_lower or (("restaurant" in context_lower or "food" in context_lower) and "type" in col_name_lower):
                cuisines = ["Italian", "Chinese", "Mexican", "Indian", "Japanese", "Thai", "French", "American", "Mediterranean", "Korean", "Vietnamese", "Greek"]
                for i in range(rows):
                    data[col_name].append(random.choice(cuisines))
            elif ("employee" in context_lower or "staff" in context_lower) and "department" in col_name_lower:
                departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", "Support", "Design", "Product", "Legal"]
                for i in range(rows):
                    data[col_name].append(random.choice(departments))
            elif "product" in col_name_lower and "name" in col_name_lower:
                categories = ["Laptop", "Smartphone", "Tablet", "Monitor", "Headphones", "Speaker", "Camera", "Printer", "Watch", "Keyboard"]
                brands = ["Pro", "Elite", "Max", "Ultra", "Prime", "Plus", "Air", "Neo", "Edge", "Core"]
                models = ["2024", "X1", "S7", "M3", "V2", "G5", "R8", "T4", "L9", "K6"]
                for i in range(rows):
                    category = random.choice(categories)
                    brand = random.choice(brands)
                    model = random.choice(models)
                    data[col_name].append(f"{category} {brand} {model}")
            elif "product" in col_name_lower and ("id" in col_name_lower or "code" in col_name_lower):
                for i in range(rows):
                    prefix = random.choice(["PRD", "ITM", "SKU", "PDT"])
                    number = str(random.randint(1000, 9999))
                    suffix = random.choice(["A", "B", "C", "X", "Y", "Z"])
                    data[col_name].append(f"{prefix}-{number}{suffix}")
            elif "name" in col_name_lower and ("employee" in context_lower or "person" in context_lower or "staff" in context_lower):
                first_names = ["Alex", "Jordan", "Casey", "Morgan", "Taylor", "Riley", "Avery", "Quinn", "Jamie", "Dakota", "Sage", "River"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Anderson"]
                for i in range(rows):
                    first = random.choice(first_names)
                    last = random.choice(last_names)
                    data[col_name].append(f"{first} {last}")
            elif "sales" in context_lower and ("rep" in col_name_lower or "person" in col_name_lower):
                first_names = ["Michael", "Sarah", "David", "Lisa", "John", "Emma", "Chris", "Amy", "Robert", "Jessica"]
                last_names = ["Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Clark", "Lewis", "Walker"]
                for i in range(rows):
                    first = random.choice(first_names)
                    last = random.choice(last_names)
                    data[col_name].append(f"{first} {last}")
            else:
                # Context-aware generic text
                if "restaurant" in context_lower or "food" in context_lower:
                    items = ["Fresh", "Organic", "Artisan", "Traditional", "Modern", "Fusion", "Seasonal", "Gourmet"]
                elif "product" in context_lower or "tech" in context_lower:
                    items = ["Premium", "Standard", "Professional", "Enterprise", "Advanced", "Basic", "Deluxe", "Elite"]
                elif "employee" in context_lower or "business" in context_lower:
                    items = ["Senior", "Junior", "Lead", "Principal", "Associate", "Manager", "Director", "Specialist"]
                else:
                    items = ["Alpha", "Beta", "Gamma", "Delta", "Premium", "Standard", "Advanced", "Professional"]
                
                suffixes = ["Pro", "Max", "Plus", "Prime", "Elite", "Core", "Edge", "Ultra"]
                for i in range(rows):
                    base = random.choice(items)
                    suffix = random.choice(suffixes)
                    data[col_name].append(f"{base} {suffix}")
    
    return data
