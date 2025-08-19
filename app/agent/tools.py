from typing import Dict, Any, List
import pandas as pd
import random
import datetime
from app.services.workbook import (
    get_sheet, set_sheet, ensure_sheet, set_cell_by_a1,
)
from app.ui.formula import evaluate_formula


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
    """Generate sample data based on column specifications and context."""
    import random
    import datetime
    from datetime import timedelta
    
    # Ensure the sheet exists
    ensure_sheet(workbook, sheet, rows + 1, len(columns))  # +1 for header
    df = get_sheet(workbook, sheet)
    
    # Clear existing data
    df.iloc[:, :] = ""
    
    # Generate data based on column specifications
    data = {}
    
    for col_spec in columns:
        col_name = col_spec["name"]
        col_type = col_spec["type"]
        
        # Add column header
        data[col_name] = []
        
        # Generate data based on type and context
        if col_type == "date":
            # Generate sequential dates (last 'rows' days)
            start_date = datetime.date.today() - timedelta(days=rows-1)
            for i in range(rows):
                data[col_name].append((start_date + timedelta(days=i)).strftime("%Y-%m-%d"))
                
        elif col_type == "number":
            if "stock" in context.lower() or "price" in col_name.lower():
                # Stock price-like data (gradual changes)
                base_price = random.uniform(150, 200)
                prices = [base_price]
                for i in range(1, rows):
                    change = random.uniform(-0.05, 0.05)  # 5% max change
                    new_price = prices[-1] * (1 + change)
                    prices.append(round(new_price, 2))
                data[col_name] = prices
            elif "cap" in col_name.lower():
                # Market cap-like data (billions)
                base_cap = random.uniform(2000, 3000)
                for i in range(rows):
                    variation = random.uniform(0.95, 1.05)
                    data[col_name].append(f"{base_cap * variation:.1f}B")
            else:
                # Generic numbers
                for i in range(rows):
                    data[col_name].append(round(random.uniform(10, 1000), 2))
                    
        elif col_type == "currency":
            # Currency values
            for i in range(rows):
                data[col_name].append(f"${random.uniform(100, 500):.2f}")
                
        elif col_type == "text":
            if "news" in col_name.lower():
                # News headlines for stock context
                news_templates = [
                    "Company reports strong Q{} earnings",
                    "Stock rises on positive analyst outlook",
                    "New product launch drives investor confidence",
                    "Market volatility affects trading volume",
                    "CEO announces expansion plans",
                    "Quarterly revenue beats expectations",
                    "Partnership deal boosts stock performance",
                    "Industry trends favor growth prospects"
                ]
                for i in range(rows):
                    template = random.choice(news_templates)
                    if "{}" in template:
                        template = template.format(random.randint(1, 4))
                    data[col_name].append(template)
            else:
                # Generic text data
                items = ["Item A", "Item B", "Item C", "Product X", "Service Y", "Plan Z"]
                for i in range(rows):
                    data[col_name].append(f"{random.choice(items)} {i+1}")
    
    # Create DataFrame from generated data
    new_df = pd.DataFrame(data)
    
    # Update the workbook sheet
    for col_idx, col_name in enumerate(new_df.columns):
        if col_idx < len(df.columns):
            df_col = df.columns[col_idx]
            # Set header
            df.iloc[0, col_idx] = col_name
            # Set data
            for row_idx, value in enumerate(new_df[col_name]):
                if row_idx + 1 < len(df):
                    df.iloc[row_idx + 1, col_idx] = value
    
    set_sheet(workbook, sheet, df)
    
    return f"Generated {rows} rows of sample data for {context} with columns: {', '.join([c['name'] for c in columns])}. Data includes realistic values based on the context."