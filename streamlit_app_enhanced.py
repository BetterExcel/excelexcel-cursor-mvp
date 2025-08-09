import os
import sys
import io
from typing import Dict, List
from datetime import datetime

# Load environment variables first, before any other imports that might need them
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd

from app.services.workbook import (
    new_workbook,
    get_sheet,
    set_sheet,
    list_sheets,
    ensure_sheet,
)
from app.ui.formula import evaluate_formula
from app.agent.agent import run_agent, probe_models
import app.charts as charts

st.set_page_config(page_title="Excel‑Cursor MVP - Enhanced", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Excel-like styling with theme support
st.markdown(
    """
<style>
/* Default light theme variables */
:root {
  --xc-surface: #ffffff;
  --xc-surface-2: #f6f8fa;
  --xc-text: #111827;
  --xc-border: rgba(0,0,0,0.12);
  --xc-accent: #0066cc;
  --xc-success: #28a745;
  --xc-warn: #ffc107;
}

/* Dark theme variables - auto detect */
@media (prefers-color-scheme: dark) {
  :root {
    --xc-surface: #0e1117;
    --xc-surface-2: #161b22;
    --xc-text: #e6edf3;
    --xc-border: rgba(255,255,255,0.18);
  }
}

/* Fix Streamlit button styling issues */
.stButton > button {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
  border-radius: 4px !important;
  padding: 0.5rem 1rem !important;
  font-weight: 500 !important;
  min-height: 2.5rem !important;
  width: 100% !important;
}

.stButton > button:hover {
  background-color: var(--xc-surface-2) !important;
  border-color: var(--xc-accent) !important;
}

/* Fix selectbox styling */
.stSelectbox > div > div > div {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix selectbox dropdown */
.stSelectbox div[data-baseweb="select"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

.stSelectbox div[data-baseweb="select"] > div {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix selectbox options/menu */
.stSelectbox ul, .stSelectbox li {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

/* More specific Streamlit component targeting */
div[data-testid="stSelectbox"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

div[data-testid="stButton"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

div[data-testid="stTextInput"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

div[data-testid="stFileUploader"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

/* Fix all input elements */
input, select, textarea {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix Streamlit's internal styling classes */
.css-1cpxqw2, .css-1d391kg, .css-1cypcdb {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
}

/* Fix any remaining text visibility issues */
span, p, div {
  color: var(--xc-text) !important;
}

/* Fix text input styling */
.stTextInput > div > div > input {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix file uploader styling */
.stFileUploader > div > div {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix checkbox styling */
.stCheckbox > label {
  color: var(--xc-text) !important;
}

/* Fix radio button styling */
.stRadio > label {
  color: var(--xc-text) !important;
}

/* Fix multiselect styling */
.stMultiSelect > div > div {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix number input styling */
.stNumberInput > div > div > input {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix text area styling */
.stTextArea > div > div > textarea {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix date input styling */
.stDateInput > div > div > input {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix time input styling */
.stTimeInput > div > div > input {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix slider styling */
.stSlider > div > div {
  color: var(--xc-text) !important;
}

/* Fix progress bar styling */
.stProgress > div > div {
  background-color: var(--xc-accent) !important;
}

/* Fix metric styling */
.metric-container {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
  border-radius: 4px !important;
  padding: 1rem !important;
}

/* Fix expander styling */
.streamlit-expanderHeader {
  background-color: var(--xc-surface-2) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

.streamlit-expanderContent {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix tabs styling */
.stTabs [data-baseweb="tab-list"] {
  background-color: var(--xc-surface-2) !important;
}

.stTabs [data-baseweb="tab"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix columns styling */
.stColumns {
  background-color: transparent !important;
}

/* Fix container styling */
.stContainer {
  background-color: transparent !important;
}

/* Fix dataframe styling */
.stDataFrame, .stDataFrame iframe {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix plotly chart background */
.js-plotly-plot {
  background-color: var(--xc-surface) !important;
}

/* Fix markdown styling */
.stMarkdown {
  color: var(--xc-text) !important;
}

/* Fix caption styling */
.stCaption {
  color: var(--xc-text) !important;
  opacity: 0.7 !important;
}

/* Fix info/success/warning/error boxes */
.stAlert {
  background-color: var(--xc-surface-2) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Fix spinner styling */
.stSpinner {
  color: var(--xc-accent) !important;
}

/***** Components *****/
.ribbon-container { 
  background: var(--xc-surface-2); 
  border-bottom: 1px solid var(--xc-border); 
  padding: 10px; 
  margin-bottom: 20px; 
}

.stDataFrame { 
  border: 1px solid var(--xc-border); 
}

.chat-container { 
  max-height: 400px; 
  overflow-y: auto; 
  border: 1px solid var(--xc-border); 
  border-radius: 6px; 
  padding: 10px; 
  margin-bottom: 10px; 
  background: var(--xc-surface-2); 
  color: var(--xc-text); 
}

.user-message { 
  background: var(--xc-surface-2); 
  color: var(--xc-text); 
  padding: 8px; 
  border-radius: 6px; 
  margin: 5px 0; 
  border-left: 3px solid var(--xc-accent); 
}

.assistant-message { 
  background: var(--xc-surface); 
  color: var(--xc-text); 
  padding: 8px; 
  border-radius: 6px; 
  margin: 5px 0; 
  border-left: 3px solid var(--xc-success); 
}

.ai-operation { 
  background: rgba(255,193,7,0.15); 
  color: var(--xc-text); 
  padding: 5px 8px; 
  border-radius: 4px; 
  margin: 2px 0; 
  border-left: 3px solid var(--xc-warn); 
  font-size: 0.9em; 
}

.status-working { color: var(--xc-warn); font-weight: bold; }
.status-success { color: var(--xc-success); font-weight: bold; }
.status-error { color: #dc3545; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state with enhanced structure
if "workbook" not in st.session_state:
    st.session_state.workbook = new_workbook()
if "current_sheet" not in st.session_state:
    st.session_state.current_sheet = list(st.session_state.workbook.keys())[0]
if "chat_histories" not in st.session_state:
    # Chat history per sheet
    st.session_state.chat_histories = {sheet: [] for sheet in st.session_state.workbook.keys()}
if "active_ribbon_tab" not in st.session_state:
    st.session_state.active_ribbon_tab = "Home"
if "ai_operations_log" not in st.session_state:
    st.session_state.ai_operations_log = []
# Model selection state
if "chat_model" not in st.session_state:
    st.session_state.chat_model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
if "model_probe" not in st.session_state:
    st.session_state.model_probe = None

# Ensure chat history exists for current sheet
if st.session_state.current_sheet not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.current_sheet] = []

# Header with title and ribbon tabs
st.title("📊 Excel‑Cursor - Enhanced AI Spreadsheet")

# Ribbon-style navigation
ribbon_tabs = ["Home", "Insert", "Formulas", "Data", "Review"]
cols = st.columns(len(ribbon_tabs))

for i, tab in enumerate(ribbon_tabs):
    with cols[i]:
        if st.button(tab, key=f"ribbon_{tab}"):
            st.session_state.active_ribbon_tab = tab

st.markdown("---")

# Sheet selector and info bar
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
with col1:
    sheet_names = list_sheets(st.session_state.workbook)
    new_sheet = st.selectbox("📋 Current Sheet", options=sheet_names, 
                           index=sheet_names.index(st.session_state.current_sheet))
    if new_sheet != st.session_state.current_sheet:
        st.session_state.current_sheet = new_sheet
        # Ensure chat history exists for new sheet
        if new_sheet not in st.session_state.chat_histories:
            st.session_state.chat_histories[new_sheet] = []

with col2:
    df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
    st.info(f"📊 Rows: {len(df)} | Columns: {len(df.columns)} | Sheet: {st.session_state.current_sheet}")

with col3:
    st.caption("🤖 AI operations will be highlighted in yellow")

# Ribbon content based on active tab
if st.session_state.active_ribbon_tab == "Home":
    st.markdown("### 🏠 Home")
    home_cols = st.columns([1,1,1,1,1])
    
    with home_cols[0]:
        st.markdown("**📁 File**")
        uploaded = st.file_uploader("Import", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded is not None:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
            st.success("✅ File imported!")
            st.rerun()
    
    with home_cols[1]:
        st.markdown("**💾 Export**")
        export_format = st.selectbox("Format", ["CSV", "XLSX"], label_visibility="collapsed")
        if st.button("Download"):
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            if export_format == "CSV":
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button("📥 Download CSV", data=buf.getvalue(), 
                                 file_name=f"{st.session_state.current_sheet}.csv", mime="text/csv")
            else:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=st.session_state.current_sheet, index=False)
                st.download_button("📥 Download XLSX", data=out.getvalue(), 
                                 file_name=f"{st.session_state.current_sheet}.xlsx", 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    with home_cols[2]:
        st.markdown("**➕ Insert**")
        if st.button("New Sheet"):
            sheet_name = f"Sheet{len(list_sheets(st.session_state.workbook)) + 1}"
            ensure_sheet(st.session_state.workbook, sheet_name, 20, 5)
            st.session_state.chat_histories[sheet_name] = []
            st.session_state.current_sheet = sheet_name
            st.rerun()
    
    with home_cols[3]:
        st.markdown("**🔧 Edit**")
        if st.button("Clear Sheet"):
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            df.iloc[:, :] = ""
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
            st.rerun()
    
    with home_cols[4]:
        st.markdown("**🎨 Format**")
        st.write("Cell formatting options")

elif st.session_state.active_ribbon_tab == "Insert":
    st.markdown("### ➕ Insert")
    insert_cols = st.columns([1,1,1,1])
    
    with insert_cols[0]:
        st.markdown("**📊 Charts**")
        if st.button("Create Chart"):
            st.session_state.show_chart_builder = True
    
    with insert_cols[1]:
        st.markdown("**🔧 Functions**")
        st.selectbox("Function", ["SUM", "AVERAGE", "COUNT", "MAX", "MIN"], key="insert_func")
        
    with insert_cols[2]:
        st.markdown("**📝 Objects**")
        st.write("Text boxes, shapes")
        
    with insert_cols[3]:
        st.markdown("**📈 Analysis**")
        if st.button("Quick Stats"):
            st.session_state.show_stats = True

elif st.session_state.active_ribbon_tab == "Data":
    st.markdown("### 📊 Data Operations")
    data_cols = st.columns([1,1,1,1])
    
    with data_cols[0]:
        st.markdown("**🔄 Sort**")
        sort_col = st.selectbox("Column", df.columns if not df.empty else [], key="sort_col")
        sort_asc = st.checkbox("Ascending", True, key="sort_asc")
        if st.button("Apply Sort") and sort_col:
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            df_sorted = df.sort_values(by=sort_col, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df_sorted)
            st.success(f"✅ Sorted by {sort_col} ({'ascending' if sort_asc else 'descending'})")
            st.rerun()
    
    with data_cols[1]:
        st.markdown("**🔍 Filter**")
        filt_col = st.selectbox("Column", df.columns if not df.empty else [], key="filt_col")
        filt_val = st.text_input("Value", key="filt_val")
        if st.button("Apply Filter") and filt_col:
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            filtered_df = df[df[filt_col].astype(str) == str(filt_val)]
            st.session_state.filtered_df = filtered_df
            st.success(f"✅ Filtered {filt_col} = {filt_val}")
    
    with data_cols[2]:
        st.markdown("**📊 Analysis**")
        if st.button("Pivot Table"):
            st.info("📊 Pivot tables coming soon!")
        
    with data_cols[3]:
        st.markdown("**🔗 Connections**")
        st.write("External data sources")

elif st.session_state.active_ribbon_tab == "Formulas":
    st.markdown("### 🧮 Formulas")
    formula_cols = st.columns([1,1,1,1])
    
    with formula_cols[0]:
        st.markdown("**📊 Statistical**")
        stat_functions = ["SUM", "AVERAGE", "STDEV", "VAR", "MEDIAN", "MODE", "COUNT"]
        selected_stat = st.selectbox("Function", stat_functions, key="stat_func")
        
    with formula_cols[1]:
        st.markdown("**📅 Date & Time**")
        date_functions = ["TODAY", "NOW", "WEEKDAY", "YEAR", "MONTH", "DAY"]
        selected_date = st.selectbox("Function", date_functions, key="date_func")
        
    with formula_cols[2]:
        st.markdown("**🔤 Text**")
        text_functions = ["CONCATENATE", "LEFT", "RIGHT", "MID", "LEN", "UPPER", "LOWER"]
        selected_text = st.selectbox("Function", text_functions, key="text_func")
        
    with formula_cols[3]:
        st.markdown("**🔍 Lookup**")
        lookup_functions = ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH"]
        selected_lookup = st.selectbox("Function", lookup_functions, key="lookup_func")

elif st.session_state.active_ribbon_tab == "Review":
    st.markdown("### 🔍 Review")
    review_cols = st.columns([1,1,1,1])
    
    with review_cols[0]:
        st.markdown("**💬 Comments**")
        st.write("Add comments to cells")
        
    with review_cols[1]:
        st.markdown("**🔐 Protection**")
        st.write("Protect sheets and cells")
        
    with review_cols[2]:
        st.markdown("**📋 Track Changes**")
        st.write("Review modifications")
        
    with review_cols[3]:
        st.markdown("**✅ Validation**")
        st.write("Data validation rules")

# Formula Bar
st.markdown("### 🧮 Formula Bar")

# Add helpful instructions
with st.expander("📖 Formula Help", expanded=False):
    st.markdown("""
    **How to use formulas:**
    
    **Basic Usage:**
    1. Enter a cell reference (e.g., A1, B2, C10)
    2. Enter a formula starting with `=`
    3. Click Apply to execute
    
    **Supported Functions:**
    - **Math:** `=A1+B1`, `=A1*B1/2`, `=A1-5`
    - **Sum:** `=SUM(A1:A5)` - adds all values in range
    - **Average:** `=AVERAGE(A1:A5)` - calculates mean
    - **Count:** `=COUNT(A1:A5)` - counts non-empty cells
    - **Min/Max:** `=MIN(A1:A5)`, `=MAX(A1:A5)`
    - **Statistics:** `=STDEV(A1:A5)`, `=MEDIAN(A1:A5)`
    - **Date:** `=TODAY()`, `=NOW()`
    - **Text:** `=CONCATENATE(A1,B1)`, `=LEN(A1)`
    
    **Examples:**
    - `=SUM(A1:A10)` - Sum of A1 through A10
    - `=A1*1.1` - Increase A1 by 10%
    - `=AVERAGE(B1:B5)*2` - Double the average
    - `=TODAY()` - Current date
    - `=CONCATENATE(A1," ",B1)` - Join A1 and B1 with space
    """)

formula_cols = st.columns([0.2, 0.5, 0.2, 0.1])
with formula_cols[0]:
    target_cell = st.text_input("📍 Cell (e.g., A1)", key="cell_addr", placeholder="A1")
with formula_cols[1]:
    formula = st.text_input("📝 Formula", key="formula_text", 
                          placeholder="=SUM(A1:A10) or =TODAY() or =CONCATENATE(A1,B1)")
with formula_cols[2]:
    apply_col = st.checkbox("Apply to column", help="Apply formula to entire column")
with formula_cols[3]:
    if st.button("⚡ Apply"):
        if target_cell and formula:
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            try:
                if str(formula).startswith("="):
                    if apply_col:
                        # Apply to entire column
                        col_letter = ''.join([c for c in target_cell if c.isalpha()])
                        if col_letter in df.columns:
                            success_count = 0
                            for r in range(len(df)):
                                try:
                                    val = evaluate_formula(formula, df, current_row=r)
                                    df.at[r, col_letter] = val
                                    success_count += 1
                                except Exception as e:
                                    # Skip rows with errors but continue
                                    continue
                            st.success(f"✅ Formula applied to {success_count} rows in column {col_letter}")
                        else:
                            st.error(f"❌ Column {col_letter} not found")
                    else:
                        # Apply to single cell
                        val = evaluate_formula(formula, df)
                        from app.services.workbook import set_cell_by_a1
                        set_cell_by_a1(df, target_cell, val)
                        st.success(f"✅ Formula applied to {target_cell}: {val}")
                else:
                    # Direct value (not a formula)
                    from app.services.workbook import set_cell_by_a1
                    set_cell_by_a1(df, target_cell, formula)
                    st.success(f"✅ Value set in {target_cell}: {formula}")
                
                set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
                
                # Log the operation
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.ai_operations_log.append({
                    'timestamp': timestamp,
                    'operation': f"Formula applied: {formula} to {target_cell}",
                    'type': 'user'
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error applying formula: {e}")
                st.info("💡 Check the formula syntax and cell references")

st.divider()

# Quick formula test section
with st.expander("🧪 Quick Formula Test", expanded=False):
    st.markdown("**Try some sample data and formulas:**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Add Sample Data"):
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            # Add some sample data for testing
            sample_data = {
                'A': [10, 20, 30, 40, 50],
                'B': [5, 15, 25, 35, 45],
                'C': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5'],
                'D': [None, None, None, None, None],  # For formula results
                'E': [None, None, None, None, None]   # For more formulas
            }
            
            for col, values in sample_data.items():
                if col in df.columns:
                    for i, val in enumerate(values):
                        if i < len(df):
                            df.at[i, col] = val
            
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
            st.success("✅ Sample data added!")
            st.rerun()
    
    with col2:
        st.markdown("""
        **Try these formulas:**
        - `=A1+B1` in cell D1
        - `=SUM(A1:A5)` in cell D2  
        - `=AVERAGE(A1:A5)` in cell D3
        - `=CONCATENATE(C1," Value: ",A1)` in cell E1
        - `=TODAY()` in cell E2
        """)

# Main spreadsheet display
st.markdown(f"### 📊 Sheet: {st.session_state.current_sheet}")

# Display the data editor
base_df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
show_df = st.session_state.get("filtered_df", base_df)

# Add row and column indicators (simplified)
st.caption("💡 **Tip:** Click on any cell to edit directly. Changes save automatically when you press Enter or click elsewhere.")

edited = st.data_editor(
    show_df, 
    num_rows="dynamic", 
    use_container_width=True,
    key="main_editor",
    hide_index=False,
    column_config={
        # Make all columns editable
        col: st.column_config.TextColumn(
            help=f"Column {col}",
            max_chars=1000,
        ) for col in show_df.columns
    }
)

# Track changes and mark as user operations
if show_df is base_df and not edited.equals(base_df):
    set_sheet(st.session_state.workbook, st.session_state.current_sheet, edited)
    
    # Log user edit
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.ai_operations_log.append({
        'timestamp': timestamp,
        'operation': "Manual data edit",
        'type': 'user'
    })
    
    # Force refresh to show changes immediately
    st.rerun()

# Chart builder section
if st.session_state.get("show_chart_builder", False):
    st.divider()
    try:
        from app.charts import display_chart_builder
        display_chart_builder(base_df)
    except ImportError:
        st.error("📊 Chart functionality requires plotly. Install with: pip install plotly")
    
    if st.button("🔽 Hide Chart Builder"):
        st.session_state.show_chart_builder = False
        st.rerun()

# Quick stats section
if st.session_state.get("show_stats", False):
    st.divider()
    st.markdown("### 📈 Quick Statistics")
    
    if not base_df.empty:
        numeric_cols = base_df.select_dtypes(include=[pd.api.types.is_numeric_dtype]).columns.tolist()
        if numeric_cols:
            stats_col = st.selectbox("📊 Select Column for Stats", numeric_cols)
            if stats_col:
                col_data = pd.to_numeric(base_df[stats_col], errors='coerce').dropna()
                if not col_data.empty:
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("📊 Count", len(col_data))
                    with stats_cols[1]:
                        st.metric("📈 Mean", f"{col_data.mean():.2f}")
                    with stats_cols[2]:
                        st.metric("📏 Std Dev", f"{col_data.std():.2f}")
                    with stats_cols[3]:
                        st.metric("🎯 Median", f"{col_data.median():.2f}")
                    
                    # Quick visualization
                    try:
                        from app.charts import quick_stats_chart
                        fig = quick_stats_chart(base_df, stats_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.info("📊 Advanced stats charts require plotly")
        else:
            st.warning("⚠️ No numeric columns found for statistics")
    
    if st.button("🔽 Hide Stats"):
        st.session_state.show_stats = False
        st.rerun()

# Sidebar for AI Chat
with st.sidebar:
    # Appearance controls
    with st.expander("🎨 Appearance", expanded=False):
        theme_choice = st.selectbox("Theme", ["Auto", "Light", "Dark"], index=0, key="theme_choice")
    
    # Apply theme overrides with higher specificity
    if st.session_state.get("theme_choice") == "Light":
        st.markdown(
            """
            <style>
            /* Force light theme - comprehensive override */
            :root, html, body, .stApp {
                --xc-surface: #ffffff !important;
                --xc-surface-2: #f6f8fa !important;
                --xc-text: #111827 !important;
                --xc-border: rgba(0,0,0,0.12) !important;
            }
            
            /* Override Streamlit's main app */
            .stApp {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            /* Force all buttons to be visible */
            .stButton > button {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.12) !important;
            }
            
            /* Force selectboxes */
            .stSelectbox > div > div {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            .stSelectbox > div > div > div {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.12) !important;
            }
            
            /* Force selectbox options */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            /* Force dropdown menu */
            .stSelectbox ul {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            .stSelectbox li {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            /* Force text inputs */
            .stTextInput > div > div > input {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            /* Force sidebar styling */
            .css-1d391kg, .css-1cypcdb, section[data-testid="stSidebar"] {
                background-color: #f6f8fa !important;
                color: #111827 !important;
            }
            
            /* Chat messages in light mode */
            .user-message, .assistant-message, .ai-operation {
                color: #111827 !important;
            }
            
            /* Fix any remaining dark elements */
            div, span, p, h1, h2, h3, h4, h5, h6 {
                color: #111827 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.get("theme_choice") == "Dark":
        st.markdown(
            """
            <style>
            /* Force dark theme - comprehensive override */
            :root, html, body, .stApp {
                --xc-surface: #0e1117 !important;
                --xc-surface-2: #161b22 !important;
                --xc-text: #e6edf3 !important;
                --xc-border: rgba(255,255,255,0.18) !important;
            }
            
            /* Override Streamlit's main app */
            .stApp {
                background-color: #0e1117 !important;
                color: #e6edf3 !important;
            }
            
            /* Force all buttons to be visible */
            .stButton > button {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force selectboxes */
            .stSelectbox > div > div {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            .stSelectbox > div > div > div {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force selectbox options */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            /* Force dropdown menu */
            .stSelectbox ul {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            .stSelectbox li {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            /* Force text inputs */
            .stTextInput > div > div > input {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            /* Force sidebar styling */
            .css-1d391kg, .css-1cypcdb, section[data-testid="stSidebar"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            /* Chat messages in dark mode */
            .user-message, .assistant-message, .ai-operation {
                color: #e6edf3 !important;
            }
            
            /* Fix any remaining light elements */
            div, span, p, h1, h2, h3, h4, h5, h6 {
                color: #e6edf3 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.header(f"🤖 AI Assistant ({st.session_state.chat_model})")
    st.caption(f"Chat for Sheet: **{st.session_state.current_sheet}**")
    
    # Model selection and probing
    with st.expander("Model settings", expanded=False):
        if st.button("Test latest models"):
            with st.spinner("Probing models..."):
                st.session_state.model_probe = probe_models()
                if st.session_state.model_probe.get("working"):
                    st.session_state.chat_model = st.session_state.model_probe["working"][0]
        probe = st.session_state.model_probe
        if probe:
            if probe.get("working"):
                st.success("Working: " + ", ".join(probe["working"]))
            if probe.get("failed"):
                st.error("Some models failed:")
                st.text("\n".join([f"{model}: {error[:100]}..." for model, error in probe["failed"].items()]))
        available = (probe["working"] if probe and probe.get("working") else [
            "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4-turbo"
        ])
        # Ensure current model is in options
        if st.session_state.chat_model not in available:
            available = [st.session_state.chat_model] + [m for m in available if m != st.session_state.chat_model]
        selected = st.selectbox("Use model", options=available, index=0, key="chat_model_select")
        st.session_state.chat_model = selected
    
    # Chat history display
    if st.session_state.chat_histories[st.session_state.current_sheet]:
        st.markdown("### 💬 Chat History")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_histories[st.session_state.current_sheet] = []
            st.rerun()
        
        # Display chat messages in a container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_histories[st.session_state.current_sheet]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class=\"user-message\">
                        <strong>👤 You:</strong> {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class=\"assistant-message\">
                        <strong>🤖 AI:</strong> {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # AI Operations Log
    if st.session_state.ai_operations_log:
        st.markdown("### 📝 Recent Operations")
        recent_ops = st.session_state.ai_operations_log[-10:]  # Show last 10
        for op in recent_ops:
            icon = "👤" if op['type'] == 'user' else "🤖"
            st.markdown(f"""
            <div class=\"ai-operation\">
                {icon} <small>{op['timestamp']}</small><br>
                {op['operation']}
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Chat input
    user_msg = st.chat_input("Ask me to perform spreadsheet operations...")
    
    if user_msg:
        # Add user message to history
        st.session_state.chat_histories[st.session_state.current_sheet].append({
            "role": "user", 
            "content": user_msg
        })
        
        # Show working status
        with st.spinner("🔄 AI is working..."):
            try:
                # Get chat history for context
                chat_history = st.session_state.chat_histories[st.session_state.current_sheet]
                
                # Call the agent with context and selected model
                reply = run_agent(
                    user_msg=user_msg,
                    workbook=st.session_state.workbook,
                    current_sheet=st.session_state.current_sheet,
                    chat_history=chat_history[:-1],  # Exclude the current message
                    model_name=st.session_state.chat_model
                )
                
                # Log AI operation
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.ai_operations_log.append({
                    'timestamp': timestamp,
                    'operation': f"AI processed: {user_msg[:50]}...",
                    'type': 'ai'
                })
                
            except Exception as e:
                reply = f"❌ Agent error: {e}. Please check your OpenAI API key and network connection."
        
        # Add assistant response to history
        st.session_state.chat_histories[st.session_state.current_sheet].append({
            "role": "assistant", 
            "content": reply
        })
        
        st.rerun()

# Status bar at bottom
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.caption(f"📊 Ready | Sheet: {st.session_state.current_sheet}")
with col2:
    st.caption(f"💬 Messages: {len(st.session_state.chat_histories[st.session_state.current_sheet])}")
with col3:
    st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
