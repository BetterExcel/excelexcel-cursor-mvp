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
# from app.explanation import ExplanationWorkflow  # FALLBACK REMOVED - CLEAN PIPELINE ONLY
from app.explanation.intelligent_workflow import IntelligentExplanationWorkflow
from app.explanation.proper_langchain_workflow import create_proper_langchain_workflow
from app.explanation.local_llm import get_local_llm, check_local_llm_availability
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

/* Specifically fix chat input textarea */
textarea[data-testid="stChatInputTextArea"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
  opacity: 1 !important;
  visibility: visible !important;
  z-index: 1000 !important;
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

/* Force chat input container visibility */
.st-emotion-cache-r0oa6g {
  background-color: var(--xc-surface) !important;
  border: 1px solid var(--xc-border) !important;
}

/* Force all chat input related elements to be visible */
div[data-testid="stChatInput"] {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  opacity: 1 !important;
  visibility: visible !important;
}

div[data-testid="stChatInput"] textarea {
  background-color: var(--xc-surface) !important;
  color: var(--xc-text) !important;
  border: 1px solid var(--xc-border) !important;
  opacity: 1 !important;
  visibility: visible !important;
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
  background: linear-gradient(135deg, var(--xc-surface-2), var(--xc-surface)); 
  color: var(--xc-text); 
  padding: 12px 16px; 
  border-radius: 12px; 
  margin: 8px 0; 
  border-left: 4px solid var(--xc-accent); 
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  animation: slideIn 0.3s ease-out;
}

.assistant-message { 
  background: linear-gradient(135deg, var(--xc-surface), var(--xc-surface-2)); 
  color: var(--xc-text); 
  padding: 12px 16px; 
  border-radius: 12px; 
  margin: 8px 0; 
  border-left: 4px solid var(--xc-success); 
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  animation: slideIn 0.3s ease-out;
}

.ai-operation { 
  background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.05)); 
  color: var(--xc-text); 
  padding: 8px 12px; 
  border-radius: 8px; 
  margin: 4px 0; 
  border-left: 3px solid var(--xc-warn); 
  font-size: 0.85em; 
  transition: all 0.2s ease;
}

.ai-operation:hover {
  background: linear-gradient(135deg, rgba(255,193,7,0.25), rgba(255,193,7,0.1));
  transform: translateX(2px);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.chat-header {
  background: linear-gradient(135deg, var(--xc-accent), #0052a3);
  color: white;
  padding: 12px 16px;
  border-radius: 8px 8px 0 0;
  margin-bottom: 0;
  font-weight: 600;
}

.chat-container {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid var(--xc-border);
  border-radius: 0 0 8px 8px;
  background: var(--xc-surface);
}

.model-badge {
  background: var(--xc-accent);
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75em;
  font-weight: 500;
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
    st.session_state.chat_model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4-turbo-2024-04-09"
if "model_probe" not in st.session_state:
    st.session_state.model_probe = None
if "enable_explanations" not in st.session_state:
    st.session_state.enable_explanations = True

# Upload tracking to prevent duplicates
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_uploaded_file_hash" not in st.session_state:
    st.session_state.last_uploaded_file_hash = None

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
    # Recent operations in header area
    if st.session_state.ai_operations_log:
        with st.expander("📝 Recent Operations", expanded=False):
            recent_ops = st.session_state.ai_operations_log[-5:]  # Show last 5
            for op in recent_ops:
                icon = "👤" if op['type'] == 'user' else "🤖"
                st.markdown(f"""
                <div class=\"ai-operation\">
                    {icon} <small>{op['timestamp']}</small><br>
                    {op['operation']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("🤖 AI operations will be highlighted in yellow")

# Ribbon content based on active tab
if st.session_state.active_ribbon_tab == "Home":
    st.markdown("### 🏠 Home")
    home_cols = st.columns([1,1,1,1,1])
    
    with home_cols[0]:
        st.markdown("**📁 File Operations**")
        
        # File Upload Section
        uploaded = st.file_uploader("📤 Import File", type=["csv", "xlsx"], label_visibility="collapsed")
        
        # Check if this is a new file upload (prevent duplicates)
        if uploaded is not None:
            # Create a unique identifier for this file
            file_hash = f"{uploaded.name}_{uploaded.size}_{uploaded.type}"
            
            if file_hash != st.session_state.last_uploaded_file_hash:
                st.session_state.last_uploaded_file = uploaded
                st.session_state.last_uploaded_file_hash = file_hash
                
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                
                set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
                
                # Auto-save imported file to data directory
                try:
                    import os
                    from datetime import datetime
                    
                    # Create data directory if it doesn't exist
                    data_dir = "data"
                    os.makedirs(data_dir, exist_ok=True)
                    
                    # Save with original filename (with timestamp if needed)
                    base_name = os.path.splitext(uploaded.name)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{base_name}_imported_{timestamp}.csv"
                    filepath = os.path.join(data_dir, filename)
                    
                    # Save the imported data
                    df.to_csv(filepath, index=False)
                    
                    st.success(f"✅ File imported and saved as {filename}!")
                except Exception as e:
                    st.success("✅ File imported!")
                    
                st.rerun()
        
        # File Selection from Data Directory
        try:
            import glob
            data_files = glob.glob("data/*.csv")
            if data_files:
                # Get just the filenames without path
                file_options = [os.path.basename(f) for f in data_files]
                selected_file = st.selectbox("📂 Load Saved File", ["Select a file..."] + file_options, 
                                           key="file_selector")
                
                if selected_file and selected_file != "Select a file..." and st.button("📖 Load Selected", key="load_file_btn"):
                    filepath = os.path.join("data", selected_file)
                    try:
                        df = pd.read_csv(filepath)
                        set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
                        st.success(f"✅ Loaded {selected_file}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
            else:
                st.info("📂 No saved files found in data directory")
        except Exception as e:
            st.warning("Could not load file list")
    
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

# Formula Bar with enhanced styling
st.markdown("### 🧮 Formula Bar")
st.markdown("*Apply formulas to individual cells or entire columns*")

# Add helpful instructions
with st.expander("📖 Formula Help", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Basic Usage:**
        1. Enter a cell reference (e.g., A1, B2, C10)
        2. Enter a formula starting with `=`
        3. Click Apply to execute
        
        **Math Functions:**
        - `=A1+B1` - Add two cells
        - `=A1*B1/2` - Multiply and divide
        - `=A1-5` - Subtract constant
        """)
    with col2:
        st.markdown("""
        **Statistical Functions:**
        - `=SUM(A1:A10)` - Sum range
        - `=AVERAGE(A1:A10)` - Average
        - `=COUNT(A1:A10)` - Count non-empty
        - `=MIN(A1:A10)` - Minimum value
        - `=MAX(A1:A10)` - Maximum value
        
        **Other Functions:**
        - `=TODAY()` - Current date
        - `=CONCATENATE(A1,B1)` - Join text
        """)

# Enhanced formula input with better layout
formula_container = st.container()
with formula_container:
    formula_cols = st.columns([0.2, 0.5, 0.2, 0.1])
    with formula_cols[0]:
        target_cell = st.text_input("📍 **Target Cell**", key="cell_addr", placeholder="A1", help="Enter the cell where you want to apply the formula")
    with formula_cols[1]:
        formula = st.text_input("📝 **Formula**", key="formula_text", 
                              placeholder="=SUM(A1:A10) or =TODAY() or =CONCATENATE(A1,B1)", 
                              help="Enter a formula starting with = or a direct value")
    with formula_cols[2]:
        apply_col = st.checkbox("📋 **Apply to Column**", help="Apply formula to entire column instead of single cell")
    with formula_cols[3]:
        apply_button = st.button("⚡ **Apply**", type="primary", help="Execute the formula")
    
    if apply_button:
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
            # Add some sample data for testing with proper data types
            sample_data = {
                'A': [10, 20, 30, 40, 50],
                'B': [5.5, 15.2, 25.7, 35.1, 45.9],
                'C': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5'],
                'D': [None, None, None, None, None],  # For formula results
                'E': [None, None, None, None, None]   # For more formulas
            }
            
            for col, values in sample_data.items():
                if col in df.columns:
                    for i, val in enumerate(values):
                        if i < len(df):
                            df.at[i, col] = val
            
            # Convert columns to appropriate types
            df['A'] = pd.to_numeric(df['A'], errors='coerce')
            df['B'] = pd.to_numeric(df['B'], errors='coerce')
            
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
            st.success("✅ Sample data added with proper data types!")
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

# Create smart column configuration based on data types
def create_column_config(df):
    """Create appropriate column configuration based on data types"""
    config = {}
    for col in df.columns:
        try:
            # Check the data type of the column
            sample_data = df[col].dropna()
            if len(sample_data) > 0:
                # Check if column contains formulas (starts with =)
                has_formulas = df[col].astype(str).str.startswith('=').any()
                
                if has_formulas:
                    # Formula column - always use text to avoid conflicts
                    config[col] = st.column_config.TextColumn(
                        help=f"Formula column {col}",
                        max_chars=1000,
                    )
                else:
                    # Check for mixed data types by converting to string and checking patterns
                    col_str = df[col].astype(str)
                    has_text = col_str.str.contains(r'[a-zA-Z]', na=False).any()
                    has_numbers = col_str.str.match(r'^-?\d+\.?\d*$', na=False).any()
                    
                    if has_text and has_numbers:
                        # Mixed data - use text column to avoid conflicts
                        config[col] = st.column_config.TextColumn(
                            help=f"Mixed data column {col}",
                            max_chars=1000,
                        )
                    elif pd.api.types.is_numeric_dtype(df[col]) and not has_text:
                        # Pure numeric data
                        if df[col].dtype == 'int64':
                            config[col] = st.column_config.NumberColumn(
                                help=f"Integer column {col}",
                                step=1,
                                format="%d"
                            )
                        else:
                            config[col] = st.column_config.NumberColumn(
                                help=f"Numeric column {col}",
                                step=0.01,
                                format="%.2f"
                            )
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        # DateTime data
                        config[col] = st.column_config.DatetimeColumn(
                            help=f"Date/time column {col}"
                        )
                    else:
                        # Text or other data - use text column
                        config[col] = st.column_config.TextColumn(
                            help=f"Text column {col}",
                            max_chars=1000,
                        )
            else:
                # Empty column - default to text
                config[col] = st.column_config.TextColumn(
                    help=f"Column {col}",
                    max_chars=1000,
                )
        except Exception:
            # If any error occurs, default to text column
            config[col] = st.column_config.TextColumn(
                help=f"Column {col}",
                max_chars=1000,
            )
    return config

# Create Excel-style display with proper row numbering
display_df = show_df.copy()
display_df.index = range(1, len(display_df) + 1)  # Start from 1 instead of 0

edited = st.data_editor(
    display_df, 
    num_rows="dynamic", 
    width='stretch',
    key="main_editor",
    hide_index=False,
    column_config=create_column_config(show_df)
)

# Convert back to 0-based indexing for internal processing
if not edited.equals(display_df):
    # Reset index back to 0-based for internal consistency
    edited.index = range(len(edited))
    show_df = edited

# Track changes and mark as user operations
if show_df is base_df and not show_df.equals(base_df):
    set_sheet(st.session_state.workbook, st.session_state.current_sheet, show_df)
    
    # Auto-save to data directory
    try:
        import os
        from datetime import datetime
        
        # Create data directory if it doesn't exist
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{st.session_state.current_sheet}_autosave_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Save the current sheet
        show_df.to_csv(filepath, index=False)
        
        # Keep only last 5 autosave files per sheet to avoid clutter
        import glob
        pattern = os.path.join(data_dir, f"{st.session_state.current_sheet}_autosave_*.csv")
        autosave_files = sorted(glob.glob(pattern))
        if len(autosave_files) > 5:
            for old_file in autosave_files[:-5]:
                try:
                    os.remove(old_file)
                except:
                    pass
    except Exception as e:
        pass  # Silently handle auto-save errors
    
    # Log user edit
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.ai_operations_log.append({
        'timestamp': timestamp,
        'operation': "Manual data edit (auto-saved)",
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
                            st.plotly_chart(fig, width='stretch')
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
    
    # Explanation agent controls
    with st.expander("🤖 AI Assistant", expanded=False):
        st.session_state.enable_explanations = st.checkbox(
            "Enable Smart Explanations", 
            value=st.session_state.enable_explanations,
            help="Show detailed explanations of what changed after each AI operation"
        )
        
        if st.session_state.enable_explanations:
            # Check LLM availability
            llm_info = check_local_llm_availability()
            
            if llm_info['is_available']:
                st.success(f"✅ Intelligent explanations enabled using {llm_info['provider_type']}")
                st.info(f"🤖 Using local LLM: {llm_info['model_name']}")
            else:
                st.warning("⚠️ Using basic explanations (no local LLM available)")
                st.info("💡 Install Ollama or other local LLM for intelligent explanations")
            
            # Show available providers
            with st.expander("🔧 Local LLM Options", expanded=False):
                providers = llm_info['available_providers']
                for provider, available in providers.items():
                    status = "✅ Available" if available else "❌ Not Available"
                    st.text(f"{provider.title()}: {status}")
                
                if not any(providers.values()):
                    st.info("💡 To enable intelligent explanations, install one of:")
                    st.text("• pip install langchain-community[ollama]")
                    st.text("• pip install langchain-community[localai]")
                    st.text("• pip install transformers torch")
        else:
            st.info("ℹ️ Explanations disabled - Only basic AI responses will be shown")
    
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
                --xc-accent: #0066cc !important;
                --xc-success: #28a745 !important;
                --xc-warn: #ffc107 !important;
            }
            
            /* Override Streamlit's main app */
            .stApp {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            /* Force all buttons to be visible - more specific selectors */
            .stButton > button, button[kind="primary"], button[kind="secondary"] {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            .stButton > button:hover {
                background-color: #f6f8fa !important;
                color: #111827 !important;
                border: 1px solid #0066cc !important;
            }
            
            /* Force primary buttons */
            button[data-testid="baseButton-primary"], .stButton > button[type="primary"] {
                background-color: #0066cc !important;
                color: #ffffff !important;
                border: 1px solid #0066cc !important;
            }
            
            /* Force selectboxes */
            .stSelectbox > div > div {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            .stSelectbox > div > div > div {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
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
            
            .stSelectbox li:hover {
                background-color: #f6f8fa !important;
                color: #111827 !important;
            }
            
            /* Force text inputs */
            .stTextInput > div > div > input {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force number inputs */
            .stNumberInput > div > div > input {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force text areas */
            .stTextArea > div > div > textarea {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force data editor/grid styling */
            .stDataFrame, .stDataFrame iframe, .stDataFrame table {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            .stDataFrame th, .stDataFrame td {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.1) !important;
            }
            
            .stDataFrame th {
                background-color: #f6f8fa !important;
                color: #111827 !important;
                font-weight: 600 !important;
            }
            
            /* Force data editor controls */
            div[data-testid="stDataFrame"] {
                background-color: #ffffff !important;
                color: #111827 !important;
            }
            
            div[data-testid="stDataFrame"] * {
                color: #111827 !important;
            }
            
            /* Force sidebar styling */
            .css-1d391kg, .css-1cypcdb, section[data-testid="stSidebar"] {
                background-color: #f6f8fa !important;
                color: #111827 !important;
            }
            
            section[data-testid="stSidebar"] * {
                color: #111827 !important;
            }
            
            /* Chat messages in light mode */
            .user-message {
                background: linear-gradient(135deg, #f6f8fa, #ffffff) !important;
                color: #111827 !important;
                border-left: 4px solid #0066cc !important;
            }
            
            .assistant-message {
                background: linear-gradient(135deg, #ffffff, #f6f8fa) !important;
                color: #111827 !important;
                border-left: 4px solid #28a745 !important;
            }
            
            .ai-operation {
                background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.05)) !important;
                color: #111827 !important;
                border-left: 3px solid #ffc107 !important;
            }
            
            /* Chat container */
            .chat-container {
                background: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #0066cc, #0052a3) !important;
                color: #ffffff !important;
            }
            
            /* Force expander styling */
            .streamlit-expanderHeader {
                background-color: #f6f8fa !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            .streamlit-expanderContent {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force file uploader */
            .stFileUploader > div {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force checkbox and radio styling */
            .stCheckbox > label, .stRadio > label {
                color: #111827 !important;
            }
            
            /* Force metric styling */
            div[data-testid="metric-container"] {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            /* Force all remaining dark elements to light */
            div, span, p, h1, h2, h3, h4, h5, h6, label {
                color: #111827 !important;
            }
            
            /* Force success/error/warning messages */
            .stSuccess {
                background-color: rgba(40, 167, 69, 0.1) !important;
                color: #155724 !important;
                border: 1px solid #28a745 !important;
            }
            
            .stError {
                background-color: rgba(220, 53, 69, 0.1) !important;
                color: #721c24 !important;
                border: 1px solid #dc3545 !important;
            }
            
            .stWarning {
                background-color: rgba(255, 193, 7, 0.1) !important;
                color: #856404 !important;
                border: 1px solid #ffc107 !important;
            }
            
            .stInfo {
                background-color: rgba(0, 102, 204, 0.1) !important;
                color: #0c4a6e !important;
                border: 1px solid #0066cc !important;
            }
            
            /* Force plotly charts */
            .js-plotly-plot {
                background-color: #ffffff !important;
            }
            
            /* Force any remaining Streamlit components */
            .stMarkdown, .stCaption, .stCode {
                color: #111827 !important;
            }
            
            /* Force tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #f6f8fa !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #f6f8fa !important;
                color: #111827 !important;
            }
            
            /* Force spinner */
            .stSpinner {
                color: #0066cc !important;
            }
            
            /* Override any remaining dark backgrounds */
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff !important;
            }
            
            [data-testid="stHeader"] {
                background-color: #ffffff !important;
            }
            
            /* Force all input fields to be visible */
            input, textarea, select {
                background-color: #ffffff !important;
                color: #111827 !important;
                border: 1px solid rgba(0,0,0,0.2) !important;
            }
            
            input:focus, textarea:focus, select:focus {
                border-color: #0066cc !important;
                box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2) !important;
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
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force number inputs */
            .stNumberInput > div > div > input {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force text areas */
            .stTextArea > div > div > textarea {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force data editor/grid styling */
            .stDataFrame, .stDataFrame iframe, .stDataFrame table {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            .stDataFrame th, .stDataFrame td {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.1) !important;
            }
            
            .stDataFrame th {
                background-color: #0e1117 !important;
                color: #e6edf3 !important;
                font-weight: 600 !important;
            }
            
            /* Force data editor controls */
            div[data-testid="stDataFrame"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            div[data-testid="stDataFrame"] * {
                color: #e6edf3 !important;
            }
            
            /* Force sidebar styling */
            .css-1d391kg, .css-1cypcdb, section[data-testid="stSidebar"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            
            section[data-testid="stSidebar"] * {
                color: #e6edf3 !important;
            }
            
            /* Chat messages in dark mode */
            .user-message {
                background: linear-gradient(135deg, #0e1117, #161b22) !important;
                color: #e6edf3 !important;
                border-left: 4px solid #0066cc !important;
            }
            
            .assistant-message {
                background: linear-gradient(135deg, #161b22, #0e1117) !important;
                color: #e6edf3 !important;
                border-left: 4px solid #28a745 !important;
            }
            
            .ai-operation {
                background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.05)) !important;
                color: #e6edf3 !important;
                border-left: 3px solid #ffc107 !important;
            }
            
            /* Chat container */
            .chat-container {
                background: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #0066cc, #0052a3) !important;
                color: #ffffff !important;
            }
            
            /* Force expander styling */
            .streamlit-expanderHeader {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            .streamlit-expanderContent {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force file uploader */
            .stFileUploader > div {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force checkbox and radio styling */
            .stCheckbox > label, .stRadio > label {
                color: #e6edf3 !important;
            }
            
            /* Force metric styling */
            div[data-testid="metric-container"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            /* Force all remaining light elements */
            div, span, p, h1, h2, h3, h4, h5, h6, label {
                color: #e6edf3 !important;
            }
            
            /* Force success/error/warning messages */
            .stSuccess {
                background-color: rgba(40, 167, 69, 0.1) !important;
                color: #155724 !important;
                border: 1px solid #28a745 !important;
            }
            
            .stError {
                background-color: rgba(220, 53, 69, 0.1) !important;
                color: #721c24 !important;
                border: 1px solid #dc3545 !important;
            }
            
            .stWarning {
                background-color: rgba(255, 193, 7, 0.1) !important;
                color: #856404 !important;
                border: 1px solid #ffc107 !important;
            }
            
            .stInfo {
                background-color: rgba(0, 102, 204, 0.1) !important;
                color: #0c4a6e !important;
                border: 1px solid #0066cc !important;
            }
            
            /* Force plotly charts */
            .js-plotly-plot {
                background-color: #161b22 !important;
            }
            
            /* Force any remaining Streamlit components */
            .stMarkdown, .stCaption, .stCode {
                color: #e6edf3 !important;
            }
            
            /* Force tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #161b22 !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #0e1117 !important;
                color: #e6edf3 !important;
            }
            
            /* Force spinner */
            .stSpinner {
                color: #0066cc !important;
            }
            
            /* Override any remaining light backgrounds */
            [data-testid="stAppViewContainer"] {
                background-color: #0e1117 !important;
            }
            
            [data-testid="stHeader"] {
                background-color: #0e1117 !important;
            }
            
            /* Force all input fields to be visible */
            input, textarea, select {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
            }
            
            input:focus, textarea:focus, select:focus {
                border-color: #0066cc !important;
                box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Enhanced AI Assistant Header
    st.markdown(f"""
    <div class="chat-header">
        🤖 AI Assistant
        <span class="model-badge">{st.session_state.chat_model}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**📋 Active Sheet:** `{st.session_state.current_sheet}`")
    
    # Show AI Context (what the AI can see about the current sheet)
    with st.expander("🔍 AI Context (What AI knows about current sheet)", expanded=False):
        current_df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        
        if current_df.empty:
            st.info("📄 Sheet is empty - AI knows this sheet has no data yet")
        else:
            st.markdown("**📊 Sheet Overview:**")
            st.markdown(f"• **Dimensions:** {len(current_df)} rows × {len(current_df.columns)} columns")
            st.markdown(f"• **Columns:** {', '.join(current_df.columns.tolist())}")
            
            # Show column details that AI can see
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📋 Column Details:**")
                for col in current_df.columns:
                    non_null = current_df[col].count()
                    total = len(current_df)
                    dtype = current_df[col].dtype
                    st.markdown(f"• **{col}:** {dtype} ({non_null}/{total} values)")
            
            with col2:
                st.markdown("**🔢 Sample Data (what AI sees):**")
                for col in current_df.columns:
                    sample_values = current_df[col].dropna().head(2).tolist()
                    if sample_values:
                        sample_str = ", ".join([str(v) for v in sample_values])
                        st.markdown(f"• **{col}:** {sample_str}...")
                    else:
                        st.markdown(f"• **{col}:** (no data)")
            
            # Show numeric summaries that AI can see
            numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                st.markdown("**📈 Numeric Data Insights:**")
                for col in numeric_cols[:2]:  # Show first 2 numeric columns
                    series = pd.to_numeric(current_df[col], errors='coerce').dropna()
                    if len(series) > 0:
                        st.markdown(f"• **{col}:** range {series.min():.1f} to {series.max():.1f}, avg {series.mean():.1f}")
        
        st.info("💡 **Tip:** The AI uses this context to provide smarter recommendations and understand your data better!")
    
    # Model selection and probing
    with st.expander("⚙️ Model Settings", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Test Models", help="Test which models work with your API key"):
                with st.spinner("Probing models..."):
                    st.session_state.model_probe = probe_models()
                    if st.session_state.model_probe.get("working"):
                        st.session_state.chat_model = st.session_state.model_probe["working"][0]
        
        probe = st.session_state.model_probe
        if probe:
            if probe.get("working"):
                st.success("✅ Working: " + ", ".join(probe["working"]))
            if probe.get("failed"):
                st.error("❌ Some models failed:")
                for model, error in probe["failed"].items():
                    st.text(f"• {model}: {error[:50]}...")
        
        available = (probe["working"] if probe and probe.get("working") else [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
        ])
        # Ensure current model is in options
        if st.session_state.chat_model not in available:
            available = [st.session_state.chat_model] + [m for m in available if m != st.session_state.chat_model]
        selected = st.selectbox("🎯 Select Model", options=available, index=0, key="chat_model_select")
        st.session_state.chat_model = selected
    
    # Chat statistics
    chat_count = len(st.session_state.chat_histories[st.session_state.current_sheet])
    if chat_count > 0:
        st.info(f"💬 **{chat_count}** messages in this chat")
    
    # Chat history display with enhanced styling
    if st.session_state.chat_histories[st.session_state.current_sheet]:
        
        # Chat controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear all chat history for this sheet"):
                st.session_state.chat_histories[st.session_state.current_sheet] = []
                st.rerun()
        with col2:
            if st.button("📤 Export Chat", help="Export chat history"):
                chat_text = "\n\n".join([
                    f"{'👤 You' if msg['role'] == 'user' else '🤖 AI'}: {msg['content']}"
                    for msg in st.session_state.chat_histories[st.session_state.current_sheet]
                ])
                st.download_button(
                    "💾 Download",
                    chat_text,
                    file_name=f"chat_{st.session_state.current_sheet}.txt",
                    mime="text/plain"
                )
        
        # Enhanced chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.chat_histories[st.session_state.current_sheet]):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💭 Start a conversation by typing below!")
    
    st.divider()
    
    # Enhanced chat input with suggestions
    st.markdown("### 💬 Ask AI Assistant")
    

    
    # Quick action buttons
    st.markdown("**🚀 Quick Actions:**")
    quick_actions = st.columns(2)
    with quick_actions[0]:
        if st.button("📊 Analyze Data", help="Get insights about your data"):
            quick_msg = "Please analyze the data in this sheet and provide insights"
            st.session_state.quick_message = quick_msg
    with quick_actions[1]:
        if st.button("📈 Create Chart", help="Generate a chart from your data"):
            quick_msg = "Create a chart to visualize this data"
            st.session_state.quick_message = quick_msg
    
    # Chat input with placeholder suggestions
    placeholder_texts = [
        "Ask me to analyze your data...",
        "Try: 'Calculate the sum of column A'",
        "Try: 'Create a pivot table'",
        "Try: 'Add a formula to calculate totals'",
        "Try: 'Sort the data by column B'",
        "Ask me anything about spreadsheets!"
    ]
    
    import random
    placeholder = random.choice(placeholder_texts)
    
    # Handle quick messages
    initial_value = st.session_state.get("quick_message", "")
    if initial_value:
        st.session_state.quick_message = ""  # Clear after use
    
    # Use Streamlit's native chat_input for better reliability
    st.markdown("### 💬 Chat with AI Assistant")
    
    # Handle quick messages first
    if initial_value and initial_value.strip():
        user_msg = initial_value.strip()
    else:
        # Use native chat input
        user_msg = st.chat_input("Ask me to edit, sort, compute, chart...", key="chat_input")
    
    # Process the message immediately if we have one
    if user_msg and user_msg.strip():
        # Add to chat history
        st.session_state.chat_histories[st.session_state.current_sheet].append({
            "role": "user", 
            "content": user_msg.strip()
        })
        
        # Clear any quick message state
        if "quick_message" in st.session_state:
            del st.session_state["quick_message"]
        
        # Log the user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.ai_operations_log.append({
            'timestamp': timestamp,
            'operation': f"User: {user_msg[:50]}{'...' if len(user_msg) > 50 else ''}",
            'type': 'user'
        })
        
        # Process AI response immediately
        try:
            # Get chat history for context (excluding the message being processed)
            current_chat = st.session_state.chat_histories[st.session_state.current_sheet]
            chat_history = current_chat[:-1]  # Exclude the current message
            
            # Validate model is accessible
            current_model = st.session_state.chat_model
            if not current_model:
                current_model = "gpt-4-turbo-2024-04-09"
                st.session_state.chat_model = current_model
            
            # Capture workbook state BEFORE AI operation for explanation
            import copy
            before_workbook = copy.deepcopy(st.session_state.workbook)
            
            # Create a working copy for the AI agent to modify
            working_workbook = copy.deepcopy(st.session_state.workbook)
            
            # Store the current sheet name to avoid reference issues
            current_sheet_name = st.session_state.current_sheet
            
            # Call the agent with context and selected model
            with st.spinner(f"🤖 AI is processing: {user_msg[:30]}..."):
                reply = run_agent(
                    user_msg=user_msg.strip(),
                    workbook=working_workbook,  # Use working copy
                    current_sheet=current_sheet_name,
                    chat_history=chat_history,
                    model_name=current_model
                )
                
                # Update the actual workbook with the AI's changes
                st.session_state.workbook = working_workbook
                
                # Ensure we're comparing the right DataFrames
                before_df = before_workbook.get(current_sheet_name, pd.DataFrame())
                after_df = working_workbook.get(current_sheet_name, pd.DataFrame())
                
                                            # Ensure we're comparing the right DataFrames
                
                # Log AI operation
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.ai_operations_log.append({
                    'timestamp': timestamp,
                    'operation': f"AI processed: {user_msg[:50]}...",
                    'type': 'ai'
                })
                
                # Generate intelligent explanation using LangChain + LangGraph (if enabled)
                if st.session_state.enable_explanations:
                    print("🔧 CLEAN PIPELINE: Starting explanation generation...")
                    try:
                        # Get local LLM for intelligent explanations
                        print("🔧 CLEAN PIPELINE: Checking for local LLM...")
                        local_llm = get_local_llm()
                        print(f"🔧 CLEAN PIPELINE: Local LLM status: {local_llm is not None}")
                        
                        if local_llm:
                            # Use PROPER LangChain workflow with OpenAI for intelligent processing
                            print("🚀 PROPER LANGCHAIN: Using ProperLangChainWorkflow with OpenAI for intelligent processing")
                            print("🔧 PROPER LANGCHAIN: Creating proper LangChain workflow instance...")
                            proper_langchain_workflow = create_proper_langchain_workflow()
                            print("🔧 PROPER LANGCHAIN: Proper LangChain workflow created successfully")
                            
                            # Determine operation type based on user message
                            operation_type = 'general'
                            user_msg_lower = user_msg.lower()
                            if any(word in user_msg_lower for word in ['create', 'add', 'generate', 'make', 'want', 'need', 'populate', 'fill', 'put', 'insert']):
                                operation_type = 'data_creation'
                            elif any(word in user_msg_lower for word in ['formula', 'calculate', 'sum', 'average', 'compute', 'math']):
                                operation_type = 'formula_application'
                            elif any(word in user_msg_lower for word in ['sort', 'order', 'arrange', 'organize']):
                                operation_type = 'sorting'
                            elif any(word in user_msg_lower for word in ['filter', 'find', 'search', 'look', 'show']):
                                operation_type = 'filtering'
                            elif any(word in user_msg_lower for word in ['chart', 'graph', 'plot', 'visualize']):
                                operation_type = 'chart_creation'
                            
                            # Debug: Show what's changing
                            print(f"🔧 CLEAN PIPELINE: Data shapes - Before: {before_df.shape}, After: {after_df.shape}")
                            if not before_df.empty and not after_df.empty:
                                print(f"🔧 CLEAN PIPELINE: Before sample: {before_df.head(2).values.tolist()}")
                                print(f"🔧 CLEAN PIPELINE: After sample:  {after_df.head(2).values.tolist()}")
                            
                            # Generate intelligent explanation using PROPER LangChain workflow
                            print("🔧 PROPER LANGCHAIN: Calling generate_explanation...")
                            explanation = proper_langchain_workflow.generate_explanation(
                                before_df=before_df,
                                after_df=after_df,
                                operation_type=operation_type,
                                operation_context=user_msg
                            )
                            
                            # Combine AI response with intelligent explanation
                            print("🔧 CLEAN PIPELINE: Combining AI response with intelligent explanation...")
                            full_response = f"{reply}\n\n---\n\n{explanation}"
                            print("🔧 CLEAN PIPELINE: Explanation generation completed successfully")
                            
                        else:
                            # No LLM available - skip explanation
                            print("⚠️ CLEAN PIPELINE: No local LLM available - skipping explanation")
                            full_response = reply
                            
                    except Exception as e:
                        # If explanation fails, just use the AI response
                        print(f"🚨 CLEAN PIPELINE: Explanation generation failed: {str(e)}")
                        st.warning(f"⚠️ Explanation generation failed: {str(e)[:100]}...")
                        full_response = reply
                else:
                    # Explanations disabled, use only AI response
                    full_response = reply
                
                # Add combined response to history
                st.session_state.chat_histories[st.session_state.current_sheet].append({
                    "role": "assistant",
                    "content": full_response
                })
                
        except Exception as e:
            error_msg = str(e)
            
            # Clean pipeline - no fallback models
            print(f"🚨 AI agent failed: {error_msg}")
            st.error(f"AI processing failed: {error_msg}")
            # Skip the rest of the processing for this message
            pass
        
        # Rerun to show the updated chat and spreadsheet
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