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
from app.agent.agent import run_agent
import app.charts as charts

st.set_page_config(page_title="Excel‑Cursor MVP - Enhanced", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Excel-like styling
st.markdown("""
<style>
/* Ribbon-style toolbar */
.ribbon-container {
    background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    border-bottom: 1px solid #dee2e6;
    padding: 10px;
    margin-bottom: 20px;
}

.ribbon-tab {
    display: inline-block;
    padding: 8px 16px;
    margin-right: 5px;
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 4px 4px 0 0;
    cursor: pointer;
    font-weight: 500;
}

.ribbon-tab.active {
    background: #0066cc;
    color: white;
    border-bottom: 1px solid #0066cc;
}

/* Excel-like grid styling */
.stDataFrame {
    border: 1px solid #d0d7de;
}

/* Chat history styling */
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
}

.user-message {
    background: #f6f8fa;
    padding: 8px;
    border-radius: 6px;
    margin: 5px 0;
    border-left: 3px solid #0066cc;
}

.assistant-message {
    background: #fff;
    padding: 8px;
    border-radius: 6px;
    margin: 5px 0;
    border-left: 3px solid #28a745;
}

.ai-operation {
    background: #fff3cd;
    padding: 5px 8px;
    border-radius: 4px;
    margin: 2px 0;
    border-left: 3px solid #ffc107;
    font-size: 0.9em;
}

/* Status indicators */
.status-working {
    color: #ffc107;
    font-weight: bold;
}

.status-success {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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
        st.markdown("**� Charts**")
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
        st.markdown("**� Text**")
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
formula_cols = st.columns([0.2, 0.5, 0.2, 0.1])
with formula_cols[0]:
    target_cell = st.text_input("📍 Cell (e.g., A1)", key="cell_addr", placeholder="A1")
with formula_cols[1]:
    formula = st.text_input("📝 Formula", key="formula_text", 
                          placeholder="=SUM(A1:A10) or =TODAY() or =CONCATENATE(A1,B1)")
with formula_cols[2]:
    apply_col = st.checkbox("Apply to column")
with formula_cols[3]:
    if st.button("⚡ Apply"):
        if target_cell and formula:
            df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
            try:
                if str(formula).startswith("="):
                    if apply_col:
                        col_letter = ''.join([c for c in target_cell if c.isalpha()])
                        for r in range(len(df)):
                            val = evaluate_formula(formula, df, current_row=r)
                            if col_letter in df.columns:
                                df.at[r, col_letter] = val
                    else:
                        val = evaluate_formula(formula, df)
                        from app.services.workbook import set_cell_by_a1
                        set_cell_by_a1(df, target_cell, val)
                else:
                    from app.services.workbook import set_cell_by_a1
                    set_cell_by_a1(df, target_cell, formula)
                
                set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
                
                # Log the operation
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.ai_operations_log.append({
                    'timestamp': timestamp,
                    'operation': f"Formula applied: {formula} to {target_cell}",
                    'type': 'user'
                })
                
                st.success("✅ Formula applied successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.divider()

# Main spreadsheet display
st.markdown(f"### 📊 Sheet: {st.session_state.current_sheet}")

# Display the data editor
base_df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
show_df = st.session_state.get("filtered_df", base_df)

# Add row and column indicators (simplified)
edited = st.data_editor(
    show_df, 
    num_rows="dynamic", 
    use_container_width=True,
    key="main_editor"
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
    st.header(f"🤖 AI Assistant (GPT-4o)")
    st.caption(f"Chat for Sheet: **{st.session_state.current_sheet}**")
    
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
                    <div class="user-message">
                        <strong>👤 You:</strong> {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
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
            <div class="ai-operation">
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
                
                # Call the agent with context
                reply = run_agent(
                    user_msg=user_msg,
                    workbook=st.session_state.workbook,
                    current_sheet=st.session_state.current_sheet,
                    chat_history=chat_history[:-1]  # Exclude the current message
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
