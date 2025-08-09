import os
import sys
import io
from typing import Dict, List

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ..services.workbook import (
    new_workbook,
    get_sheet,
    set_sheet,
    list_sheets,
    ensure_sheet,
)
from .formula import evaluate_formula
from ..agent.agent import run_agent
import app.charts as charts

load_dotenv()

st.set_page_config(page_title="Excel‑Cursor MVP", layout="wide")

# ---------- Session State ----------
if "workbook" not in st.session_state:
    st.session_state.workbook = new_workbook()
if "current_sheet" not in st.session_state:
    st.session_state.current_sheet = list(st.session_state.workbook.keys())[0]
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict] = []

# ---------- Top Bar ----------
left, mid, right = st.columns([0.6, 0.2, 0.2])
with left:
    st.title("Excel‑Cursor — Python MVP")
with mid:
    # Sheet selector
    sheet_names = list_sheets(st.session_state.workbook)
    curr = st.selectbox("Sheet", options=sheet_names, index=sheet_names.index(st.session_state.current_sheet))
    if curr != st.session_state.current_sheet:
        st.session_state.current_sheet = curr
with right:
    st.caption("Agent will operate on the selected sheet unless told otherwise.")

st.divider()

# ---------- Toolbar ----------
btns1 = st.columns([1,1,1,1,1,1,1])
with btns1[0]:
    if st.button("➕ New sheet"):
        name = st.text_input("New sheet name", value="Sheet2", key="new_sheet_name")
        rows = st.number_input("Rows", 1, 1000, 20, key="new_sheet_rows")
        cols = st.number_input("Cols", 1, 100, 5, key="new_sheet_cols")
        if st.button("Create", key="create_sheet_go"):
            ensure_sheet(st.session_state.workbook, name, rows, cols)
            st.session_state.current_sheet = name
            st.experimental_rerun()
with btns1[1]:
    uploaded = st.file_uploader("Import CSV/XLSX", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
        st.success("Imported into current sheet.")
with btns1[2]:
    if st.button("💾 Export CSV"):
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download CSV", data=buf.getvalue(), file_name=f"{st.session_state.current_sheet}.csv", mime="text/csv")
with btns1[3]:
    if st.button("📗 Export XLSX"):
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=st.session_state.current_sheet, index=False)
        st.download_button("Download XLSX", data=out.getvalue(), file_name=f"{st.session_state.current_sheet}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with btns1[4]:
    st.markdown("**Sort**")
    sort_col = st.text_input("Column name", value="", key="sort_col")
    sort_asc = st.checkbox("Ascending", value=True, key="sort_asc")
    if st.button("Apply sort") and sort_col:
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
            set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
        else:
            st.warning("Column not found.")
with btns1[5]:
    st.markdown("**Filter (equals)**")
    filt_col = st.text_input("Column", value="", key="filt_col")
    filt_val = st.text_input("Value", value="", key="filt_val")
    if st.button("Apply filter") and filt_col:
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        if filt_col in df.columns:
            st.session_state.filtered_df = df[df[filt_col].astype(str) == str(filt_val)]
        else:
            st.warning("Column not found.")
with btns1[6]:
    st.markdown("**Charts**")
    xcol = st.text_input("X", value="", key="chart_x")
    ycols = st.text_input("Y (comma‑sep)", value="", key="chart_y")
    kind = st.selectbox("Type", ["line", "bar"], key="chart_kind")
    if st.button("Plot") and xcol and ycols:
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        fig = charts.quick_plot(df, xcol, [c.strip() for c in ycols.split(",")], kind)
        st.pyplot(fig, clear_figure=True)

st.divider()

# ---------- Formula Bar ----------
form_cols = st.columns([0.2, 0.3, 0.3, 0.2])
with form_cols[0]:
    target_cell = st.text_input("Cell (e.g., A1)", key="cell_addr")
with form_cols[1]:
    formula = st.text_input("Formula or value (e.g., =SUM(A1:A10) or 123)", key="formula_text")
with form_cols[2]:
    apply_col = st.checkbox("Apply to entire column (use cell's column)")
with form_cols[3]:
    if st.button("Apply"):
        df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
        try:
            if not target_cell:
                st.warning("Provide a target cell like A1.")
            else:
                if str(formula).startswith("="):
                    if apply_col:
                        # apply same formula per row, interpreting A1 refs per row
                        col_letter = ''.join([c for c in target_cell if c.isalpha()])
                        for r in range(len(df)):
                            val = evaluate_formula(formula, df, current_row=r)
                            df.at[r, col_letter] = val if col_letter in df.columns else val
                    else:
                        val = evaluate_formula(formula, df)
                        # write into provided cell address
                        from app.services.workbook import set_cell_by_a1
                        set_cell_by_a1(df, target_cell, val)
                else:
                    from app.services.workbook import set_cell_by_a1
                    set_cell_by_a1(df, target_cell, formula)
                set_sheet(st.session_state.workbook, st.session_state.current_sheet, df)
                st.success("Applied.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------- Grid ----------
st.subheader(f"Sheet: {st.session_state.current_sheet}")
base_df = get_sheet(st.session_state.workbook, st.session_state.current_sheet)
show_df = st.session_state.get("filtered_df", base_df)
edited = st.data_editor(show_df, num_rows="dynamic", use_container_width=True)
# If user edited filtered view, don't overwrite base directly; here we overwrite only if no filter
if show_df is base_df and not edited.equals(base_df):
    set_sheet(st.session_state.workbook, st.session_state.current_sheet, edited)

# ---------- Sidebar Agent ----------
st.sidebar.header("Agent (GPT‑4o)")
for m in st.session_state.chat_history:
    with st.sidebar:
        if m["role"] == "user":
            st.chat_message("user").markdown(m["content"])
        else:
            st.chat_message("assistant").markdown(m["content"])

user_msg = st.sidebar.chat_input("Ask me to edit, sort, compute, chart…")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    try:
        reply = run_agent(
            user_msg=user_msg,
            workbook=st.session_state.workbook,
            current_sheet=st.session_state.current_sheet,
        )
    except Exception as e:
        reply = f"Agent error: {e}. Check your OPENAI_API_KEY and network."
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.experimental_rerun()