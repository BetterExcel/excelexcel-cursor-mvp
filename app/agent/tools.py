from typing import Dict, Any, List
import pandas as pd
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