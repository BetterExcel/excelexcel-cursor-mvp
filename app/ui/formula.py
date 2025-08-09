import re
import math
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

# Minimal A1 helpers
COL_RE = re.compile(r"([A-Za-z]+)")
CELL_RE = re.compile(r"([A-Za-z]+)(\d+)")
RANGE_RE = re.compile(r"([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)")


def col_to_idx(col: str) -> int:
    col = col.upper()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def cell_to_rc(cell: str) -> Tuple[int, int]:
    m = CELL_RE.fullmatch(cell.strip())
    if not m:
        raise ValueError(f"Bad cell ref: {cell}")
    col, row = m.group(1), int(m.group(2))
    return row - 1, col_to_idx(col)


def get_cell_value(df: pd.DataFrame, cell: str):
    r, c = cell_to_rc(cell)
    try:
        col_name = df.columns[c]
        return df.iloc[r][col_name]
    except Exception:
        return np.nan


def range_values(df: pd.DataFrame, rng: str) -> List[Any]:
    m = RANGE_RE.fullmatch(rng.strip())
    if not m:
        raise ValueError(f"Bad range: {rng}")
    c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    r1 -= 1; r2 -= 1
    c1i = col_to_idx(c1); c2i = col_to_idx(c2)
    vals = []
    for r in range(min(r1, r2), max(r1, r2)+1):
        for ci in range(min(c1i, c2i), max(c1i, c2i)+1):
            if ci < len(df.columns) and r < len(df):
                vals.append(df.iloc[r, ci])
    return vals


# Very small formula evaluator
# Supports: =A1+B2-3*D4/2, SUM(range), AVERAGE, MIN, MAX, COUNT

def evaluate_formula(expr: str, df: pd.DataFrame, current_row: int | None = None):
    if not isinstance(expr, str) or not expr.startswith('='):
        return expr
    expr = expr[1:].strip()

    # Functions with ranges
    def fn_sum(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(sum(nums))

    def fn_avg(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(sum(nums)/len(nums) if nums else float('nan'))

    def fn_min(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(min(nums) if nums else float('nan'))

    def fn_max(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(max(nums) if nums else float('nan'))

    def fn_cnt(m):
        vals = range_values(df, m.group(1))
        return str(len([x for x in vals if str(x).strip() != '']))

    # Replace function(range) first
    patterns = [
        (re.compile(r"SUM\(([^)]+)\)", re.I), fn_sum),
        (re.compile(r"AVERAGE\(([^)]+)\)", re.I), fn_avg),
        (re.compile(r"MIN\(([^)]+)\)", re.I), fn_min),
        (re.compile(r"MAX\(([^)]+)\)", re.I), fn_max),
        (re.compile(r"COUNT\(([^)]+)\)", re.I), fn_cnt),
    ]
    for pat, fn in patterns:
        while True:
            m = pat.search(expr)
            if not m:
                break
            expr = expr[:m.start()] + fn(m) + expr[m.end():]

    # Replace A1 refs with values
    def repl_cell(m):
        val = get_cell_value(df, m.group(0))
        return str(float(val)) if _is_number(val) else "nan"

    expr = re.sub(r"\b[A-Za-z]+\d+\b", repl_cell, expr)

    # Safe arithmetic eval
    allowed = {k: getattr(math, k) for k in ["sqrt", "ceil", "floor", "exp", "log", "log10", "sin", "cos", "tan"]}
    allowed.update({"nan": float('nan')})
    try:
        return eval(expr, {"__builtins__": {}}, allowed)
    except Exception:
        return float('nan')


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False