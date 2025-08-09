from typing import Dict
import pandas as pd
import numpy as np


def new_workbook() -> Dict[str, pd.DataFrame]:
    # initial single sheet with 20x5 and A,B,C,D,E columns
    cols = [chr(ord('A') + i) for i in range(5)]
    df = pd.DataFrame([[None]*5 for _ in range(20)], columns=cols)
    return {"Sheet1": df}


def list_sheets(workbook: Dict[str, pd.DataFrame]):
    return list(workbook.keys())


def ensure_sheet(workbook: Dict[str, pd.DataFrame], name: str, rows: int=20, cols: int=5):
    if name in workbook:
        return workbook[name]
    labels = []
    c = cols
    i = 0
    while len(labels) < cols:
        # Generate Excel-like column labels up to double letters
        if i < 26:
            labels.append(chr(ord('A') + i))
        else:
            first = (i // 26) - 1
            second = i % 26
            labels.append(chr(ord('A') + first) + chr(ord('A') + second))
        i += 1
    workbook[name] = pd.DataFrame([[None]*cols for _ in range(rows)], columns=labels)
    return workbook[name]


def get_sheet(workbook: Dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    if name not in workbook:
        raise KeyError(f"Sheet '{name}' not found")
    return workbook[name]


def set_sheet(workbook: Dict[str, pd.DataFrame], name: str, df: pd.DataFrame):
    workbook[name] = df


def set_cell_by_a1(df: pd.DataFrame, cell: str, value):
    # Resolve A1 into row/col index
    from app.ui.formula import cell_to_rc
    r, c = cell_to_rc(cell)
    # Ensure DataFrame big enough
    while r >= len(df):
        df.loc[len(df)] = [None]*len(df.columns)
    while c >= len(df.columns):
        df[df.columns[-1] + "_"] = None
    col_name = df.columns[c]
    df.at[r, col_name] = value