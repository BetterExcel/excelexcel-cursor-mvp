from typing import List
import matplotlib.pyplot as plt
import pandas as pd


def quick_plot(df: pd.DataFrame, x: str, ys: List[str], kind: str = "line"):
    if x not in df.columns:
        raise ValueError(f"x column '{x}' not found")
    for y in ys:
        if y not in df.columns:
            raise ValueError(f"y column '{y}' not found: {y}")

    fig = plt.figure()
    if kind == "line":
        for y in ys:
            plt.plot(df[x], df[y], label=y)
    else:
        # naive multi-series bar: stacked by index
        for y in ys:
            plt.bar(df[x], df[y], label=y)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig