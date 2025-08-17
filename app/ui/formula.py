import re
import math
import statistics
from datetime import datetime, timedelta
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

    # Statistical functions
    def fn_stdev(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(statistics.stdev(nums) if len(nums) > 1 else float('nan'))

    def fn_var(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(statistics.variance(nums) if len(nums) > 1 else float('nan'))

    def fn_median(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        return str(statistics.median(nums) if nums else float('nan'))

    def fn_mode(m):
        vals = range_values(df, m.group(1))
        nums = [float(x) for x in vals if _is_number(x)]
        try:
            return str(statistics.mode(nums))
        except statistics.StatisticsError:
            return str(float('nan'))

    # Date/Time functions
    def fn_today(m):
        return f'"{datetime.now().strftime("%Y-%m-%d")}"'  # Return quoted string to avoid syntax error

    def fn_now(m):
        return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def fn_weekday(m):
        # WEEKDAY(date) - returns 1-7 for Mon-Sun
        date_str = m.group(1).strip('"\'')
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return str(date_obj.weekday() + 1)  # Python weekday is 0-6, Excel is 1-7
        except:
            return str(float('nan'))

    # Text functions
    def fn_concatenate(m):
        # CONCATENATE(text1, text2, ...)
        # First, let's extract the arguments more carefully
        arg_str = m.group(1)
        
        # Split by comma but respect quotes
        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0
        
        i = 0
        while i < len(arg_str):
            char = arg_str[i]
            
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == '(' and not in_quotes:
                paren_depth += 1
                current_arg += char
            elif char == ')' and not in_quotes:
                paren_depth -= 1
                current_arg += char
            elif char == ',' and not in_quotes and paren_depth == 0:
                # End of argument
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
            i += 1
        
        # Add the last argument
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Process each argument
        result_parts = []
        for arg in args:
            # Check if it's a cell reference (no quotes, matches cell pattern)
            if re.match(r'^[A-Za-z]+\d+$', arg):
                cell_val = get_cell_value(df, arg)
                if cell_val is not np.nan:
                    result_parts.append(str(cell_val))
                else:
                    result_parts.append('')
            else:
                # It's a literal string, remove quotes
                result_parts.append(arg.strip('"\''))
        
        result = ''.join(result_parts)
        return f'"{result}"'

    def fn_left(m):
        # LEFT(text, num_chars)
        args = [arg.strip() for arg in m.group(1).split(',')]
        if len(args) >= 2:
            text = args[0].strip('"\'')
            try:
                num_chars = int(args[1])
                return f'"{text[:num_chars]}"'
            except:
                pass
        return '""'

    def fn_right(m):
        # RIGHT(text, num_chars)
        args = [arg.strip() for arg in m.group(1).split(',')]
        if len(args) >= 2:
            text = args[0].strip('"\'')
            try:
                num_chars = int(args[1])
                return f'"{text[-num_chars:]}"'
            except:
                pass
        return '""'

    def fn_mid(m):
        # MID(text, start, num_chars)
        args = [arg.strip() for arg in m.group(1).split(',')]
        if len(args) >= 3:
            text = args[0].strip('"\'')
            try:
                start = int(args[1]) - 1  # Excel is 1-indexed
                num_chars = int(args[2])
                return f'"{text[start:start+num_chars]}"'
            except:
                pass
        return '""'

    def fn_len(m):
        # LEN(text)
        text = m.group(1).strip().strip('"\'')
        return str(len(text))

    # Lookup functions (simplified)
    def fn_vlookup(m):
        # VLOOKUP(lookup_value, table_range, col_index, [range_lookup])
        args = [arg.strip() for arg in m.group(1).split(',')]
        if len(args) >= 3:
            lookup_val = args[0].strip('"\'')
            table_range = args[1].strip()
            try:
                col_index = int(args[2]) - 1  # Excel is 1-indexed
                # Simple lookup in first column of range
                vals = range_values(df, table_range)
                # This is a simplified implementation
                return f'"{lookup_val}_result"'  # Placeholder
            except:
                pass
        return str(float('nan'))

    # Replace function(range) first
    patterns = [
        # Basic statistical functions
        (re.compile(r"SUM\(([^)]+)\)", re.I), fn_sum),
        (re.compile(r"AVERAGE\(([^)]+)\)", re.I), fn_avg),
        (re.compile(r"MIN\(([^)]+)\)", re.I), fn_min),
        (re.compile(r"MAX\(([^)]+)\)", re.I), fn_max),
        (re.compile(r"COUNT\(([^)]+)\)", re.I), fn_cnt),
        
        # Advanced statistical functions
        (re.compile(r"STDEV\(([^)]+)\)", re.I), fn_stdev),
        (re.compile(r"VAR\(([^)]+)\)", re.I), fn_var),
        (re.compile(r"MEDIAN\(([^)]+)\)", re.I), fn_median),
        (re.compile(r"MODE\(([^)]+)\)", re.I), fn_mode),
        
        # Date/Time functions
        (re.compile(r"TODAY\(\)", re.I), fn_today),
        (re.compile(r"NOW\(\)", re.I), fn_now),
        (re.compile(r"WEEKDAY\(([^)]+)\)", re.I), fn_weekday),
        
        # Text functions
        (re.compile(r"CONCATENATE\(([^)]+)\)", re.I), fn_concatenate),
        (re.compile(r"LEFT\(([^)]+)\)", re.I), fn_left),
        (re.compile(r"RIGHT\(([^)]+)\)", re.I), fn_right),
        (re.compile(r"MID\(([^)]+)\)", re.I), fn_mid),
        (re.compile(r"LEN\(([^)]+)\)", re.I), fn_len),
        
        # Lookup functions
        (re.compile(r"VLOOKUP\(([^)]+)\)", re.I), fn_vlookup),
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
        if _is_number(val):
            return str(float(val))
        elif isinstance(val, str):
            return f'"{val}"'  # Quote string values for proper handling in expressions
        else:
            return "nan"

    expr = re.sub(r"\b[A-Za-z]+\d+\b", repl_cell, expr)

    # Safe arithmetic eval
    allowed = {k: getattr(math, k) for k in ["sqrt", "ceil", "floor", "exp", "log", "log10", "sin", "cos", "tan"]}
    allowed.update({"nan": float('nan')})
    
    # Check for unknown functions before evaluation
    if re.search(r'[A-Z_]+\(', expr, re.I):
        # There are still function calls that weren't processed
        unknown_funcs = re.findall(r'([A-Z_]+)\(', expr, re.I)
        if unknown_funcs:
            raise ValueError(f"Unknown function(s): {', '.join(set(unknown_funcs))}")
    
    try:
        return eval(expr, {"__builtins__": {}}, allowed)
    except Exception as e:
        raise ValueError(f"Formula evaluation error: {str(e)}")


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False