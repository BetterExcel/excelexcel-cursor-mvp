from typing import Dict, Any, List
import os
import json
import pandas as pd
from openai import OpenAI

from app.agent import tools as tool_impl
from app.services.workbook import list_sheets


client = OpenAI()

# Preferred models to try in order
PREFERRED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
]


def probe_models(candidates: List[str] | None = None, timeout_sec: float = 10.0) -> Dict[str, Any]:
    """Try a quick minimal chat call on candidate models and return which work.
    Returns a dict: { 'working': [models], 'failed': {model: error_str} }
    """
    import time
    models = candidates or PREFERRED_MODELS
    working: List[str] = []
    failed: Dict[str, str] = {}
    for m in models:
        try:
            # Minimal, fast ping
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=4,
            )
            _ = resp.choices[0].message.content
            working.append(m)
        except Exception as e:
            failed[m] = str(e)
    return {"working": working, "failed": failed}


def pick_default_model() -> str:
    """Pick a default model based on env or probing results."""
    env_model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL")
    if env_model:
        return env_model
    # Probe and return first working, else last fallback
    result = probe_models()
    if result["working"]:
        return result["working"][0]
    # Final fallback – allow code to raise later
    return PREFERRED_MODELS[0]


SYSTEM_PROMPT = (
    "You are an advanced Excel assistant for a web spreadsheet application. "
    "You have detailed information about the current active sheet including its structure, data types, and content. "
    "Use this context to provide informed recommendations and perform operations intelligently. "
    
    "IMPORTANT GUIDELINES: "
    "• Always explain your actions step-by-step and provide detailed feedback about what you're doing "
    "• When performing operations, break down your process and explain each step clearly "
    "• Use the provided tools to perform actions rather than replying with manual steps "
    "• Reference specific columns and data types when making recommendations "
    "• Default to the current sheet unless the user names another "
    "• Ask clarification if column names/targets are ambiguous "
    "• Always acknowledge successful operations and explain what was accomplished "
    "• If an operation fails, explain why and suggest alternatives "
    "• Use the sheet context to suggest relevant operations (e.g., if you see numeric data, suggest calculations) "
    "• When analyzing data, reference the actual column names and data types you can see "
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_cell",
            "description": "Set a cell's value in A1 notation on a sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "cell": {"type": "string", "description": "A1 ref e.g., B2"},
                    "value": {"type": "string"}
                },
                "required": ["sheet", "cell", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell",
            "description": "Read a cell's value (A1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "cell": {"type": "string"}
                },
                "required": ["sheet", "cell"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_formula",
            "description": "Apply a formula (e.g., '=A1+B1' or '=SUM(A1:A10)') to a cell; can apply to entire column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "cell": {"type": "string"},
                    "formula": {"type": "string"},
                    "by_column": {"type": "boolean", "default": False}
                },
                "required": ["sheet", "cell", "formula"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_sheet",
            "description": "Sort a sheet by a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "column": {"type": "string"},
                    "ascending": {"type": "boolean", "default": True}
                },
                "required": ["sheet", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_equals",
            "description": "Preview rows where column equals a value (does not overwrite).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "column": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["sheet", "column", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_sheet",
            "description": "Create a new empty sheet with rows x cols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rows": {"type": "integer", "default": 20},
                    "cols": {"type": "integer", "default": 5}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Create a line or bar chart (displayed in UI).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "x": {"type": "string"},
                    "ys": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": ["line", "bar"], "default": "line"}
                },
                "required": ["sheet", "x", "ys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_sheet",
            "description": "Export a sheet to csv or markdown and return text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "fmt": {"type": "string", "enum": ["csv", "markdown"], "default": "csv"}
                },
                "required": ["sheet"]
            }
        }
    }
]


def get_sheet_context(df: pd.DataFrame, sheet_name: str) -> str:
    """Generate detailed context about the active sheet for the LLM."""
    if df.empty:
        return f"Sheet '{sheet_name}' is empty (no data)."
    
    context_parts = []
    
    # Basic info
    context_parts.append(f"Active Sheet: '{sheet_name}'")
    context_parts.append(f"Dimensions: {len(df)} rows × {len(df.columns)} columns")
    context_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Data types and sample content
    context_parts.append("\nColumn Details:")
    for col in df.columns:
        non_null_count = df[col].count()
        null_count = len(df) - non_null_count
        dtype = df[col].dtype
        
        # Get sample values (first 3 non-null)
        sample_values = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(v) for v in sample_values]) if sample_values else "no data"
        
        context_parts.append(f"  • {col}: {dtype} ({non_null_count} values, {null_count} null) - Sample: {sample_str}")
    
    # Data summary
    if len(df) > 0:
        context_parts.append(f"\nData Range: Row 1-{len(df)} (Excel-style numbering)")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            context_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
            for col in numeric_cols[:3]:  # Show stats for first 3 numeric columns
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    context_parts.append(f"  • {col}: min={series.min():.2f}, max={series.max():.2f}, mean={series.mean():.2f}")
    
    # Recent changes or patterns
    non_empty_rows = df.dropna(how='all')
    if len(non_empty_rows) != len(df):
        empty_rows = len(df) - len(non_empty_rows)
        context_parts.append(f"\nNote: {empty_rows} empty rows detected")
    
    return "\n".join(context_parts)


def run_agent(user_msg: str, workbook: Dict[str, pd.DataFrame], current_sheet: str, chat_history: List[Dict] = None, model_name: str | None = None) -> str:
    """Run a single chat turn with function/tool calling and side‑effect the workbook."""

    model = model_name or os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"

    # Get detailed context about the current sheet
    current_df = workbook.get(current_sheet, pd.DataFrame())
    sheet_context = get_sheet_context(current_df, current_sheet)
    
    # Start with system messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Available sheets: {', '.join(list_sheets(workbook))}\n\n{sheet_context}"},
    ]

    # Add recent chat history for context (last 20 messages)
    if chat_history:
        recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history
        messages.extend(recent_history)

    # Add current user message
    messages.append({"role": "user", "content": user_msg})

    for iteration in range(6):  # small loop to allow a few tool calls
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            # Add the assistant's message with tool calls to the conversation
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in msg.tool_calls
                ]
            })

            # execute each tool call and append results
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                # default sheet to current if not provided
                if "sheet" in args and not args["sheet"]:
                    args["sheet"] = current_sheet
                if name == "set_cell":
                    out = tool_impl.tool_set_cell(workbook, args.get("sheet", current_sheet), args["cell"], str(args["value"]))
                elif name == "get_cell":
                    out = tool_impl.tool_get_cell(workbook, args.get("sheet", current_sheet), args["cell"])
                elif name == "apply_formula":
                    out = tool_impl.tool_apply_formula(workbook, args.get("sheet", current_sheet), args["cell"], args["formula"], bool(args.get("by_column", False)))
                elif name == "sort_sheet":
                    out = tool_impl.tool_sort(workbook, args.get("sheet", current_sheet), args["column"], bool(args.get("ascending", True)))
                elif name == "filter_equals":
                    out = tool_impl.tool_filter_equals(workbook, args.get("sheet", current_sheet), args["column"], str(args["value"]))
                elif name == "add_sheet":
                    out = tool_impl.tool_add_sheet(workbook, args["name"], int(args.get("rows", 20)), int(args.get("cols", 5)))
                elif name == "make_chart":
                    out = tool_impl.tool_make_chart(workbook, args.get("sheet", current_sheet), args["x"], args["ys"], args.get("kind", "line"))
                elif name == "export_sheet":
                    out = tool_impl.tool_export(workbook, args.get("sheet", current_sheet), args.get("fmt", "csv"))
                else:
                    out = f"Unknown tool {name}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": out,
                })
            # then continue loop for model to see tool outputs
            continue
        else:
            # final assistant reply
            return msg.content or "(no reply)"

    return "Done. If you expected more, please be more specific about the target sheet/columns/cells."