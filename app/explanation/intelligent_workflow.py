"""
Intelligent Explanation Workflow using LangChain and LangGraph

This is the REAL implementation that uses LangChain to generate intelligent,
context-aware explanations of spreadsheet changes.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Import our ChangeDetector for proper content analysis
from .change_detector import ChangeDetector

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.callbacks import CallbackManagerForChainRun
    
    # LangGraph imports
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Clean pipeline requires LangChain.")


class IntelligentExplanationWorkflow:
    """
    Real LangChain + LangGraph powered explanation system.
    
    This class uses LangChain to:
    1. Intelligently analyze data changes
    2. Generate contextual insights
    3. Provide meaningful recommendations
    4. Handle complex data patterns
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize the intelligent workflow.
        
        Args:
            llm: LangChain language model (required for clean pipeline)
        """
        self.llm = llm
        self.change_detector = ChangeDetector()  # Add our proper change detector
        self.workflow = None
        
        if LANGCHAIN_AVAILABLE and llm:
            self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow."""
        print("🔧 CLEAN PIPELINE: Building LangGraph workflow...")
        # Create the state graph with dictionary-based state
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("analyze_changes", self._analyze_changes_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("create_explanation", self._create_explanation_node)
        workflow.add_node("format_output", self._format_output_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_changes")
        workflow.add_edge("analyze_changes", "generate_insights")
        workflow.add_edge("generate_insights", "create_explanation")
        workflow.add_edge("create_explanation", "format_output")
        workflow.add_edge("format_output", END)
        
        # Compile the workflow
        self.workflow = workflow.compile()
        print("✅ CLEAN PIPELINE: LangGraph workflow built and compiled successfully!")
    
    def _analyze_changes_node(self, state):
        """Intelligently analyze what changed in the data."""
        try:
            print("🔍 CLEAN PIPELINE: Analyzing changes with LLM...")
            print(f"🔍 CLEAN PIPELINE: State received: {list(state.keys())}")
            print(f"🔍 CLEAN PIPELINE: Before shape: {state.get('before_df', 'None')}")
            print(f"🔍 CLEAN PIPELINE: After shape: {state.get('after_df', 'None')}")
            print(f"🔍 CLEAN PIPELINE: Operation type: {state.get('operation_type', 'None')}")
            
            # Use LangChain to analyze changes
            analysis_prompt = PromptTemplate.from_template("""
            You are an AI spreadsheet assistant. Analyze ONLY the actual changes you made to the spreadsheet.
            
            CRITICAL: You must be 100% accurate. Do NOT make up or assume anything.
            
            Before DataFrame Shape: {before_shape}
            After DataFrame Shape: {after_shape}
            Operation Type: {operation_type}
            User Request: {user_request}
            Formulas Used: {formulas_info}
            
            ACTUAL DATA CONTENT:
            Before Data:
            {before_data}
            
            After Data:
            {after_data}
            
            STRICT RULES:
            1. If before_shape equals after_shape, you did NOT add/remove rows or columns
            2. If formulas_info is "none", you did NOT use any formulas
            3. ONLY describe what you can see in the ACTUAL DATA above
            4. Do NOT make up numbers, ranges, or statistics
            5. Do NOT claim to have done things you didn't do
            6. Use the ACTUAL DATA CONTENT to analyze what happened
            
            Analyze ONLY what actually happened:
            1. **Structural Changes**: Did you actually change rows/columns? (Compare before_shape vs after_shape)
            2. **Data Content**: What type of data did you actually create? (Look at the ACTUAL DATA above)
            3. **Formulas**: Did you actually use formulas? (Check formulas_info)
            4. **Accomplishment**: What did you actually accomplish based on the user request?
            5. **Observations**: What can you actually observe about the REAL DATA shown above?
            
            Respond with ONLY valid JSON. Start with {{ and end with }}.
            
            {{
                "structural_changes": "exact description of structural changes (or 'no structural changes' if shapes are identical)",
                "data_patterns": "description of actual data types created (be specific about what you see in the ACTUAL DATA)",
                "formulas_applied": "exact formulas used (or 'no formulas used' if formulas_info is 'none')",
                "accomplishment": "what you actually accomplished based on the user request",
                "observations": "observations about the ACTUAL DATA shown above (no made-up statistics)"
            }}
            """)
            
            if self.llm:
                print("🔍 LLM available, using single-prompt approach with better model...")
                
                # Get formula information from change analysis
                formulas_info = "none"
                if state.get("change_analysis") and "formulas_detected" in state["change_analysis"]:
                    formulas = state["change_analysis"]["formulas_detected"]
                    if formulas:
                        formula_descriptions = [f"{f['cell']}: {f['formula']}" for f in formulas[:3]]  # Show first 3
                        formulas_info = "; ".join(formula_descriptions)
                        if len(formulas) > 3:
                            formulas_info += f" (and {len(formulas) - 3} more)"
                
                # Get actual data content for LLM analysis with better formatting
                before_data_sample = "No data"
                after_data_sample = "No data"
                
                if state.get("before_df") is not None:
                    before_df = state["before_df"]
                    # Get first 5 rows with better formatting
                    before_sample = before_df.head(5)
                    # Convert to a more readable format
                    before_data_sample = "First 5 rows:\n"
                    for idx, row in before_sample.iterrows():
                        row_data = []
                        for col, val in row.items():
                            if pd.isna(val) or val is None:
                                row_data.append("None")
                            else:
                                row_data.append(str(val))
                        before_data_sample += f"Row {idx}: {', '.join(row_data)}\n"
                
                if state.get("after_df") is not None:
                    after_df = state["after_df"]
                    # Get first 5 rows with better formatting
                    after_sample = after_df.head(5)
                    # Convert to a more readable format
                    after_data_sample = "First 5 rows:\n"
                    for idx, row in after_sample.iterrows():
                        row_data = []
                        for col, val in row.items():
                            if pd.isna(val) or val is None:
                                row_data.append("None")
                            else:
                                row_data.append(str(val))
                        after_data_sample += f"Row {idx}: {', '.join(row_data)}\n"
                
                # Single comprehensive prompt with all information
                before_shape = state["before_df"].shape if state.get("before_df") is not None else (0, 0)
                after_shape = state["after_df"].shape if state.get("after_df") is not None else (0, 0)
                
                analysis_prompt = f"""You are an AI spreadsheet assistant. Analyze ONLY the actual changes made to a spreadsheet.

SHAPE COMPARISON (CRITICAL):
- Before Shape: {before_shape}
- After Shape: {after_shape}
- Shape Changed: {before_shape != after_shape}

OPERATION DETAILS:
- Operation Type: {state.get("operation_type", "unknown")}
- User Request: {state.get("operation_context", {}).get("user_request", "unknown") if state.get("operation_context") else "unknown"}
- Formulas Used: {formulas_info}

ACTUAL DATA CONTENT:
BEFORE DATA:
{before_data_sample}

AFTER DATA:
{after_data_sample}

MANDATORY ANALYSIS RULES:
1. STRUCTURAL CHANGES: If before_shape == after_shape, you did NOT add/remove rows or columns. Say "no structural changes" if shapes are identical.
2. DATA PATTERNS: Describe ONLY what you can see in the ACTUAL DATA above. Reference specific values like "151.45", "2023-11-01", etc.
3. FORMULAS: If formulas_info is "none", you did NOT use any formulas.
4. ACCOMPLISHMENT: What you actually accomplished based on the user request.
5. OBSERVATIONS: Only observations about the ACTUAL DATA shown above. No made-up statistics.

VALIDATION CHECKLIST:
- Did I add rows? {before_shape[0] != after_shape[0]}
- Did I add columns? {before_shape[1] != after_shape[1]}
- Did I use formulas? {formulas_info != "none"}
- Do I have actual data to reference? {len(after_data_sample) > 10}

Respond with ONLY valid JSON. Start with {{ and end with }}.

{{
    "structural_changes": "exact description (or 'no structural changes' if shapes are identical)",
    "data_patterns": "description of actual data types with specific values from ACTUAL DATA above",
    "formulas_applied": "exact formulas used (or 'no formulas used' if formulas_info is 'none')",
    "accomplishment": "what you actually accomplished based on the user request",
    "observations": "observations about the ACTUAL DATA shown above (reference specific values)"
}}"""

                print("🔍 Sending comprehensive prompt to LLM...")
                analysis_result = self.llm.invoke(analysis_prompt)
                print(f"🔍 LLM Response: {analysis_result[:200]}...")
                
                # DEBUG: Show if LLM is using the actual data
                print("🔍 DEBUG: Checking if LLM used actual data...")
                
                # Dynamic check based on actual data content
                data_indicators = []
                if state.get("after_df") is not None:
                    after_df = state["after_df"]
                    # Get some actual values from the data for validation
                    for col in after_df.columns:
                        for val in after_df[col].dropna().head(3):
                            if isinstance(val, (int, float)) and val != 0:
                                data_indicators.append(str(val))
                            elif isinstance(val, str) and len(val) > 2:
                                data_indicators.append(val)
                
                # Check if LLM referenced any actual data
                llm_used_data = any(indicator in analysis_result for indicator in data_indicators[:5])
                
                if llm_used_data:
                    print("   ✅ LLM appears to be using actual data")
                else:
                    print("   ❌ LLM is NOT using actual data - making up fake information")
                    print(f"   🔍 Available data indicators: {data_indicators[:5]}")
                
                # Parse the JSON response - handle various formatting
                try:
                    # Clean the response - remove various prefixes and markdown
                    cleaned_result = analysis_result.strip()
                    
                    # Remove common prefixes
                    prefixes_to_remove = [
                        "Here is the analysis in JSON format:",
                        "```json",
                        "```",
                        "Analysis:",
                        "JSON:"
                    ]
                    
                    for prefix in prefixes_to_remove:
                        if cleaned_result.startswith(prefix):
                            cleaned_result = cleaned_result[len(prefix):].strip()
                    
                    # Remove trailing ```
                    if cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[:-3].strip()
                    
                    print(f"🔍 Cleaned JSON: {cleaned_result[:200]}...")
                    
                    analysis_data = json.loads(cleaned_result)
                    
                    # VALIDATION: Check if LLM made obvious errors
                    validation_errors = []
                    
                    # Check 1: Structural changes validation
                    if "structural_changes" in analysis_data:
                        structural_text = analysis_data["structural_changes"].lower()
                        before_shape = state["before_df"].shape if state.get("before_df") is not None else (0, 0)
                        after_shape = state["after_df"].shape if state.get("after_df") is not None else (0, 0)
                        
                        # If shapes are identical, LLM should not claim rows/columns were added
                        if before_shape == after_shape:
                            if any(word in structural_text for word in ["added", "new rows", "new columns", "increased"]):
                                validation_errors.append(f"LLM claimed structural changes when shapes are identical: {before_shape} == {after_shape}")
                    
                    # Check 2: Data patterns validation
                    if "data_patterns" in analysis_data and not llm_used_data:
                        validation_errors.append("LLM provided data patterns without referencing actual data")
                    
                    # Check 3: Structural accuracy validation
                    if "data_patterns" in analysis_data and state.get("after_df") is not None:
                        after_df = state["after_df"]
                        data_patterns_text = analysis_data["data_patterns"].lower()
                        
                        # Extract actual data structure
                        actual_columns_with_data = []
                        actual_rows_with_data = []
                        
                        for col_idx, col in enumerate(after_df.columns):
                            col_has_data = not after_df[col].isna().all()
                            if col_has_data:
                                actual_columns_with_data.append(f"column {chr(65 + col_idx)}")  # A, B, C, etc.
                        
                        for row_idx in range(len(after_df)):
                            row_has_data = not after_df.iloc[row_idx].isna().all()
                            if row_has_data:
                                actual_rows_with_data.append(row_idx)
                        
                        # Check if LLM correctly identifies columns
                        if actual_columns_with_data:
                            # Check if LLM mentions wrong columns
                            mentioned_columns = []
                            for col_letter in ['a', 'b', 'c', 'd', 'e']:
                                if f"column {col_letter}" in data_patterns_text or f"column {col_letter.upper()}" in data_patterns_text:
                                    mentioned_columns.append(col_letter.upper())
                            
                            # If LLM mentions columns that don't have data
                            wrong_columns = [col for col in mentioned_columns if col not in [c.split()[-1] for c in actual_columns_with_data]]
                            if wrong_columns:
                                validation_errors.append(f"LLM mentioned columns {wrong_columns} that don't contain data. Actual columns with data: {[c.split()[-1] for c in actual_columns_with_data]}")
                        
                        # Check if LLM correctly identifies row ranges
                        if actual_rows_with_data:
                            min_row, max_row = min(actual_rows_with_data), max(actual_rows_with_data)
                            # Check if LLM mentions wrong row ranges
                            if "rows 1-9" in data_patterns_text and (min_row != 1 or max_row != 9):
                                validation_errors.append(f"LLM mentioned 'rows 1-9' but actual data is in rows {min_row}-{max_row}")
                            elif "rows 0-4" in data_patterns_text and (min_row != 0 or max_row != 4):
                                validation_errors.append(f"LLM mentioned 'rows 0-4' but actual data is in rows {min_row}-{max_row}")
                    
                    if validation_errors:
                        print(f"🚨 LLM VALIDATION FAILED: {validation_errors}")
                        print("🔄 Using fallback analysis instead...")
                        
                        # Use ChangeDetector for accurate analysis
                        try:
                            changes = self.change_detector.detect_changes(
                                state["before_df"], 
                                state["after_df"], 
                                state.get("operation_type", "general"),
                                state.get("operation_context", {})
                            )
                            
                            # Create accurate analysis based on actual changes
                            after_df = state["after_df"]
                            
                            # Extract actual data structure for accurate description
                            actual_columns_with_data = []
                            actual_rows_with_data = []
                            
                            for col_idx, col in enumerate(after_df.columns):
                                col_has_data = not after_df[col].isna().all()
                                if col_has_data:
                                    actual_columns_with_data.append(chr(65 + col_idx))  # A, B, C, etc.
                            
                            for row_idx in range(len(after_df)):
                                row_has_data = not after_df.iloc[row_idx].isna().all()
                                if row_has_data:
                                    actual_rows_with_data.append(row_idx)
                            
                            # Create accurate data patterns description
                            if actual_columns_with_data and actual_rows_with_data:
                                min_row, max_row = min(actual_rows_with_data), max(actual_rows_with_data)
                                data_patterns_desc = f"Data added in column{'s' if len(actual_columns_with_data) > 1 else ''} {', '.join(actual_columns_with_data)}, rows {min_row}-{max_row}"
                            else:
                                data_patterns_desc = "No data changes detected"
                            
                            state["change_analysis"] = {
                                "structural_changes": f"Rows: {changes['before_shape'][0]} → {changes['after_shape'][0]} ({changes['rows_added']:+d}), Columns: {changes['before_shape'][1]} → {changes['after_shape'][1]} ({changes['columns_added']:+d})",
                                "data_patterns": data_patterns_desc,
                                "formulas_applied": f"{len(changes.get('formulas_detected', []))} formulas used" if changes.get('formulas_detected') else "no formulas used",
                                "accomplishment": changes.get('summary', 'Operation completed successfully'),
                                "observations": f"Data structure: {changes['before_shape']} → {changes['after_shape']}"
                            }
                            print("✅ Fallback analysis completed successfully!")
                        except Exception as e:
                            print(f"🚨 Fallback analysis failed: {str(e)}")
                            # Use LLM result despite validation errors
                            state["change_analysis"] = analysis_data
                    else:
                        state["change_analysis"] = analysis_data
                        print("🔍 Analysis data parsed and validated successfully!")
                except json.JSONDecodeError as e:
                    print(f"🔍 JSON parsing failed: {str(e)}")
                    print(f"🔍 Raw result: {analysis_result[:300]}...")
                    # Fallback if JSON parsing fails
                    state["change_analysis"] = {
                        "structural_changes": "Data structure modified",
                        "data_patterns": "New data patterns detected",
                        "user_accomplishment": "Operation completed successfully",
                        "observations": analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
                    }
            else:
                print("🔍 No LLM available - clean pipeline requires LLM")
                state["change_analysis"] = {"error": "LLM required for clean pipeline"}
                
        except Exception as e:
            print(f"🔍 Analysis node error: {str(e)}")
            state["change_analysis"] = {
                "structural_changes": "Analysis failed",
                "data_patterns": "Unable to detect patterns",
                "user_accomplishment": "Operation completed",
                "observations": f"Error in analysis: {str(e)}"
            }
        
        return state
    
    def _generate_insights_node(self, state):
        """Generate intelligent insights about the data."""
        try:
            print("💡 Generating insights with LLM...")
            print(f"💡 State keys: {list(state.keys())}")
            print(f"💡 Change analysis available: {'change_analysis' in state}")
            if self.llm:
                insights_prompt = PromptTemplate.from_template("""
                As an AI spreadsheet assistant, analyze the actual data patterns and extremes you created.
                
                Change Analysis: {change_analysis}
                Operation Type: {operation_type}
                User Request: {user_request}
                
                Focus on meaningful data insights - not obvious statements. Analyze:
                1. **Data Extremes**: What are the highest/lowest values, ranges, or notable patterns?
                2. **Interesting Patterns**: Any trends, cycles, or unusual data points?
                3. **Practical Observations**: What stands out about this specific dataset?
                4. **Actionable Insights**: What can be done with these specific values?
                
                Avoid generic statements like "data is numeric" or "high quality". Be specific about the actual data.
                
                Provide insights in JSON format:
                {{
                    "data_extremes": "specific high/low values, ranges, or notable data points",
                    "patterns": "interesting trends, cycles, or patterns in the data",
                    "practical_insights": "what stands out about this specific dataset",
                    "next_steps": "1-2 specific actions based on the actual data values"
                }}
                """)
                
                insights_chain = insights_prompt | self.llm | StrOutputParser()
                
                insights_input = {
                    "change_analysis": json.dumps(state.change_analysis),
                    "operation_type": state.operation_type or "unknown",
                    "user_request": state.operation_context.get("user_request", "unknown") if state.operation_context else "unknown"
                }
                
                insights_result = insights_chain.invoke(insights_input)
                
                try:
                    insights_data = json.loads(insights_result)
                    state["data_insights"] = insights_data
                except json.JSONDecodeError:
                    state["data_insights"] = {
                        "data_meaning": "Data structure updated",
                        "use_cases": "Various applications possible",
                        "quality_notes": "Data appears complete",
                        "next_steps": "Consider adding formulas or charts"
                    }
            else:
                print("💡 No LLM available - clean pipeline requires LLM")
                state["data_insights"] = {"error": "LLM required for clean pipeline"}
                
        except Exception as e:
            state["data_insights"] = {
                "data_meaning": "Data updated",
                "use_cases": "Multiple applications",
                "quality_notes": "Standard quality",
                "next_steps": "Explore the data"
            }
        
        return state
    
    def _create_explanation_node(self, state):
        """Create an intelligent explanation using LangChain."""
        try:
            print("📝 Creating explanation with LLM...")
            if self.llm:
                print("📝 Using single-prompt approach for explanation generation...")
                
                # Get actual data for explanation generation
                after_data_for_explanation = "No data"
                if state.get("after_df") is not None:
                    after_df = state["after_df"]
                    after_data_for_explanation = after_df.head(10).to_string()
                
                # Single comprehensive explanation prompt
                explanation_prompt = f"""You are an AI spreadsheet assistant speaking directly to the person who requested changes. Create an explanation of what you accomplished.

CONTEXT:
- Change Analysis: {json.dumps(state.get("change_analysis", {}))}
- Data Insights: {json.dumps(state.get("data_insights", {}))}
- Operation Type: {state.get("operation_type", "unknown")}
- User Request: {state.get("operation_context", {}).get("user_request", "unknown") if state.get("operation_context") else "unknown"}

ACTUAL DATA YOU CREATED:
{after_data_for_explanation}

CRITICAL RULES:
1. ONLY describe what you actually did - no made-up details
2. If you didn't add rows/columns, don't say you did
3. If you didn't use formulas, don't mention formulas
4. If you don't have actual data statistics, don't make them up
5. Be honest about what you accomplished vs. what you didn't
6. Use the ACTUAL DATA shown above

Write a direct, conversational explanation that:
1. **Accurately explains what you changed** - only the actual modifications you made
2. **Honestly describes what you accomplished** - what you actually created for them
3. **Provides real data insights** - only if you have actual data to analyze from the ACTUAL DATA above
4. **Offers practical recommendations** - based on what you actually created
5. **Speaks directly to the person** - use "you" and "I" conversationally

Structure your response with clear sections:
- **Summary of Changes** (be accurate about what you actually did)
- **What This Means** (explain the practical value of what you created)
- **Data Insights** (only if you have real data to analyze from the ACTUAL DATA above - no made-up statistics)
- **Next Steps** (suggest 1-2 specific actions based on what you actually created, end with "What would you like to change next?")

Be honest and specific. If you don't know something, don't make it up. Use the ACTUAL DATA above."""

                print("📝 Sending comprehensive explanation prompt to LLM...")
                explanation_result = self.llm.invoke(explanation_prompt)
                print(f"📝 LLM Response: {explanation_result[:200]}...")
                
                state["explanation"] = explanation_result
                
            else:
                # FALLBACK COMMENTED OUT - CLEAN PIPELINE ONLY
                # state["explanation"] = self._fallback_explanation(state)
                state["explanation"] = "LLM analysis failed - no fallback available"
                
        except Exception as e:
            state["explanation"] = f"Explanation generation failed: {str(e)}"
        
        return state
    
    def _format_output_node(self, state):
        """Format the final output."""
        try:
            print("🎨 Formatting final output...")
            # Create a structured, professional output
            output_parts = [
                "**Intelligent Analysis Summary**",
                "",
                state.get("explanation", "No explanation available"),
                "",
                f"*Generated at: {datetime.now().isoformat()}*",
                f"*Operation: {state.get('operation_type', 'unknown')}*"
            ]
            
            state["final_output"] = "\n".join(output_parts)
            
        except Exception as e:
            state["final_output"] = f"Output formatting failed: {str(e)}"
        
        return state
    
    # FALLBACK METHODS COMMENTED OUT - CLEAN PIPELINE ONLY
    # def _fallback_analysis(self, state):
        """Fallback analysis without LLM - now uses proper ChangeDetector."""
        if state.before_df is None or state.after_df is None:
            return {
                "structural_changes": "No data available for comparison",
                "data_patterns": "Unable to detect patterns",
                "user_accomplishment": "Operation completed",
                "observations": "Limited analysis possible"
            }
        
        # Use our proper ChangeDetector for accurate analysis
        try:
            changes = self.change_detector.detect_changes(
                state.before_df, 
                state.after_df, 
                state.operation_type or 'general',
                state.operation_context or {}
            )
            
            # Extract meaningful information from the change detection
            structural_info = f"Rows: {changes['before_shape'][0]} → {changes['after_shape'][0]} ({changes['rows_added']:+d}), Columns: {changes['before_shape'][1]} → {changes['after_shape'][1]} ({changes['columns_added']:+d})"
            
            if changes['total_cells_changed'] > 0:
                content_info = f"Content changes: {changes['total_cells_changed']} cells modified"
                if changes['cells_modified']:
                    sample_changes = changes['cells_modified'][:3]
                    sample_info = "; ".join([f"{c['cell']}: {c['before']} → {c['after']}" for c in sample_changes])
                    content_info += f" (Sample: {sample_info})"
            else:
                content_info = "No content changes detected"
            
            return {
                "structural_changes": structural_info,
                "data_patterns": content_info,
                "user_accomplishment": changes.get('summary', 'Operation completed successfully'),
                "observations": f"Total cells: {changes['before_shape'][0] * changes['before_shape'][1]} → {changes['after_shape'][0] * changes['after_shape'][1]}"
            }
            
        except Exception as e:
            # Fallback to simple analysis if ChangeDetector fails
            before_shape = state.before_df.shape if state.before_df is not None else (0, 0)
            after_shape = state.after_df.shape if state.after_df is not None else (0, 0)
            
            rows_diff = after_shape[0] - before_shape[0]
            cols_diff = after_shape[1] - before_shape[1]
            
            return {
                "structural_changes": f"Rows: {before_shape[0]} → {after_shape[0]} ({rows_diff:+d}), Columns: {before_shape[1]} → {after_shape[1]} ({cols_diff:+d})",
                "data_patterns": f"Data structure changed from {before_shape[0]}x{before_shape[1]} to {after_shape[0]}x{after_shape[1]}",
                "user_accomplishment": f"Modified spreadsheet structure",
                "observations": f"Total cells: {before_shape[0] * before_shape[1]} → {after_shape[0] * after_shape[1]}"
            }
    
    # def _fallback_insights(self, state):
        """Fallback insights without LLM."""
        return {
            "data_meaning": "Spreadsheet data structure updated",
            "use_cases": "Data organization and analysis",
            "quality_notes": "Structure appears complete",
            "next_steps": "Consider adding content and formulas"
        }
    
    # def _fallback_explanation(self, state):
        """Fallback explanation without LLM."""
        analysis = state.change_analysis or {}
        
        return f"""
**Summary of Changes**
{analysis.get('structural_changes', 'Data structure modified')}

**What This Means**
{analysis.get('user_accomplishment', 'Operation completed successfully')}

**Suggested Next Steps**
1. Review the updated data structure
2. Add relevant content to the new cells
3. Consider applying formulas or formatting
        """.strip()
    
    def generate_intelligent_explanation(
        self,
        operation_type: str,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an intelligent explanation using LangChain + LangGraph.
        
        Args:
            operation_type: Type of operation performed
            before_df: DataFrame before the operation
            after_df: DataFrame after the operation
            operation_context: Additional context about the operation
            
        Returns:
            Intelligent explanation string
        """
        if not LANGCHAIN_AVAILABLE or not self.llm:
            # Clean pipeline requires LangChain and LLM
            print(f"🚨 Clean pipeline requirements not met: LANGCHAIN_AVAILABLE={LANGCHAIN_AVAILABLE}, llm={self.llm is not None}")
            return f"Clean pipeline requires LangChain and local LLM. Current status: LangChain={LANGCHAIN_AVAILABLE}, LLM={self.llm is not None}"
        
        try:
            print("🚀 Running LangGraph workflow with LLM...")
            print(f"🚀 Workflow available: {self.workflow is not None}")
            print(f"🚀 LLM available: {self.llm is not None}")
            
            # Initialize workflow state
            initial_state = {
                "before_df": before_df,
                "after_df": after_df,
                "operation_type": operation_type,
                "operation_context": operation_context or {},
                "change_analysis": None,
                "data_insights": None,
                "explanation": None,
                "final_output": None
            }
            print(f"🚀 Initial state keys: {list(initial_state.keys())}")
            print(f"🚀 Before shape: {before_df.shape if before_df is not None else 'None'}")
            print(f"🚀 After shape: {after_df.shape if after_df is not None else 'None'}")
            
            # Run the workflow
            print("🚀 Invoking workflow...")
            result = self.workflow.invoke(initial_state)
            print(f"🚀 Workflow result type: {type(result)}")
            print(f"🚀 Workflow result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            print("✅ LangGraph workflow completed successfully!")
            return result.get("final_output", "No final output generated")
            
        except Exception as e:
            # FALLBACK COMMENTED OUT - CLEAN PIPELINE ONLY
            print(f"🚨 LangGraph workflow failed: {str(e)}")
            print(f"🚨 No fallback available - returning error message")
            return f"Intelligent explanation failed: {str(e)}"
    
    # FALLBACK METHOD COMMENTED OUT - CLEAN PIPELINE ONLY
    # def _generate_fallback_explanation(
    #     self,
    #     operation_type: str,
    #     before_df: pd.DataFrame,
    #     after_df: pd.DataFrame,
    #     operation_context: Optional[Dict[str, Any]] = None
    # ) -> str:
    #     """Generate fallback explanation without LangChain."""
    #     analysis = self._fallback_analysis(type('State', (), {
    #         'before_df': before_df,
    #         'after_df': after_df,
    #         'operation_type': operation_type,
    #         'operation_context': operation_context or {}
    #     })())
    #     
    #     insights = self._fallback_insights(type('State', (), {
    #         'change_analysis': analysis
    #     })())
    #     
    #     explanation = self._fallback_explanation(type('State', (), {
    #         'change_analysis': analysis
    #     })())
    #     
    #     # Use our enhanced change detector for better explanations
    #     try:
    #         changes = self.change_detector.detect_changes(
    #             before_df, after_df, operation_type, operation_context or {}
    #         )
    #         
    #         # Use the template system for consistent formatting
    #         from .templates import ExplanationTemplates
    #         templates = ExplanationTemplates()
    #         explanation = templates.generate_explanation(operation_type, changes)
    #         
    #         return explanation
    #         
    #     except Exception as e:
    #         # Ultimate fallback
    #         output_parts = [
    #             "**Analysis Summary**",
    #             "",
    #             explanation,
    #             "",
    #             f"*Generated at: {datetime.now().isoformat()}*",
    #             f"*Operation: {operation_type}*"
    #         ]
    #         
    #         return "\n".join(output_parts)


# Convenience function for quick intelligent explanations
def generate_intelligent_explanation(
    operation_type: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    operation_context: Optional[Dict[str, Any]] = None,
    llm: Optional[BaseLanguageModel] = None
) -> str:
    """
    Quick function to generate intelligent explanations.
    
    Args:
        operation_type: Type of operation performed
        before_df: DataFrame before the operation
        after_df: DataFrame after the operation
        operation_context: Additional context about the operation
        llm: Optional LangChain language model
        
    Returns:
        Intelligent explanation string
    """
    workflow = IntelligentExplanationWorkflow(llm)
    return workflow.generate_intelligent_explanation(
        operation_type, before_df, after_df, operation_context
    )
