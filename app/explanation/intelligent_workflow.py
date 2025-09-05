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
    print("Warning: LangChain not available. Using fallback mode.")


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
            llm: LangChain language model (if None, uses fallback)
        """
        self.llm = llm
        self.change_detector = ChangeDetector()  # Add our proper change detector
        self.workflow = None
        
        if LANGCHAIN_AVAILABLE and llm:
            self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow."""
        print("🔧 Building LangGraph workflow...")
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
        print("✅ LangGraph workflow built and compiled successfully!")
    
    def _analyze_changes_node(self, state):
        """Intelligently analyze what changed in the data."""
        try:
            print("🔍 Analyzing changes with LLM...")
            print(f"🔍 State received: {list(state.keys())}")
            print(f"🔍 Before shape: {state.get('before_df', 'None')}")
            print(f"🔍 After shape: {state.get('after_df', 'None')}")
            print(f"🔍 Operation type: {state.get('operation_type', 'None')}")
            
            # Use LangChain to analyze changes
            analysis_prompt = PromptTemplate.from_template("""
            You are an expert spreadsheet analyst. Analyze the changes between two spreadsheet states and provide intelligent, actionable insights.
            
            Before DataFrame Shape: {before_shape}
            After DataFrame Shape: {after_shape}
            Operation Type: {operation_type}
            User Request: {user_request}
            
            As a spreadsheet expert, analyze:
            1. **Structural Changes**: What changed in the table structure (rows/columns added/removed)
            2. **Data Patterns**: What type of data was created (numeric, text, formulas, patterns)
            3. **User Achievement**: What the user successfully accomplished
            4. **Smart Insights**: Professional observations about data quality, potential issues, or opportunities
            
            Provide your analysis in JSON format:
            {{
                "structural_changes": "detailed description of structural changes",
                "data_patterns": "analysis of data types and patterns created",
                "user_accomplishment": "what the user successfully achieved",
                "observations": "professional insights and recommendations"
            }}
            """)
            
            if self.llm:
                print("🔍 LLM available, creating analysis chain...")
                # Use LangChain LLM for intelligent analysis
                analysis_chain = analysis_prompt | self.llm | StrOutputParser()
                print("🔍 Analysis chain created successfully!")
                
                analysis_input = {
                    "before_shape": str(state["before_df"].shape) if state.get("before_df") is not None else "None",
                    "after_shape": str(state["after_df"].shape) if state.get("after_df") is not None else "None",
                    "operation_type": state.get("operation_type", "unknown"),
                    "user_request": state.get("operation_context", {}).get("user_request", "unknown") if state.get("operation_context") else "unknown"
                }
                print(f"🔍 Analysis input: {analysis_input}")
                
                print("🔍 Invoking LLM for analysis...")
                analysis_result = analysis_chain.invoke(analysis_input)
                print(f"🔍 LLM analysis result: {analysis_result[:200]}...")
                
                # Parse the JSON response - handle markdown formatting
                try:
                    # Clean the response - remove markdown formatting
                    cleaned_result = analysis_result.strip()
                    if cleaned_result.startswith("```json"):
                        cleaned_result = cleaned_result[7:]  # Remove ```json
                    if cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[:-3]  # Remove ```
                    cleaned_result = cleaned_result.strip()
                    
                    analysis_data = json.loads(cleaned_result)
                    state["change_analysis"] = analysis_data
                    print("🔍 Analysis data parsed successfully!")
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
                print("🔍 No LLM available, using fallback analysis")
                # Fallback analysis without LLM
                state["change_analysis"] = self._fallback_analysis(state)
                
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
                Based on the change analysis, generate intelligent insights about the spreadsheet data.
                
                Change Analysis: {change_analysis}
                Operation Type: {operation_type}
                User Request: {user_request}
                
                Generate insights about:
                1. What the data represents
                2. Potential use cases
                3. Data quality observations
                4. Next logical steps
                
                Provide insights in JSON format:
                {{
                    "data_meaning": "what the data represents",
                    "use_cases": "potential applications",
                    "quality_notes": "data quality observations",
                    "next_steps": "logical next actions"
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
                state["data_insights"] = self._fallback_insights(state)
                
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
                explanation_prompt = PromptTemplate.from_template("""
                You are a professional spreadsheet consultant. Create an intelligent, comprehensive explanation of what happened in the spreadsheet.
                
                Change Analysis: {change_analysis}
                Data Insights: {data_insights}
                Operation Type: {operation_type}
                User Request: {user_request}
                
                Write a detailed, professional explanation that:
                1. **Clearly explains what changed** - specific details about the modifications
                2. **Highlights key accomplishments** - what the user successfully achieved
                3. **Provides data insights** - analysis of the data types, patterns, and quality
                4. **Offers actionable recommendations** - specific next steps based on the data
                5. **Uses professional language** - clear, concise, and helpful
                
                Structure your response with clear sections:
                - **Summary of Changes**
                - **What This Means**
                - **Data Insights**
                - **Recommended Next Steps**
                
                Be specific about numbers, locations, and data types. Make it intelligent and contextual, not just a template.
                """)
                
                explanation_chain = explanation_prompt | self.llm | StrOutputParser()
                
                explanation_input = {
                    "change_analysis": json.dumps(state.get("change_analysis", {})),
                    "data_insights": json.dumps(state.get("data_insights", {})),
                    "operation_type": state.get("operation_type", "unknown"),
                    "user_request": state.get("operation_context", {}).get("user_request", "unknown") if state.get("operation_context") else "unknown"
                }
                
                explanation_result = explanation_chain.invoke(explanation_input)
                state["explanation"] = explanation_result
                
            else:
                state["explanation"] = self._fallback_explanation(state)
                
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
    
    def _fallback_analysis(self, state):
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
    
    def _fallback_insights(self, state):
        """Fallback insights without LLM."""
        return {
            "data_meaning": "Spreadsheet data structure updated",
            "use_cases": "Data organization and analysis",
            "quality_notes": "Structure appears complete",
            "next_steps": "Consider adding content and formulas"
        }
    
    def _fallback_explanation(self, state):
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
            # Use fallback mode
            print(f"🚨 Intelligent workflow fallback: LANGCHAIN_AVAILABLE={LANGCHAIN_AVAILABLE}, llm={self.llm is not None}")
            return self._generate_fallback_explanation(
                operation_type, before_df, after_df, operation_context
            )
        
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
            # Fallback if workflow fails
            print(f"🚨 LangGraph workflow failed: {str(e)}")
            print(f"🚨 Falling back to template-based explanation")
            return self._generate_fallback_explanation(
                operation_type, before_df, after_df, operation_context
            )
    
    def _generate_fallback_explanation(
        self,
        operation_type: str,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate fallback explanation without LangChain."""
        analysis = self._fallback_analysis(type('State', (), {
            'before_df': before_df,
            'after_df': after_df,
            'operation_type': operation_type,
            'operation_context': operation_context or {}
        })())
        
        insights = self._fallback_insights(type('State', (), {
            'change_analysis': analysis
        })())
        
        explanation = self._fallback_explanation(type('State', (), {
            'change_analysis': analysis
        })())
        
        # Use our enhanced change detector for better explanations
        try:
            changes = self.change_detector.detect_changes(
                before_df, after_df, operation_type, operation_context or {}
            )
            
            # Use the template system for consistent formatting
            from .templates import ExplanationTemplates
            templates = ExplanationTemplates()
            explanation = templates.generate_explanation(operation_type, changes)
            
            return explanation
            
        except Exception as e:
            # Ultimate fallback
            output_parts = [
                "**Analysis Summary**",
                "",
                explanation,
                "",
                f"*Generated at: {datetime.now().isoformat()}*",
                f"*Operation: {operation_type}*"
            ]
            
            return "\n".join(output_parts)


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
