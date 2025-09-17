"""
PROPER LangChain Implementation with OpenAI
This uses LangChain agents, tools, and chains in their full glory!
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import re

from .change_detector import ChangeDetector


class ExplanationOutput(BaseModel):
    """Structured output for explanations"""
    summary: str = Field(description="Brief summary of what changed")
    details: str = Field(description="Detailed explanation of the changes")
    insights: str = Field(description="Key insights from the data")
    next_steps: str = Field(description="Recommended next steps")


class DataAnalysisTool(BaseTool):
    """LangChain tool for analyzing DataFrame structure and content"""
    name: str = "analyze_data"
    description: str = "Analyze DataFrame structure, content, and statistics. Use this to understand what data is in the spreadsheet."
    workflow_instance: Any = None
    
    def __init__(self, workflow_instance=None, **kwargs):
        super().__init__(**kwargs)
        self.workflow_instance = workflow_instance
    
    def _run(self, context: str = "") -> str:
        """Analyze the DataFrame and return structured information"""
        try:
            # Get the DataFrame from the workflow instance
            df = self.workflow_instance.after_df if self.workflow_instance else None
            if df is None:
                return "No data available for analysis"
            
            # Basic structure analysis
            structure_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
            }
            
            # Content analysis with comprehensive statistics
            content_info = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    content_info[col] = {
                        "type": "numeric",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "count": int(df[col].count()),
                        "range": float(df[col].max() - df[col].min())
                    }
                else:
                    # For categorical data, show more comprehensive info
                    unique_vals = df[col].dropna().unique()
                    content_info[col] = {
                        "type": "categorical",
                        "unique_values": df[col].nunique(),
                        "total_count": int(df[col].count()),
                        "sample_values": df[col].dropna().head(5).tolist(),  # Show 5 instead of 3
                        "all_unique_values": unique_vals.tolist() if len(unique_vals) <= 20 else unique_vals[:20].tolist()  # Show all if <= 20, otherwise first 20
                    }
            
            # Enhanced sample data - show more rows and include statistics
            sample_data = []
            # Show first 5 rows instead of 3
            for _, row in df.head(5).iterrows():
                sample_row = {}
                for col, value in row.items():
                    if pd.isna(value):
                        sample_row[str(col)] = None
                    elif isinstance(value, (int, float, str, bool)):
                        sample_row[str(col)] = value
                    else:
                        sample_row[str(col)] = str(value)
                sample_data.append(sample_row)
            
            # Add data summary statistics
            data_summary = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "date_range": None,
                "numeric_columns": [],
                "categorical_columns": []
            }
            
            # Identify column types
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    data_summary["numeric_columns"].append(col)
                else:
                    data_summary["categorical_columns"].append(col)
            
            # Try to detect date range if first column looks like dates
            if len(df) > 0:
                first_col = df.columns[0]
                first_val = str(df[first_col].iloc[0])
                if any(date_indicator in first_val.lower() for date_indicator in ['date', '2023', '2024', '2025']):
                    try:
                        # Get first and last non-null values
                        non_null_values = df[first_col].dropna()
                        if len(non_null_values) > 0:
                            data_summary["date_range"] = {
                                "first": str(non_null_values.iloc[0]),
                                "last": str(non_null_values.iloc[-1]),
                                "total_days": len(non_null_values)
                            }
                    except:
                        pass
            
            return json.dumps({
                "structure": structure_info,
                "content": content_info,
                "sample_data": sample_data,
                "data_summary": data_summary,
                "context": context
            }, indent=2)
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"


class ChangeDetectionTool(BaseTool):
    """LangChain tool for detecting changes between DataFrames"""
    name: str = "detect_changes"
    description: str = "Detect and analyze changes between before and after DataFrame states. Use this to understand what changed in the spreadsheet."
    workflow_instance: Any = None
    
    def __init__(self, workflow_instance=None, **kwargs):
        super().__init__(**kwargs)
        self.workflow_instance = workflow_instance
    
    def _run(self, operation_type: str = "general") -> str:
        """Detect changes between two DataFrames"""
        try:
            # Get the DataFrames from the workflow instance
            before_df = self.workflow_instance.before_df if self.workflow_instance else None
            after_df = self.workflow_instance.after_df if self.workflow_instance else None
            
            if before_df is None or after_df is None:
                return "No data available for change detection"
            
            change_detector = ChangeDetector()
            changes = change_detector.detect_changes(before_df, after_df, operation_type)
            
            # Format changes for LLM consumption
            change_summary = {
                "structural_changes": {
                    "rows_added": changes.get('rows_added', 0),
                    "rows_removed": changes.get('rows_removed', 0),
                    "columns_added": changes.get('columns_added', []),
                    "columns_removed": changes.get('columns_removed', [])
                },
                "content_changes": {
                    "cells_changed": changes.get('total_cells_changed', 0),  # Fixed: use total_cells_changed
                    "formulas_added": len(changes.get('formulas_detected', [])),  # Fixed: count formulas
                    "formulas_removed": 0  # Not tracked yet
                },
                "data_types_changed": changes.get('data_types_changed', {}),
                "memory_usage_change": changes.get('memory_usage_change', 0),
                "change_summary": changes.get('summary', ''),  # Add summary
                "key_info": changes.get('key_info', ''),  # Add key info
                "insights": changes.get('insights', '')  # Add insights
            }
            
            return json.dumps(change_summary, indent=2)
            
        except Exception as e:
            return f"Error detecting changes: {str(e)}"


class ValidationTool(BaseTool):
    """LangChain tool for validating LLM responses"""
    name: str = "validate_explanation"
    description: str = "Validate explanation accuracy against actual data to catch hallucinations and ensure accuracy."
    
    def _run(self, explanation: str, actual_data_info: str) -> str:
        """Validate the explanation against actual data"""
        try:
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Parse actual data info
            data_info = json.loads(actual_data_info)
            actual_rows = data_info["structure"]["shape"][0]
            actual_columns = data_info["structure"]["shape"][1]
            
            # Check for row count accuracy
            row_numbers = re.findall(r'\b(\d+)\s*(?:rows?|days?|entries?)\b', explanation.lower())
            for num in row_numbers:
                if int(num) != actual_rows:
                    validation_results["errors"].append(f"LLM mentioned {num} rows but actual data has {actual_rows} rows")
                    validation_results["is_valid"] = False
            
            # Check for column count accuracy
            col_numbers = re.findall(r'\b(\d+)\s*columns?\b', explanation.lower())
            for num in col_numbers:
                if int(num) != actual_columns:
                    validation_results["errors"].append(f"LLM mentioned {num} columns but actual data has {actual_columns} columns")
                    validation_results["is_valid"] = False
            
            # Check for statistical accuracy
            for col, info in data_info["content"].items():
                if info["type"] == "numeric":
                    # Check min/max claims
                    min_claims = re.findall(rf'{col}.*?(\d+\.?\d*).*?(?:lowest|minimum|min)', explanation.lower())
                    max_claims = re.findall(rf'{col}.*?(\d+\.?\d*).*?(?:highest|maximum|max)', explanation.lower())
                    
                    for claim in min_claims:
                        if abs(float(claim) - info["min"]) > 0.01:
                            validation_results["warnings"].append(f"LLM mentioned {claim} as min for {col} but actual min is {info['min']}")
                    
                    for claim in max_claims:
                        if abs(float(claim) - info["max"]) > 0.01:
                            validation_results["warnings"].append(f"LLM mentioned {claim} as max for {col} but actual max is {info['max']}")
            
            return json.dumps(validation_results, indent=2)
            
        except Exception as e:
            return f"Error validating explanation: {str(e)}"


class ProperLangChainWorkflow:
    """PROPER LangChain implementation using OpenAI with real agents and tool calling"""
    
    def __init__(self):
        # Use OpenAI for proper LangChain agent support
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for proper LangChain workflow")
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",  # Use the model you have access to
            temperature=0.1,  # Low temperature for consistent results
            api_key=api_key
        )
        
        # Create memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools with reference to this workflow instance
        self.tools = [
            DataAnalysisTool(workflow_instance=self),
            ChangeDetectionTool(workflow_instance=self),
            ValidationTool()
        ]
        
        # Create prompt template for the agent
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a smart Excel Analysis Agent that provides structured, actionable insights.

🎯 YOUR MISSION:
- Provide clear, structured analysis in bullet points
- Give actionable insights and recommendations
- Be context-aware: simple queries get simple responses
- Focus on what matters to the user

🛠️ AVAILABLE TOOLS:
1. analyze_data – Get data structure, statistics, and sample values
2. detect_changes – Compare before/after to find what changed
3. validate_explanation – Ensure accuracy

🧠 HOW TO THINK:
- ALWAYS use detect_changes to see what actually changed
- For financial data: focus on investment insights, trends, valuations
- For inventory data: focus on stock levels, reorder points, trends  
- For sales data: focus on performance, patterns, opportunities
- Use ACTUAL numbers from the data (don't make up statistics)

📝 OUTPUT FORMAT - ALWAYS USE THIS STRUCTURE:

**📊 What Changed**
• [Specific changes with actual numbers]
• [Key data points that were modified]

**💡 Key Insights** 
• [Important patterns or findings]
• [Notable statistics with actual values]
• [Trends or comparisons worth noting]

**🎯 Actionable Recommendations**
• [Specific actions the user can take]
• [Investment advice for stocks, business decisions for other data]
• [Next steps based on the data]

**❓ Quick Response Rule:**
- If user just says "hi" or simple greeting → Just respond with a friendly greeting, no analysis
- If user asks simple question → Give brief, direct answer
- If user makes data changes → Use full structured format above

CRITICAL: Use bullet points, not paragraphs. Be specific with numbers. Give actionable advice."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

        
        # Create the PROPER LangChain agent with tool calling
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create the PROPER agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        # Create output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ExplanationOutput)
    
    def generate_explanation(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_type: str,
        operation_context: str
    ) -> str:
        """Generate explanation using PROPER LangChain agents and tool calling"""
        try:
            print("🚀 PROPER LANGCHAIN: Starting real LangChain workflow with OpenAI...")
            
            # Store DataFrames for tool access
            self.before_df = before_df
            self.after_df = after_df
            
            # Determine response type based on user input
            user_input = operation_context.lower().strip()
            is_simple_greeting = any(greeting in user_input for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon'])
            is_simple_question = len(user_input.split()) < 10 and '?' in user_input
            
            # Prepare context-aware input for the agent
            if is_simple_greeting:
                agent_input = f"""
The user just said: "{operation_context}"

This is a simple greeting. Respond with a friendly greeting back. No need for data analysis.
"""
            elif is_simple_question:
                agent_input = f"""
The user asked: "{operation_context}"

This is a simple question. Give a brief, direct answer. Only use tools if absolutely necessary.
"""
            else:
                agent_input = f"""
Operation: {operation_type}
User Request: {operation_context}

TASK: Analyze the spreadsheet changes and provide a structured response.

STEPS:
1. Use detect_changes tool to see what actually changed
2. Use analyze_data tool to get key statistics and insights  
3. Provide response in the structured format specified in your system prompt

CONTEXT CLUES for better insights:
- If this involves stocks/financial data → Focus on investment recommendations
- If this involves inventory/products → Focus on stock management insights
- If this involves sales/revenue → Focus on performance analysis

Remember: Use the structured format with bullet points, not paragraphs!
"""

            
            # Execute the PROPER LangChain agent
            result = self.agent_executor.invoke({
                "input": agent_input
            })
            
            print("✅ PROPER LANGCHAIN: Agent execution completed")
            
            # Format the output
            explanation = result.get("output", "No explanation generated")
            
            # Return the explanation directly (it's already structured)
            formatted_output = explanation
            
            print("✅ PROPER LANGCHAIN: Workflow completed successfully!")
            return formatted_output
            
        except Exception as e:
            print(f"❌ PROPER LANGCHAIN: Error in workflow: {str(e)}")
            return f"Error generating explanation: {str(e)}"


def create_proper_langchain_workflow():
    """Factory function to create PROPER LangChain workflow"""
    return ProperLangChainWorkflow()
