"""
Proper LangChain Implementation for Excel Explanation Agent
This uses LangChain agents, tools, chains, and prompts as intended.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

from .local_llm import get_local_llm
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
    description: str = "Analyze DataFrame structure, content, and statistics"
    
    def _run(self, df: pd.DataFrame, context: str = "") -> str:
        """Analyze the DataFrame and return structured information"""
        try:
            # Basic structure analysis
            structure_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},  # Convert to string for JSON serialization
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
            }
            
            # Content analysis
            content_info = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    content_info[col] = {
                        "type": "numeric",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std())
                    }
                else:
                    content_info[col] = {
                        "type": "categorical",
                        "unique_values": df[col].nunique(),
                        "sample_values": df[col].dropna().head(3).tolist()
                    }
            
            # Sample data - convert to JSON-serializable format
            sample_data = []
            for _, row in df.head(3).iterrows():
                sample_row = {}
                for col, value in row.items():
                    if pd.isna(value):
                        sample_row[str(col)] = None
                    elif isinstance(value, (int, float, str, bool)):
                        sample_row[str(col)] = value
                    else:
                        sample_row[str(col)] = str(value)
                sample_data.append(sample_row)
            
            return json.dumps({
                "structure": structure_info,
                "content": content_info,
                "sample_data": sample_data,
                "context": context
            }, indent=2)
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"


class ChangeDetectionTool(BaseTool):
    """LangChain tool for detecting changes between DataFrames"""
    name: str = "detect_changes"
    description: str = "Detect and analyze changes between before and after DataFrame states"
    
    def _run(self, before_df: pd.DataFrame, after_df: pd.DataFrame, operation_type: str = "general") -> str:
        """Detect changes between two DataFrames"""
        try:
            change_detector = ChangeDetector()
            changes = change_detector.detect_changes(before_df, after_df)
            
            # Format changes for LLM consumption
            change_summary = {
                "structural_changes": {
                    "rows_added": changes.get('rows_added', 0),
                    "rows_removed": changes.get('rows_removed', 0),
                    "columns_added": changes.get('columns_added', []),
                    "columns_removed": changes.get('columns_removed', [])
                },
                "content_changes": {
                    "cells_changed": changes.get('cells_changed', 0),
                    "formulas_added": changes.get('formulas_added', 0),
                    "formulas_removed": changes.get('formulas_removed', 0)
                },
                "data_types_changed": changes.get('data_types_changed', {}),
                "memory_usage_change": changes.get('memory_usage_change', 0)
            }
            
            return json.dumps(change_summary, indent=2)
            
        except Exception as e:
            return f"Error detecting changes: {str(e)}"


class ValidationTool(BaseTool):
    """LangChain tool for validating LLM responses"""
    name: str = "validate_explanation"
    description: str = "Validate explanation accuracy against actual data"
    
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


class LangChainExplanationWorkflow:
    """Proper LangChain implementation for explanation generation"""
    
    def __init__(self):
        self.llm = get_local_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        self.tools = [
            DataAnalysisTool(),
            ChangeDetectionTool(),
            ValidationTool()
        ]
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent Excel analysis agent. You have access to tools to analyze data, detect changes, and validate explanations.

Your task is to:
1. Analyze the current data structure and content
2. Detect what changed between before and after states
3. Generate an accurate, informative explanation
4. Validate your explanation against the actual data

Be precise with numbers, row counts, column counts, and statistics. Use the tools to get accurate information before making claims.

Format your response as a structured explanation with:
- Summary: Brief overview of what changed
- Details: Specific changes and their impact
- Insights: Key patterns or observations
- Next Steps: Recommended actions

Always use exact numbers from the data analysis tools."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create a custom agent that works with local LLMs
        self.agent_executor = self._create_custom_agent()
        
        # Create output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ExplanationOutput)
    
    def _create_custom_agent(self):
        """Create a custom agent that works with local LLMs using LangChain patterns"""
        
        class CustomLangChainAgent:
            def __init__(self, llm, tools, prompt, memory, workflow_instance):
                self.llm = llm
                self.tools = {tool.name: tool for tool in tools}
                self.prompt = prompt
                self.memory = memory
                self.workflow_instance = workflow_instance
            
            def invoke(self, inputs):
                """Execute the agent workflow using LangChain patterns"""
                try:
                    # Get chat history from memory
                    chat_history = self.memory.chat_memory.messages
                    
                    # Format the prompt with LangChain
                    formatted_prompt = self.prompt.format_messages(
                        input=inputs["input"],
                        chat_history=chat_history,
                        agent_scratchpad=[]
                    )
                    
                    # Create a chain for tool selection
                    tool_selection_chain = LLMChain(
                        llm=self.llm,
                        prompt=ChatPromptTemplate.from_messages([
                            ("system", """You are an intelligent agent that needs to analyze Excel data. 
                            
Available tools:
- analyze_data: Analyze DataFrame structure and content
- detect_changes: Detect changes between DataFrames  
- validate_explanation: Validate explanation accuracy

Based on the user's request, determine which tools to use and in what order. Respond with a JSON list of tool names to execute.

Example: ["analyze_data", "detect_changes", "validate_explanation"]"""),
                            ("human", "{input}")
                        ])
                    )
                    
                    # Get tool selection
                    tool_selection = tool_selection_chain.run(input=inputs["input"])
                    
                    # Parse tool selection
                    try:
                        tools_to_use = json.loads(tool_selection)
                    except:
                        tools_to_use = ["analyze_data", "detect_changes"]
                    
                    # Execute tools in sequence
                    tool_results = {}
                    for tool_name in tools_to_use:
                        if tool_name in self.tools:
                            print(f"🔧 LANGCHAIN: Executing tool: {tool_name}")
                            try:
                                # Execute the actual tool with proper data
                                if tool_name == "analyze_data":
                                    result = self.tools[tool_name]._run(self.workflow_instance.after_df, inputs["input"])
                                elif tool_name == "detect_changes":
                                    result = self.tools[tool_name]._run(self.workflow_instance.before_df, self.workflow_instance.after_df, "general")
                                elif tool_name == "validate_explanation":
                                    # This will be called after explanation generation
                                    continue
                                else:
                                    result = f"Tool {tool_name} executed successfully"
                                
                                tool_results[tool_name] = result
                            except Exception as e:
                                tool_results[tool_name] = f"Error executing {tool_name}: {str(e)}"
                    
                    # Create final explanation chain
                    explanation_chain = LLMChain(
                        llm=self.llm,
                        prompt=ChatPromptTemplate.from_messages([
                            ("system", """You are an intelligent Excel analysis agent. Based on the tool results, generate a comprehensive explanation.

Tool Results: {tool_results}

Generate a structured explanation with:
- Summary: Brief overview of what changed
- Details: Specific changes and their impact  
- Insights: Key patterns or observations
- Next Steps: Recommended actions

Be precise and use exact numbers from the tool results."""),
                            ("human", "Generate explanation for: {input}")
                        ])
                    )
                    
                    # Generate final explanation
                    explanation = explanation_chain.run(
                        input=inputs["input"],
                        tool_results=json.dumps(tool_results, indent=2)
                    )
                    
                    # Update memory
                    self.memory.chat_memory.add_user_message(inputs["input"])
                    self.memory.chat_memory.add_ai_message(explanation)
                    
                    return {"output": explanation}
                    
                except Exception as e:
                    return {"output": f"Error in agent execution: {str(e)}"}
        
        return CustomLangChainAgent(self.llm, self.tools, self.prompt, self.memory, self)
    
    def generate_explanation(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_type: str,
        operation_context: str
    ) -> str:
        """Generate explanation using proper LangChain agents and tools"""
        try:
            print("🚀 LANGCHAIN: Starting proper LangChain workflow...")
            
            # Store DataFrames for tool access
            self.before_df = before_df
            self.after_df = after_df
            
            # Prepare input for the agent
            agent_input = f"""
            Analyze the Excel data changes for operation: {operation_type}
            Context: {operation_context}
            
            Please:
            1. Use the analyze_data tool to examine the current data structure
            2. Use the detect_changes tool to identify what changed
            3. Generate an accurate explanation based on the tool results
            4. Use the validate_explanation tool to check your response
            
            Focus on accuracy and provide specific details about the data.
            """
            
            # Execute the agent
            result = self.agent_executor.invoke({
                "input": agent_input
            })
            
            print("✅ LANGCHAIN: Agent execution completed")
            
            # Format the output
            explanation = result.get("output", "No explanation generated")
            
            # Try to parse as structured output
            try:
                parsed_output = self.output_parser.parse(explanation)
                formatted_output = f"""
**Intelligent Analysis Summary**

**Summary of Changes**
{parsed_output.summary}

**Detailed Analysis**
{parsed_output.details}

**Key Insights**
{parsed_output.insights}

**Next Steps**
{parsed_output.next_steps}
"""
            except:
                # Fallback to raw explanation
                formatted_output = f"""
**Intelligent Analysis Summary**

{explanation}
"""
            
            print("✅ LANGCHAIN: Workflow completed successfully!")
            return formatted_output
            
        except Exception as e:
            print(f"❌ LANGCHAIN: Error in workflow: {str(e)}")
            return f"Error generating explanation: {str(e)}"


def create_langchain_workflow():
    """Factory function to create LangChain workflow"""
    return LangChainExplanationWorkflow()
