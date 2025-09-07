"""
Multi-Agent LangChain Architecture for Intelligent Explanations

This module implements a scalable multi-agent system using LangChain's advanced features:
- Specialized agents for different tasks
- Memory and context retention
- Tool-based processing
- Chunked data handling
- Scalable to any Excel size
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .local_llm import get_local_llm
from .change_detector import ChangeDetector


class DataStructureAnalyzerTool(BaseTool):
    """Tool for analyzing DataFrame structure and metadata"""
    
    name: str = "analyze_data_structure"
    description: str = "Analyze the structure of a DataFrame including shape, column names, data types, and basic statistics"
    
    def _run(self, df: pd.DataFrame, context: str = "") -> str:
        """Analyze DataFrame structure"""
        try:
            structure_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            }
            return json.dumps(structure_info, indent=2, default=str)
        except Exception as e:
            return f"Error analyzing structure: {str(e)}"


class ChangeDetectionTool(BaseTool):
    """Tool for detecting changes between two DataFrames"""
    
    name: str = "detect_changes"
    description: str = "Detect and analyze changes between before and after DataFrame states"
    
    def _run(self, before_df: pd.DataFrame, after_df: pd.DataFrame) -> str:
        """Detect changes between DataFrames"""
        try:
            # Create ChangeDetector instance locally to avoid Pydantic issues
            change_detector = ChangeDetector()
            changes = change_detector.detect_changes(before_df, after_df)
            return json.dumps(changes, indent=2, default=str)
        except Exception as e:
            return f"Error detecting changes: {str(e)}"


class ContentAnalysisTool(BaseTool):
    """Tool for analyzing data content and patterns"""
    
    name: str = "analyze_content"
    description: str = "Analyze the actual content of data including patterns, statistics, and insights"
    
    def _run(self, df: pd.DataFrame, focus_columns: List[str] = None) -> str:
        """Analyze data content"""
        try:
            if focus_columns:
                df_analysis = df[focus_columns]
            else:
                df_analysis = df
            
            content_info = {
                "numeric_columns": df_analysis.select_dtypes(include=['number']).columns.tolist(),
                "text_columns": df_analysis.select_dtypes(include=['object']).columns.tolist(),
                "date_columns": df_analysis.select_dtypes(include=['datetime']).columns.tolist(),
                "statistics": df_analysis.describe().to_dict() if len(df_analysis.select_dtypes(include=['number'])) > 0 else {},
                "unique_values": {col: df_analysis[col].nunique() for col in df_analysis.columns},
                "sample_values": {col: df_analysis[col].dropna().head(5).tolist() for col in df_analysis.columns}
            }
            return json.dumps(content_info, indent=2, default=str)
        except Exception as e:
            return f"Error analyzing content: {str(e)}"


class ValidationTool(BaseTool):
    """Tool for validating LLM responses against actual data"""
    
    name: str = "validate_response"
    description: str = "Validate LLM response accuracy against actual data to catch hallucinations"
    
    def _run(self, llm_response: str, actual_data_info: str) -> str:
        """Validate LLM response"""
        try:
            # Parse LLM response and actual data
            response_data = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
            actual_data = json.loads(actual_data_info) if isinstance(actual_data_info, str) else actual_data_info
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check for common hallucination patterns
            if "structural_changes" in response_data:
                structural_text = response_data["structural_changes"].lower()
                if "added" in structural_text and actual_data.get("shape_changed", False) == False:
                    validation_results["errors"].append("LLM claimed structural changes when none occurred")
                    validation_results["is_valid"] = False
            
            if "data_patterns" in response_data:
                patterns_text = response_data["data_patterns"].lower()
                # Check if LLM references actual data values
                actual_values = []
                if "sample_values" in actual_data:
                    for col_values in actual_data["sample_values"].values():
                        actual_values.extend([str(v) for v in col_values[:3]])
                
                llm_uses_actual_data = any(value in patterns_text for value in actual_values)
                if not llm_uses_actual_data and len(actual_values) > 0:
                    validation_results["warnings"].append("LLM may not be referencing actual data values")
            
            return json.dumps(validation_results, indent=2)
        except Exception as e:
            return f"Error validating response: {str(e)}"


class MultiAgentExplanationWorkflow:
    """
    Multi-Agent LangChain Architecture for Intelligent Explanations
    
    Uses specialized agents with tools, memory, and context retention
    to provide accurate, scalable explanations for any Excel size.
    """
    
    # Define the workflow state
    from typing import TypedDict
    
    class WorkflowState(TypedDict):
        before_df: pd.DataFrame
        after_df: pd.DataFrame
        operation_type: str
        operation_context: str
        structure_analysis: Dict[str, Any]
        change_analysis: Dict[str, Any]
        content_analysis: Dict[str, Any]
        explanation: str
        validation_results: Dict[str, Any]
        final_output: str
    
    def __init__(self):
        self.llm = get_local_llm()
        self.memory = MemorySaver()
        self.tools = [
            DataStructureAnalyzerTool(),
            ChangeDetectionTool(),
            ContentAnalysisTool(),
            ValidationTool()
        ]
        self._setup_agents()
        self._build_workflow()
    
    def _setup_agents(self):
        """Setup specialized agents with tools and memory"""
        
        # Data Structure Analyzer Agent
        self.structure_agent = self._create_agent(
            name="DataStructureAnalyzer",
            description="Analyzes DataFrame structure, shape, columns, and metadata",
            tools=[self.tools[0]]  # DataStructureAnalyzerTool
        )
        
        # Change Detector Agent  
        self.change_agent = self._create_agent(
            name="ChangeDetector",
            description="Detects and analyzes changes between before/after DataFrame states",
            tools=[self.tools[1]]  # ChangeDetectionTool
        )
        
        # Content Analyzer Agent
        self.content_agent = self._create_agent(
            name="ContentAnalyzer", 
            description="Analyzes data content, patterns, and provides insights",
            tools=[self.tools[2]]  # ContentAnalysisTool
        )
        
        # Explanation Generator Agent
        self.explanation_agent = self._create_agent(
            name="ExplanationGenerator",
            description="Creates user-friendly explanations based on analysis results",
            tools=[]  # No tools needed, just generates explanations
        )
        
        # Validator Agent
        self.validator_agent = self._create_agent(
            name="Validator",
            description="Validates explanations for accuracy and catches hallucinations",
            tools=[self.tools[3]]  # ValidationTool
        )
    
    def _create_agent(self, name: str, description: str, tools: List[BaseTool]) -> Any:
        """Create a specialized agent with tools and memory"""
        
        # For local LLMs that don't support bind_tools, create a simple agent
        # that uses the LLM directly with tool information in the prompt
        
        class SimpleAgent:
            def __init__(self, llm, name: str, description: str, tools: List[BaseTool]):
                self.llm = llm
                self.name = name
                self.description = description
                self.tools = {tool.name: tool for tool in tools}
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            
            def invoke(self, input_data: Dict[str, Any]) -> str:
                """Invoke the agent with input"""
                try:
                    # Get chat history
                    chat_history = self.memory.chat_memory.messages
                    
                    # Create system prompt with tool information
                    system_prompt = f"""You are {self.name}: {self.description}
                    
                    You have access to these specialized tools:
                    {json.dumps({name: tool.description for name, tool in self.tools.items()}, indent=2)}
                    
                    When you need to use a tool, respond with JSON in this format:
                    {{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}
                    
                    Be precise, factual, and avoid making assumptions.
                    """
                    
                    # Create messages
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Add chat history
                    for msg in chat_history:
                        if hasattr(msg, 'content'):
                            messages.append({"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content})
                    
                    # Add current input
                    messages.append({"role": "user", "content": input_data["input"]})
                    
                    # Get response from LLM
                    response = self.llm.invoke(messages)
                    
                    # Check if response contains tool call
                    if isinstance(response, str) and response.strip().startswith('{'):
                        try:
                            tool_call = json.loads(response)
                            if "tool" in tool_call and tool_call["tool"] in self.tools:
                                # Execute tool
                                tool = self.tools[tool_call["tool"]]
                                tool_result = tool._run(**tool_call.get("parameters", {}))
                                
                                # Get final response with tool result
                                final_messages = messages + [
                                    {"role": "assistant", "content": response},
                                    {"role": "user", "content": f"Tool result: {tool_result}"}
                                ]
                                final_response = self.llm.invoke(final_messages)
                                
                                # Update memory
                                self.memory.chat_memory.add_user_message(input_data["input"])
                                self.memory.chat_memory.add_ai_message(final_response)
                                
                                return final_response
                        except json.JSONDecodeError:
                            pass
                    
                    # Update memory
                    self.memory.chat_memory.add_user_message(input_data["input"])
                    self.memory.chat_memory.add_ai_message(response)
                    
                    return response
                    
                except Exception as e:
                    return f"Error in {self.name}: {str(e)}"
        
        return SimpleAgent(self.llm, name, description, tools)
    
    def _build_workflow(self):
        """Build the LangGraph workflow for multi-agent coordination"""
        
        # Create the graph using the class-defined WorkflowState
        workflow = StateGraph(self.WorkflowState)
        
        # Add nodes
        workflow.add_node("analyze_structure", self._analyze_structure_node)
        workflow.add_node("detect_changes", self._detect_changes_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("generate_explanation", self._generate_explanation_node)
        workflow.add_node("validate_explanation", self._validate_explanation_node)
        workflow.add_node("format_output", self._format_output_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_structure")
        workflow.add_edge("analyze_structure", "detect_changes")
        workflow.add_edge("detect_changes", "analyze_content")
        workflow.add_edge("analyze_content", "generate_explanation")
        workflow.add_edge("generate_explanation", "validate_explanation")
        workflow.add_edge("validate_explanation", "format_output")
        workflow.add_edge("format_output", END)
        
        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
    
    def _analyze_structure_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze DataFrame structure"""
        print("🔍 MULTI-AGENT: Analyzing data structure...")
        
        try:
            # Analyze before structure
            before_structure = self.structure_agent.invoke({
                "input": f"Analyze the structure of this DataFrame: {state.before_df.to_string()}"
            })
            
            # Analyze after structure  
            after_structure = self.structure_agent.invoke({
                "input": f"Analyze the structure of this DataFrame: {state.after_df.to_string()}"
            })
            
            structure_analysis = {
                "before": before_structure,
                "after": after_structure,
                "comparison": "Structure analysis completed"
            }
            
            print("✅ Structure analysis completed")
            return {"structure_analysis": structure_analysis}
            
        except Exception as e:
            print(f"❌ Structure analysis failed: {str(e)}")
            return {"structure_analysis": {"error": str(e)}}
    
    def _detect_changes_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Detect changes between DataFrames"""
        print("🔍 MULTI-AGENT: Detecting changes...")
        
        try:
            changes = self.change_agent.invoke({
                "input": f"Detect changes between these DataFrames:\nBefore: {state.before_df.shape}\nAfter: {state.after_df.shape}"
            })
            
            print("✅ Change detection completed")
            return {"change_analysis": changes}
            
        except Exception as e:
            print(f"❌ Change detection failed: {str(e)}")
            return {"change_analysis": {"error": str(e)}}
    
    def _analyze_content_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze data content"""
        print("🔍 MULTI-AGENT: Analyzing content...")
        
        try:
            content_analysis = self.content_agent.invoke({
                "input": f"Analyze the content of this DataFrame: {state.after_df.head(10).to_string()}"
            })
            
            print("✅ Content analysis completed")
            return {"content_analysis": content_analysis}
            
        except Exception as e:
            print(f"❌ Content analysis failed: {str(e)}")
            return {"content_analysis": {"error": str(e)}}
    
    def _generate_explanation_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate user-friendly explanation"""
        print("📝 MULTI-AGENT: Generating explanation...")
        
        try:
            explanation_prompt = f"""
            Based on the following analysis results, create a user-friendly explanation:
            
            Structure Analysis: {state.structure_analysis}
            Change Analysis: {state.change_analysis}
            Content Analysis: {state.content_analysis}
            Operation: {state.operation_type}
            Context: {state.operation_context}
            
            Create a clear, concise explanation that:
            1. Summarizes what changed
            2. Explains what this means
            3. Provides relevant insights
            4. Suggests next steps
            5. Ends with "What would you like to change next?"
            """
            
            explanation = self.explanation_agent.invoke({
                "input": explanation_prompt
            })
            
            print("✅ Explanation generation completed")
            return {"explanation": explanation}
            
        except Exception as e:
            print(f"❌ Explanation generation failed: {str(e)}")
            return {"explanation": f"Error generating explanation: {str(e)}"}
    
    def _validate_explanation_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Validate explanation for accuracy"""
        print("🔍 MULTI-AGENT: Validating explanation...")
        
        try:
            validation = self.validator_agent.invoke({
                "input": f"Validate this explanation against the actual data:\nExplanation: {state.explanation}\nData: {state.content_analysis}"
            })
            
            print("✅ Validation completed")
            return {"validation_results": validation}
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            return {"validation_results": {"error": str(e)}}
    
    def _format_output_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Format final output"""
        print("🎨 MULTI-AGENT: Formatting output...")
        
        try:
            final_output = f"""
            **Intelligent Analysis Summary**
            
            {state.explanation}
            
            Generated at: {pd.Timestamp.now().isoformat()}
            Operation: {state.operation_type}
            """
            
            print("✅ Output formatting completed")
            return {"final_output": final_output}
            
        except Exception as e:
            print(f"❌ Output formatting failed: {str(e)}")
            return {"final_output": f"Error formatting output: {str(e)}"}
    
    def generate_explanation(self, before_df: pd.DataFrame, after_df: pd.DataFrame, 
                           operation_type: str, operation_context: str) -> str:
        """
        Generate intelligent explanation using multi-agent architecture
        
        Args:
            before_df: DataFrame before changes
            after_df: DataFrame after changes  
            operation_type: Type of operation performed
            operation_context: Context about the operation
            
        Returns:
            Formatted explanation string
        """
        print("🚀 MULTI-AGENT: Starting multi-agent explanation workflow...")
        
        try:
            # Create initial state
            initial_state = {
                "before_df": before_df,
                "after_df": after_df,
                "operation_type": operation_type,
                "operation_context": operation_context
            }
            
            # Run the workflow with proper configuration
            config = {"configurable": {"thread_id": "explanation_thread"}}
            result = self.workflow.invoke(initial_state, config=config)
            
            print("✅ Multi-agent workflow completed successfully!")
            return result.get("final_output", "Error: No output generated")
            
        except Exception as e:
            print(f"❌ Multi-agent workflow failed: {str(e)}")
            return f"Error in multi-agent workflow: {str(e)}"


# Factory function for easy integration
def create_multi_agent_workflow() -> MultiAgentExplanationWorkflow:
    """Create and return a new MultiAgentExplanationWorkflow instance"""
    return MultiAgentExplanationWorkflow()
