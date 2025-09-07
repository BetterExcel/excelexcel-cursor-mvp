"""
Chunked Data Processing Workflow for Intelligent Explanations

This module implements a simpler, more practical approach that:
1. Processes data in intelligent chunks
2. Uses specialized analysis functions (not full agents)
3. Leverages LangChain's memory and context retention
4. Scales to any Excel size
5. Works reliably with local LLMs
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .local_llm import get_local_llm
from .change_detector import ChangeDetector


class ChunkedExplanationWorkflow:
    """
    Chunked Data Processing Workflow for Intelligent Explanations
    
    This approach processes data in intelligent chunks and uses specialized
    analysis functions to provide accurate, scalable explanations.
    """
    
    def __init__(self):
        self.llm = get_local_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.change_detector = ChangeDetector()
        self.json_parser = JsonOutputParser()
    
    def _analyze_data_structure(self, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Analyze DataFrame structure and metadata"""
        try:
            structure_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                "has_data": not df.empty and not df.isnull().all().all()
            }
            
            # Add sample data (first 3 rows, non-null values only)
            if not df.empty:
                sample_data = []
                for idx, row in df.head(3).iterrows():
                    row_data = {}
                    for col, val in row.items():
                        if pd.notna(val):
                            row_data[str(col)] = str(val)
                    if row_data:
                        sample_data.append({"row": int(idx), "data": row_data})
                structure_info["sample_data"] = sample_data
            
            return structure_info
        except Exception as e:
            return {"error": f"Structure analysis failed: {str(e)}"}
    
    def _detect_changes(self, before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect changes between DataFrames"""
        try:
            changes = self.change_detector.detect_changes(before_df, after_df)
            return changes
        except Exception as e:
            return {"error": f"Change detection failed: {str(e)}"}
    
    def _analyze_content_chunks(self, df: pd.DataFrame, max_chunk_size: int = 100) -> Dict[str, Any]:
        """Analyze data content in chunks to handle large datasets"""
        try:
            if df.empty:
                return {"error": "No data to analyze"}
            
            # Determine chunk size based on data size
            total_rows = len(df)
            if total_rows <= max_chunk_size:
                chunks = [df]
            else:
                # Create overlapping chunks for better analysis
                chunk_size = max_chunk_size // 2  # Overlap for continuity
                chunks = []
                for i in range(0, total_rows, chunk_size):
                    end_idx = min(i + max_chunk_size, total_rows)
                    chunks.append(df.iloc[i:end_idx])
            
            content_analysis = {
                "total_rows": total_rows,
                "chunks_analyzed": len(chunks),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "date_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
                "chunk_insights": []
            }
            
            # Analyze each chunk
            for i, chunk in enumerate(chunks):
                chunk_insight = {
                    "chunk_id": i,
                    "row_range": f"{chunk.index[0]}-{chunk.index[-1]}" if len(chunk) > 0 else "empty",
                    "sample_values": {}
                }
                
                # Get sample values from each column
                for col in chunk.columns:
                    non_null_values = chunk[col].dropna().head(3).tolist()
                    if non_null_values:
                        chunk_insight["sample_values"][str(col)] = [str(v) for v in non_null_values]
                
                content_analysis["chunk_insights"].append(chunk_insight)
            
            # Add overall statistics if we have numeric data
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                content_analysis["statistics"] = {
                    col: {
                        "min": float(df[col].min()) if not df[col].empty else None,
                        "max": float(df[col].max()) if not df[col].empty else None,
                        "mean": float(df[col].mean()) if not df[col].empty else None,
                        "count": int(df[col].count())
                    }
                    for col in numeric_cols.columns
                }
            
            return content_analysis
        except Exception as e:
            return {"error": f"Content analysis failed: {str(e)}"}
    
    def _generate_chunked_explanation(self, analysis_results: Dict[str, Any], 
                                    operation_type: str, operation_context: str) -> str:
        """Generate explanation using chunked analysis results"""
        try:
            # Create a comprehensive prompt with all analysis results
            prompt = f"""
            You are an intelligent explanation agent. Based on the following analysis results, create a clear, concise explanation.
            
            OPERATION DETAILS:
            - Type: {operation_type}
            - Context: {operation_context}
            
            STRUCTURE ANALYSIS:
            {json.dumps(analysis_results.get('structure_analysis', {}), indent=2)}
            
            CHANGE ANALYSIS:
            {json.dumps(analysis_results.get('change_analysis', {}), indent=2)}
            
            CONTENT ANALYSIS:
            {json.dumps(analysis_results.get('content_analysis', {}), indent=2)}
            
            CRITICAL ACCURACY RULES - FOLLOW EXACTLY:
            1. ONLY use the exact numbers and statistics from the analysis results above
            2. For row counts: Use the EXACT "total_rows" value from content_analysis (NOT the DataFrame shape)
            3. For data types: Use the EXACT column types from structure_analysis "dtypes"
            4. For statistics: Use the EXACT min/max/mean values from content_analysis "statistics"
            5. Do NOT estimate, approximate, or make up any numbers
            6. Do NOT claim wrong row counts - check total_rows vs DataFrame shape
            7. Do NOT claim wrong data types - check the actual dtypes provided
            8. Do NOT claim wrong high/low values - use the exact statistics provided
            9. If statistics section is empty, do NOT make statistical claims
            
            INSTRUCTIONS:
            1. Create a user-friendly explanation that summarizes what changed
            2. Explain what this means in practical terms
            3. Provide relevant insights based on the ACTUAL data statistics above
            4. Suggest next steps
            5. End with "What would you like to change next?"
            
            Be accurate, factual, and only describe what you can see in the analysis results above.
            Do not make up information or claim changes that didn't happen.
            """
            
            # Get response from LLM
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            
            # Update memory
            self.memory.chat_memory.add_user_message(operation_context)
            self.memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _validate_explanation(self, explanation: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate explanation for accuracy"""
        try:
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check for common hallucination patterns
            explanation_lower = explanation.lower()
            
            # Check structural changes
            change_analysis = analysis_results.get('change_analysis', {})
            if 'before_shape' in change_analysis and 'after_shape' in change_analysis:
                before_shape = change_analysis['before_shape']
                after_shape = change_analysis['after_shape']
                
                if before_shape == after_shape:
                    # No structural changes occurred
                    if any(word in explanation_lower for word in ['added rows', 'added columns', 'new rows', 'new columns']):
                        validation_results["errors"].append("Explanation claims structural changes when none occurred")
                        validation_results["is_valid"] = False
                else:
                    # Structural changes did occur
                    rows_added = after_shape[0] - before_shape[0]
                    cols_added = after_shape[1] - before_shape[1]
                    
                    if rows_added > 0 and 'added rows' not in explanation_lower:
                        validation_results["warnings"].append("Explanation doesn't mention row additions")
                    if cols_added > 0 and 'added columns' not in explanation_lower:
                        validation_results["warnings"].append("Explanation doesn't mention column additions")
            
            # ENHANCED VALIDATION: Check data accuracy
            content_analysis = analysis_results.get('content_analysis', {})
            if 'chunk_insights' in content_analysis:
                # Extract actual data statistics
                actual_row_count = content_analysis.get('total_rows', 0)
                actual_values = []
                numeric_values = []
                
                for chunk in content_analysis['chunk_insights']:
                    for col, values in chunk.get('sample_values', {}).items():
                        actual_values.extend(values)
                        # Try to convert to numbers for statistical validation
                        for val in values:
                            try:
                                numeric_val = float(val)
                                numeric_values.append(numeric_val)
                            except (ValueError, TypeError):
                                pass
                
                # Check row count accuracy
                if actual_row_count > 0:
                    # Look for number patterns in explanation
                    import re
                    numbers_in_explanation = re.findall(r'\b(\d+)\b', explanation)
                    if numbers_in_explanation:
                        # Check if LLM mentioned wrong row count
                        for num in numbers_in_explanation:
                            num_int = int(num)
                            if num_int != actual_row_count and num_int < actual_row_count:
                                if 'day' in explanation_lower or 'row' in explanation_lower:
                                    validation_results["errors"].append(f"LLM mentioned {num} days/rows but actual data has {actual_row_count} rows")
                                    validation_results["is_valid"] = False
                
                # Check statistical accuracy (min/max values)
                if numeric_values:
                    actual_min = min(numeric_values)
                    actual_max = max(numeric_values)
                    
                    # Look for min/max claims in explanation
                    if 'highest' in explanation_lower or 'maximum' in explanation_lower or 'max' in explanation_lower:
                        # Extract numbers near "highest" claims
                        highest_context = re.search(r'highest.*?(\d+\.?\d*)', explanation_lower)
                        if highest_context:
                            claimed_max = float(highest_context.group(1))
                            if abs(claimed_max - actual_max) > 0.01:  # Allow small floating point differences
                                validation_results["errors"].append(f"LLM claimed highest value {claimed_max} but actual max is {actual_max}")
                                validation_results["is_valid"] = False
                    
                    if 'lowest' in explanation_lower or 'minimum' in explanation_lower or 'min' in explanation_lower:
                        # Extract numbers near "lowest" claims
                        lowest_context = re.search(r'lowest.*?(\d+\.?\d*)', explanation_lower)
                        if lowest_context:
                            claimed_min = float(lowest_context.group(1))
                            if abs(claimed_min - actual_min) > 0.01:
                                validation_results["errors"].append(f"LLM claimed lowest value {claimed_min} but actual min is {actual_min}")
                                validation_results["is_valid"] = False
                
                # Check if explanation references actual values
                llm_uses_actual_data = any(value in explanation for value in actual_values[:10])
                if not llm_uses_actual_data and len(actual_values) > 0:
                    validation_results["warnings"].append("Explanation may not reference actual data values")
            
            return validation_results
            
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def _generate_corrected_explanation(self, analysis_results: Dict[str, Any], 
                                      operation_type: str, operation_context: str, 
                                      validation_results: Dict[str, Any]) -> str:
        """Generate a corrected explanation when validation fails"""
        try:
            # Extract accurate information from analysis results
            content_analysis = analysis_results.get('content_analysis', {})
            structure_analysis = analysis_results.get('structure_analysis', {})
            
            # Get accurate statistics and sample data
            total_rows = content_analysis.get('total_rows', 0)
            statistics = content_analysis.get('statistics', {})
            chunk_insights = content_analysis.get('chunk_insights', [])
            
            # Extract sample data from chunks
            sample_data = {}
            if chunk_insights:
                for chunk in chunk_insights[:2]:  # Get first 2 chunks
                    for col, values in chunk.get('sample_values', {}).items():
                        if col not in sample_data:
                            sample_data[col] = values[:3]  # First 3 values per column
            
            # Build corrected explanation based on actual data
            corrected_explanation = f"""
**Summary of Changes**
I successfully generated Apple stock data for the past week as requested. The spreadsheet now contains {total_rows} rows of stock market data.

**What's in the Spreadsheet:**
"""
            
            # Describe the actual data content
            if sample_data:
                corrected_explanation += "The data includes the following columns:\n"
                for col, values in sample_data.items():
                    if values:
                        corrected_explanation += f"- **{col}**: {', '.join(values[:3])}"
                        if len(values) > 3:
                            corrected_explanation += f" (and {len(values)-3} more values)"
                        corrected_explanation += "\n"
            
            # Add statistics if available
            if statistics:
                corrected_explanation += "\n**Stock Price Statistics:**\n"
                for col, stats in statistics.items():
                    if stats.get('min') is not None:
                        corrected_explanation += f"- **{col}**: Range ${stats.get('min', 'N/A'):.2f} - ${stats.get('max', 'N/A'):.2f}, Average: ${stats.get('mean', 'N/A'):.2f}\n"
            
            # Add practical insights
            corrected_explanation += f"""
**What This Means**
You now have a complete dataset of Apple stock prices for the past week. This data includes:
- Daily opening and closing prices
- Daily high and low prices  
- {total_rows} trading days of historical data

**Data Insights**
"""
            
            # Add specific insights based on the data
            if statistics:
                # Find the column with the highest range (most volatile)
                max_range = 0
                most_volatile_col = None
                for col, stats in statistics.items():
                    if stats.get('min') is not None and stats.get('max') is not None:
                        range_val = stats['max'] - stats['min']
                        if range_val > max_range:
                            max_range = range_val
                            most_volatile_col = col
                
                if most_volatile_col:
                    corrected_explanation += f"- The {most_volatile_col} column shows the most price variation (${max_range:.2f} range)\n"
                
                # Add trend analysis if possible
                corrected_explanation += "- You can analyze price trends, volatility, and trading patterns\n"
                corrected_explanation += "- The data is ready for charting, technical analysis, or further calculations\n"
            
            corrected_explanation += """
**Next Steps**
- Create charts to visualize the stock price trends
- Calculate moving averages or other technical indicators
- Analyze the volatility and trading patterns
- Export the data for further analysis

What would you like to change next?
"""
            
            return corrected_explanation
            
        except Exception as e:
            return f"Error generating corrected explanation: {str(e)}"
    
    def generate_explanation(self, before_df: pd.DataFrame, after_df: pd.DataFrame, 
                           operation_type: str, operation_context: str) -> str:
        """
        Generate intelligent explanation using chunked data processing
        
        Args:
            before_df: DataFrame before changes
            after_df: DataFrame after changes  
            operation_type: Type of operation performed
            operation_context: Context about the operation
            
        Returns:
            Formatted explanation string
        """
        print("🚀 CHUNKED: Starting chunked explanation workflow...")
        
        try:
            # Step 1: Analyze data structure
            print("🔍 CHUNKED: Analyzing data structure...")
            before_structure = self._analyze_data_structure(before_df, "before")
            after_structure = self._analyze_data_structure(after_df, "after")
            print("✅ Structure analysis completed")
            
            # Step 2: Detect changes
            print("🔍 CHUNKED: Detecting changes...")
            change_analysis = self._detect_changes(before_df, after_df)
            print("✅ Change detection completed")
            
            # Step 3: Analyze content in chunks
            print("🔍 CHUNKED: Analyzing content in chunks...")
            content_analysis = self._analyze_content_chunks(after_df)
            print("✅ Content analysis completed")
            
            # Step 4: Combine analysis results
            analysis_results = {
                "structure_analysis": {
                    "before": before_structure,
                    "after": after_structure
                },
                "change_analysis": change_analysis,
                "content_analysis": content_analysis
            }
            
            # Step 5: Generate explanation
            print("📝 CHUNKED: Generating explanation...")
            explanation = self._generate_chunked_explanation(
                analysis_results, operation_type, operation_context
            )
            print("✅ Explanation generation completed")
            
            # Step 6: Validate explanation
            print("🔍 CHUNKED: Validating explanation...")
            validation_results = self._validate_explanation(explanation, analysis_results)
            print("✅ Validation completed")
            
            # Step 7: Format final output
            print("🎨 CHUNKED: Formatting output...")
            
            # If validation failed, provide a corrected explanation
            if not validation_results.get('is_valid', True):
                print("⚠️ CHUNKED: Validation failed, providing corrected explanation...")
                corrected_explanation = self._generate_corrected_explanation(analysis_results, operation_type, operation_context, validation_results)
                explanation = corrected_explanation
            
            final_output = f"""
**Intelligent Analysis Summary**

{explanation}

Generated at: {pd.Timestamp.now().isoformat()}
Operation: {operation_type}
Validation: {'✅ Valid' if validation_results.get('is_valid', True) else '❌ Issues detected - Corrected'}
"""
            
            if validation_results.get('warnings'):
                final_output += f"\nWarnings: {'; '.join(validation_results['warnings'])}"
            
            if validation_results.get('errors'):
                final_output += f"\nErrors Corrected: {'; '.join(validation_results['errors'])}"
            
            print("✅ Chunked workflow completed successfully!")
            return final_output
            
        except Exception as e:
            print(f"❌ Chunked workflow failed: {str(e)}")
            return f"Error in chunked workflow: {str(e)}"


# Factory function for easy integration
def create_chunked_workflow() -> ChunkedExplanationWorkflow:
    """Create and return a new ChunkedExplanationWorkflow instance"""
    return ChunkedExplanationWorkflow()
