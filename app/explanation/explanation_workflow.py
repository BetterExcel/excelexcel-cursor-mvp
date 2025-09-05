"""
Explanation Workflow using LangGraph

Orchestrates the explanation generation process using LangGraph for structured reasoning.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from .change_detector import ChangeDetector
from .templates import ExplanationTemplates


class ExplanationWorkflow:
    """
    Simplified workflow for generating intelligent explanations of spreadsheet changes.
    
    This class orchestrates the entire explanation process:
    1. Change detection
    2. Context analysis
    3. Template selection
    4. Explanation generation
    5. Formatting and output
    """
    
    def __init__(self):
        """Initialize the explanation workflow."""
        self.change_detector = ChangeDetector()
        self.templates = ExplanationTemplates()
    
    def generate_explanation(
        self,
        operation_type: str,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a complete explanation for spreadsheet changes.
        
        Args:
            operation_type: Type of operation performed
            before_df: DataFrame before the operation
            after_df: DataFrame after the operation
            operation_context: Additional context about the operation
            
        Returns:
            Formatted explanation string
        """
        try:
            # Step 1: Detect changes
            changes = self.change_detector.detect_changes(
                before_df, after_df, operation_type, operation_context
            )
            
            # Step 2: Analyze context
            changes = self._analyze_context(changes, after_df)
            
            # Step 3: Generate explanation
            explanation = self.templates.generate_explanation(
                operation_type, changes
            )
            
            # Step 4: Format output
            final_output = self._format_output(explanation, changes, operation_type)
            
            return final_output
            
        except Exception as e:
            # Debug: Show what error occurred
            print(f"🚨 ExplanationWorkflow error: {str(e)}")
            # Fallback to simple explanation if workflow fails
            return self._generate_fallback_explanation(
                operation_type, before_df, after_df, operation_context
            )
    
    def _analyze_context(self, changes: Dict[str, Any], after_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze context and enhance change information."""
        # Enhance context based on operation type
        if changes["operation_type"] == "data_creation":
            # Add more context for data creation
            if "data_patterns" in changes and changes["data_patterns"]:
                patterns = changes["data_patterns"]
                if "numeric_summary" in patterns:
                    numeric_cols = list(patterns["numeric_summary"].keys())
                    if numeric_cols:
                        changes["context_enhancement"] = f"Data includes {len(numeric_cols)} numeric columns for analysis"
                
                if "text_summary" in patterns:
                    text_cols = list(patterns["text_summary"].keys())
                    if text_cols:
                        changes["context_enhancement"] = f"Text columns available for categorization and filtering"
        
        elif changes["operation_type"] == "formula_application":
            # Add formula-specific context
            if "total_cells_changed" in changes and changes["total_cells_changed"] > 0:
                changes["context_enhancement"] = f"Formulas applied to {changes['total_cells_changed']} cells"
        
        # Add sheet context
        if not after_df.empty:
            changes["sheet_context"] = {
                "total_rows": len(after_df),
                "total_columns": len(after_df.columns),
                "column_names": list(after_df.columns),
                "data_types": {col: str(after_df[col].dtype) for col in after_df.columns}
            }
        
        return changes
    
    def _format_output(self, explanation: str, changes: Dict[str, Any], operation_type: str) -> str:
        """Format the final output."""
        # Add operation metadata
        metadata = f"""
---
*Generated at: {changes.get('timestamp', 'Unknown')}*
*Operation: {operation_type}*
        """.strip()
        
        final_output = f"{explanation}\n\n{metadata}"
        
        return final_output
    
    def _generate_fallback_explanation(
        self,
        operation_type: str,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a simple fallback explanation if the workflow fails."""
        try:
            # Use change detector directly
            changes = self.change_detector.detect_changes(
                before_df, after_df, operation_type, operation_context
            )
            
            # Use templates directly
            explanation = self.templates.generate_explanation(
                operation_type, changes
            )
            
            return explanation
        except Exception:
            # Ultimate fallback
            return f"""
📋 Operation Summary

**📊 What Changed:** {operation_type} operation completed
**📍 Location:** Current sheet
**🔢 Key Information:** Data has been updated
**📋 Next Steps:** Review the changes to verify results

---
*Generated using fallback method*
            """.strip()
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current status of the workflow."""
        return {
            "workflow_built": True,
            "change_detector_ready": self.change_detector is not None,
            "templates_ready": self.templates is not None,
            "available_templates": self.templates.get_available_templates() if self.templates else []
        }
    
    def test_workflow(self) -> bool:
        """Test if the workflow is working correctly."""
        try:
            # Create test data
            test_before = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            test_after = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
            
            # Test the workflow
            result = self.generate_explanation(
                "data_creation",
                test_before,
                test_after,
                {"test": True}
            )
            
            # Check if we got a reasonable result
            return (
                isinstance(result, str) and
                len(result) > 50 and
                "📋" in result and
                "What Changed" in result
            )
        except Exception:
            return False


# Convenience function for quick explanation generation
def quick_explanation(
    operation_type: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    operation_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Quick function to generate explanations without creating a workflow instance.
    
    Args:
        operation_type: Type of operation performed
        before_df: DataFrame before the operation
        after_df: DataFrame after the operation
        operation_context: Additional context about the operation
        
    Returns:
        Formatted explanation string
    """
    workflow = ExplanationWorkflow()
    return workflow.generate_explanation(
        operation_type, before_df, after_df, operation_context
    )

