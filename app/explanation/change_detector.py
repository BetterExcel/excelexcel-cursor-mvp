"""
Change Detection System

Detects and analyzes changes between spreadsheet states to provide context for explanations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


class ChangeDetector:
    """
    Detects and analyzes changes between two DataFrame states.
    
    This class provides comprehensive change detection for spreadsheet operations,
    identifying modifications, additions, deletions, and patterns in the data.
    """
    
    def __init__(self):
        """Initialize the change detector."""
        self.change_history = []
        self.operation_metadata = {}
    
    def detect_changes(
        self, 
        before_df: pd.DataFrame, 
        after_df: pd.DataFrame, 
        operation_type: str,
        operation_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main method to detect what changed between two DataFrame states.
        
        Args:
            before_df: DataFrame before the operation
            after_df: DataFrame after the operation
            operation_type: Type of operation performed
            operation_context: Additional context about the operation
            
        Returns:
            Dictionary containing detailed change information
        """
        changes = {
            'operation_type': operation_type,
            'operation_context': operation_context or {},
            'timestamp': datetime.now().isoformat(),
            'cells_modified': [],
            'rows_added': 0,
            'columns_added': 0,
            'data_patterns': {},
            'key_values': [],
            'summary': '',
            'location': '',
            'key_info': '',
            'suggestions': [],
            'before_shape': before_df.shape,
            'after_shape': after_df.shape,
            'change_impact': 'low'
        }
        
        # Detect structural changes
        changes.update(self._detect_structural_changes(before_df, after_df))
        
        # Detect data changes
        changes.update(self._detect_data_changes(before_df, after_df))
        
        # Detect patterns and insights
        changes.update(self._detect_patterns(after_df, operation_type))
        
        # Generate summary and suggestions
        changes.update(self._generate_summary(changes, operation_type))
        
        # Store in history
        self.change_history.append(changes)
        
        return changes
    
    def _detect_structural_changes(
        self, 
        before_df: pd.DataFrame, 
        after_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect structural changes like new rows/columns."""
        changes = {}
        
        # Row changes
        before_rows, before_cols = before_df.shape
        after_rows, after_cols = after_df.shape
        
        changes['rows_added'] = max(0, after_rows - before_rows)
        changes['columns_added'] = max(0, after_cols - before_cols)
        
        # Column changes
        if before_cols != after_cols:
            new_columns = set(after_df.columns) - set(before_df.columns)
            removed_columns = set(before_df.columns) - set(after_df.columns)
            changes['new_columns'] = list(new_columns)
            changes['removed_columns'] = list(removed_columns)
        
        return changes
    
    def _detect_data_changes(
        self, 
        before_df: pd.DataFrame, 
        after_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect changes in data values."""
        changes = {}
        
        # Ensure both DataFrames have the same structure for comparison
        common_columns = list(set(before_df.columns) & set(after_df.columns))
        min_rows = min(len(before_df), len(after_df))
        
        if not common_columns or min_rows == 0:
            changes['cells_modified'] = []
            changes['data_summary'] = "No comparable data found"
            return changes
        
        # Compare common data
        modified_cells = []
        total_changes = 0
        formulas_detected = []
        
        for col in common_columns:
            for row_idx in range(min_rows):
                before_val = before_df.iloc[row_idx][col]
                after_val = after_df.iloc[row_idx][col]
                
                # Handle NaN/None values properly
                before_is_empty = pd.isna(before_val) or before_val is None or str(before_val).strip() == ''
                after_is_empty = pd.isna(after_val) or after_val is None or str(after_val).strip() == ''
                
                # Check if the new value is a formula
                is_formula = False
                if not after_is_empty and str(after_val).strip().startswith('='):
                    is_formula = True
                    formulas_detected.append({
                        'cell': f"{col}{row_idx + 1}",
                        'formula': str(after_val).strip(),
                        'result': str(after_val)  # The evaluated result
                    })
                
                if before_is_empty and after_is_empty:
                    continue
                elif before_is_empty or after_is_empty:
                    change_type = 'formula_added' if is_formula else 'value_change'
                    modified_cells.append({
                        'cell': f"{col}{row_idx + 1}",
                        'before': "empty" if before_is_empty else str(before_val),
                        'after': "empty" if after_is_empty else str(after_val),
                        'change_type': change_type,
                        'is_formula': is_formula
                    })
                    total_changes += 1
                elif str(before_val).strip() != str(after_val).strip():
                    change_type = 'formula_added' if is_formula else 'value_change'
                    modified_cells.append({
                        'cell': f"{col}{row_idx + 1}",
                        'before': str(before_val),
                        'after': str(after_val),
                        'change_type': change_type,
                        'is_formula': is_formula
                    })
                    total_changes += 1
        
        changes['cells_modified'] = modified_cells
        changes['total_cells_changed'] = total_changes
        changes['formulas_detected'] = formulas_detected
        
        # Debug output - show what actually changed
        if total_changes > 0:
            print(f"🔍 ChangeDetector: {total_changes} cells changed")
            for i, change in enumerate(modified_cells[:5]):  # Show first 5 changes
                formula_indicator = " [FORMULA]" if change.get('is_formula', False) else ""
                print(f"   {change['cell']}: '{change['before']}' → '{change['after']}'{formula_indicator}")
            if len(modified_cells) > 5:
                print(f"   ... and {len(modified_cells) - 5} more changes")
            
            # Show formula information
            if formulas_detected:
                print(f"🔍 ChangeDetector: {len(formulas_detected)} formulas detected:")
                for formula in formulas_detected[:3]:  # Show first 3 formulas
                    print(f"   {formula['cell']}: {formula['formula']}")
                if len(formulas_detected) > 3:
                    print(f"   ... and {len(formulas_detected) - 3} more formulas")
        else:
            print("🔍 ChangeDetector: No content changes detected")
        
        # Add new data if rows were added
        if len(after_df) > len(before_df):
            new_data = after_df.iloc[len(before_df):]
            changes['new_data_summary'] = f"Added {len(new_data)} new rows"
        
        return changes
    
    def _detect_patterns(
        self, 
        df: pd.DataFrame, 
        operation_type: str
    ) -> Dict[str, Any]:
        """Detect patterns and insights in the data."""
        patterns = {}
        
        if df.empty:
            return patterns
        
        # Detect numeric patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            patterns['numeric_columns'] = numeric_cols
            patterns['numeric_summary'] = {}
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    patterns['numeric_summary'][col] = {
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'mean': float(series.mean()),
                        'count': len(series)
                    }
        
        # Detect text patterns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            patterns['text_columns'] = text_cols
            patterns['text_summary'] = {}
            
            for col in text_cols[:3]:  # Limit to first 3 text columns
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    unique_values = non_null.nunique()
                    patterns['text_summary'][col] = {
                        'unique_values': int(unique_values),
                        'total_values': len(non_null),
                        'sample_values': non_null.head(3).tolist()
                    }
        
        # Detect date patterns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            patterns['date_columns'] = date_cols
        
        # Detect empty patterns
        empty_rows = df.isna().all(axis=1).sum()
        empty_cols = df.isna().all(axis=0).sum()
        patterns['empty_patterns'] = {
            'empty_rows': int(empty_rows),
            'empty_columns': int(empty_cols),
            'total_cells': int(df.size),
            'filled_cells': int(df.size - df.isna().sum().sum())
        }
        
        return {'data_patterns': patterns}
    
    def _generate_summary(
        self, 
        changes: Dict[str, Any], 
        operation_type: str
    ) -> Dict[str, Any]:
        """Generate human-readable summary and suggestions."""
        summary_parts = []
        location_parts = []
        key_info_parts = []
        suggestions = []
        
        # Generate summary based on operation type
        if operation_type == 'data_creation':
            if changes.get('total_cells_changed', 0) > 0:
                summary_parts.append(f"Created data with {changes['total_cells_changed']} cells filled")
            else:
                summary_parts.append(f"Created data with {changes['after_shape'][0]} rows and {changes['after_shape'][1]} columns")
            
            if changes['rows_added'] > 0:
                summary_parts.append(f"Added {changes['rows_added']} new rows")
            if changes['columns_added'] > 0:
                summary_parts.append(f"Added {changes['columns_added']} new columns")
        
        elif operation_type == 'formula_application':
            if changes['total_cells_changed'] > 0:
                summary_parts.append(f"Applied formulas to {changes['total_cells_changed']} cells")
            else:
                summary_parts.append("Formula applied successfully")
        
        elif operation_type == 'data_modification':
            if changes['total_cells_changed'] > 0:
                summary_parts.append(f"Modified {changes['total_cells_changed']} cells")
            else:
                summary_parts.append("Data modified successfully")
        
        # Generate location information
        if changes['after_shape'][0] > 0 and changes['after_shape'][1] > 0:
            location_parts.append(f"Sheet dimensions: {changes['after_shape'][0]} rows × {changes['after_shape'][1]} columns")
            
            if 'new_columns' in changes and changes['new_columns']:
                location_parts.append(f"New columns: {', '.join(changes['new_columns'])}")
        
        # Generate key information
        if changes['total_cells_changed'] > 0:
            key_info_parts.append(f"Total changes: {changes['total_cells_changed']} cells")
            
            # Show formula information if any
            if 'formulas_detected' in changes and changes['formulas_detected']:
                formula_count = len(changes['formulas_detected'])
                key_info_parts.append(f"Formulas applied: {formula_count} cells")
                
                # Show sample formulas
                sample_formulas = changes['formulas_detected'][:2]  # Show first 2 formulas
                formula_descriptions = []
                for formula in sample_formulas:
                    formula_descriptions.append(f"{formula['cell']}: {formula['formula']}")
                if formula_descriptions:
                    key_info_parts.append(f"Sample formulas: {'; '.join(formula_descriptions)}")
            
            # Show sample of what changed
            if changes['cells_modified']:
                sample_changes = changes['cells_modified'][:3]  # Show first 3 changes
                change_descriptions = []
                for change in sample_changes:
                    if change['before'] == 'empty':
                        formula_indicator = " (formula)" if change.get('is_formula', False) else ""
                        change_descriptions.append(f"{change['cell']}: added {change['after']}{formula_indicator}")
                    elif change['after'] == 'empty':
                        change_descriptions.append(f"{change['cell']}: removed {change['before']}")
                    else:
                        formula_indicator = " (formula)" if change.get('is_formula', False) else ""
                        change_descriptions.append(f"{change['cell']}: {change['before']} → {change['after']}{formula_indicator}")
                
                if change_descriptions:
                    key_info_parts.append(f"Sample changes: {'; '.join(change_descriptions)}")
            
            # Add data type analysis
            if changes['cells_modified']:
                data_types = {}
                for change in changes['cells_modified']:
                    if change['after'] != 'empty':
                        try:
                            float(change['after'])
                            data_types['numeric'] = data_types.get('numeric', 0) + 1
                        except:
                            data_types['text'] = data_types.get('text', 0) + 1
                
                if data_types:
                    type_info = []
                    if 'numeric' in data_types:
                        type_info.append(f"{data_types['numeric']} numeric values")
                    if 'text' in data_types:
                        type_info.append(f"{data_types['text']} text values")
                    key_info_parts.append(f"Data types: {', '.join(type_info)}")
        
        if 'data_patterns' in changes and changes['data_patterns']:
            patterns = changes['data_patterns']
            if 'numeric_summary' in patterns:
                numeric_cols = list(patterns['numeric_summary'].keys())[:2]  # Show first 2
                key_info_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
            
            if 'empty_patterns' in patterns:
                empty_info = patterns['empty_patterns']
                key_info_parts.append(f"Data coverage: {empty_info['filled_cells']}/{empty_info['total_cells']} cells filled")
        
        # Generate suggestions based on operation type and data
        if operation_type == 'data_creation':
            suggestions.extend([
                "Add formulas to calculate totals, averages, or other statistics",
                "Create charts to visualize the data patterns and trends", 
                "Apply formatting to highlight important values or ranges",
                "Use filters to explore specific data subsets or patterns"
            ])
            
            # Add specific suggestions based on data types
            if changes['total_cells_changed'] > 0 and changes['cells_modified']:
                has_numeric = any(change['after'] != 'empty' and str(change['after']).replace('.', '').replace('-', '').isdigit() 
                                for change in changes['cells_modified'])
                has_text = any(change['after'] != 'empty' and not str(change['after']).replace('.', '').replace('-', '').isdigit() 
                             for change in changes['cells_modified'])
                
                if has_numeric:
                    suggestions.append("Consider creating summary statistics (SUM, AVERAGE, MIN, MAX)")
                if has_text:
                    suggestions.append("Try text analysis functions (LEN, UPPER, LOWER, CONCATENATE)")
        
        elif operation_type == 'formula_application':
            suggestions.extend([
                "Verify formula results look correct",
                "Try applying similar formulas to other columns",
                "Create summary statistics for the calculated data"
            ])
        
        # Add general suggestions based on data patterns
        if 'data_patterns' in changes and changes['data_patterns']:
            patterns = changes['data_patterns']
            if 'numeric_summary' in patterns and len(patterns['numeric_summary']) > 1:
                suggestions.append("Create correlation analysis between numeric columns")
            
            if 'empty_patterns' in patterns and patterns['empty_patterns']['empty_rows'] > 0:
                suggestions.append("Consider filling empty cells or removing empty rows/columns")
        
        # Generate insights
        insights = []
        if changes['total_cells_changed'] > 0:
            insights.append(f"Successfully populated {changes['total_cells_changed']} cells with new data")
            
            if changes['cells_modified']:
                # Analyze the data pattern
                numeric_count = sum(1 for change in changes['cells_modified'] 
                                  if change['after'] != 'empty' and str(change['after']).replace('.', '').replace('-', '').isdigit())
                text_count = changes['total_cells_changed'] - numeric_count
                
                if numeric_count > 0:
                    insights.append(f"Generated {numeric_count} numeric values")
                if text_count > 0:
                    insights.append(f"Created {text_count} text entries")
        
        return {
            'summary': '; '.join(summary_parts) if summary_parts else "Operation completed successfully",
            'location': '; '.join(location_parts) if location_parts else "Data updated in current sheet",
            'key_info': '; '.join(key_info_parts) if key_info_parts else "Data structure modified",
            'insights': '; '.join(insights) if insights else "Data structure updated successfully",
            'suggestions': suggestions[:4] if suggestions else ["Explore the updated data to verify changes"]
        }
    
    def get_change_history(self) -> List[Dict[str, Any]]:
        """Get the history of all detected changes."""
        return self.change_history.copy()
    
    def get_last_change(self) -> Optional[Dict[str, Any]]:
        """Get the most recent change."""
        return self.change_history[-1] if self.change_history else None
    
    def clear_history(self):
        """Clear the change history."""
        self.change_history.clear()
