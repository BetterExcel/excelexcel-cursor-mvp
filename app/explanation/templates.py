"""
Explanation Templates

Provides structured templates for generating user-friendly explanations of spreadsheet changes.
"""

from typing import Dict, Any, List, Optional
import re


class ExplanationTemplates:
    """
    Manages explanation templates for different types of spreadsheet operations.
    
    Templates provide consistent, professional explanations with clear structure
    and actionable information for users.
    """
    
    def __init__(self):
        """Initialize the explanation templates."""
        self.templates = self._load_default_templates()
        self.custom_templates = {}
    
    def _load_default_templates(self) -> Dict[str, Dict[str, str]]:
        """Load the default explanation templates."""
        return {
            'data_creation': {
                'title': '📊 Data Creation Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Key Data:** {key_info}',
                'insights': '**💡 Insights:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'formula_application': {
                'title': '🧮 Formula Application Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Results:** {key_info}',
                'insights': '**💡 Formula Insights:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'data_modification': {
                'title': '✏️ Data Modification Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Changes:** {key_info}',
                'insights': '**💡 Impact:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'sorting': {
                'title': '🔄 Data Sorting Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Sort Details:** {key_info}',
                'insights': '**💡 Order:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'filtering': {
                'title': '🔍 Data Filtering Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Filter Results:** {key_info}',
                'insights': '**💡 Filtered Data:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'sheet_management': {
                'title': '📋 Sheet Management Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Sheet Info:** {key_info}',
                'insights': '**💡 Structure:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'chart_creation': {
                'title': '📈 Chart Creation Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Chart Details:** {key_info}',
                'insights': '**💡 Visualization:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'data_import': {
                'title': '📥 Data Import Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Import Details:** {key_info}',
                'insights': '**💡 Data Quality:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'data_export': {
                'title': '📤 Data Export Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Export Details:** {key_info}',
                'insights': '**💡 Export Status:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            },
            'general': {
                'title': '📋 Operation Summary',
                'what_changed': '**📊 What Changed:** {summary}',
                'location': '**📍 Location:** {location}',
                'key_data': '**🔢 Key Information:** {key_info}',
                'insights': '**💡 Insights:** {insights}',
                'next_steps': '**📋 Next Steps:** {suggestions}'
            }
        }
    
    def get_template(self, operation_type: str) -> Dict[str, str]:
        """
        Get the template for a specific operation type.
        
        Args:
            operation_type: Type of operation (e.g., 'data_creation', 'formula_application')
            
        Returns:
            Template dictionary for the operation type
        """
        # Try to get the specific template
        if operation_type in self.templates:
            return self.templates[operation_type]
        
        # Try to find a template by partial match
        for key in self.templates.keys():
            if operation_type.lower() in key.lower() or key.lower() in operation_type.lower():
                return self.templates[key]
        
        # Return general template as fallback
        return self.templates['general']
    
    def generate_explanation(
        self, 
        operation_type: str, 
        changes: Dict[str, Any],
        custom_insights: Optional[str] = None
    ) -> str:
        """
        Generate a complete explanation using the appropriate template.
        
        Args:
            operation_type: Type of operation performed
            changes: Change detection results
            custom_insights: Optional custom insights to include
            
        Returns:
            Formatted explanation string
        """
        template = self.get_template(operation_type)
        
        # Prepare the data for template filling
        template_data = {
            'summary': changes.get('summary', 'Operation completed'),
            'location': changes.get('location', 'Current sheet'),
            'key_info': changes.get('key_info', 'Data updated'),
            'insights': custom_insights or self._generate_insights(changes),
            'suggestions': self._format_suggestions(changes.get('suggestions', []))
        }
        
        # Build the explanation
        explanation_parts = [template['title']]
        
        # Add each section if it has content
        for section_key in ['what_changed', 'location', 'key_data', 'insights', 'next_steps']:
            if section_key in template:
                section_content = template[section_key].format(**template_data)
                if section_content and not section_content.endswith('None'):
                    explanation_parts.append(section_content)
        
        return '\n\n'.join(explanation_parts)
    
    def _generate_insights(self, changes: Dict[str, Any]) -> str:
        """Generate insights based on the detected changes."""
        insights = []
        
        # Data pattern insights
        if 'data_patterns' in changes and changes['data_patterns']:
            patterns = changes['data_patterns']
            
            # Numeric insights
            if 'numeric_summary' in patterns and patterns['numeric_summary']:
                numeric_cols = list(patterns['numeric_summary'].keys())
                if len(numeric_cols) > 1:
                    insights.append(f"Multiple numeric columns ({', '.join(numeric_cols[:2])}) available for analysis")
                else:
                    insights.append(f"Single numeric column ({numeric_cols[0]}) ready for calculations")
            
            # Text insights
            if 'text_summary' in patterns and patterns['text_summary']:
                text_cols = list(patterns['text_summary'].keys())
                insights.append(f"Text columns ({', '.join(text_cols[:2])}) available for categorization")
            
            # Empty data insights
            if 'empty_patterns' in patterns:
                empty_info = patterns['empty_patterns']
                if empty_info['filled_cells'] > 0:
                    coverage = (empty_info['filled_cells'] / empty_info['total_cells']) * 100
                    insights.append(f"Data coverage: {coverage:.1f}% ({empty_info['filled_cells']}/{empty_info['total_cells']} cells)")
        
        # Change impact insights
        if changes.get('total_cells_changed', 0) > 0:
            if changes['total_cells_changed'] < 10:
                insights.append("Small-scale changes - easy to review and verify")
            elif changes['total_cells_changed'] < 100:
                insights.append("Medium-scale changes - consider using filters to review")
            else:
                insights.append("Large-scale changes - use summary statistics to verify results")
        
        # Structural insights
        if changes.get('rows_added', 0) > 0:
            insights.append(f"Added {changes['rows_added']} new rows to expand dataset")
        
        if changes.get('columns_added', 0) > 0:
            insights.append(f"Added {changes['columns_added']} new columns for additional data")
        
        return '; '.join(insights) if insights else "Data structure updated successfully"
    
    def _format_suggestions(self, suggestions: List[str]) -> str:
        """Format suggestions into a readable string."""
        if not suggestions:
            return "Explore the updated data to understand the changes"
        
        # Format each suggestion with a bullet point
        formatted_suggestions = []
        for i, suggestion in enumerate(suggestions[:3], 1):  # Limit to 3 suggestions
            formatted_suggestions.append(f"{i}. {suggestion}")
        
        return '\n'.join(formatted_suggestions)
    
    def add_custom_template(self, operation_type: str, template: Dict[str, str]):
        """
        Add a custom template for a specific operation type.
        
        Args:
            operation_type: Name of the operation type
            template: Template dictionary with sections
        """
        self.custom_templates[operation_type] = template
    
    def get_available_templates(self) -> List[str]:
        """Get list of all available template types."""
        all_templates = list(self.templates.keys()) + list(self.custom_templates.keys())
        return sorted(all_templates)
    
    def customize_template(self, operation_type: str, section: str, new_content: str):
        """
        Customize a specific section of an existing template.
        
        Args:
            operation_type: Type of operation to customize
            section: Section to modify (e.g., 'what_changed', 'next_steps')
            new_content: New content for the section
        """
        if operation_type in self.templates:
            self.templates[operation_type][section] = new_content
        elif operation_type in self.custom_templates:
            self.custom_templates[operation_type][section] = new_content
        else:
            # Create new custom template
            self.custom_templates[operation_type] = {
                'title': f'📋 {operation_type.title()} Summary',
                section: new_content
            }
    
    def get_template_preview(self, operation_type: str) -> str:
        """
        Get a preview of how a template will look with sample data.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Preview string with sample data
        """
        sample_changes = {
            'summary': 'Sample operation completed successfully',
            'location': 'Sheet1, columns A-E, rows 1-10',
            'key_info': '5 columns, 10 rows, 50 total cells',
            'suggestions': ['Try adding formulas', 'Create charts', 'Apply filters']
        }
        
        return self.generate_explanation(operation_type, sample_changes, "Sample insights for demonstration")
