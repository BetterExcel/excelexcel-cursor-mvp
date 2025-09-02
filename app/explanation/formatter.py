"""
Explanation Formatter

Handles final output formatting, styling, and presentation of explanations.
"""

from typing import Dict, Any, List, Optional
import re


class ExplanationFormatter:
    """
    Formats and styles explanation output for optimal user experience.
    
    This class handles:
    - Consistent styling and emojis
    - Mobile-friendly formatting
    - Customization options
    - Output validation
    """
    
    def __init__(self, style_preferences: Optional[Dict[str, Any]] = None):
        """
        Initialize the formatter with optional style preferences.
        
        Args:
            style_preferences: Dictionary of styling preferences
        """
        self.style_preferences = style_preferences or self._get_default_preferences()
        self.emoji_mapping = self._get_emoji_mapping()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default formatting preferences."""
        return {
            'use_emojis': True,
            'compact_mode': False,
            'highlight_key_info': True,
            'include_metadata': True,
            'max_suggestions': 3,
            'line_breaks': 'double',
            'bullet_style': 'numbered'
        }
    
    def _get_emoji_mapping(self) -> Dict[str, str]:
        """Get mapping of operation types to appropriate emojis."""
        return {
            'data_creation': '📊',
            'formula_application': '🧮',
            'data_modification': '✏️',
            'sorting': '🔄',
            'filtering': '🔍',
            'sheet_management': '📋',
            'chart_creation': '📈',
            'data_import': '📥',
            'data_export': '📤',
            'general': '📋'
        }
    
    def format_explanation(
        self, 
        explanation: str, 
        operation_type: str = 'general',
        custom_style: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format an explanation with consistent styling.
        
        Args:
            explanation: Raw explanation text
            operation_type: Type of operation for emoji selection
            custom_style: Optional custom styling overrides
            
        Returns:
            Formatted explanation string
        """
        # Merge custom style with preferences
        style = {**self.style_preferences, **(custom_style or {})}
        
        # Add operation emoji if enabled
        if style.get('use_emojis', True):
            emoji = self.emoji_mapping.get(operation_type, self.emoji_mapping['general'])
            explanation = f"{emoji} {explanation}"
        
        # Apply formatting
        formatted = self._apply_formatting(explanation, style)
        
        # Validate output
        formatted = self._validate_output(formatted)
        
        return formatted
    
    def _apply_formatting(self, explanation: str, style: Dict[str, Any]) -> str:
        """Apply formatting based on style preferences."""
        formatted = explanation
        
        # Handle line breaks
        if style.get('line_breaks') == 'single':
            formatted = formatted.replace('\n\n', '\n')
        elif style.get('line_breaks') == 'compact':
            formatted = re.sub(r'\n\s*\n', '\n', formatted)
        
        # Highlight key information
        if style.get('highlight_key_info', True):
            formatted = self._highlight_key_info(formatted)
        
        # Format suggestions
        if style.get('bullet_style') == 'dashed':
            formatted = formatted.replace('1. ', '- ')
            formatted = formatted.replace('2. ', '- ')
            formatted = formatted.replace('3. ', '- ')
        
        # Compact mode
        if style.get('compact_mode', False):
            formatted = self._make_compact(formatted)
        
        return formatted
    
    def _highlight_key_info(self, text: str) -> str:
        """Highlight key information in the text."""
        # Highlight numbers and percentages
        text = re.sub(r'(\d+(?:\.\d+)?%)', r'**\1**', text)
        
        # Highlight cell references
        text = re.sub(r'([A-Z]+\d+)', r'`\1`', text)
        
        # Highlight column ranges
        text = re.sub(r'([A-Z]+:[A-Z]+)', r'`\1`', text)
        
        # Highlight formulas
        text = re.sub(r'(=SUM\([^)]+\))', r'`\1`', text)
        text = re.sub(r'(=AVERAGE\([^)]+\))', r'`\1`', text)
        text = re.sub(r'(=COUNT\([^)]+\))', r'`\1`', text)
        
        return text
    
    def _make_compact(self, text: str) -> str:
        """Make the text more compact for space-constrained displays."""
        # Reduce multiple line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Reduce spacing around sections
        text = re.sub(r'\n\*\*([^*]+)\*\*:\s*\n', r'\n**\1:** ', text)
        
        return text
    
    def _validate_output(self, text: str) -> str:
        """Validate and clean the output text."""
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        
        # Ensure consistent line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove empty lines at start/end
        text = text.strip()
        
        # Ensure minimum length
        if len(text) < 50:
            text = f"{text}\n\n*Explanation generated successfully*"
        
        return text
    
    def create_summary_card(
        self, 
        changes: Dict[str, Any], 
        operation_type: str = 'general'
    ) -> str:
        """
        Create a summary card format for quick overview.
        
        Args:
            changes: Change detection results
            operation_type: Type of operation
            
        Returns:
            Formatted summary card
        """
        emoji = self.emoji_mapping.get(operation_type, '📋')
        
        card_parts = [
            f"{emoji} **{operation_type.replace('_', ' ').title()}**",
            f"📊 **Changes:** {changes.get('summary', 'Operation completed')}",
            f"📍 **Location:** {changes.get('location', 'Current sheet')}"
        ]
        
        # Add key data if available
        if changes.get('key_info'):
            card_parts.append(f"🔢 **Data:** {changes['key_info']}")
        
        # Add suggestions if available
        suggestions = changes.get('suggestions', [])
        if suggestions:
            card_parts.append(f"💡 **Next:** {suggestions[0]}")
        
        return '\n'.join(card_parts)
    
    def create_detailed_report(
        self, 
        changes: Dict[str, Any], 
        operation_type: str = 'general',
        include_metadata: bool = True
    ) -> str:
        """
        Create a detailed report format with all information.
        
        Args:
            changes: Change detection results
            operation_type: Type of operation
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted detailed report
        """
        emoji = self.emoji_mapping.get(operation_type, '📋')
        
        report_parts = [
            f"{emoji} **{operation_type.replace('_', ' ').title()} Report**",
            "=" * 50
        ]
        
        # Add main sections
        if changes.get('summary'):
            report_parts.append(f"\n**📊 What Changed:**\n{changes['summary']}")
        
        if changes.get('location'):
            report_parts.append(f"\n**📍 Location:**\n{changes['location']}")
        
        if changes.get('key_info'):
            report_parts.append(f"\n**🔢 Key Data:**\n{changes['key_info']}")
        
        # Add data patterns if available
        if 'data_patterns' in changes and changes['data_patterns']:
            patterns = changes['data_patterns']
            report_parts.append(f"\n**📈 Data Patterns:**")
            
            if 'numeric_summary' in patterns:
                numeric_info = []
                for col, stats in list(patterns['numeric_summary'].items())[:3]:
                    numeric_info.append(f"  • {col}: {stats['count']} values, range {stats['min']:.2f}-{stats['max']:.2f}")
                report_parts.append('\n'.join(numeric_info))
        
        # Add suggestions
        suggestions = changes.get('suggestions', [])
        if suggestions:
            report_parts.append(f"\n**💡 Recommendations:**")
            for i, suggestion in enumerate(suggestions[:5], 1):
                report_parts.append(f"  {i}. {suggestion}")
        
        # Add metadata if requested
        if include_metadata and changes.get('timestamp'):
            report_parts.append(f"\n---\n*Generated: {changes['timestamp']}*")
        
        return '\n'.join(report_parts)
    
    def customize_style(self, **kwargs):
        """
        Customize the formatting style.
        
        Args:
            **kwargs: Style preferences to update
        """
        self.style_preferences.update(kwargs)
    
    def get_style_preferences(self) -> Dict[str, Any]:
        """Get current style preferences."""
        return self.style_preferences.copy()
    
    def reset_to_defaults(self):
        """Reset style preferences to defaults."""
        self.style_preferences = self._get_default_preferences()
    
    def preview_formatting(self, explanation: str, operation_type: str = 'general') -> str:
        """
        Preview how an explanation will look with current formatting.
        
        Args:
            explanation: Raw explanation text
            operation_type: Type of operation
            
        Returns:
            Preview of formatted explanation
        """
        return self.format_explanation(explanation, operation_type)


# Convenience functions for quick formatting
def format_explanation_quick(
    explanation: str, 
    operation_type: str = 'general'
) -> str:
    """
    Quick formatting function without creating a formatter instance.
    
    Args:
        explanation: Raw explanation text
        operation_type: Type of operation
        
    Returns:
        Formatted explanation string
    """
    formatter = ExplanationFormatter()
    return formatter.format_explanation(explanation, operation_type)


def create_summary_card_quick(
    changes: Dict[str, Any], 
    operation_type: str = 'general'
) -> str:
    """
    Quick summary card creation without creating a formatter instance.
    
    Args:
        changes: Change detection results
        operation_type: Type of operation
        
    Returns:
        Formatted summary card
    """
    formatter = ExplanationFormatter()
    return formatter.create_summary_card(changes, operation_type)
