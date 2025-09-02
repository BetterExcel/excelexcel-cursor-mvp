"""
Explanation Agent Module

Provides intelligent explanations for AI spreadsheet operations using LangGraph and LangChain.
"""

from .change_detector import ChangeDetector
from .templates import ExplanationTemplates
from .explanation_workflow import ExplanationWorkflow
from .formatter import ExplanationFormatter
from .intelligent_workflow import IntelligentExplanationWorkflow
from .local_llm import get_local_llm, check_local_llm_availability

__all__ = [
    'ChangeDetector',
    'ExplanationTemplates', 
    'ExplanationWorkflow',
    'ExplanationFormatter',
    'IntelligentExplanationWorkflow',
    'get_local_llm',
    'check_local_llm_availability'
]
