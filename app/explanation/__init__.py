"""
Explanation Agent Module

Provides intelligent explanations for AI spreadsheet operations using LangGraph and LangChain.
"""

from .change_detector import ChangeDetector
# from .templates import ExplanationTemplates  # UNUSED - CLEAN PIPELINE ONLY
# from .explanation_workflow import ExplanationWorkflow  # FALLBACK REMOVED - CLEAN PIPELINE ONLY
# from .formatter import ExplanationFormatter  # UNUSED - CLEAN PIPELINE ONLY
from .intelligent_workflow import IntelligentExplanationWorkflow
from .local_llm import get_local_llm, check_local_llm_availability

__all__ = [
    'ChangeDetector',
    # 'ExplanationTemplates',  # UNUSED - CLEAN PIPELINE ONLY
    # 'ExplanationWorkflow',  # FALLBACK REMOVED - CLEAN PIPELINE ONLY
    # 'ExplanationFormatter',  # UNUSED - CLEAN PIPELINE ONLY
    'IntelligentExplanationWorkflow',
    'get_local_llm',
    'check_local_llm_availability'
]
