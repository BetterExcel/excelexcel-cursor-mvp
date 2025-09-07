"""
Local LLM Integration for Intelligent Explanations

This module provides local LLM capabilities using various local models
without requiring external API calls.
"""

import os
import sys
from typing import Optional, Any, Dict
import logging

# Try to import various local LLM options
try:
    # Option 1: Ollama (most popular local option)
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    # Option 2: LocalAI
    from langchain_community.llms import LocalAI
    LOCALAI_AVAILABLE = True
except ImportError:
    LOCALAI_AVAILABLE = False

try:
    # Option 3: HuggingFace Transformers
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    # Option 4: GPT4All
    from langchain_community.llms import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMProvider:
    """
    Provider for local LLM capabilities.
    
    This class tries to initialize various local LLM options
    and provides a unified interface for the explanation system.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the local LLM provider.
        
        Args:
            model_name: Specific model to use (if None, auto-detects)
        """
        self.llm = None
        self.provider_type = None
        self.model_name = model_name or self._auto_detect_model()
        
        self._initialize_llm()
    
    def _auto_detect_model(self) -> str:
        """Auto-detect the best available local model."""
        if OLLAMA_AVAILABLE:
            return "ollama"
        elif LOCALAI_AVAILABLE:
            return "localai"
        elif HUGGINGFACE_AVAILABLE:
            return "huggingface"
        elif GPT4ALL_AVAILABLE:
            return "gpt4all"
        else:
            return "none"
    
    def _initialize_ollama(self) -> bool:
        """Initialize Ollama LLM."""
        try:
            # Try to connect to Ollama
            self.llm = OllamaLLM(
                model="llama3.2:3b",  # Lightweight model
                base_url="http://localhost:11434"
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            if test_response:
                self.provider_type = "ollama"
                logger.info("✅ Ollama LLM initialized successfully")
                return True
                
        except Exception as e:
            logger.warning(f"❌ Ollama initialization failed: {e}")
            return False
    
    def _initialize_localai(self) -> bool:
        """Initialize LocalAI LLM."""
        try:
            # Try to connect to LocalAI
            self.llm = LocalAI(
                endpoint="http://localhost:8080",
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            if test_response:
                self.provider_type = "localai"
                logger.info("✅ LocalAI LLM initialized successfully")
                return True
                
        except Exception as e:
            logger.warning(f"❌ LocalAI initialization failed: {e}")
            return False
    
    def _initialize_huggingface(self) -> bool:
        """Initialize HuggingFace Transformers LLM."""
        try:
            # Use a lightweight model for local processing
            model_name = "microsoft/DialoGPT-small"  # ~117M parameters
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            if test_response:
                self.provider_type = "huggingface"
                logger.info("✅ HuggingFace LLM initialized successfully")
                return True
                
        except Exception as e:
            logger.warning(f"❌ HuggingFace initialization failed: {e}")
            return False
    
    def _initialize_gpt4all(self) -> bool:
        """Initialize GPT4All LLM."""
        try:
            # Try to use GPT4All
            model_path = os.path.expanduser("~/.local/share/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin")
            
            if os.path.exists(model_path):
                self.llm = GPT4All(
                    model=model_path,
                    backend="gpt4all",
                    verbose=True
                )
                
                # Test the connection
                test_response = self.llm.invoke("Hello")
                if test_response:
                    self.provider_type = "gpt4all"
                    logger.info("✅ GPT4All LLM initialized successfully")
                    return True
            else:
                logger.warning("❌ GPT4All model file not found")
                return False
                
        except Exception as e:
            logger.warning(f"❌ GPT4All initialization failed: {e}")
            return False
    
    def _initialize_llm(self):
        """Initialize the best available local LLM."""
        if self.model_name == "ollama" and OLLAMA_AVAILABLE:
            if self._initialize_ollama():
                return
        
        if self.model_name == "localai" and LOCALAI_AVAILABLE:
            if self._initialize_localai():
                return
        
        if self.model_name == "huggingface" and HUGGINGFACE_AVAILABLE:
            if self._initialize_huggingface():
                return
        
        if self.model_name == "gpt4all" and GPT4ALL_AVAILABLE:
            if self._initialize_gpt4all():
                return
        
        # Try auto-detection
        if OLLAMA_AVAILABLE and self._initialize_ollama():
            return
        elif LOCALAI_AVAILABLE and self._initialize_localai():
            return
        elif HUGGINGFACE_AVAILABLE and self._initialize_huggingface():
            return
        elif GPT4ALL_AVAILABLE and self._initialize_gpt4all():
            return
        
        # No LLM available
        logger.warning("❌ No local LLM available. Clean pipeline requires local LLM.")
        self.llm = None
        self.provider_type = "none"
    
    def get_llm(self):
        """Get the initialized LLM instance."""
        return self.llm
    
    def is_available(self) -> bool:
        """Check if a local LLM is available."""
        return self.llm is not None and self.provider_type != "none"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider."""
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "is_available": self.is_available(),
            "available_providers": {
                "ollama": OLLAMA_AVAILABLE,
                "localai": LOCALAI_AVAILABLE,
                "huggingface": HUGGINGFACE_AVAILABLE,
                "gpt4all": GPT4ALL_AVAILABLE
            }
        }


def get_local_llm(model_name: Optional[str] = None) -> Optional[Any]:
    """
    Get a local LLM instance.
    
    Args:
        model_name: Specific model to use (if None, auto-detects)
        
    Returns:
        LangChain LLM instance or None if none available
    """
    provider = LocalLLMProvider(model_name)
    return provider.get_llm()


def check_local_llm_availability() -> Dict[str, Any]:
    """
    Check what local LLM options are available.
    
    Returns:
        Dictionary with availability information
    """
    provider = LocalLLMProvider()
    return provider.get_provider_info()


# Convenience function for quick LLM access
def quick_local_llm() -> Optional[Any]:
    """Quick access to a local LLM."""
    return get_local_llm()
