"""
Control prompting technique implementation - serves as a baseline for comparison.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_control(ollama_service,
                       generator_model: str,
                       prompt: str,
                       template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Control technique to a prompt - simply returns the original prompt unchanged.
    
    Args:
        ollama_service: The OllamaService instance (unused in this technique)
        generator_model: The model to use (unused in this technique)
        prompt: The original prompt
        template_params: Optional parameters (unused in this technique)
        
    Returns:
        The original prompt unchanged
    """
    return prompt 