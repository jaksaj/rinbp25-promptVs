"""
Self-Consistency prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_self_consistency(ollama_service,
                               generator_model: str,
                               prompt: str,
                               num_paths: int = 3,
                               template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Self-Consistency technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        num_paths: Number of reasoning paths to request (default: 3)
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Self-Consistency technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
    
    # Create meta-prompt for the LLM to transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in self-consistency prompting techniques.
Your task is to transform a prompt to ask for multiple reasoning paths and find the most consistent answer.

{template_info}

Original prompt:
"{prompt}"

Create a self-consistency prompt that:
1. Instructs the model to generate {num_paths} different reasoning approaches to the problem
2. Asks the model to start with different assumptions or perspectives for each approach
3. Directs the model to identify consistent elements across all reasoning paths
4. Requests a final synthesis based on the consistency analysis
5. Maintains the original task or question

Format your response as a single complete prompt with no additional text, explanations, or JSON formatting.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error transforming prompt with Self-Consistency: {str(e)}")
        # Fallback to a simpler approach if generation fails
        fallback_prompt = f"""Explore multiple reasoning paths for this question and find the most consistent answer:

{prompt}

Generate {num_paths} different approaches to solve this problem, then determine which answer is most consistent across approaches."""
        return fallback_prompt
