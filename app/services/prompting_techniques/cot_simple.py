"""
Chain of Thought (Simple) prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_cot_simple(ollama_service, 
                          generator_model: str,
                          prompt: str, 
                          template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Chain of Thought (Simple) technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Chain of Thought (Simple) technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
    
    # Create meta-prompt for the LLM to transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in Chain of Thought prompting.
Your task is to modify the prompt to include a "Let's think step by step" instruction.

{template_info}

Original prompt:
"{prompt}"

Guidelines:
1. Add a clear instruction for the model to think through the problem step by step
2. Do not change the core meaning or task in the original prompt
3. Ensure the modified prompt encourages methodical reasoning
4. Keep it simple - use phrases like "Let's think step by step" or "Think through this step by step"

Format your response as a single string, with no additional text, explanations, or JSON formatting.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error transforming prompt with CoT Simple: {str(e)}")
        # Fallback to a simpler approach if generation fails
        return f"Let's think step by step: {prompt}"
