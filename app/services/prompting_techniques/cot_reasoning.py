"""
Chain of Thought (Reasoning) prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_cot_reasoning(ollama_service,
                             generator_model: str,
                             prompt: str,
                             template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Chain of Thought (Reasoning) technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Chain of Thought (Reasoning) technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
    
    # Create meta-prompt for the LLM to transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in Chain of Thought reasoning techniques.
Your task is to transform this prompt to explicitly elicit detailed, structured reasoning.

{template_info}

Original prompt:
"{prompt}"

Transform this prompt to:
1. Break down the reasoning process into explicit steps
2. Encourage exploration of multiple perspectives or aspects
3. Ask for thorough analysis before reaching conclusions
4. Maintain the original task/question but structure the approach
5. Include specific instructions for step-by-step reasoning

Format your response as a single complete prompt with no additional text, explanations, or JSON formatting.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error transforming prompt with CoT Reasoning: {str(e)}")
        # Fallback to a simpler approach if generation fails
        return f"Think through this problem step-by-step with detailed reasoning: {prompt}"
