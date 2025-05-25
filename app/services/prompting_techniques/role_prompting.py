"""
Role Prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_role_prompting(ollama_service,
                              generator_model: str,
                              prompt: str,
                              template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Role Prompting technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Role Prompting technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
      # Create meta-prompt for the LLM to transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in role prompting techniques.
Your task is to transform a prompt by adding the most appropriate expert role or persona for the model to adopt.

{template_info}

Original prompt:
"{prompt}"

Transform this prompt to:
1. Choose the most appropriate expert role/persona based on the task and content
2. Add a clear instruction for the model to respond as if it were that specific expert or persona
3. Use phrases like "You are a [specific role]" or "Act as a [specific expert]"
4. Ensure the role is relevant and will enhance the quality of the response
5. Maintain the original task/question but frame it from the expert's perspective
6. Consider what specific knowledge, experience, or perspective that role would bring

Guidelines for role selection:
- Choose roles that have relevant expertise for the task
- Be specific (e.g., "pediatric nurse" rather than just "nurse")
- Consider professional roles, subject matter experts, or personas with relevant experience
- Examples: professional chef, data scientist, historian, software architect, financial advisor, etc.

Format your response as a single complete prompt with no additional text, explanations, or JSON formatting.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error transforming prompt with Role Prompting: {str(e)}")
        # Fallback to a simpler approach if generation fails
        return f"You are an expert. {prompt}"
