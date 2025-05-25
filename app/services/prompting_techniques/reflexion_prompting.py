"""
Reflexion Prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_reflexion_prompting(ollama_service,
                                   generator_model: str,
                                   prompt: str,
                                   template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Reflexion Prompting technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Reflexion Prompting technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
    
    # Create meta-prompt for the LLM to transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in reflexion prompting techniques.
Your task is to transform a prompt to include a reflexion component where the model solves a problem and then critically reviews its own solution.

{template_info}

Original prompt:
"{prompt}"

Transform this prompt to:
1. First ask the model to solve the problem or answer the question
2. Then ask the model to critically review its own solution/answer
3. Include instructions to check for errors, logical inconsistencies, or inefficiencies
4. Ask the model to identify potential improvements or alternative approaches
5. Finally, ask the model to provide a revised or corrected solution if needed
6. Use clear section headers or separators to distinguish between initial solution and reflection

The reflexion process should include:
- Error checking: Look for factual errors, logical inconsistencies, or calculation mistakes
- Completeness review: Check if all aspects of the problem have been addressed
- Efficiency analysis: Consider if there are more efficient or elegant solutions
- Alternative approaches: Think about different methods or perspectives
- Improvement identification: Suggest specific enhancements

Structure the prompt with clear phases:
Phase 1: Initial Solution
Phase 2: Critical Reflection and Review
Phase 3: Revised Solution (if improvements are identified)

Format your response as a single complete prompt with no additional text, explanations, or JSON formatting.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error transforming prompt with Reflexion Prompting: {str(e)}")
        # Fallback to a simpler approach if generation fails
        return f"""
{prompt}

Please solve this step by step, then critically review your solution:

Phase 1 - Initial Solution:
[Provide your initial solution here]

Phase 2 - Critical Review:
- Check for any errors or inconsistencies in your solution
- Consider if there are more efficient approaches
- Identify any missing aspects or improvements

Phase 3 - Final Answer:
[Provide your final, refined solution incorporating any improvements from your review]
"""
