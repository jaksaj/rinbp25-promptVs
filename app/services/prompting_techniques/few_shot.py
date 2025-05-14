"""
Few-Shot prompting technique implementation.
"""
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def apply_few_shot(ollama_service,
                        generator_model: str,
                        prompt: str,
                        domain: Optional[str] = None,
                        num_examples: int = 2,
                        template_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply the Few-Shot Prompting technique to a prompt.
    
    Args:
        ollama_service: The OllamaService instance to use for generating prompts
        generator_model: The model to use for generating prompts
        prompt: The original prompt to apply the technique to
        domain: Optional domain/topic to focus the examples on
        num_examples: Number of examples to generate (default: 2)
        template_params: Optional parameters to be preserved in the template
        
    Returns:
        The prompt with the Few-Shot Prompting technique applied
    """
    # Identify and preserve template parameters
    template_info = ""
    if template_params:
        template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
    
    # Add domain context if provided
    domain_info = f"in the domain of {domain}" if domain else ""
    
    # Create meta-prompt for the LLM to generate examples and transform the original prompt
    meta_prompt = f"""
You are an expert prompt engineer specializing in few-shot prompting techniques.
Your task is to transform a prompt by adding {num_examples} high-quality examples to guide the model's reasoning.

{template_info}

Original prompt:
"{prompt}"

Create a few-shot prompt {domain_info} by:
1. Creating {num_examples} diverse examples that demonstrate the expected reasoning for similar questions
2. Each example should include a question and a detailed, step-by-step answer
3. Format each example with "Question:" and "Answer:" labels for clarity
4. After the examples, include the original question to be answered with the same approach

Format your response as a single complete prompt ready to be sent to an LLM.
Do not include additional explanations, comments, or metadata outside the prompt itself.
"""
    
    try:
        # Generate the transformed prompt using the LLM
        transformed_prompt = ollama_service.process_prompt(meta_prompt, generator_model)
        
        # Clean up any extra quotes or whitespace
        return transformed_prompt.strip().strip('"').strip()
    except Exception as e:
        logger.error(f"Error generating few-shot prompt: {str(e)}")
        # Fallback to a simpler approach if generation fails
        return f"Please answer the following question with clear step-by-step reasoning: {prompt}"
