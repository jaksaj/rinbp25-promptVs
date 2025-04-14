"""
Prompt Generator Service - Generates variations of prompts using LLMs.
"""
from typing import List, Dict, Any, Optional, Tuple
from .ollama import OllamaService
import logging
import json
import uuid

logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, ollama_service: OllamaService, generator_model: str = "llama3.1:8b"):
        """
        Initialize the PromptGenerator with an Ollama service and a default model.
        
        Args:
            ollama_service: The OllamaService instance to use for generating prompts
            generator_model: The model to use for generating prompt variations
        """
        self.ollama_service = ollama_service
        self.generator_model = generator_model
        self.variation_instructions = {
            "rephrase": "Rephrase this prompt while keeping the exact same meaning and preserving any template parameters",
            "add_context": "Add some relevant context to this prompt while preserving any template parameters",
            "formalize": "Rewrite this prompt in a more formal tone while preserving any template parameters",
            "simplify": "Simplify this prompt for easier understanding while preserving any template parameters",
            "casual": "Rewrite this prompt in a more casual, conversational tone while preserving any template parameters",
            "multiple_choice": "Convert this open-ended question into a multiple-choice format while preserving any template parameters",
            "add_irrelevant": "Add some irrelevant information to this prompt to test model robustness while preserving any template parameters",
            "step_by_step": "Rewrite this prompt to explicitly request a step-by-step response while preserving any template parameters"
        }
        
    def add_variation_type(self, variation_type: str, instruction: str) -> Dict[str, str]:
        """
        Add a new variation type with its instruction.
        
        Args:
            variation_type: The name of the variation type
            instruction: The instruction for generating this variation
            
        Returns:
            Dictionary of all available variation types
        """
        if variation_type in self.variation_instructions:
            logger.warning(f"Variation type '{variation_type}' already exists and will be overwritten")
            
        self.variation_instructions[variation_type] = instruction
        return self.variation_instructions
        
    def get_variation_types(self) -> Dict[str, str]:
        """
        Get all available variation types and their instructions.
        
        Returns:
            Dictionary of all variation types and their instructions
        """
        return self.variation_instructions
        
    async def generate_single_variation(self, 
                                        base_prompt: str, 
                                        variation_type: str,
                                        template_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a single variation of a base prompt using specified transformation type.
        
        Args:
            base_prompt: The original prompt to create a variation of
            variation_type: Type of variation to apply (rephrase, add_context, formalize, etc.)
            template_params: Optional parameters to be preserved in the template
            
        Returns:
            A prompt variation
        """
        instruction = self.variation_instructions.get(
            variation_type, self.variation_instructions.get("rephrase", "Rephrase this prompt"))
        
        # Identify and preserve template parameters
        template_info = ""
        if template_params:
            template_info = f"The prompt contains these template parameters that MUST be preserved exactly as they are: {json.dumps(template_params)}"
        
        # Construct meta-prompt for generating a variation
        meta_prompt = f"""
You are an expert prompt engineer. Your task is to generate a variation of the following prompt.

{instruction}.

{template_info}

Original prompt:
"{base_prompt}"

Generate exactly 1 variation. The variation should be a complete, standalone prompt.
Format your response as a single string, with no additional text, explanations, or JSON formatting.
"""
        
        try:
            # Generate the variation using the specified model
            response = self.ollama_service.process_prompt(meta_prompt, self.generator_model)
            
            # Clean up the response - remove quotes and unnecessary whitespace
            cleaned_response = response.strip().strip('"').strip()
            
            return cleaned_response
                
        except Exception as e:
            logger.error(f"Error generating prompt variation: {str(e)}")
            return f"Error generating {variation_type} variation: {str(e)}"
            
    async def generate_variations(self, 
                                 base_prompt: str,
                                 variation_types: List[str],
                                 template_params: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
        """
        Generate variations of a base prompt using specified transformation types.
        
        Args:
            base_prompt: The original prompt to create variations of
            variation_types: List of variation types to apply
            template_params: Optional parameters to be preserved in the template
            
        Returns:
            A list of tuples containing (variation_type, variation)
        """
        results = []
        
        # Generate a variation for each variation type
        for variation_type in variation_types:
            variation = await self.generate_single_variation(
                base_prompt=base_prompt,
                variation_type=variation_type,
                template_params=template_params
            )
            
            results.append((variation_type, variation))
            
        return results