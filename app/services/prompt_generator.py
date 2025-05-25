"""
Prompt Generator Service - Implements advanced prompting techniques for LLMs.
"""
from typing import List, Dict, Any, Optional, Tuple
from .ollama import OllamaService
import logging
import json

# Import techniques from the techniques package
from .prompting_techniques import (
    apply_cot_simple,
    apply_cot_reasoning,
    apply_few_shot,
    apply_self_consistency,
    apply_role_prompting,
    apply_reflexion_prompting
)

logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, ollama_service: OllamaService, generator_model: str = "llama3.1:8b"):
        """
        Initialize the PromptGenerator with an Ollama service and a default model.
        
        Args:
            ollama_service: The OllamaService instance to use for generating prompts
            generator_model: The default model to use for generating prompts with techniques
        """
        self.ollama_service = ollama_service
        self.generator_model = generator_model
        # Define the available prompting techniques
        self.prompting_techniques = {
            "cot_simple": {
                "name": "Chain of Thought (Simple)",
                "description": "Adds 'Let's think step by step' to encourage methodical reasoning",
                "complexity": "low"
            },
            "cot_reasoning": {
                "name": "Chain of Thought (Reasoning)",
                "description": "Structures the prompt to ask for detailed reasoning with multiple steps",
                "complexity": "medium"
            },
            "few_shot": {
                "name": "Few-Shot Prompting",
                "description": "Provides example Q&A pairs to demonstrate desired reasoning and output format",
                "complexity": "high"
            },
            "self_consistency": {
                "name": "Self-Consistency",
                "description": "Asks for multiple reasoning paths to find the most consistent answer",
                "complexity": "high"
            },            
            "role_prompting": {
                "name": "Role Prompting",
                "description": "Ask the model to respond as if it were a specific expert or persona",
                "complexity": "low"
            },
            "reflexion_prompting": {
                "name": "Reflexion Prompting",
                "description": "Ask the model to solve a problem and then critically review its own solution",
                "complexity": "medium"
            }
        }
    
    def get_available_techniques(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available prompting techniques with their descriptions.
        
        Returns:
            Dictionary of all available prompting techniques and their metadata
        """
        return self.prompting_techniques
    
    def get_technique_info(self, technique_id: str) -> Optional[Dict[str, str]]:
        """
        Get information about a specific prompting technique.
        
        Args:
            technique_id: The ID of the prompting technique
            
        Returns:
            Information about the technique or None if not found
        """
        return self.prompting_techniques.get(technique_id)
    async def apply_technique(self,
                             prompt: str,
                             technique: str,
                             domain: Optional[str] = None,
                             num_paths: int = 3,
                             num_examples: int = 2,
                             template_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply a specific prompting technique to a prompt.
        
        Args:
            prompt: The original prompt to apply the technique to
            technique: The technique to apply
            domain: Optional domain context for specialized techniques
            num_paths: Number of reasoning paths for self-consistency (default: 3)
            num_examples: Number of examples for few-shot (default: 2)
            template_params: Optional parameters to be preserved in the template
            
        Returns:
            The prompt with the technique applied, or the original prompt if technique is not found
        """
        # Check if the technique exists
        if technique not in self.prompting_techniques:
            logger.warning(f"Unknown prompting technique: {technique}")
            return prompt
        
        # Apply the appropriate technique
        if technique == "cot_simple":
            return await apply_cot_simple(
                self.ollama_service,
                self.generator_model,
                prompt, 
                template_params
            )
        elif technique == "cot_reasoning":
            return await apply_cot_reasoning(
                self.ollama_service,
                self.generator_model,
                prompt, 
                template_params
            )
        elif technique == "few_shot":
            return await apply_few_shot(
                self.ollama_service,
                self.generator_model,
                prompt,
                domain,
                num_examples,
                template_params
            )
        elif technique == "self_consistency":
            return await apply_self_consistency(
                self.ollama_service,
                self.generator_model,
                prompt,
                num_paths,
                template_params
            )
        elif technique == "role_prompting":
            return await apply_role_prompting(
                self.ollama_service,
                self.generator_model,
                prompt,
                template_params
            )
        elif technique == "reflexion_prompting":
            return await apply_reflexion_prompting(
                self.ollama_service,
                self.generator_model,
                prompt,
                template_params
            )
        else:
            # Should not reach here given the earlier check, but as a fallback
            return prompt
    
    async def apply_techniques_batch(self,
                                    prompts: List[str],
                                    techniques: List[str],
                                    domain: Optional[str] = None,
                                    num_paths: int = 3,
                                    num_examples: int = 2,
                                    template_params: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str, str]]:
        """
        Apply prompting techniques to a batch of prompts.
        
        Args:
            prompts: List of original prompts to apply techniques to
            techniques: List of techniques to apply
            domain: Optional domain context for specialized techniques
            num_paths: Number of reasoning paths for self-consistency (default: 3)
            num_examples: Number of examples for few-shot (default: 2)
            template_params: Optional parameters to be preserved in the template
            
        Returns:
            List of tuples containing (original_prompt, technique, modified_prompt)
        """
        results = []
        
        for prompt in prompts:
            for technique in techniques:
                modified_prompt = await self.apply_technique(
                    prompt=prompt,
                    technique=technique,
                    domain=domain,
                    num_paths=num_paths,
                    num_examples=num_examples,
                    template_params=template_params
                )
                
                results.append((prompt, technique, modified_prompt))
        
        return results
    
    async def get_technique_examples(self) -> Dict[str, Dict[str, str]]:
        """
        Get examples of how each prompting technique transforms a prompt.
        
        Returns:
            Dictionary with examples for each technique
        """
        # Base example prompt to transform
        base_prompt = "What are the main factors that led to the Industrial Revolution?"
        
        examples = {}
        
        for technique_id, info in self.prompting_techniques.items():
            try:
                transformed_prompt = await self.apply_technique(base_prompt, technique_id)
                examples[technique_id] = {
                    "name": info["name"],
                    "description": info["description"],
                    "original_prompt": base_prompt,
                    "transformed_prompt": transformed_prompt
                }
            except Exception as e:
                logger.error(f"Error generating example for {technique_id}: {str(e)}")
                examples[technique_id] = {
                    "name": info["name"],
                    "description": info["description"],
                    "original_prompt": base_prompt,
                    "transformed_prompt": f"Error generating example: {str(e)}"
                }
        
        return examples
