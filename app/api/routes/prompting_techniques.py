"""
API Routes for prompting techniques operations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
import uuid

from app.api.models.prompt import (
    PromptingTechniquesResponse,
    PromptingTechniqueInfo,
    ApplyTechniqueRequest,
    ApplyTechniqueResponse,
    BatchApplyTechniqueRequest,
    BatchApplyTechniqueResponse,
    TechniqueResult,
    TechniqueExampleResponse,
    PromptVersionCreate
)
from app.core.dependencies import get_prompt_generator, get_neo4j_service
from app.services.prompt_generator import PromptGenerator
from app.services.neo4j import Neo4jService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/techniques", response_model=PromptingTechniquesResponse)
async def get_prompting_techniques(
    prompt_generator: PromptGenerator = Depends(get_prompt_generator)
):
    """
    Get all available prompting techniques.
    
    Returns:
        Dictionary of all available prompting techniques with their details
    """
    try:
        techniques_dict = prompt_generator.get_available_techniques()
        
        # Convert to the correct response format
        response_dict = {}
        for key, value in techniques_dict.items():
            response_dict[key] = PromptingTechniqueInfo(
                name=value["name"],
                description=value["description"],
                complexity=value["complexity"]
            )
        
        return PromptingTechniquesResponse(techniques=response_dict)
    except Exception as e:
        logger.error(f"Error getting prompting techniques: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/techniques/examples", response_model=Dict[str, TechniqueExampleResponse])
async def get_technique_examples(
    prompt_generator: PromptGenerator = Depends(get_prompt_generator)
):
    """
    Get examples of how each prompting technique transforms a prompt.
    
    Returns:
        Dictionary with examples for each technique
    """
    try:
        examples_dict = await prompt_generator.get_technique_examples()
        
        # Convert to the correct response format
        response_dict = {}
        for key, value in examples_dict.items():
            response_dict[key] = TechniqueExampleResponse(
                name=value["name"],
                description=value["description"],
                original_prompt=value["original_prompt"],
                transformed_prompt=value["transformed_prompt"]
            )
        
        return response_dict
    except Exception as e:
        logger.error(f"Error getting technique examples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply", response_model=ApplyTechniqueResponse)
async def apply_technique(
    request: ApplyTechniqueRequest,
    prompt_generator: PromptGenerator = Depends(get_prompt_generator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Apply a prompting technique to a specific prompt.
    Optionally save the transformed prompt as a new prompt version.
    
    Args:
        request: Contains prompt_id, technique, save flag, etc.
    
    Returns:
        The original and transformed prompts, and optionally the version ID if saved
    """
    try:
        # Get the original prompt from Neo4j
        original_prompt = neo4j_service.get_prompt(request.prompt_id)
        if not original_prompt:
            raise HTTPException(status_code=404, detail=f"Prompt with ID {request.prompt_id} not found")
        
        # Override the generator model if specified
        if request.generator_model:
            original_model = prompt_generator.generator_model
            prompt_generator.generator_model = request.generator_model
          # Apply the technique
        transformed_prompt = await prompt_generator.apply_technique(
            prompt=original_prompt["content"],
            technique=request.technique,
            domain=request.domain,
            num_paths=request.num_paths,
            num_examples=request.num_examples,
            template_params=request.template_params
        )
        
        # Reset the generator model if it was overridden
        if request.generator_model:
            prompt_generator.generator_model = original_model
        
        # Save as a new version if requested
        version_id = None
        if request.save_as_version:
            version = PromptVersionCreate(
                prompt_id=request.prompt_id,
                content=transformed_prompt,
                version=f"{request.technique}_technique",
                notes=f"Auto-generated using {request.technique} prompting technique"
            )
            
            version_id = neo4j_service.create_prompt_version(version)
        
        return ApplyTechniqueResponse(
            prompt_id=request.prompt_id,
            technique=request.technique,
            original_prompt=original_prompt["content"],
            transformed_prompt=transformed_prompt,
            version_id=version_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying prompting technique: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-apply", response_model=BatchApplyTechniqueResponse)
async def batch_apply_techniques(
    request: BatchApplyTechniqueRequest,
    prompt_generator: PromptGenerator = Depends(get_prompt_generator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Apply prompting techniques to multiple prompts.
    Optionally save the transformed prompts as new prompt versions.
    
    Args:
        request: Contains prompt_ids, techniques, save flag, etc.
    
    Returns:
        The results of applying each technique to each prompt
    """
    try:
        # Override the generator model if specified
        if request.generator_model:
            original_model = prompt_generator.generator_model
            prompt_generator.generator_model = request.generator_model
        
        results = []
        total_transformations = 0
        
        for prompt_id in request.prompt_ids:
            try:
                # Get the original prompt from Neo4j
                original_prompt = neo4j_service.get_prompt(prompt_id)
                if not original_prompt:
                    logger.warning(f"Prompt with ID {prompt_id} not found, skipping")
                    continue
                
                for technique in request.techniques:
                    try:                        # Apply the technique
                        transformed_prompt = await prompt_generator.apply_technique(
                            prompt=original_prompt["content"],
                            technique=technique,
                            domain=request.domain,
                            num_paths=request.num_paths,
                            num_examples=request.num_examples,
                            template_params=request.template_params
                        )
                        
                        # Save as a new version if requested
                        version_id = None
                        if request.save_as_versions:
                            version = PromptVersionCreate(
                                prompt_id=prompt_id,
                                content=transformed_prompt,
                                version=f"{technique}_technique",
                                notes=f"Auto-generated using {technique} prompting technique"
                            )
                            
                            version_id = neo4j_service.create_prompt_version(version)
                        
                        results.append(TechniqueResult(
                            prompt_id=prompt_id,
                            technique=technique,
                            original_prompt=original_prompt["content"],
                            transformed_prompt=transformed_prompt,
                            version_id=version_id
                        ))
                        
                        total_transformations += 1
                    except Exception as e:
                        logger.error(f"Error applying technique {technique} to prompt {prompt_id}: {str(e)}")
                        # Continue with other techniques even if one fails
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
                # Continue with other prompts even if one fails
        
        # Reset the generator model if it was overridden
        if request.generator_model:
            prompt_generator.generator_model = original_model
            
        return BatchApplyTechniqueResponse(
            results=results,
            total_transformations=total_transformations
        )
    except Exception as e:
        logger.error(f"Error in batch application of prompting techniques: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
