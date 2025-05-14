"""
API Routes for prompt generation operations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
import uuid

from app.api.models.prompt import (
    VariationTypeRequest,
    VariationTypeResponse,
    GenerateVariationsRequest,
    GenerateVariationsResponse,
    BatchGenerateVariationsRequest,
    BatchGenerateVariationsResponse,
    BatchVariationResult,
    PromptVariation,
    PromptVersionCreate
)
from app.core.dependencies import get_prompt_generator, get_neo4j_service
from app.services.prompt_generator import PromptGenerator
from app.services.neo4j import Neo4jService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/variation-types", response_model=VariationTypeResponse)
async def add_variation_type(
    request: VariationTypeRequest,
    prompt_generator: PromptGenerator = Depends(get_prompt_generator)
):
    """
    Add a new variation type with its instruction to the system.
    
    Args:
        request: Contains the variation_type name and instruction for generating variations
    
    Returns:
        Dictionary of all available variation types
    """
    try:
        variation_types = prompt_generator.add_variation_type(
            variation_type=request.variation_type,
            instruction=request.instruction
        )
        
        return VariationTypeResponse(variation_types=variation_types)
    except Exception as e:
        logger.error(f"Error adding variation type: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/variation-types", response_model=VariationTypeResponse)
async def get_variation_types(
    prompt_generator: PromptGenerator = Depends(get_prompt_generator)
):
    """
    Get all available variation types and their instructions.
    
    Returns:
        Dictionary of all variation types and their instructions
    """
    try:
        variation_types = prompt_generator.get_variation_types()
        return VariationTypeResponse(variation_types=variation_types)
    except Exception as e:
        logger.error(f"Error getting variation types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=GenerateVariationsResponse)
async def generate_prompt_variations(
    request: GenerateVariationsRequest,
    prompt_generator: PromptGenerator = Depends(get_prompt_generator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Generate variations of a prompt using specified transformation types.
    Optionally save the variations as new prompt versions.
    
    Args:
        request: Contains prompt_uuid, variation_types, save flag
    
    Returns:
        Generated variations and optionally their UUIDs if saved
    """
    try:
        # Get the original prompt from Neo4j
        original_prompt = neo4j_service.get_prompt(request.prompt_uuid)
        if not original_prompt:
            raise HTTPException(status_code=404, detail=f"Prompt with ID {request.prompt_uuid} not found")
        
        # Override the generator model if specified
        if request.generator_model:
            original_model = prompt_generator.generator_model
            prompt_generator.generator_model = request.generator_model
        
        # Generate variations
        variations_tuples = await prompt_generator.generate_variations(
            base_prompt=original_prompt["content"],
            variation_types=request.variation_types,
            template_params=request.template_params,
            domain=request.domain
        )
        
        # Reset the generator model if it was overridden
        if request.generator_model:
            prompt_generator.generator_model = original_model
        
        # Process and optionally save the variations
        prompt_variations = []
        for variation_type, content in variations_tuples:
            variation = PromptVariation(
                variation_type=variation_type,
                content=content
            )
            
            # Only save the variation if requested AND there's no error in the content
            if request.save and not content.startswith("Error generating"):
                version = PromptVersionCreate(
                    prompt_id=request.prompt_uuid,
                    content=content,
                    version=f"{variation_type}_variation",
                    notes=f"Auto-generated {variation_type} variation"
                )
                
                version_id = neo4j_service.create_prompt_version(version)
                variation.uuid = version_id
                
            prompt_variations.append(variation)
        
        return GenerateVariationsResponse(
            prompt_uuid=request.prompt_uuid,
            variations=prompt_variations,
            original_prompt=original_prompt["content"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prompt variations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-generate", response_model=BatchGenerateVariationsResponse)
async def batch_generate_prompt_variations(
    request: BatchGenerateVariationsRequest,
    prompt_generator: PromptGenerator = Depends(get_prompt_generator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Generate variations for multiple prompts using specified transformation types.
    Optionally save the variations as new prompt versions.
    
    Args:
        request: Contains prompt_uuids, variation_types, save flag
    
    Returns:
        Generated variations for each prompt and optionally their UUIDs if saved
    """
    try:
        # Override the generator model if specified
        if request.generator_model:
            original_model = prompt_generator.generator_model
            prompt_generator.generator_model = request.generator_model
        
        results = []
        total_variations = 0
        
        for prompt_uuid in request.prompt_uuids:
            try:
                # Get the original prompt from Neo4j
                original_prompt = neo4j_service.get_prompt(prompt_uuid)
                if not original_prompt:
                    logger.warning(f"Prompt with ID {prompt_uuid} not found, skipping")
                    continue
                
                # Generate variations
                variations_tuples = await prompt_generator.generate_variations(
                    base_prompt=original_prompt["content"],
                    variation_types=request.variation_types,
                    template_params=request.template_params,
                    domain=request.domain
                )
                
                # Process and optionally save the variations
                prompt_variations = []
                for variation_type, content in variations_tuples:
                    variation = PromptVariation(
                        variation_type=variation_type,
                        content=content
                    )
                    
                    # Save the variation if requested AND there's no error in the content
                    if request.save and not content.startswith("Error generating"):
                        version = PromptVersionCreate(
                            prompt_id=prompt_uuid,
                            content=content,
                            version=f"{variation_type}_variation",
                            notes=f"Auto-generated {variation_type} variation"
                        )
                        
                        version_id = neo4j_service.create_prompt_version(version)
                        variation.uuid = version_id
                        
                    prompt_variations.append(variation)
                    total_variations += 1
                
                results.append(BatchVariationResult(
                    prompt_uuid=prompt_uuid,
                    variations=prompt_variations,
                    original_prompt=original_prompt["content"]
                ))
                
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_uuid}: {str(e)}")
                # Continue with other prompts even if one fails
        
        # Reset the generator model if it was overridden
        if request.generator_model:
            prompt_generator.generator_model = original_model
            
        return BatchGenerateVariationsResponse(
            results=results,
            total_variations=total_variations
        )
    except Exception as e:
        logger.error(f"Error in batch generation of prompt variations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))