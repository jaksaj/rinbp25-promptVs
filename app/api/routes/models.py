from fastapi import APIRouter, HTTPException, Depends
from app.api.models.model import ModelRequest, ModelResponse, ModelsListResponse, RunningModelsResponse, ModelStatusResponse
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
from app.core.config import DEFAULT_MODELS
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
async def list_available_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """List all available models."""
    try:
        models = ollama_service.list_available_models()
        return ModelsListResponse(models=models)
    except Exception as e:
        logger.error("Error listing models: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status", response_model=List[ModelStatusResponse])
async def get_models_status(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Get status of all default models including pull and running status."""
    try:
        statuses = []
        available_models = ollama_service.list_available_models()
        running_models = [model.name for model in ollama_service.list_running_models()]
        
        for model_name in DEFAULT_MODELS:
            status = ModelStatusResponse(
                name=model_name,
                is_pulled=model_name in available_models,
                is_running=model_name in running_models,
                status_message=""
            )
            
            # Set status message based on model state
            if model_name in ollama_service.pulling_models:
                status.status_message = "Pulling in progress..."
            elif model_name in ollama_service.starting_models:
                status.status_message = "Starting in progress..."
            elif model_name in ollama_service.stopping_models:
                status.status_message = "Stopping in progress..."
            elif not status.is_pulled:
                status.status_message = "Not pulled"
            elif status.is_running:
                status.status_message = "Running"
            else:
                status.status_message = "Pulled but not running"
                
            statuses.append(status)
            
        return statuses
    except Exception as e:
        logger.error("Error getting models status: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/running", response_model=RunningModelsResponse)
async def list_running_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """List all currently running models."""
    try:
        running_models = ollama_service.list_running_models()
        return RunningModelsResponse(running_models=running_models)
    except Exception as e:
        logger.error("Error listing running models: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/pull", response_model=ModelResponse)
async def pull_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Pull a specific model."""
    try:
        success = ollama_service.pull_model(request.model_name)
        if success:
            return ModelResponse(message=f"Successfully pulled model: {request.model_name}")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model: {request.model_name}")
    except Exception as e:
        logger.error("Error pulling model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/start", response_model=ModelResponse)
async def start_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Start a specific model."""
    try:
        success = ollama_service.add_running_model(request.model_name)
        if success:
            return ModelResponse(
                message=f"Successfully started model: {request.model_name}",
                model_name=request.model_name
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start model: {request.model_name}")
    except ValueError as e:
        logger.error("Validation error starting model: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error starting model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/stop", response_model=ModelResponse)
async def stop_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Stop a specific model."""
    try:
        success = ollama_service.remove_running_model(request.model_name)
        if success:
            return ModelResponse(message=f"Successfully stopped model: {request.model_name}")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to stop model: {request.model_name}")
    except ValueError as e:
        logger.error("Validation error stopping model: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error stopping model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 