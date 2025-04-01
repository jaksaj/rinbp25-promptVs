from fastapi import APIRouter, HTTPException, Depends
from app.api.models.model import ModelRequest, ModelResponse, ModelsListResponse, RunningModelsResponse, ModelStatusResponse, BatchModelRequest
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

@router.post("/models/pull/batch", response_model=List[ModelResponse])
async def pull_models(
    request: BatchModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Pull multiple models."""
    try:
        results = []
        for model_name in request.models:
            try:
                success = ollama_service.pull_model(model_name)
                if success:
                    results.append(ModelResponse(
                        message=f"Successfully pulled model: {model_name}",
                        model_name=model_name
                    ))
                else:
                    results.append(ModelResponse(
                        message=f"Failed to pull model: {model_name}",
                        model_name=model_name
                    ))
            except Exception as e:
                logger.error("Error pulling model %s: %s", model_name, str(e))
                results.append(ModelResponse(
                    message=f"Error pulling model: {str(e)}",
                    model_name=model_name
                ))
        return results
    except Exception as e:
        logger.error("Error in batch pull operation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/start", response_model=ModelResponse)
async def start_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Start a specific model."""
    try:
        # First check if model exists
        available_models = ollama_service.list_available_models()
        if request.model_name not in available_models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} is not available")
            
        # Check if model is already running
        running_models = [model.name for model in ollama_service.list_running_models()]
        if request.model_name in running_models:
            raise HTTPException(status_code=400, detail=f"Model {request.model_name} is already running")
            
        # Try to start the model
        success = ollama_service.add_running_model(request.model_name)
        if success:
            return ModelResponse(
                message=f"Successfully started model: {request.model_name}",
                model_name=request.model_name
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start model: {request.model_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error starting model: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error starting model: {request.model_name}")

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

@router.post("/models/start/batch", response_model=List[ModelResponse])
async def start_models(
    request: BatchModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Start multiple models."""
    try:
        results = []
        available_models = ollama_service.list_available_models()
        running_models = [model.name for model in ollama_service.list_running_models()]
        
        for model_name in request.models:
            try:
                # Check if model exists
                if model_name not in available_models:
                    results.append(ModelResponse(
                        message=f"Model {model_name} is not available",
                        model_name=model_name
                    ))
                    continue
                    
                # Check if model is already running
                if model_name in running_models:
                    results.append(ModelResponse(
                        message=f"Model {model_name} is already running",
                        model_name=model_name
                    ))
                    continue
                    
                # Try to start the model
                success = ollama_service.add_running_model(model_name)
                if success:
                    results.append(ModelResponse(
                        message=f"Successfully started model: {model_name}",
                        model_name=model_name
                    ))
                else:
                    results.append(ModelResponse(
                        message=f"Failed to start model: {model_name}",
                        model_name=model_name
                    ))
            except Exception as e:
                logger.error("Error starting model %s: %s", model_name, str(e))
                results.append(ModelResponse(
                    message=f"Error starting model: {str(e)}",
                    model_name=model_name
                ))
        return results
    except Exception as e:
        logger.error("Error in batch start operation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/stop/batch", response_model=List[ModelResponse])
async def stop_models(
    request: BatchModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Stop multiple models."""
    try:
        results = []
        for model_name in request.models:
            try:
                success = ollama_service.remove_running_model(model_name)
                if success:
                    results.append(ModelResponse(
                        message=f"Successfully stopped model: {model_name}",
                        model_name=model_name
                    ))
                else:
                    results.append(ModelResponse(
                        message=f"Failed to stop model: {model_name}",
                        model_name=model_name
                    ))
            except ValueError as e:
                logger.error("Validation error stopping model %s: %s", model_name, str(e))
                results.append(ModelResponse(
                    message=f"Validation error: {str(e)}",
                    model_name=model_name
                ))
            except Exception as e:
                logger.error("Error stopping model %s: %s", model_name, str(e))
                results.append(ModelResponse(
                    message=f"Error stopping model: {str(e)}",
                    model_name=model_name
                ))
        return results
    except Exception as e:
        logger.error("Error in batch stop operation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 