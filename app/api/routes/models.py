from fastapi import APIRouter, HTTPException, Depends
from app.api.models.model import ModelRequest, ModelResponse, ModelsListResponse, RunningModelsResponse
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
from app.core.config import DEFAULT_MODELS
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
async def list_available_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """List all available models and pull default models if needed."""
    try:
        # Get currently available models
        available_models = ollama_service.list_available_models()
        
        # Pull default models that aren't available
        for model_name in DEFAULT_MODELS:
            if model_name not in available_models:
                try:
                    logger.info("Pulling default model: %s", model_name)
                    ollama_service.pull_model(model_name)
                    available_models.append(model_name)
                except Exception as e:
                    logger.warning("Failed to pull model %s: %s", model_name, str(e))
        
        return ModelsListResponse(models=available_models)
    except Exception as e:
        logger.error("Error listing models: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/running", response_model=RunningModelsResponse)
async def list_running_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """List all running models and their status."""
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
        ollama_service.pull_model(request.model_name)
        return ModelResponse(
            message="Model pulled successfully",
            model_name=request.model_name
        )
    except Exception as e:
        logger.error("Error pulling model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/start", response_model=ModelResponse)
async def start_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Start a model for processing prompts."""
    try:
        # Verify the model exists
        available_models = ollama_service.list_available_models()
        if request.model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found. Available models: {', '.join(available_models)}"
            )
        
        ollama_service.add_running_model(request.model_name)
        return ModelResponse(
            message="Model started successfully",
            model_name=request.model_name
        )
    except Exception as e:
        logger.error("Error starting model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/stop", response_model=ModelResponse)
async def stop_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Stop a running model."""
    try:
        ollama_service.remove_running_model(request.model_name)
        return ModelResponse(
            message="Model stopped successfully",
            model_name=request.model_name
        )
    except Exception as e:
        logger.error("Error stopping model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 