from fastapi import APIRouter, HTTPException, Depends
from app.api.models.model import ModelRequest, ModelResponse, ModelsListResponse
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """List all available models."""
    try:
        models = ollama_service.list_models()
        return ModelsListResponse(
            models=models,
            current_model=ollama_service.current_model
        )
    except Exception as e:
        logger.error("Error listing models: %s", str(e))
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

@router.post("/models/select", response_model=ModelResponse)
async def select_model(
    request: ModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Select a model to use for prompts."""
    try:
        # Verify the model exists
        models = ollama_service.list_models()
        if request.model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found. Available models: {', '.join(models)}"
            )
        
        ollama_service.set_current_model(request.model_name)
        return ModelResponse(
            message="Model selected successfully",
            model_name=request.model_name
        )
    except Exception as e:
        logger.error("Error selecting model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 