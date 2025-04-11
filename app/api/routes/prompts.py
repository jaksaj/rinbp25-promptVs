from fastapi import APIRouter, HTTPException, Depends
from app.api.models.prompt import PromptRequest, SimplePromptResponse, BatchPromptRequest, BatchPromptResponse, ModelPromptResponse
from app.api.models.model import ModelRequest
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/batch", response_model=BatchPromptResponse)
async def process_batch_prompt(
    request: BatchPromptRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Process a prompt using multiple models."""
    try:
        results = []
        running_models = [model.name for model in ollama_service.list_running_models()]
        
        for model_name in request.models:
            if model_name not in running_models:
                results.append(ModelPromptResponse(
                    model=model_name,
                    response=f"Error: Model {model_name} is not running"
                ))
                continue
                
            try:
                response = ollama_service.process_prompt(request.prompt, model_name)
                results.append(ModelPromptResponse(
                    model=model_name,
                    response=response
                ))
            except Exception as e:
                logger.error("Error processing prompt for model %s: %s", model_name, str(e))
                results.append(ModelPromptResponse(
                    model=model_name,
                    response=f"Error: {str(e)}"
                ))
        
        return BatchPromptResponse(results=results)
    except Exception as e:
        logger.error("Error processing batch prompt: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/single/{model_name}", response_model=SimplePromptResponse)
async def process_prompt(
    model_name: str,
    request: PromptRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Process a prompt using a specific model."""
    try:
        result = ollama_service.process_prompt(request.prompt, model_name)
        return SimplePromptResponse(result=result)
    except Exception as e:
        logger.error("Error processing prompt: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))