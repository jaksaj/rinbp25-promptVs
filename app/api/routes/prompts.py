from fastapi import APIRouter, HTTPException, Depends
from app.api.models.prompt import PromptRequest, PromptResponse
from app.api.models.model import ModelRequest
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/prompt/{model_name}", response_model=PromptResponse)
async def process_prompt(
    model_name: str,
    request: PromptRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Process a prompt using a specific model."""
    try:
        result = ollama_service.process_prompt(request.prompt, model_name)
        return PromptResponse(result=result)
    except Exception as e:
        logger.error("Error processing prompt: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 