from fastapi import APIRouter, HTTPException, Depends
from app.api.models.prompt import PromptRequest, PromptResponse
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/prompt", response_model=PromptResponse)
async def process_prompt(
    request: PromptRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Process a prompt using the local Ollama instance."""
    try:
        if not ollama_service.ready:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not ready yet. Please try again in a few moments."
            )
            
        logger.info("Received prompt request: %s", request.prompt)
        result = ollama_service.process_prompt(request.prompt)
        return PromptResponse(result=result)
    except Exception as e:
        logger.error("Error processing prompt: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 