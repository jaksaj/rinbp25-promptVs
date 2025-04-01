from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import prompts, models
from app.core.dependencies import get_ollama_service
from app.services.ollama import OllamaService
from app.core.config import API_TITLE, API_VERSION, API_DESCRIPTION
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prompts.router, prefix="/api")
app.include_router(models.router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint that returns the API status."""
    ollama_service = get_ollama_service()
    if ollama_service.check_installation():
        return {"message": "API is running", "status": "ready"}
    else:
        return {"message": "API is running but Ollama is not installed", "status": "not_ready"}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        ollama_service = get_ollama_service()
        if not ollama_service.check_installation():
            logger.error("Ollama is not installed. Please install Ollama first.")
        else:
            logger.info("Ollama is installed and ready")
    except Exception as e:
        logger.error("Error during startup: %s", str(e))
        raise 