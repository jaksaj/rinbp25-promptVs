from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging import setup_logging
from app.core.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.core.dependencies import ollama_service
from app.api.routes import prompts, models
import platform

# Configure logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up PromptVs API...")
    
    try:
        # Start Ollama server
        ollama_service.start_server()
        
    except Exception as e:
        logger.error("Failed to initialize Ollama: %s", str(e))
        ollama_service.stop_server()
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PromptVs API...")
    ollama_service.stop_server()

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

# Include routers
app.include_router(prompts.router, prefix="/api", tags=["prompts"])
app.include_router(models.router, prefix="/api", tags=["models"])

# Add root endpoint directly to the app
@app.get("/")
async def root():
    """Get the status of the API and Ollama."""
    status = "ready" if ollama_service.ready else "initializing"
    return {
        "message": f"Welcome to {API_TITLE}",
        "status": status,
        "ollama_installed": ollama_service.check_installation(),
        "system": platform.system()
    } 