from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import prompts, models, prompt_versioning
from app.core.dependencies import get_ollama_service, get_neo4j_service
from app.services.ollama import OllamaService
from app.services.neo4j import Neo4jService
from app.core.config import API_TITLE, API_VERSION, API_DESCRIPTION
import logging
import platform

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
app.include_router(prompt_versioning.router, prefix="/api")  # Add new router

@app.get("/")
async def root():
    """Root endpoint that returns the API status."""
    ollama_service = get_ollama_service()
    neo4j_service = get_neo4j_service()
    
    status = {
        "message": "API is running",
        "status": "ready",
        "system": platform.system(),
        "ollama_installed": ollama_service.check_installation(),
        "components": {
            "ollama": "not_ready",
            "neo4j": "not_ready"
        }
    }
    
    # Check Ollama status
    if ollama_service.check_installation():
        status["components"]["ollama"] = "ready"
    
    # Check Neo4j status (if the driver exists, we've already validated the connection)
    if neo4j_service.driver:
        status["components"]["neo4j"] = "ready"
        
    # Overall status is ready only if both components are ready
    if all(value == "ready" for value in status["components"].values()):
        status["status"] = "ready"
    else:
        status["status"] = "not_ready"
    
    return status

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Check Ollama
        ollama_service = get_ollama_service()
        if not ollama_service.check_installation():
            logger.error("Ollama is not installed. Please install Ollama first.")
        else:
            logger.info("Ollama is installed and ready")
            
        # Check Neo4j
        neo4j_service = get_neo4j_service()
        if not neo4j_service.driver:
            logger.error("Neo4j connection failed. Please check Neo4j settings.")
        else:
            logger.info("Neo4j connection established")
            # Create necessary indexes and constraints
            neo4j_service.create_indexes()
            
    except Exception as e:
        logger.error("Error during startup: %s", str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown."""
    try:
        # Close Neo4j connection
        neo4j_service = get_neo4j_service()
        if neo4j_service.driver:
            neo4j_service.close()
            logger.info("Neo4j connection closed")
            
    except Exception as e:
        logger.error("Error during shutdown: %s", str(e))