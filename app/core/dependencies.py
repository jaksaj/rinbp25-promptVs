from app.services.ollama import OllamaService
from app.services.neo4j import Neo4jService
from app.services.prompt_generator import PromptGenerator
from app.services.ab_evaluator import ABTestingEvaluator

# Create single instances of services
ollama_service = OllamaService()
neo4j_service = Neo4jService()
prompt_generator = PromptGenerator(ollama_service)
ab_evaluator = ABTestingEvaluator(ollama_service, neo4j_service=neo4j_service)

def get_ollama_service() -> OllamaService:
    """Dependency function to get the shared OllamaService instance."""
    return ollama_service

def get_neo4j_service() -> Neo4jService:
    """Dependency function to get the shared Neo4jService instance."""
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("get_neo4j_service called")
    service = Neo4jService()
    logger.debug(f"get_neo4j_service returning: {service}")
    return service

def get_prompt_generator() -> PromptGenerator:
    """Dependency function to get the shared PromptGenerator instance."""
    return prompt_generator

def get_ab_evaluator() -> ABTestingEvaluator:
    """Dependency function to get the shared ABTestingEvaluator instance."""
    return ab_evaluator