from app.services.ollama import OllamaService
from app.services.neo4j import Neo4jService
from app.services.prompt_generator import PromptGenerator

# Create single instances of services
ollama_service = OllamaService()
neo4j_service = Neo4jService()
prompt_generator = PromptGenerator(ollama_service)

def get_ollama_service() -> OllamaService:
    """Dependency function to get the shared OllamaService instance."""
    return ollama_service

def get_neo4j_service() -> Neo4jService:
    """Dependency function to get the shared Neo4jService instance."""
    return neo4j_service

def get_prompt_generator() -> PromptGenerator:
    """Dependency function to get the shared PromptGenerator instance."""
    return prompt_generator