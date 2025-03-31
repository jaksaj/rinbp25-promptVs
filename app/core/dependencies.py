from app.services.ollama import OllamaService

# Create a single instance of OllamaService
ollama_service = OllamaService()

def get_ollama_service() -> OllamaService:
    """Dependency function to get the shared OllamaService instance."""
    return ollama_service 