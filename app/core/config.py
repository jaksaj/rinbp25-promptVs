import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Settings
API_TITLE = "PromptVs API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API for processing prompts using Ollama"

# Ollama Settings
OLLAMA_MODEL = "deepseek-r1:1.5b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300  # 5 minutes
MAX_RUNNING_MODELS = 2  # Maximum number of models that can run simultaneously

# Default Models
DEFAULT_MODELS = [
    "deepseek-r1:1.5b",
    "gemma3:1b",
    "llama3.2:1b",
    "gemma3:4b",
    "llama3.1:8b"
]

# Windows-specific paths
WINDOWS_OLLAMA_PATH = r'C:\Users\JJ\AppData\Local\Programs\Ollama\ollama.exe'

# Neo4j Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")