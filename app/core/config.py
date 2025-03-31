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

# Windows-specific paths
WINDOWS_OLLAMA_PATH = r'C:\Users\JJ\AppData\Local\Programs\Ollama\ollama.exe' 