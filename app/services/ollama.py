import subprocess
import time
import platform
import os
import requests
import logging
from typing import Optional, List, Dict
from app.core.config import OLLAMA_HOST, WINDOWS_OLLAMA_PATH, MAX_RUNNING_MODELS, DEFAULT_MODELS

logger = logging.getLogger(__name__)

class ModelStatus:
    def __init__(self, name: str, is_running: bool = False):
        self.name = name
        self.is_running = is_running
        self.process: Optional[subprocess.Popen] = None

class OllamaService:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.ready = False
        self.models: Dict[str, ModelStatus] = {}
        self.max_running_models = MAX_RUNNING_MODELS
        self._initialize_models()

    def _initialize_models(self):
        """Initialize default models during service startup."""
        try:
            for model_name in DEFAULT_MODELS:
                if model_name not in self.models:
                    self.models[model_name] = ModelStatus(model_name)
                    logger.info("Initialized model: %s", model_name)
        except Exception as e:
            logger.error("Error initializing models: %s", str(e))
            raise

    def check_installation(self) -> bool:
        """Check if Ollama is installed locally."""
        try:
            # On Windows, try to find ollama in the default installation path
            if platform.system() == 'Windows':
                if os.path.exists(WINDOWS_OLLAMA_PATH):
                    return True
                # Also check if ollama is in PATH
                result = subprocess.run(['where', 'ollama'], capture_output=True, text=True)
                return result.returncode == 0
            else:
                subprocess.run(['ollama', '--version'], capture_output=True, check=True)
                return True
        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            return False

    def start_server(self) -> bool:
        """Start the Ollama server locally."""
        if not self.check_installation():
            raise Exception("Ollama is not installed. Please install Ollama first.")
        
        try:
            # On Windows, use the full path to ollama
            if platform.system() == 'Windows':
                if os.path.exists(WINDOWS_OLLAMA_PATH):
                    self.process = subprocess.Popen([WINDOWS_OLLAMA_PATH, 'serve'])
                else:
                    self.process = subprocess.Popen(['ollama', 'serve'])
            else:
                self.process = subprocess.Popen(['ollama', 'serve'])
                
            logger.info("Started Ollama server")
            
            # Wait for server to be ready
            time.sleep(5)
            
            # Verify server is running
            response = requests.get(f'{OLLAMA_HOST}/api/version')
            if response.status_code == 200:
                self.ready = True
                logger.info("Ollama server is ready")
                return True
            else:
                raise Exception("Failed to start Ollama server")
                
        except Exception as e:
            logger.error("Error starting Ollama server: %s", str(e))
            if self.process:
                self.process.terminate()
            raise

    def stop_server(self) -> None:
        """Stop the Ollama server."""
        if self.process:
            self.process.terminate()
            self.process = None
            self.ready = False
            logger.info("Stopped Ollama server")

    def pull_model(self, model_name: str) -> None:
        """Pull a specific Ollama model."""
        logger.info("Pulling model: %s", model_name)
        try:
            if platform.system() == 'Windows':
                if os.path.exists(WINDOWS_OLLAMA_PATH):
                    subprocess.run([WINDOWS_OLLAMA_PATH, 'pull', model_name], check=True)
                else:
                    subprocess.run(['ollama', 'pull', model_name], check=True)
            else:
                subprocess.run(['ollama', 'pull', model_name], check=True)
            logger.info("Model pulled successfully")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to pull model: %s", str(e))
            raise Exception(f"Failed to pull model: {str(e)}")

    def list_available_models(self) -> List[str]:
        """List all available models."""
        try:
            response = requests.get(f'{OLLAMA_HOST}/api/tags')
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                raise Exception("Failed to get model list")
        except Exception as e:
            logger.error("Error listing models: %s", str(e))
            raise

    def list_running_models(self) -> List[Dict[str, bool]]:
        """List all running models and their status."""
        return [
            {"name": name, "is_running": status.is_running}
            for name, status in self.models.items()
        ]

    def add_running_model(self, model_name: str) -> None:
        """Add a model to the running models list."""
        if len([m for m in self.models.values() if m.is_running]) >= self.max_running_models:
            raise Exception(f"Maximum number of running models ({self.max_running_models}) reached")
        
        if model_name not in self.models:
            self.models[model_name] = ModelStatus(model_name)
        
        self.models[model_name].is_running = True
        logger.info("Added running model: %s", model_name)

    def remove_running_model(self, model_name: str) -> None:
        """Remove a model from the running models list."""
        if model_name in self.models:
            self.models[model_name].is_running = False
            logger.info("Removed running model: %s", model_name)
        else:
            raise Exception(f"Model {model_name} not found in running models")

    def process_prompt(self, prompt: str, model_name: str) -> str:
        """Process a prompt using a specific model."""
        if not self.ready:
            raise Exception("Ollama is not ready yet. Please try again in a few moments.")
            
        if model_name not in self.models or not self.models[model_name].is_running:
            raise Exception(f"Model {model_name} is not running")
            
        logger.info("Processing prompt with model %s: %s", model_name, prompt)
        
        try:
            # Prepare the API request
            api_request = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Send request to Ollama API
            response = requests.post(
                f'{OLLAMA_HOST}/api/generate',
                json=api_request,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to get response from Ollama: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error("Network error while processing prompt: %s", str(e))
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Error processing prompt: %s", str(e))
            raise 