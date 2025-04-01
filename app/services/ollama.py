import subprocess
import time
import platform
import os
import requests
import logging
import json
from typing import Optional, List, Dict, Set
from app.core.config import OLLAMA_HOST, WINDOWS_OLLAMA_PATH, MAX_RUNNING_MODELS, DEFAULT_MODELS
from app.api.models.model import RunningModelStatus

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
        self.pulling_models: Set[str] = set()
        self.starting_models: Set[str] = set()
        self.stopping_models: Set[str] = set()
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
        """Check if Ollama is installed and accessible."""
        try:
            if platform.system() == "Windows":
                if not os.path.exists(WINDOWS_OLLAMA_PATH):
                    logger.error("Ollama not found at: %s", WINDOWS_OLLAMA_PATH)
                    return False
                return True
            else:
                result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
                return result.returncode == 0
        except Exception as e:
            logger.error("Error checking Ollama installation: %s", str(e))
            return False

    def verify_server(self) -> bool:
        """Verify that the Ollama server is running and accessible."""
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error("Error verifying Ollama server: %s", str(e))
            return False

    def list_available_models(self) -> List[str]:
        """List all available models."""
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error("Error listing models: %s", str(e))
            return []

    def list_running_models(self) -> List[RunningModelStatus]:
        """List all currently running models."""
        return [
            RunningModelStatus(name=name, is_running=status.is_running)
            for name, status in self.models.items()
            if status.is_running
        ]

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama."""
        try:
            if model_name in self.pulling_models:
                raise ValueError(f"Model {model_name} is already being pulled")
                
            self.pulling_models.add(model_name)
            logger.info("Pulling model: %s", model_name)
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/pull",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                logger.info("Successfully pulled model: %s", model_name)
                return True
            else:
                logger.error("Failed to pull model %s: %s", model_name, response.text)
                return False
        except Exception as e:
            logger.error("Error pulling model %s: %s", model_name, str(e))
            return False
        finally:
            self.pulling_models.remove(model_name)

    def add_running_model(self, model_name: str) -> bool:
        """Add a model to the list of running models."""
        try:
            if model_name in self.starting_models:
                raise ValueError(f"Model {model_name} is already being started")
                
            if len(self.list_running_models()) >= self.max_running_models:
                raise ValueError(f"Maximum number of running models ({self.max_running_models}) reached")
                
            self.starting_models.add(model_name)
            logger.info("Starting model: %s", model_name)
            
            # Start the model process
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    [WINDOWS_OLLAMA_PATH, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            self.models[model_name].process = process
            self.models[model_name].is_running = True
            
            # Wait for the server to be ready
            time.sleep(2)
            if self.verify_server():
                logger.info("Successfully started model: %s", model_name)
                return True
            else:
                logger.error("Failed to start model %s: Server not ready", model_name)
                return False
        except Exception as e:
            logger.error("Error starting model %s: %s", model_name, str(e))
            return False
        finally:
            self.starting_models.remove(model_name)

    def remove_running_model(self, model_name: str) -> bool:
        """Remove a model from the list of running models."""
        try:
            if model_name in self.stopping_models:
                raise ValueError(f"Model {model_name} is already being stopped")
                
            if model_name not in self.models or not self.models[model_name].is_running:
                raise ValueError(f"Model {model_name} is not running")
                
            self.stopping_models.add(model_name)
            logger.info("Stopping model: %s", model_name)
            
            # Stop the model process
            if self.models[model_name].process:
                self.models[model_name].process.terminate()
                self.models[model_name].process.wait()
            
            self.models[model_name].process = None
            self.models[model_name].is_running = False
            
            logger.info("Successfully stopped model: %s", model_name)
            return True
        except Exception as e:
            logger.error("Error stopping model %s: %s", model_name, str(e))
            return False
        finally:
            self.stopping_models.remove(model_name)

    def process_prompt(self, prompt: str, model_name: str) -> str:
        """Process a prompt using a specific model."""
        try:
            if model_name not in self.models or not self.models[model_name].is_running:
                raise ValueError(f"Model {model_name} is not running")
                
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = line.decode('utf-8')
                            if chunk.startswith('data: '):
                                chunk = chunk[6:]  # Remove 'data: ' prefix
                            if chunk == '[DONE]':
                                break
                            json_chunk = json.loads(chunk)
                            if 'response' in json_chunk:
                                full_response += json_chunk['response']
                        except Exception as e:
                            logger.error("Error processing chunk: %s", str(e))
                            continue
                return full_response
            else:
                raise ValueError(f"Failed to process prompt: {response.text}")
        except Exception as e:
            logger.error("Error processing prompt: %s", str(e))
            raise 