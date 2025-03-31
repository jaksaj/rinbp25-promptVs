import subprocess
import time
import platform
import os
import requests
import logging
from typing import Optional
from app.core.config import OLLAMA_MODEL, OLLAMA_HOST, WINDOWS_OLLAMA_PATH

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.ready = False

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

    def pull_model(self) -> None:
        """Pull the Ollama model."""
        logger.info("Pulling model...")
        try:
            if platform.system() == 'Windows':
                if os.path.exists(WINDOWS_OLLAMA_PATH):
                    subprocess.run([WINDOWS_OLLAMA_PATH, 'pull', OLLAMA_MODEL], check=True)
                else:
                    subprocess.run(['ollama', 'pull', OLLAMA_MODEL], check=True)
            else:
                subprocess.run(['ollama', 'pull', OLLAMA_MODEL], check=True)
            logger.info("Model pulled successfully")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to pull model: %s", str(e))
            raise Exception(f"Failed to pull model: {str(e)}")

    def process_prompt(self, prompt: str) -> str:
        """Process a prompt using the local Ollama instance."""
        if not self.ready:
            raise Exception("Ollama is not ready yet. Please try again in a few moments.")
            
        logger.info("Processing prompt: %s", prompt)
        
        try:
            # Prepare the API request
            api_request = {
                "model": OLLAMA_MODEL,
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