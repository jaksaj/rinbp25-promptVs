from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import json
import requests
import subprocess
import time
import platform
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
ollama_process = None
ollama_ready = False

def check_ollama_installation():
    """Check if Ollama is installed locally."""
    try:
        # On Windows, try to find ollama in the default installation path
        if platform.system() == 'Windows':
            ollama_path = r'C:\Users\JJ\AppData\Local\Programs\Ollama\ollama.exe'
            if os.path.exists(ollama_path):
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

def start_ollama_server():
    """Start the Ollama server locally."""
    global ollama_process, ollama_ready
    
    if not check_ollama_installation():
        raise Exception("Ollama is not installed. Please install Ollama first.")
    
    try:
        # On Windows, use the full path to ollama
        if platform.system() == 'Windows':
            ollama_path = r'C:\Users\JJ\AppData\Local\Programs\Ollama\ollama.exe'
            if os.path.exists(ollama_path):
                ollama_process = subprocess.Popen([ollama_path, 'serve'])
            else:
                ollama_process = subprocess.Popen(['ollama', 'serve'])
        else:
            ollama_process = subprocess.Popen(['ollama', 'serve'])
            
        logger.info("Started Ollama server")
        
        # Wait for server to be ready
        time.sleep(5)
        
        # Verify server is running
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code == 200:
            ollama_ready = True
            logger.info("Ollama server is ready")
            return True
        else:
            raise Exception("Failed to start Ollama server")
            
    except Exception as e:
        logger.error("Error starting Ollama server: %s", str(e))
        if ollama_process:
            ollama_process.terminate()
        raise

def stop_ollama_server():
    """Stop the Ollama server."""
    global ollama_process, ollama_ready
    
    if ollama_process:
        ollama_process.terminate()
        ollama_process = None
        ollama_ready = False
        logger.info("Stopped Ollama server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up PromptVs API...")
    
    try:
        # Start Ollama server
        start_ollama_server()
        
        # Pull the model
        logger.info("Pulling model...")
        if platform.system() == 'Windows':
            ollama_path = r'C:\Users\JJ\AppData\Local\Programs\Ollama\ollama.exe'
            if os.path.exists(ollama_path):
                subprocess.run([ollama_path, 'pull', 'deepseek-r1:1.5b'], check=True)
            else:
                subprocess.run(['ollama', 'pull', 'deepseek-r1:1.5b'], check=True)
        else:
            subprocess.run(['ollama', 'pull', 'deepseek-r1:1.5b'], check=True)
            
        logger.info("Model pulled successfully")
        
    except Exception as e:
        logger.error("Failed to initialize Ollama: %s", str(e))
        stop_ollama_server()
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PromptVs API...")
    stop_ollama_server()

app = FastAPI(title="PromptVs API", lifespan=lifespan)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
    """Process a prompt using the local Ollama instance."""
    try:
        if not ollama_ready:
            raise HTTPException(status_code=503, detail="Ollama is not ready yet. Please try again in a few moments.")
            
        logger.info("Received prompt request: %s", request.prompt)
        
        # Prepare the API request
        api_request = {
            "model": "deepseek-r1:1.5b",
            "prompt": request.prompt,
            "stream": False
        }
        
        # Send request to Ollama API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=api_request
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get response from Ollama")
        
        result = response.json()
        return {"result": result.get("response", "")}
    
    except Exception as e:
        logger.error("Error processing prompt: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Get the status of the API and Ollama."""
    status = "ready" if ollama_ready else "initializing"
    return {
        "message": "Welcome to PromptVs API",
        "status": status,
        "ollama_installed": check_ollama_installation(),
        "system": platform.system()
    } 