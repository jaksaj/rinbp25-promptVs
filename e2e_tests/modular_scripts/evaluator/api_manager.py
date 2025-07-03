#!/usr/bin/env python3
"""
API Manager
==========

Handles API server management, Ollama service management, and model operations.
"""

import logging
import subprocess
import time
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class APIManager:
    """Manages API server, Ollama service, and model operations"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", evaluation_model: str = "llama3.1:8b"):
        self.api_base_url = api_base_url.rstrip('/')
        self.evaluation_model = evaluation_model
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None, timeout: int = 120) -> requests.Response:
        """Make HTTP request to API with error handling"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            logger.error(f"API request timed out after {timeout}s: {method} {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def check_api_health(self) -> bool:
        """Check if API is running and healthy"""
        try:
            response = self._make_request('GET', '/', timeout=10)
            status_data = response.json()
            
            if status_data.get('status') == 'ready':
                logger.info("API server is healthy")
                return True
            else:
                logger.warning(f"API server status: {status_data.get('status')}")
                return False
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Start the API server if not running"""
        if self.check_api_health():
            logger.info("API server is already running")
            return True
            
        logger.info("Starting API server...")
        try:
            subprocess.Popen(
                ["python", "run.py"],
                cwd="../../",  # Run from project root
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            for attempt in range(30):
                time.sleep(2)
                if self.check_api_health():
                    logger.info("API server started successfully")
                    return True
                logger.debug(f"Waiting for API server... (attempt {attempt + 1}/30)")
            
            logger.error("API server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service if not running"""
        try:
            # Check if Ollama is running by trying to list models
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Ollama service is already running")
                return True
        except:
            pass
        
        logger.info("Starting Ollama service...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)  # Give Ollama time to start
            
            # Verify it's running
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Ollama service started successfully")
                return True
            else:
                logger.error("Failed to verify Ollama service startup")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def stop_all_models(self) -> bool:
        """Stop all running models to conserve resources"""
        logger.info("Stopping all running models to conserve resources...")
        try:
            response = self._make_request('GET', '/api/models/running')
            running_models = response.json().get('running_models', [])
            
            if not running_models:
                logger.info("No models are currently running")
                return True
            
            for model in running_models:
                model_name = model['name']
                try:
                    self._make_request('POST', '/api/models/stop', {'model_name': model_name})
                    logger.info(f"Stopped model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to stop model {model_name}: {e}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to stop all models: {e}")
            return False
    
    def ensure_evaluation_model_available(self) -> bool:
        """Ensure evaluation model is pulled and running"""
        logger.info(f"Ensuring evaluation model is available: {self.evaluation_model}")
        
        try:
            # Check if model exists
            response = self._make_request('GET', '/api/models/status')
            model_statuses = response.json()
            
            # Handle both list and dict responses
            if isinstance(model_statuses, list):
                available_models = model_statuses
            else:
                available_models = model_statuses.get('available_models', [])
            
            model_exists = any(
                model.get('name') == self.evaluation_model if isinstance(model, dict) else model == self.evaluation_model
                for model in available_models
            )
            
            if not model_exists:
                logger.info(f"Model {self.evaluation_model} not found, pulling...")
                self._make_request('POST', '/api/models/pull', {'model_name': self.evaluation_model}, timeout=600)
                logger.info(f"Successfully pulled model: {self.evaluation_model}")
            
            # Check if model is running
            response = self._make_request('GET', '/api/models/running')
            running_response = response.json()
            
            # Handle both list and dict responses
            if isinstance(running_response, list):
                running_models = [m.get('name') if isinstance(m, dict) else m for m in running_response]
            else:
                running_models = [m['name'] for m in running_response.get('running_models', [])]
            
            if self.evaluation_model not in running_models:
                logger.info(f"Starting evaluation model: {self.evaluation_model}")
                response = self._make_request('POST', '/api/models/start', {'model_name': self.evaluation_model})
                logger.info(f"Successfully started model: {self.evaluation_model}")
            
            # Verify model is running
            time.sleep(3)
            response = self._make_request('GET', '/api/models/running')
            running_response = response.json()
            
            # Handle both list and dict responses for verification
            if isinstance(running_response, list):
                running_models = [m.get('name') if isinstance(m, dict) else m for m in running_response]
            else:
                running_models = [m['name'] for m in running_response.get('running_models', [])]
            
            if self.evaluation_model in running_models:
                logger.info(f"Evaluation model {self.evaluation_model} is ready")
                return True
            else:
                logger.error(f"Failed to start evaluation model {self.evaluation_model}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to ensure evaluation model is available: {e}")
            return False
        
    def get_test_run_model(self, test_run: dict) -> str:
        """Return model name from test run dict (strict, raises if missing)"""
        if not isinstance(test_run, dict):
            raise ValueError(f"Test run must be a dict, got {type(test_run)}: {test_run}")
        if 'model_name' not in test_run or not test_run['model_name']:
            raise ValueError(f"Test run dict missing 'model_name': {test_run}")
        return test_run['model_name']