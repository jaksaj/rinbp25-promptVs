#!/usr/bin/env python3
"""
Script 1: Create Prompts and Versions
=====================================

This script handles the initial setup phase:
1. Start API server and Ollama
2. Stop unnecessary models to conserve resources
3. Start required models for version creation
4. Create prompt group and prompts
5. Create prompt versions using different techniques
6. Save results for next script

Usage:
    python 01_create_prompts_and_versions.py

Output:
    - prompts_and_versions.json: Contains prompt group, prompts, and versions data
"""

import json
import logging
import time
import requests
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'../logs/01_create_prompts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PromptCreator:
    """Creates prompts and versions for testing"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", config_file: str = "../config/test_prompts_config.json"):
        self.api_base_url = api_base_url.rstrip('/')
        self.config_file = config_file
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Data storage
        self.prompt_group_id: Optional[str] = None
        self.prompt_ids: List[str] = []
        self.prompt_versions: Dict[str, List[str]] = {}  # prompt_id -> [version_ids]
        
        # Load configuration
        self.config = self._load_config()
        
        # Model for version creation
        self.generator_model = "llama3.1:8b"
        
    def _load_config(self) -> Dict:
        """Load test configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None, timeout: int = 60) -> requests.Response:
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
            response = self._make_request('GET', '/')
            status_data = response.json()
            logger.info(f"API Status: {status_data}")
            return status_data.get('status') == 'ready'
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Start the API server if not running"""
        if self.check_api_health():
            logger.info("API server is already running")
            return True
        logger.info("Starting API server...")
        try:
            # Start the API server in background - go up to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            subprocess.Popen([
                sys.executable, "run.py"
            ], cwd=project_root)
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_api_health():
                    logger.info("API server started successfully")
                    return True
            
            logger.error("API server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service if not running"""
        try:
            # Check if Ollama is already running
            response = self._make_request('GET', '/api/models/running')
            logger.info("Ollama is already running")
            return True
        except:
            logger.info("Starting Ollama service...")
            try:
                # Try to start Ollama (Windows)
                subprocess.Popen(['ollama', 'serve'], shell=True)
                time.sleep(5)  # Give it time to start
                
                # Verify it started
                response = self._make_request('GET', '/api/models/running')
                logger.info("Ollama service started successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to start Ollama service: {e}")
                return False
    
    def stop_all_models(self) -> bool:
        """Stop all running models to conserve resources"""
        logger.info("Stopping all running models to conserve resources...")
        try:
            response = self._make_request('GET', '/api/models/running')
            running_models = response.json().get('running_models', [])
            
            for model in running_models:
                model_name = model['name']
                logger.info(f"Stopping model: {model_name}")
                try:
                    self._make_request('POST', '/api/models/stop', {'model_name': model_name})
                    logger.info(f"Stopped model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to stop model {model_name}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to stop models: {e}")
            return False
    
    def ensure_model_available(self, model: str) -> bool:
        """Ensure a specific model is pulled and running"""
        logger.info(f"Ensuring model is available: {model}")
        
        try:
            # Get current model status
            response = self._make_request('GET', '/api/models/status')
            model_statuses = response.json()
            
            # Check if model needs to be pulled
            status_dict = {status['name']: status for status in model_statuses}
            
            if model not in status_dict or not status_dict[model]['is_pulled']:
                logger.info(f"Pulling model: {model}")
                response = self._make_request('POST', '/api/models/pull', {'model_name': model})
                logger.info(f"Successfully pulled model: {model}")
            
            # Start the model
            logger.info(f"Starting model: {model}")
            response = self._make_request('POST', '/api/models/start', {'model_name': model})
            logger.info(f"Successfully started model: {model}")
            
            # Verify model is running
            time.sleep(2)
            response = self._make_request('GET', '/api/models/running')
            running_models = [m['name'] for m in response.json()['running_models']]
            
            if model in running_models:
                logger.info(f"Model {model} is now running")
                return True
            else:
                logger.error(f"Model {model} failed to start")
                return False
            
        except Exception as e:
            logger.error(f"Failed to ensure model {model} is available: {e}")
            return False
    
    def create_prompt_group(self) -> str:
        """Create a prompt group for testing"""
        logger.info("Creating prompt group...")
        
        group_data = self.config['prompt_group']
        response = self._make_request('POST', '/api/prompt-groups', group_data)
        
        self.prompt_group_id = response.json()['id']
        logger.info(f"Created prompt group with ID: {self.prompt_group_id}")
        return self.prompt_group_id
    
    def create_prompts(self) -> List[str]:
        """Create prompts in the prompt group using batch endpoint"""
        logger.info("Creating prompts (batch)...")
        prompt_payloads = []
        for prompt_config in self.config['prompts']:
            prompt_data = {
                'prompt_group_id': self.prompt_group_id,
                'content': prompt_config['content'],
                'name': prompt_config['name'],
                'description': prompt_config['description'],
                'expected_solution': prompt_config.get('expected_solution'),
                'tags': prompt_config.get('tags', [])
            }
            prompt_payloads.append(prompt_data)

        batch_data = {'prompts': prompt_payloads}
        response = self._make_request('POST', '/api/prompts/batch', batch_data)
        ids = response.json()['ids']
        self.prompt_ids = ids
        for prompt_config, prompt_id in zip(self.config['prompts'], ids):
            logger.info(f"Created prompt '{prompt_config['name']}' with ID: {prompt_id}")
        return self.prompt_ids
    
    def create_prompt_versions(self, techniques: List[str] = ['control', 'cot_reasoning', 'cot_simple','role_prompting','few_shot']) -> Dict[str, List[str]]:
        """Create prompt versions using different techniques via batch endpoint and poll for results."""
        logger.info(f"Creating prompt versions using techniques: {techniques} (batch mode)")
        batch_data = {
            'prompt_ids': self.prompt_ids,
            'techniques': techniques,
            'generator_model': self.generator_model
        }
        response = self._make_request('POST', '/api/techniques/batch-apply', batch_data)
        job_id = response.json()['job_id']
        logger.info(f"Batch job submitted: {job_id}")
        # Poll for completion
        status = None
        results = None
        for attempt in range(600):  # up to 10 minutes
            time.sleep(1)
            status_resp = self._make_request('GET', f'/api/techniques/batch-status/{job_id}')
            status_json = status_resp.json()
            status = status_json['status']
            if status == 'completed':
                results = status_json['results']
                logger.info(f"Batch job completed after {attempt+1} seconds")
                break
            elif status == 'failed':
                logger.error(f"Batch job failed: {status_json.get('error')}")
                raise Exception(f"Batch job failed: {status_json.get('error')}")
            elif attempt % 10 == 0:
                logger.info(f"Waiting for batch job... ({attempt+1}s)")
        if status != 'completed':
            raise Exception("Batch job did not complete in time")
        # Collect version IDs
        for prompt_id in self.prompt_ids:
            self.prompt_versions[prompt_id] = []
            for technique in techniques:
                version_id = results.get(prompt_id, {}).get(technique)
                if version_id:
                    self.prompt_versions[prompt_id].append(version_id)
                    logger.info(f"Created version {version_id} for prompt {prompt_id} using technique '{technique}'")
                else:
                    logger.error(f"Failed to create version for prompt {prompt_id} using technique '{technique}'")
        return self.prompt_versions
    
    def save_results(self, filename: str = "../output/prompts_and_versions.json") -> str:
        """Save prompt and version data for next script"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_file,
            'api_base_url': self.api_base_url,
            'prompt_group_id': self.prompt_group_id,
            'prompt_ids': self.prompt_ids,
            'prompt_versions': self.prompt_versions,
            'generator_model': self.generator_model,
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def run(self) -> Dict:
        """Run the complete prompt and version creation process"""
        logger.info("Starting prompt and version creation...")
        start_time = time.time()
        
        try:
            # Step 1: Start API server
            if not self.start_api_server():
                raise Exception("Failed to start API server")
            
            # Step 2: Start Ollama service
            if not self.start_ollama_service():
                raise Exception("Failed to start Ollama service")
            
            # Step 3: Stop all models to conserve resources
            self.stop_all_models()
            
            # Step 4: Ensure generator model is available
            if not self.ensure_model_available(self.generator_model):
                raise Exception(f"Failed to ensure {self.generator_model} is available")
            
            # Step 5: Create prompt group
            self.create_prompt_group()
            
            # Step 6: Create prompts
            self.create_prompts()
            
            # Step 7: Create prompt versions
            self.create_prompt_versions()
            
            # Step 8: Save results
            results_file = self.save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Prompt and version creation completed successfully in {total_time:.2f} seconds")
            logger.info(f"Results saved to: {results_file}")
            
            # Stop the generator model to free resources
            self.stop_all_models()
            
            summary = {
                'success': True,
                'prompt_group_id': self.prompt_group_id,
                'total_prompts': len(self.prompt_ids),
                'total_versions': sum(len(versions) for versions in self.prompt_versions.values()),
                'execution_time': total_time,
                'results_file': results_file
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Prompt and version creation failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create prompts and versions for testing')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--config', default='../config/test_prompts_config.json', help='Test configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    creator = PromptCreator(api_base_url=args.api_url, config_file=args.config)
    
    try:
        summary = creator.run()
        print("\n✅ Prompt and version creation completed successfully!")
        print(f"Created {summary['total_prompts']} prompts with {summary['total_versions']} versions")
        print(f"Results saved to: {summary['results_file']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ Prompt and version creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
