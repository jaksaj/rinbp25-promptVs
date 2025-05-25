#!/usr/bin/env python3
"""
Script 2: Create Test Runs
==========================

This script handles the test execution phase:
1. Load prompts and versions from previous script
2. Start API server and Ollama
3. Stop unnecessary models
4. Start required test models
5. Run tests for each version multiple times
6. Save test run results for evaluation script

Usage:
    python 02_create_test_runs.py

Input:
    - prompts_and_versions.json: Data from script 1

Output:
    - test_runs.json: Contains test run data
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
        logging.FileHandler(f'../logs/02_test_runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestRunCreator:
    """Creates test runs for prompt versions"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", input_file: str = "../output/prompts_and_versions.json"):
        self.api_base_url = api_base_url.rstrip('/')
        self.input_file = input_file
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Data from previous script
        self.prompt_group_id: Optional[str] = None
        self.prompt_ids: List[str] = []
        self.prompt_versions: Dict[str, List[str]] = {}
        
        # Test run data
        self.test_runs: Dict[str, List[str]] = {}  # version_id -> [test_run_ids]
        
        # Load data from previous script
        self.load_input_data()
        
        # Models for testing (lightweight models for faster execution)
        self.test_models = ["llama3.2:1b", "gemma3:1b"]
        
    def load_input_data(self) -> Dict:
        """Load data from previous script"""
        try:
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            
            self.prompt_group_id = data['prompt_group_id']
            self.prompt_ids = data['prompt_ids']
            self.prompt_versions = data['prompt_versions']
            
            logger.info(f"Loaded data from {self.input_file}")
            logger.info(f"Prompt group: {self.prompt_group_id}")
            logger.info(f"Prompts: {len(self.prompt_ids)}")
            logger.info(f"Total versions: {sum(len(versions) for versions in self.prompt_versions.values())}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"Input file {self.input_file} not found. Please run script 1 first.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            raise
    
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
    
    def ensure_models_available(self, models: List[str]) -> bool:
        """Ensure required models are pulled and running"""
        logger.info(f"Ensuring models are available: {models}")
        
        for model in models:
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
                
            except Exception as e:
                logger.error(f"Failed to ensure model {model} is available: {e}")
                return False
        
        # Verify all models are running
        time.sleep(3)
        response = self._make_request('GET', '/api/models/running')
        running_models = [m['name'] for m in response.json()['running_models']]
        
        all_running = all(model in running_models for model in models)
        if all_running:
            logger.info(f"All required models are running: {running_models}")
        else:
            missing = [model for model in models if model not in running_models]
            logger.error(f"Some models failed to start: {missing}")
        
        return all_running
    
    def run_test_runs(self, runs_per_version: int = 2) -> Dict[str, List[str]]:
        """Run test runs for each prompt version"""
        logger.info(f"Running {runs_per_version} test runs per version...")
        
        total_versions = sum(len(versions) for versions in self.prompt_versions.values())
        total_runs = total_versions * len(self.test_models) * runs_per_version
        current_run = 0
        
        logger.info(f"Total test runs to create: {total_runs}")
        
        for prompt_id, version_ids in self.prompt_versions.items():
            for version_id in version_ids:
                self.test_runs[version_id] = []
                
                for model in self.test_models:
                    for run_num in range(runs_per_version):
                        current_run += 1
                        logger.info(f"Creating test run {current_run}/{total_runs}: version {version_id}, model {model}, run {run_num + 1}")
                        
                        try:
                            # Run test and save with longer timeout for model inference
                            response = self._make_request(
                                'POST',
                                f'/api/test-prompt-and-save',
                                params={
                                    'version_id': version_id,
                                    'model_name': model
                                },
                                timeout=120  # Increased timeout for model inference
                            )
                            result = response.json()
                            test_run_id = result['run_id']
                            self.test_runs[version_id].append(test_run_id)
                            
                            logger.info(f"Created test run {test_run_id} successfully")
                            
                        except Exception as e:
                            logger.error(f"Failed to run test for version {version_id} with model {model}: {e}")
                            # Continue with other tests even if one fails
        
        return self.test_runs
    
    def save_results(self, filename: str = "../output/test_runs.json") -> str:
        """Save test run data for next script"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'api_base_url': self.api_base_url,
            'prompt_group_id': self.prompt_group_id,
            'prompt_ids': self.prompt_ids,
            'prompt_versions': self.prompt_versions,
            'test_runs': self.test_runs,
            'test_models': self.test_models,
            'total_test_runs': sum(len(runs) for runs in self.test_runs.values())
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def run(self, runs_per_version: int = 2) -> Dict:
        """Run the complete test run creation process"""
        logger.info("Starting test run creation...")
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
            
            # Step 4: Ensure test models are available
            if not self.ensure_models_available(self.test_models):
                raise Exception("Failed to ensure test models are available")
            
            # Step 5: Run test runs
            self.run_test_runs(runs_per_version)
            
            # Step 6: Save results
            results_file = self.save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Test run creation completed successfully in {total_time:.2f} seconds")
            logger.info(f"Results saved to: {results_file}")
            
            # Stop models to free resources
            self.stop_all_models()
            
            total_test_runs = sum(len(runs) for runs in self.test_runs.values())
            
            summary = {
                'success': True,
                'total_versions': sum(len(versions) for versions in self.prompt_versions.values()),
                'total_test_runs': total_test_runs,
                'execution_time': total_time,
                'results_file': results_file
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Test run creation failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create test runs for prompt versions')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--input', default='../output/prompts_and_versions.json', help='Input file from script 1')
    parser.add_argument('--runs-per-version', type=int, default=2, help='Number of test runs per version')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    creator = TestRunCreator(api_base_url=args.api_url, input_file=args.input)
    
    try:
        summary = creator.run(runs_per_version=args.runs_per_version)
        print("\n✅ Test run creation completed successfully!")
        print(f"Created {summary['total_test_runs']} test runs for {summary['total_versions']} versions")
        print(f"Results saved to: {summary['results_file']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ Test run creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
