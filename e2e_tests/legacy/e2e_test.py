#!/usr/bin/env python3
"""
End-to-End Test Script for PromptVs API
======================================

This script performs a complete end-to-end test of the PromptVs API workflow:
1. Start Ollama and required models
2. Create prompt group and prompts
3. Create prompt versions using different techniques (cot_simple, cot_reasoning)
4. Run each version multiple times
5. Evaluate and compare results

Usage:
    python e2e_test.py

Requirements:
    - API server running on localhost:8000
    - Ollama installed and accessible
    - Neo4j database running
    - test_prompts_config.json file with prompt definitions
"""

import asyncio
import json
import logging
import time
import requests
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'e2e_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class E2ETestRunner:
    """End-to-end test runner for PromptVs API"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", config_file: str = "test_prompts_config.json"):
        self.api_base_url = api_base_url.rstrip('/')
        self.config_file = config_file
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
          # Test data storage
        self.prompt_group_id: Optional[str] = None
        self.prompt_ids: List[str] = []
        self.prompt_versions: Dict[str, List[str]] = {}  # prompt_id -> [version_ids]
        self.test_runs: Dict[str, List[str]] = {}  # version_id -> [test_run_ids]
        self.comparison_results: List[Dict] = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Models to use for testing
        self.test_models = ["llama3.2:1b", "gemma3:1b"]
        self.evaluation_model = "gemma3:4b"
        
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
    
    def ensure_models_available(self, models: List[str]) -> bool:
        """Ensure required models are pulled and running"""
        logger.info(f"Ensuring models are available: {models}")
        
        try:
            # Get current model status
            response = self._make_request('GET', '/api/models/status')
            model_statuses = response.json()
            
            models_to_pull = []
            models_to_start = []
            
            # Check which models need to be pulled or started
            status_dict = {status['name']: status for status in model_statuses}
            
            for model in models:
                if model not in status_dict:
                    models_to_pull.append(model)
                    continue
                    
                status = status_dict[model]
                if not status['is_pulled']:
                    models_to_pull.append(model)
                elif not status['is_running']:
                    models_to_start.append(model)
            
            # Pull missing models
            for model in models_to_pull:
                logger.info(f"Pulling model: {model}")
                try:
                    response = self._make_request('POST', '/api/models/pull', {'model_name': model})
                    logger.info(f"Successfully pulled model: {model}")
                    models_to_start.append(model)  # Add to start list after pulling
                except Exception as e:
                    logger.error(f"Failed to pull model {model}: {e}")
                    return False
            
            # Start models that aren't running
            for model in models_to_start:
                logger.info(f"Starting model: {model}")
                try:
                    response = self._make_request('POST', '/api/models/start', {'model_name': model})
                    logger.info(f"Successfully started model: {model}")
                except Exception as e:
                    logger.error(f"Failed to start model {model}: {e}")
                    return False
            
            # Verify all models are running
            time.sleep(3)  # Give models time to start
            response = self._make_request('GET', '/api/models/running')
            running_models = [model['name'] for model in response.json()['running_models']]
            
            all_running = all(model in running_models for model in models)
            if all_running:
                logger.info(f"All required models are running: {running_models}")
            else:
                missing = [model for model in models if model not in running_models]
                logger.error(f"Some models failed to start: {missing}")
            
            return all_running
            
        except Exception as e:
            logger.error(f"Failed to ensure models are available: {e}")
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
        """Create prompts in the prompt group"""
        logger.info("Creating prompts...")
        
        for prompt_config in self.config['prompts']:
            prompt_data = {
                'prompt_group_id': self.prompt_group_id,
                'content': prompt_config['content'],
                'name': prompt_config['name'],
                'description': prompt_config['description'],
                'expected_solution': prompt_config.get('expected_solution'),
                'tags': prompt_config.get('tags', [])
            }
            
            response = self._make_request('POST', '/api/prompts', prompt_data)
            prompt_id = response.json()['id']
            self.prompt_ids.append(prompt_id)
            logger.info(f"Created prompt '{prompt_config['name']}' with ID: {prompt_id}")
        
        return self.prompt_ids
    
    def create_prompt_versions(self, techniques: List[str] = ['cot_simple', 'cot_reasoning']) -> Dict[str, List[str]]:
        """Create prompt versions using different techniques"""
        logger.info(f"Creating prompt versions using techniques: {techniques}")
        
        for prompt_id in self.prompt_ids:
            self.prompt_versions[prompt_id] = []
            
            for technique in techniques:
                logger.info(f"Applying technique '{technique}' to prompt {prompt_id}")
                
                # Apply technique and save as version
                technique_data = {
                    'prompt_id': prompt_id,
                    'technique': technique,
                    'save_as_version': True,
                    'generator_model': self.evaluation_model
                }
                
                try:
                    response = self._make_request('POST', '/api/techniques/apply', technique_data)
                    result = response.json()
                    version_id = result['version_id']
                    
                    if version_id:
                        self.prompt_versions[prompt_id].append(version_id)
                        logger.info(f"Created version {version_id} using technique '{technique}'")
                    else:
                        logger.error(f"Failed to create version for technique '{technique}'")
                        
                except Exception as e:
                    logger.error(f"Failed to apply technique '{technique}' to prompt {prompt_id}: {e}")
        
        return self.prompt_versions
    
    def run_test_runs(self, runs_per_version: int = 2) -> Dict[str, List[str]]:
        """Run test runs for each prompt version"""
        logger.info(f"Running {runs_per_version} test runs per version...")
        
        for prompt_id, version_ids in self.prompt_versions.items():
            for version_id in version_ids:
                self.test_runs[version_id] = []
                
                for model in self.test_models:
                    for run_num in range(runs_per_version):
                        logger.info(f"Running test {run_num + 1}/{runs_per_version} for version {version_id} with model {model}")
                        
                        try:                            # Run test and save with longer timeout for model inference
                            logger.info(f"Making API call to test-prompt-and-save...")
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
    
    def compare_prompt_versions(self) -> List[Dict]:
        """Compare prompt versions within each prompt using A/B testing"""
        logger.info("Comparing prompt versions using A/B testing...")
        
        for prompt_id, version_ids in self.prompt_versions.items():
            if len(version_ids) < 2:
                logger.warning(f"Prompt {prompt_id} has less than 2 versions, skipping comparison")
                continue
            
            # Get test runs for each version
            version_test_runs = {}
            for version_id in version_ids:
                version_test_runs[version_id] = self.test_runs.get(version_id, [])
            
            # Compare test runs between versions
            for i, version_id1 in enumerate(version_ids):
                for version_id2 in version_ids[i+1:]:
                    test_runs1 = version_test_runs[version_id1]
                    test_runs2 = version_test_runs[version_id2]
                    
                    # Compare each test run from version1 with each from version2
                    for test_run_id1 in test_runs1:
                        for test_run_id2 in test_runs2:
                            logger.info(f"Comparing test runs {test_run_id1} vs {test_run_id2}")
                            
                            try:
                                comparison_data = {
                                    'test_run_id1': test_run_id1,
                                    'test_run_id2': test_run_id2,
                                    'compare_within_version': False
                                }
                                
                                response = self._make_request(
                                    'POST',
                                    '/api/ab-testing/compare',
                                    comparison_data
                                )
                                
                                comparison_result = response.json()
                                comparison_result['prompt_id'] = prompt_id
                                comparison_result['version_id1'] = version_id1
                                comparison_result['version_id2'] = version_id2
                                self.comparison_results.append(comparison_result)
                                
                                logger.info(f"Comparison completed. Winner: {comparison_result.get('winner_test_run_id')}")
                                
                            except Exception as e:
                                logger.error(f"Failed to compare test runs {test_run_id1} vs {test_run_id2}: {e}")
        
        return self.comparison_results
    
    def find_best_versions(self) -> Dict[str, Dict]:
        """Find the best performing version for each prompt based on A/B testing results"""
        logger.info("Finding best performing versions based on comparison results...")
        
        best_versions = {}
        
        for prompt_id in self.prompt_ids:
            # Count wins for each version in this prompt
            version_wins = {}
            version_ids = self.prompt_versions.get(prompt_id, [])
            
            for version_id in version_ids:
                version_wins[version_id] = 0
            
            # Count wins from comparison results
            prompt_comparisons = [
                comp for comp in self.comparison_results 
                if comp.get('prompt_id') == prompt_id
            ]
            
            for comparison in prompt_comparisons:
                winner_test_run_id = comparison.get('winner_test_run_id')
                version_id1 = comparison.get('version_id1')
                version_id2 = comparison.get('version_id2')
                
                # Find which version the winning test run belongs to
                if winner_test_run_id:
                    for version_id in [version_id1, version_id2]:
                        if winner_test_run_id in self.test_runs.get(version_id, []):
                            version_wins[version_id] = version_wins.get(version_id, 0) + 1
                            break
            
            # Find the version with most wins
            if version_wins:
                best_version_id = max(version_wins.keys(), key=lambda v: version_wins[v])
                best_versions[prompt_id] = {
                    'best_version_id': best_version_id,
                    'wins': version_wins[best_version_id],
                    'total_comparisons': len(prompt_comparisons),
                    'win_rate': version_wins[best_version_id] / len(prompt_comparisons) if prompt_comparisons else 0,                    'all_version_wins': version_wins
                }
                logger.info(f"Best version for prompt {prompt_id}: {best_version_id} (wins: {version_wins[best_version_id]}/{len(prompt_comparisons)})")
        
        return best_versions
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report based on A/B testing results"""
        logger.info("Generating test report...")
        
        best_versions = self.find_best_versions()
        
        # Calculate summary statistics
        total_versions = sum(len(versions) for versions in self.prompt_versions.values())
        total_test_runs = sum(len(runs) for runs in self.test_runs.values())
        total_comparisons = len(self.comparison_results)
        
        # Technique performance based on comparison wins
        technique_wins = {}
        technique_totals = {}
        
        for comparison in self.comparison_results:
            prompt_id = comparison.get('prompt_id')
            version_id1 = comparison.get('version_id1')
            version_id2 = comparison.get('version_id2')
            winner_test_run_id = comparison.get('winner_test_run_id')
            
            if prompt_id and version_id1 and version_id2 and winner_test_run_id:
                # Determine which version won
                winner_version_id = None
                if winner_test_run_id in self.test_runs.get(version_id1, []):
                    winner_version_id = version_id1
                elif winner_test_run_id in self.test_runs.get(version_id2, []):
                    winner_version_id = version_id2
                
                # Map version to technique
                version_ids = self.prompt_versions.get(prompt_id, [])
                for version_id in [version_id1, version_id2]:
                    if version_id in version_ids:
                        technique_idx = version_ids.index(version_id)
                        technique = ['cot_simple', 'cot_reasoning'][technique_idx] if technique_idx < 2 else 'unknown'
                        
                        if technique not in technique_totals:
                            technique_totals[technique] = 0
                            technique_wins[technique] = 0
                        
                        technique_totals[technique] += 1
                        if version_id == winner_version_id:
                            technique_wins[technique] += 1
        
        # Calculate win rates
        technique_performance = {}
        for technique in technique_totals:
            win_rate = technique_wins[technique] / technique_totals[technique] if technique_totals[technique] > 0 else 0
            technique_performance[technique] = {
                'wins': technique_wins[technique],
                'total_comparisons': technique_totals[technique],
                'win_rate': win_rate
            }
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_prompts': len(self.prompt_ids),
                'total_versions': total_versions,
                'total_test_runs': total_test_runs,
                'total_comparisons': total_comparisons,
                'models_tested': self.test_models,
                'evaluation_model': self.evaluation_model,
                'evaluation_method': 'A/B Testing Comparisons'
            },
            'technique_performance': technique_performance,
            'best_versions_per_prompt': best_versions,
            'detailed_results': {
                'comparisons': self.comparison_results,
                'test_runs_by_version': self.test_runs,
                'versions_by_prompt': self.prompt_versions
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"e2e_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test report saved to {filename}")
        return filename
    
    async def run_full_test(self) -> Dict:
        """Run the complete end-to-end test"""
        logger.info("Starting end-to-end test...")
        start_time = time.time()
        
        try:
            # Step 1: Check API health
            if not self.check_api_health():
                raise Exception("API is not healthy")
            
            # Step 2: Start Ollama and ensure models are available
            if not self.start_ollama_service():
                raise Exception("Failed to start Ollama service")
            
            if not self.ensure_models_available(self.test_models + [self.evaluation_model]):
                raise Exception("Failed to ensure all required models are available")
            
            # Step 3: Create prompt group
            self.create_prompt_group()
            
            # Step 4: Create prompts
            self.create_prompts()
            
            # Step 5: Create prompt versions with different techniques
            self.create_prompt_versions()
              # Step 6: Run test runs for each version
            self.run_test_runs()
            
            # Step 7: Compare prompt versions using A/B testing
            self.compare_prompt_versions()
            
            # Step 8: Generate and save report
            report = self.generate_report()
            report_file = self.save_report(report)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"End-to-end test completed successfully in {total_time:.2f} seconds")
            logger.info(f"Report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            raise

def main():
    """Main function to run the end-to-end test"""
    parser = argparse.ArgumentParser(description='Run end-to-end test for PromptVs API')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--config', default='test_prompts_config.json', help='Test configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test runner
    runner = E2ETestRunner(api_base_url=args.api_url, config_file=args.config)
    
    try:
        # Run the test
        asyncio.run(runner.run_full_test())
        print("\n✅ End-to-end test completed successfully!")
        print(f"Check the log file and report for detailed results.")
        
    except Exception as e:
        print(f"\n❌ End-to-end test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
