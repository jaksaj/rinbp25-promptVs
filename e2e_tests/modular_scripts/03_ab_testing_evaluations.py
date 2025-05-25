#!/usr/bin/env python3
"""
Script 3: A/B Testing Evaluations
=================================

This script handles the evaluation phase:
1. Load test runs from previous script
2. Start API server and Ollama
3. Start evaluation model
4. Perform A/B testing comparisons
5. Generate comprehensive report

Usage:
    python 03_ab_testing_evaluations.py

Input:
    - test_runs.json: Data from script 2

Output:
    - evaluation_report.json: Comprehensive test report
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
        logging.FileHandler(f'../logs/03_evaluations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ABTestingEvaluator:
    """Performs A/B testing evaluations and generates reports"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", input_file: str = "../output/test_runs.json"):
        self.api_base_url = api_base_url.rstrip('/')
        self.input_file = input_file
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Data from previous scripts
        self.prompt_group_id: Optional[str] = None
        self.prompt_ids: List[str] = []
        self.prompt_versions: Dict[str, List[str]] = {}
        self.test_runs: Dict[str, List[str]] = {}
        self.test_models: List[str] = []
        
        # Evaluation data
        self.comparison_results: List[Dict] = []
        
        # Load data from previous script
        self.load_input_data()
        
        # Model for evaluation (larger model for better evaluation quality)
        self.evaluation_model = "gemma3:4b"
        
    def load_input_data(self) -> Dict:
        """Load data from previous script"""
        try:
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            
            self.prompt_group_id = data['prompt_group_id']
            self.prompt_ids = data['prompt_ids']
            self.prompt_versions = data['prompt_versions']
            self.test_runs = data['test_runs']
            self.test_models = data['test_models']
            
            total_test_runs = sum(len(runs) for runs in self.test_runs.values())
            
            logger.info(f"Loaded data from {self.input_file}")
            logger.info(f"Prompt group: {self.prompt_group_id}")
            logger.info(f"Prompts: {len(self.prompt_ids)}")
            logger.info(f"Versions: {sum(len(versions) for versions in self.prompt_versions.values())}")
            logger.info(f"Test runs: {total_test_runs}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"Input file {self.input_file} not found. Please run script 2 first.")
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
    
    def ensure_evaluation_model_available(self) -> bool:
        """Ensure evaluation model is pulled and running"""
        logger.info(f"Ensuring evaluation model is available: {self.evaluation_model}")
        
        try:
            # Get current model status
            response = self._make_request('GET', '/api/models/status')
            model_statuses = response.json()
            
            # Check if model needs to be pulled
            status_dict = {status['name']: status for status in model_statuses}
            
            if self.evaluation_model not in status_dict or not status_dict[self.evaluation_model]['is_pulled']:
                logger.info(f"Pulling evaluation model: {self.evaluation_model}")
                response = self._make_request('POST', '/api/models/pull', {'model_name': self.evaluation_model})
                logger.info(f"Successfully pulled model: {self.evaluation_model}")
            
            # Start the model
            logger.info(f"Starting evaluation model: {self.evaluation_model}")
            response = self._make_request('POST', '/api/models/start', {'model_name': self.evaluation_model})
            logger.info(f"Successfully started model: {self.evaluation_model}")
            
            # Verify model is running
            time.sleep(3)
            response = self._make_request('GET', '/api/models/running')
            running_models = [m['name'] for m in response.json()['running_models']]
            
            if self.evaluation_model in running_models:
                logger.info(f"Evaluation model {self.evaluation_model} is now running")
                return True
            else:
                logger.error(f"Evaluation model {self.evaluation_model} failed to start")
                return False
            
        except Exception as e:
            logger.error(f"Failed to ensure evaluation model is available: {e}")
            return False
    
    def compare_prompt_versions(self) -> List[Dict]:
        """Compare prompt versions within each prompt using A/B testing"""
        logger.info("Comparing prompt versions using A/B testing...")
        
        total_comparisons = 0
        for prompt_id, version_ids in self.prompt_versions.items():
            if len(version_ids) >= 2:
                for i in range(len(version_ids)):
                    for j in range(i+1, len(version_ids)):
                        version_test_runs1 = self.test_runs.get(version_ids[i], [])
                        version_test_runs2 = self.test_runs.get(version_ids[j], [])
                        total_comparisons += len(version_test_runs1) * len(version_test_runs2)
        
        logger.info(f"Total comparisons to perform: {total_comparisons}")
        current_comparison = 0
        
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
                            current_comparison += 1
                            logger.info(f"Comparison {current_comparison}/{total_comparisons}: {test_run_id1} vs {test_run_id2}")
                            
                            try:
                                comparison_data = {
                                    'test_run_id1': test_run_id1,
                                    'test_run_id2': test_run_id2,
                                    'compare_within_version': False
                                }
                                
                                response = self._make_request(
                                    'POST',
                                    '/api/ab-testing/compare',
                                    comparison_data,
                                    timeout=180  # Longer timeout for evaluation
                                )
                                
                                comparison_result = response.json()
                                comparison_result['prompt_id'] = prompt_id
                                comparison_result['version_id1'] = version_id1
                                comparison_result['version_id2'] = version_id2
                                self.comparison_results.append(comparison_result)
                                
                                winner = comparison_result.get('winner_test_run_id', 'unknown')
                                logger.info(f"Comparison completed. Winner: {winner}")
                                
                            except Exception as e:
                                logger.error(f"Failed to compare test runs {test_run_id1} vs {test_run_id2}: {e}")
                                # Continue with other comparisons
        
        logger.info(f"Completed {len(self.comparison_results)} comparisons")
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
                    'win_rate': version_wins[best_version_id] / len(prompt_comparisons) if prompt_comparisons else 0,
                    'all_version_wins': version_wins
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
        """Save evaluation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../output/evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filename}")
        return filename
    
    def print_summary(self, report: Dict):
        """Print a human-readable summary of the results"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        summary = report['test_summary']
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Prompts: {summary['total_prompts']}")
        print(f"Total Versions: {summary['total_versions']}")
        print(f"Total Test Runs: {summary['total_test_runs']}")
        print(f"Total Comparisons: {summary['total_comparisons']}")
        print(f"Models Tested: {', '.join(summary['models_tested'])}")
        print(f"Evaluation Model: {summary['evaluation_model']}")
        
        print("\nTECHNIQUE PERFORMANCE:")
        print("-" * 40)
        technique_perf = report['technique_performance']
        for technique, perf in technique_perf.items():
            win_rate = perf['win_rate'] * 100
            print(f"{technique:15}: {perf['wins']:3}/{perf['total_comparisons']:3} wins ({win_rate:5.1f}%)")
        
        print("\nBEST VERSIONS PER PROMPT:")
        print("-" * 40)
        best_versions = report['best_versions_per_prompt']
        for prompt_id, best_info in best_versions.items():
            win_rate = best_info['win_rate'] * 100
            print(f"Prompt {prompt_id}: Version {best_info['best_version_id']} ({best_info['wins']}/{best_info['total_comparisons']} wins, {win_rate:.1f}%)")
        
        print("="*80)
    
    def run(self) -> Dict:
        """Run the complete A/B testing evaluation process"""
        logger.info("Starting A/B testing evaluations...")
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
            
            # Step 4: Ensure evaluation model is available
            if not self.ensure_evaluation_model_available():
                raise Exception(f"Failed to ensure {self.evaluation_model} is available")
            
            # Step 5: Perform A/B testing comparisons
            self.compare_prompt_versions()
            
            # Step 6: Generate and save report
            report = self.generate_report()
            report_file = self.save_report(report)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"A/B testing evaluations completed successfully in {total_time:.2f} seconds")
            logger.info(f"Report saved to: {report_file}")
            
            # Stop evaluation model to free resources
            self.stop_all_models()
            
            # Print summary
            self.print_summary(report)
            
            summary = {
                'success': True,
                'total_comparisons': len(self.comparison_results),
                'execution_time': total_time,
                'report_file': report_file,
                'report': report
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"A/B testing evaluations failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Perform A/B testing evaluations')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--input', default='../output/test_runs.json', help='Input file from script 2')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = ABTestingEvaluator(api_base_url=args.api_url, input_file=args.input)
    
    try:
        summary = evaluator.run()
        print("\n✅ A/B testing evaluations completed successfully!")
        print(f"Performed {summary['total_comparisons']} comparisons")
        print(f"Report saved to: {summary['report_file']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ A/B testing evaluations failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
