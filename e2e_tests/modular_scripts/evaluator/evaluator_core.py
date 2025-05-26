#!/usr/bin/env python3
"""
Evaluator Core
=============

Main evaluator class that orchestrates the entire A/B testing evaluation process.
"""

import json
import logging
import time
from typing import Dict, List
from .api_manager import APIManager
from .comparison_engine import ComparisonEngine
from .analysis_engine import AnalysisEngine

logger = logging.getLogger(__name__)


class ABTestingEvaluator:
    """Main A/B Testing Evaluator that coordinates all components"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", input_file: str = "../output/test_runs.json", 
                 max_comparisons_per_version_pair: int = 3, enable_model_comparison: bool = True):
        self.input_file = input_file
        self.max_comparisons_per_version_pair = max_comparisons_per_version_pair
        self.enable_model_comparison = enable_model_comparison
        
        # Initialize components
        self.api_manager = APIManager(api_base_url)
        self.comparison_engine = ComparisonEngine(
            self.api_manager, max_comparisons_per_version_pair, enable_model_comparison
        )
        self.analysis_engine = AnalysisEngine(self.api_manager)
        
        # Data storage
        self.prompt_group_id = None
        self.prompt_ids = []
        self.prompt_versions = {}
        self.test_runs = {}
        self.test_models = []
        
        # Load input data
        self.load_input_data()
    
    def load_input_data(self) -> Dict:
        """Load test run data from input file"""
        logger.info(f"Loading test run data from {self.input_file}")
        
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
        except Exception as e:
            logger.error(f"Failed to load input data: {e}")
            raise
    
    def run(self) -> Dict:
        """Run the complete A/B testing evaluation process"""
        logger.info("Starting A/B testing evaluations...")
        start_time = time.time()
        
        try:
            # Step 1: Start API server
            if not self.api_manager.start_api_server():
                raise Exception("Failed to start API server")
            
            # Step 2: Start Ollama service
            if not self.api_manager.start_ollama_service():
                raise Exception("Failed to start Ollama service")
            
            # Step 3: Stop all models to conserve resources
            self.api_manager.stop_all_models()
            
            # Step 4: Ensure evaluation model is available
            if not self.api_manager.ensure_evaluation_model_available():
                raise Exception(f"Failed to ensure {self.api_manager.evaluation_model} is available")
            
            # Step 5: Perform model comparisons within versions (if enabled)
            if self.enable_model_comparison:
                logger.info("Starting model comparison within versions...")
                self.comparison_engine.compare_models_within_versions(self.test_runs, self.prompt_versions)
                logger.info(f"Completed {len(self.comparison_engine.model_comparison_results)} model comparisons")
            
            # Step 6: Perform A/B testing comparisons between versions
            logger.info("Starting prompt version comparisons...")
            self.comparison_engine.compare_prompt_versions(self.prompt_ids, self.prompt_versions, self.test_runs)
            logger.info(f"Completed {len(self.comparison_engine.comparison_results)} version comparisons")
            
            # Step 7: Generate and save report
            report = self.analysis_engine.generate_report(
                self.prompt_ids, self.prompt_versions, self.test_runs, self.test_models,
                self.comparison_engine.comparison_results, self.comparison_engine.model_comparison_results,
                self.max_comparisons_per_version_pair
            )
            report_file = self.analysis_engine.save_report(report)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"A/B testing evaluations completed successfully in {total_time:.2f} seconds")
            logger.info(f"Report saved to: {report_file}")
            
            # Stop evaluation model to free resources
            self.api_manager.stop_all_models()
            
            # Print summary
            self.analysis_engine.print_summary(report)
            
            summary = {
                'success': True,
                'total_version_comparisons': len(self.comparison_engine.comparison_results),
                'total_model_comparisons': len(self.comparison_engine.model_comparison_results) if self.enable_model_comparison else 0,
                'total_comparisons': len(self.comparison_engine.comparison_results) + (len(self.comparison_engine.model_comparison_results) if self.enable_model_comparison else 0),
                'execution_time': total_time,
                'report_file': report_file,
                'report': report
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"A/B testing evaluations failed: {e}")
            raise
