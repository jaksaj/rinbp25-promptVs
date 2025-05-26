#!/usr/bin/env python3
"""
Comparison Engine
================

Handles A/B testing comparisons between prompt versions and models.
"""

import logging
from typing import List, Dict, Any
from .sampling_strategies import SamplingStrategies

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """Handles A/B testing comparisons for prompt versions and models"""
    
    def __init__(self, api_manager, max_comparisons_per_version_pair: int = 3, 
                 enable_model_comparison: bool = True):
        self.api_manager = api_manager
        self.max_comparisons_per_version_pair = max_comparisons_per_version_pair
        self.enable_model_comparison = enable_model_comparison
        self.comparison_results = []
        self.model_comparison_results = []
    
    def compare_models_within_versions(self, test_runs: Dict, prompt_versions: Dict) -> List[Dict]:
        """Compare different LLM models for the same prompt version"""
        if not self.enable_model_comparison:
            logger.info("Model comparison disabled, skipping...")
            return []
        
        logger.info("Comparing models within each version...")
        
        total_model_comparisons = 0
        for version_id, test_run_list in test_runs.items():
            runs_by_model = SamplingStrategies.group_test_runs_by_model(
                test_run_list, self.api_manager.get_test_run_model
            )
            if len(runs_by_model) >= 2:
                model_pairs = len(runs_by_model) * (len(runs_by_model) - 1) // 2
                total_model_comparisons += model_pairs * self.max_comparisons_per_version_pair
        
        logger.info(f"Total model comparisons to perform: {total_model_comparisons}")
        current_comparison = 0
        
        for version_id, test_run_list in test_runs.items():
            runs_by_model = SamplingStrategies.group_test_runs_by_model(
                test_run_list, self.api_manager.get_test_run_model
            )
            
            if len(runs_by_model) < 2:
                logger.debug(f"Version {version_id} has less than 2 models, skipping model comparison")
                continue
            
            # Compare each model pair within this version
            models = list(runs_by_model.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    # Use smart sampling to select representative test run pairs
                    selected_pairs = SamplingStrategies.smart_sampling_strategy(
                        runs_by_model[model1], runs_by_model[model2], 
                        self.max_comparisons_per_version_pair,
                        self.api_manager.get_test_run_model
                    )
                    
                    logger.info(f"Comparing {len(selected_pairs)} pairs between {model1} and {model2} for version {version_id}")
                    
                    for test_run_id1, test_run_id2 in selected_pairs:
                        current_comparison += 1
                        logger.info(f"Model comparison {current_comparison}/{total_model_comparisons}: {test_run_id1} vs {test_run_id2}")
                        
                        try:
                            comparison_data = {
                                'test_run_id1': test_run_id1,
                                'test_run_id2': test_run_id2,
                                'compare_within_version': True,
                                'comparison_type': 'model_comparison'
                            }
                            
                            response = self.api_manager._make_request(
                                'POST',
                                '/api/ab-testing/compare',
                                comparison_data,
                                timeout=180
                            )
                            
                            comparison_result = response.json()
                            comparison_result['version_id'] = version_id
                            comparison_result['model1'] = model1
                            comparison_result['model2'] = model2
                            comparison_result['comparison_type'] = 'model_comparison'
                            self.model_comparison_results.append(comparison_result)
                            
                            winner = comparison_result.get('winner_test_run_id', 'unknown')
                            logger.info(f"Model comparison completed. Winner: {winner}")
                            
                        except Exception as e:
                            logger.error(f"Failed to compare test runs {test_run_id1} vs {test_run_id2}: {e}")
                            continue
        
        logger.info(f"Completed {len(self.model_comparison_results)} model comparisons")
        return self.model_comparison_results
    
    def compare_prompt_versions(self, prompt_ids: List[str], prompt_versions: Dict, test_runs: Dict) -> List[Dict]:
        """Compare prompt versions within each prompt using A/B testing with optimized sampling"""
        logger.info("Comparing prompt versions using A/B testing with optimized sampling...")
        
        # Calculate budget using new method
        budget = SamplingStrategies.calculate_comparison_budget(
            prompt_versions, test_runs, self.max_comparisons_per_version_pair,
            self.enable_model_comparison, self.api_manager.get_test_run_model
        )
        current_comparison = 0
        
        for prompt_id in prompt_ids:
            version_ids = prompt_versions.get(prompt_id, [])
            if len(version_ids) < 2:
                logger.warning(f"Prompt {prompt_id} has less than 2 versions, skipping comparison")
                continue
            
            # Get test runs for each version
            version_test_runs = {}
            for version_id in version_ids:
                version_test_runs[version_id] = test_runs.get(version_id, [])
            
            # Compare test runs between versions with optimized sampling
            for i, version_id1 in enumerate(version_ids):
                for version_id2 in version_ids[i+1:]:
                    test_runs1 = version_test_runs[version_id1]
                    test_runs2 = version_test_runs[version_id2]
                    
                    # Use smart sampling strategy instead of exhaustive comparison
                    selected_pairs = SamplingStrategies.smart_sampling_strategy(
                        test_runs1, test_runs2, self.max_comparisons_per_version_pair,
                        self.api_manager.get_test_run_model
                    )
                    
                    logger.info(f"Comparing {len(selected_pairs)} selected pairs between versions {version_id1} and {version_id2}")
                    
                    # Compare selected test run pairs
                    for test_run_id1, test_run_id2 in selected_pairs:
                        current_comparison += 1
                        logger.info(f"Version comparison {current_comparison}/{budget['version_comparisons']}: {test_run_id1} vs {test_run_id2}")
                        
                        try:
                            comparison_data = {
                                'test_run_id1': test_run_id1,
                                'test_run_id2': test_run_id2,
                                'compare_within_version': False,
                                'comparison_type': 'version_comparison'
                            }
                            
                            response = self.api_manager._make_request(
                                'POST',
                                '/api/ab-testing/compare',
                                comparison_data,
                                timeout=180  # Longer timeout for evaluation
                            )
                            
                            comparison_result = response.json()
                            comparison_result['prompt_id'] = prompt_id
                            comparison_result['version_id1'] = version_id1
                            comparison_result['version_id2'] = version_id2
                            comparison_result['comparison_type'] = 'version_comparison'
                            self.comparison_results.append(comparison_result)
                            winner = comparison_result.get('winner_test_run_id', 'unknown')
                            logger.info(f"Version comparison completed. Winner: {winner}")
                            
                        except Exception as e:
                            logger.error(f"Failed to compare test runs {test_run_id1} vs {test_run_id2}: {e}")
                            # Continue with other comparisons
        
        logger.info(f"Completed {len(self.comparison_results)} version comparisons")
        return self.comparison_results
