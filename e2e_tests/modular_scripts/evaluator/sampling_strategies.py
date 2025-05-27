#!/usr/bin/env python3
"""
Sampling Strategies
==================

Smart sampling algorithms for A/B testing comparisons.
"""

import random
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class SamplingStrategies:
    """Smart sampling strategies for test run comparisons"""
    
    @staticmethod
    def group_test_runs_by_model(test_run_objs: List[dict], get_model_func) -> Dict[str, List[str]]:
        """Group test runs by their model (expects list of test run dicts)"""
        runs_by_model = {}
        for run in test_run_objs:
            model = get_model_func(run)
            run_id = run['run_id'] if isinstance(run, dict) and 'run_id' in run else run
            if model:
                if model not in runs_by_model:
                    runs_by_model[model] = []
                runs_by_model[model].append(run_id)
        return runs_by_model
    
    @staticmethod
    def smart_sampling_strategy(version_test_runs1: List[dict], version_test_runs2: List[dict], 
                              max_comparisons: int = 3, get_model_func=None) -> List[Tuple[str, str]]:
        """Select representative test run pairs for comparison with smart sampling (expects test run dicts)"""
        
        if not get_model_func:
            all_pairs = [(r1['run_id'], r2['run_id']) for r1 in version_test_runs1 for r2 in version_test_runs2]
            random.shuffle(all_pairs)
            return all_pairs[:max_comparisons]
        
        # Group by model to ensure we compare across different models
        runs1_by_model = SamplingStrategies.group_test_runs_by_model(version_test_runs1, get_model_func)
        runs2_by_model = SamplingStrategies.group_test_runs_by_model(version_test_runs2, get_model_func)
        
        selected_pairs = []
        
        # Priority 1: Cross-model comparisons (most important)
        for model1 in runs1_by_model:
            for model2 in runs2_by_model:
                if model1 != model2 and len(selected_pairs) < max_comparisons:
                    # Select one random pair from each model combination
                    run1 = random.choice(runs1_by_model[model1])
                    run2 = random.choice(runs2_by_model[model2])
                    selected_pairs.append((run1, run2))
        
        # Priority 2: Fill remaining slots with random pairs if needed
        if len(selected_pairs) < max_comparisons:
            all_pairs = [(r1['run_id'], r2['run_id']) for r1 in version_test_runs1 for r2 in version_test_runs2]
            remaining_pairs = [pair for pair in all_pairs if pair not in selected_pairs]
            random.shuffle(remaining_pairs)
            
            needed = max_comparisons - len(selected_pairs)
            selected_pairs.extend(remaining_pairs[:needed])
        
        return selected_pairs
    
    @staticmethod
    def calculate_comparison_budget(prompt_versions: Dict, test_runs: Dict, 
                                  max_comparisons_per_version_pair: int, 
                                  enable_model_comparison: bool, get_model_func) -> Dict[str, int]:
        """Calculate and log the total number of comparisons for different evaluation types"""
        version_pairs = 0
        for prompt_id, version_ids in prompt_versions.items():
            if len(version_ids) >= 2:
                pairs = len(version_ids) * (len(version_ids) - 1) // 2
                version_pairs += pairs
        
        version_comparisons = version_pairs * max_comparisons_per_version_pair
        
        # Calculate model comparisons if enabled
        model_comparisons = 0
        if enable_model_comparison and get_model_func:
            for version_id, test_run_list in test_runs.items():
                runs_by_model = SamplingStrategies.group_test_runs_by_model(test_run_list, get_model_func)
                if len(runs_by_model) >= 2:
                    model_pairs = len(runs_by_model) * (len(runs_by_model) - 1) // 2
                    model_comparisons += model_pairs * max_comparisons_per_version_pair
        
        budget = {
            'version_pairs': version_pairs,
            'version_comparisons': version_comparisons,
            'model_comparisons': model_comparisons,
            'total_comparisons': version_comparisons + model_comparisons
        }
        
        logger.info(f"Comparison Budget Analysis:")
        logger.info(f"- Version pairs: {budget['version_pairs']}")
        logger.info(f"- Version comparisons: {budget['version_comparisons']}")
        logger.info(f"- Model comparisons: {budget['model_comparisons']}")
        logger.info(f"- Total comparisons: {budget['total_comparisons']}")
        logger.info(f"- Comparisons per pair: {max_comparisons_per_version_pair}")
        
        return budget
