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
        """Compare different LLM models for the same prompt version using async bulk endpoint."""
        if not self.enable_model_comparison:
            logger.info("Model comparison disabled, skipping...")
            return []
        logger.info("Comparing models within each version (BULK async)...")
        all_pairs = []
        pair_metadata = []
        for version_id, test_run_list in test_runs.items():
            runs_by_model = SamplingStrategies.group_test_runs_by_model(
                test_run_list, self.api_manager.get_test_run_model
            )
            if len(runs_by_model) < 2:
                logger.debug(f"Version {version_id} has less than 2 models, skipping model comparison")
                continue
            models = list(runs_by_model.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    selected_pairs = SamplingStrategies.smart_sampling_strategy(
                        [run for run in test_run_list if self.api_manager.get_test_run_model(run) == model1],
                        [run for run in test_run_list if self.api_manager.get_test_run_model(run) == model2],
                        self.max_comparisons_per_version_pair,
                        self.api_manager.get_test_run_model
                    )
                    for test_run_id1, test_run_id2 in selected_pairs:
                        all_pairs.append({
                            "test_run_id1": test_run_id1,
                            "test_run_id2": test_run_id2,
                            "compare_within_version": True
                        })
                        pair_metadata.append({
                            "version_id": version_id,
                            "model1": model1,
                            "model2": model2
                        })
        if not all_pairs:
            logger.info("No model pairs to compare.")
            return []
        logger.info(f"Sending {len(all_pairs)} model pairs for async bulk comparison...")
        # Submit job
        response = self.api_manager._make_request(
            'POST',
            '/api/ab-testing/compare/bulk',
            all_pairs,
            timeout=30
        )
        job_info = response.json()
        job_id = job_info.get("job_id")
        if not job_id:
            logger.error(f"No job_id returned from bulk compare endpoint: {job_info}")
            return []
        logger.info(f"Bulk comparison job submitted. job_id={job_id}")
        # Poll for completion
        import time
        poll_interval = 5
        max_wait = 3600  # 1 hour max
        waited = 0
        while True:
            status_resp = self.api_manager._make_request(
                'GET',
                f'/api/ab-testing/compare/bulk/status/{job_id}',
                None,
                timeout=30
            )
            status = status_resp.json()
            if status.get("status") == "completed":
                results = status.get("results", [])
                logger.info(f"Bulk comparison job completed with {len(results)} results.")
                for result, meta in zip(results, pair_metadata):
                    result['version_id'] = meta['version_id']
                    result['model1'] = meta['model1']
                    result['model2'] = meta['model2']
                    result['comparison_type'] = 'model_comparison'
                    self.model_comparison_results.append(result)
                    winner = result.get('winner_test_run_id', 'unknown')
                    logger.info(f"Model comparison completed. Winner: {winner}")
                break
            elif status.get("status") == "failed":
                logger.error(f"Bulk comparison job failed: {status.get('error')}")
                break
            else:
                logger.info(f"Bulk comparison job status: {status.get('status')}, waiting {poll_interval}s...")
                time.sleep(poll_interval)
                waited += poll_interval
                if waited > max_wait:
                    logger.error("Bulk comparison job timed out.")
                    break
        logger.info(f"Completed {len(self.model_comparison_results)} model comparisons")
        return self.model_comparison_results
    
    def compare_prompt_versions(self, prompt_ids: List[str], prompt_versions: Dict, test_runs: Dict) -> List[Dict]:
        """Compare prompt versions within each prompt using async bulk endpoint for speed."""
        logger.info("Comparing prompt versions using async bulk endpoint...")
        # Calculate budget using new method
        budget = SamplingStrategies.calculate_comparison_budget(
            prompt_versions, test_runs, self.max_comparisons_per_version_pair,
            self.enable_model_comparison, self.api_manager.get_test_run_model
        )
        all_pairs = []
        pair_metadata = []
        for prompt_id in prompt_ids:
            version_ids = prompt_versions.get(prompt_id, [])
            if len(version_ids) < 2:
                logger.warning(f"Prompt {prompt_id} has less than 2 versions, skipping comparison")
                continue
            # Get test runs for each version
            version_test_runs = {version_id: test_runs.get(version_id, []) for version_id in version_ids}
            # Compare test runs between versions with optimized sampling
            for i, version_id1 in enumerate(version_ids):
                for version_id2 in version_ids[i+1:]:
                    test_runs1 = version_test_runs[version_id1]
                    test_runs2 = version_test_runs[version_id2]
                    selected_pairs = SamplingStrategies.smart_sampling_strategy(
                        test_runs1, test_runs2, self.max_comparisons_per_version_pair,
                        self.api_manager.get_test_run_model
                    )
                    logger.info(f"Comparing {len(selected_pairs)} selected pairs between versions {version_id1} and {version_id2}")
                    for test_run_id1, test_run_id2 in selected_pairs:
                        all_pairs.append({
                            'test_run_id1': test_run_id1,
                            'test_run_id2': test_run_id2,
                            'compare_within_version': False
                        })
                        pair_metadata.append({
                            'prompt_id': prompt_id,
                            'version_id1': version_id1,
                            'version_id2': version_id2
                        })
        if not all_pairs:
            logger.info("No version pairs to compare.")
            return []
        logger.info(f"Sending {len(all_pairs)} version pairs for async bulk comparison...")
        # Submit job
        response = self.api_manager._make_request(
            'POST',
            '/api/ab-testing/compare/bulk',
            all_pairs,
            timeout=30
        )
        job_info = response.json()
        job_id = job_info.get("job_id")
        if not job_id:
            logger.error(f"No job_id returned from bulk compare endpoint: {job_info}")
            return []
        logger.info(f"Bulk comparison job submitted. job_id={job_id}")
        # Poll for completion
        import time
        poll_interval = 5
        max_wait = 3600  # 1 hour max
        waited = 0
        while True:
            status_resp = self.api_manager._make_request(
                'GET',
                f'/api/ab-testing/compare/bulk/status/{job_id}',
                None,
                timeout=30
            )
            status = status_resp.json()
            if status.get("status") == "completed":
                results = status.get("results", [])
                logger.info(f"Bulk comparison job completed with {len(results)} results.")
                for result, meta in zip(results, pair_metadata):
                    result['prompt_id'] = meta['prompt_id']
                    result['version_id1'] = meta['version_id1']
                    result['version_id2'] = meta['version_id2']
                    result['comparison_type'] = 'version_comparison'
                    self.comparison_results.append(result)
                    winner = result.get('winner_test_run_id', 'unknown')
                    logger.info(f"Version comparison completed. Winner: {winner}")
                break
            elif status.get("status") == "failed":
                logger.error(f"Bulk comparison job failed: {status.get('error')}")
                break
            else:
                logger.info(f"Bulk comparison job status: {status.get('status')}, waiting {poll_interval}s...")
                time.sleep(poll_interval)
                waited += poll_interval
                if waited > max_wait:
                    logger.error("Bulk comparison job timed out.")
                    break
        logger.info(f"Completed {len(self.comparison_results)} version comparisons")
        return self.comparison_results
