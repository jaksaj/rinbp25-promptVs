#!/usr/bin/env python3
"""
ELO Data Collector
==================

Handles data collection from the API for ELO rating analysis.
"""

import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)


class EloDataCollector:
    """Collects ELO rating data and metadata from the API"""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
    def _make_request(self, method: str, endpoint: str, params: Dict = None, timeout: int = 60) -> requests.Response:
        """Make HTTP request to API with error handling"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise
    
    def collect_all_elo_ratings(self, test_run_ids: List[str]) -> Dict[str, Dict]:
        """Collect ELO ratings for all test runs using the new bulk endpoint"""
        logger.info(f"Collecting ELO ratings for {len(test_run_ids)} test runs (bulk)...")
        elo_ratings = {}
        try:
            response = self.session.post(
                f"{self.api_base_url}/ab-testing/elo-ratings/bulk",
                json={"test_run_ids": test_run_ids},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                test_run_id = result.get("test_run_id")
                if test_run_id:
                    elo_ratings[test_run_id] = result
            # Fill in missing test_run_ids with default values
            for test_run_id in test_run_ids:
                if test_run_id not in elo_ratings:
                    elo_ratings[test_run_id] = {
                        'test_run_id': test_run_id,
                        'elo_score': 1000,
                        'version_elo_score': 1000,
                        'global_elo_score': 1000,
                        'missing_data': True
                    }
            logger.info(f"ELO ratings collected (bulk): {len(elo_ratings)} total")
            return elo_ratings
        except Exception as e:
            logger.error(f"Bulk API request for ELO ratings failed: {e}")
            # Fallback: mark all as missing
            for test_run_id in test_run_ids:
                elo_ratings[test_run_id] = {
                    'test_run_id': test_run_id,
                    'elo_score': 1000,
                    'version_elo_score': 1000,
                    'global_elo_score': 1000,
                    'missing_data': True
                }
            return elo_ratings
    
    def collect_test_run_metadata(self, test_run_ids: List[str]) -> Dict[str, Dict]:
        """Collect test run metadata using a bulk endpoint"""
        logger.info(f"Collecting test run metadata for {len(test_run_ids)} test runs (bulk)...")
        test_runs = {}
        try:
            response = self.session.post(
                f"{self.api_base_url}/batch/test-runs/bulk",
                json={"test_run_ids": test_run_ids},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                test_run_id = result.get("id")
                if test_run_id:
                    test_runs[test_run_id] = result
            # Fill in missing test_run_ids with empty dicts
            for test_run_id in test_run_ids:
                if test_run_id not in test_runs:
                    test_runs[test_run_id] = {"id": test_run_id, "missing_data": True}
            logger.info(f"Test run metadata collected (bulk): {len(test_runs)} total")
            return test_runs
        except Exception as e:
            logger.error(f"Bulk API request for test run metadata failed: {e}")
            # Fallback: mark all as missing
            for test_run_id in test_run_ids:
                test_runs[test_run_id] = {"id": test_run_id, "missing_data": True}
            return test_runs
    
    def collect_prompt_version_metadata(self, version_ids: List[str]) -> Dict[str, Dict]:
        """Collect prompt version metadata using a bulk endpoint"""
        logger.info(f"Collecting prompt version metadata for {len(version_ids)} versions (bulk)...")
        prompt_versions = {}
        try:
            response = self.session.post(
                f"{self.api_base_url}/prompt-versions/bulk",
                json={"version_ids": version_ids},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                version_id = result.get("id")
                if version_id:
                    prompt_versions[version_id] = result
            # Fill in missing version_ids with empty dicts
            for version_id in version_ids:
                if version_id not in prompt_versions:
                    prompt_versions[version_id] = {"id": version_id, "missing_data": True}
            logger.info(f"Prompt version metadata collected (bulk): {len(prompt_versions)} total")
            return prompt_versions
        except Exception as e:
            logger.error(f"Bulk API request for prompt version metadata failed: {e}")
            # Fallback: mark all as missing
            for version_id in version_ids:
                prompt_versions[version_id] = {"id": version_id, "missing_data": True}
            return prompt_versions
    
    def collect_comparison_results(self, test_run_ids: List[str]) -> List[Dict]:
        """Collect comparison results for test runs using a bulk endpoint"""
        logger.info(f"Collecting comparison results for {len(test_run_ids)} test runs (bulk)...")
        try:
            response = self.session.post(
                f"{self.api_base_url}/ab-testing/comparisons/bulk",
                json={"test_run_ids": test_run_ids},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            logger.info(f"Collected {len(results)} comparison results (bulk)")
            return results
        except Exception as e:
            logger.error(f"Bulk API request for comparison results failed: {e}")
            # Fallback: try to collect individually
            all_comparisons = []
            for test_run_id in test_run_ids:
                try:
                    response = self._make_request('GET', f'/ab-testing/comparisons/{test_run_id}')
                    comparisons = response.json()
                    if isinstance(comparisons, list):
                        all_comparisons.extend(comparisons)
                    else:
                        all_comparisons.append(comparisons)
                except Exception as e2:
                    logger.warning(f"Failed to get comparisons for test run {test_run_id}: {e2}")
            logger.info(f"Collected {len(all_comparisons)} comparison results (fallback)")
            return all_comparisons
