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
        """Collect ELO ratings for all test runs"""
        logger.info(f"Collecting ELO ratings for {len(test_run_ids)} test runs...")
        
        elo_ratings = {}
        successful_requests = 0
        failed_requests = 0
        
        for test_run_id in test_run_ids:
            try:
                response = self._make_request('GET', f'/ab-testing/elo-rating/{test_run_id}')
                elo_data = response.json()
                elo_ratings[test_run_id] = elo_data
                successful_requests += 1
            except Exception as e:
                logger.warning(f"Failed to get ELO rating for test run {test_run_id}: {e}")
                failed_requests += 1
                # Set default ELO scores for missing data
                elo_ratings[test_run_id] = {
                    'test_run_id': test_run_id,
                    'elo_score': 1000,
                    'version_elo_score': 1000,
                    'global_elo_score': 1000,
                    'missing_data': True
                }
        
        logger.info(f"ELO ratings collected: {successful_requests} successful, {failed_requests} failed")
        return elo_ratings
    
    def collect_test_run_metadata(self, test_run_ids: List[str]) -> Dict[str, Dict]:
        """Collect test run metadata"""
        logger.info(f"Collecting test run metadata for {len(test_run_ids)} test runs...")
        
        test_runs = {}
        successful_requests = 0
        failed_requests = 0
        
        for test_run_id in test_run_ids:
            try:
                response = self._make_request('GET', f'/test-runs/{test_run_id}')
                test_run_data = response.json()
                test_runs[test_run_id] = test_run_data
                successful_requests += 1
            except Exception as e:
                logger.warning(f"Failed to get test run metadata for {test_run_id}: {e}")
                failed_requests += 1
        
        logger.info(f"Test run metadata collected: {successful_requests} successful, {failed_requests} failed")
        return test_runs
    
    def collect_prompt_version_metadata(self, version_ids: List[str]) -> Dict[str, Dict]:
        """Collect prompt version metadata"""
        logger.info(f"Collecting prompt version metadata for {len(version_ids)} versions...")
        
        prompt_versions = {}
        successful_requests = 0
        failed_requests = 0
        
        for version_id in version_ids:
            try:
                response = self._make_request('GET', f'/prompt-versions/{version_id}')
                version_data = response.json()
                prompt_versions[version_id] = version_data
                successful_requests += 1
            except Exception as e:
                logger.warning(f"Failed to get prompt version metadata for {version_id}: {e}")
                failed_requests += 1
        
        logger.info(f"Prompt version metadata collected: {successful_requests} successful, {failed_requests} failed")
        return prompt_versions
    
    def collect_comparison_results(self, test_run_ids: List[str]) -> List[Dict]:
        """Collect comparison results for test runs"""
        logger.info(f"Collecting comparison results for {len(test_run_ids)} test runs...")
        
        all_comparisons = []
        
        for test_run_id in test_run_ids:
            try:
                response = self._make_request('GET', f'/ab-testing/comparisons/{test_run_id}')
                comparisons = response.json()
                if isinstance(comparisons, list):
                    all_comparisons.extend(comparisons)
                else:
                    all_comparisons.append(comparisons)
            except Exception as e:
                logger.warning(f"Failed to get comparisons for test run {test_run_id}: {e}")
        
        logger.info(f"Collected {len(all_comparisons)} comparison results")
        return all_comparisons
