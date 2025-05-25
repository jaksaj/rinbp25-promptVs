#!/usr/bin/env python3
"""
Debug E2E Test Script
===================

This script runs a simplified version of the E2E test for debugging.
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"

def test_single_prompt():
    """Test creating and running a single prompt to debug issues"""
    
    # Step 1: Check API health
    logger.info("1. Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            logger.info("✅ API is healthy")
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API health check failed: {e}")
        return False
    
    # Step 2: Check models
    logger.info("2. Checking available models...")
    try:
        response = requests.get(f"{API_BASE}/api/models/running")
        running_models = response.json()['running_models']
        logger.info(f"✅ Running models: {[m['name'] for m in running_models]}")
        
        if not running_models:
            logger.error("❌ No models are running!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to check models: {e}")
        return False
    
    # Step 3: Create prompt group
    logger.info("3. Creating prompt group...")
    try:
        group_data = {
            "name": "Debug Test Group",
            "description": "Debug test group",
            "tags": ["debug"]
        }
        response = requests.post(f"{API_BASE}/api/prompt-groups", json=group_data)
        group_id = response.json()['id']
        logger.info(f"✅ Created group: {group_id}")
    except Exception as e:
        logger.error(f"❌ Failed to create group: {e}")
        return False
    
    # Step 4: Create simple prompt
    logger.info("4. Creating simple prompt...")
    try:
        prompt_data = {
            "prompt_group_id": group_id,
            "content": "What is 2 + 2? Answer with just the number.",
            "name": "Simple Math",
            "description": "Very simple math test",
            "expected_solution": "4",
            "tags": ["math", "simple"]
        }
        response = requests.post(f"{API_BASE}/api/prompts", json=prompt_data)
        prompt_id = response.json()['id']
        logger.info(f"✅ Created prompt: {prompt_id}")
    except Exception as e:
        logger.error(f"❌ Failed to create prompt: {e}")
        return False
    
    # Step 5: Create simple version (no technique)
    logger.info("5. Creating prompt version...")
    try:
        version_data = {
            "prompt_id": prompt_id,
            "content": "What is 2 + 2? Answer with just the number.",
            "version": "v1_simple",
            "notes": "Simple version for debugging"
        }
        response = requests.post(f"{API_BASE}/api/prompt-versions", json=version_data)
        version_id = response.json()['id']
        logger.info(f"✅ Created version: {version_id}")
    except Exception as e:
        logger.error(f"❌ Failed to create version: {e}")
        return False
    
    # Step 6: Test with smallest/fastest model
    logger.info("6. Testing prompt with model...")
    try:
        model_name = running_models[0]['name']  # Use first available model
        logger.info(f"Testing with model: {model_name}")
        
        # Add timeout and detailed logging
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/test-prompt-and-save",
            params={
                'version_id': version_id,
                'model_name': model_name
            },
            timeout=60  # 60 second timeout
        )
        end_time = time.time()
        
        result = response.json()
        test_run_id = result['run_id']
        output = result.get('result', 'No result')
        
        logger.info(f"✅ Test completed in {end_time - start_time:.2f}s")
        logger.info(f"✅ Test run ID: {test_run_id}")
        logger.info(f"✅ Model output: {output}")
        
    except requests.exceptions.Timeout:
        logger.error("❌ Request timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to test prompt: {e}")
        return False
    
    logger.info("🎉 Debug test completed successfully!")
    return True

if __name__ == "__main__":
    print("🔍 Debug E2E Test")
    print("=" * 30)
    
    success = test_single_prompt()
    
    if success:
        print("\n✅ Debug test passed! You can now try the full E2E test.")
    else:
        print("\n❌ Debug test failed. Check the logs above for issues.")
