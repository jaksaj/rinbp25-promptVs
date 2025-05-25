#!/usr/bin/env python3
"""
E2E Test Runner Script
======================

This script sets up the environment and runs the end-to-end test.
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = ['requests', 'asyncio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def start_api_server():
    """Start the API server if not already running"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            logger.info("API server is already running")
            return True
    except:
        pass
    
    logger.info("Starting API server...")
    try:
    # Start the API server in the background
        api_process = subprocess.Popen([
            sys.executable, "../run.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        time.sleep(10)
        
        # Check if it's running
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                logger.info("API server started successfully")
                return True
        except:
            pass
        
        logger.error("Failed to start API server")
        return False
        
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return False

def main():
    """Main runner function"""
    print("🚀 Starting E2E Test for PromptVs API")
    print("=" * 50)
    
    # Check dependencies
    if not check_python_dependencies():
        sys.exit(1)
    
    # Start API server
    if not start_api_server():
        logger.error("Cannot proceed without API server")
        sys.exit(1)
    
    # Run the end-to-end test
    try:
        import e2e_test
        import asyncio
        
        runner = e2e_test.E2ETestRunner()
        asyncio.run(runner.run_full_test())
        
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
