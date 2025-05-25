#!/usr/bin/env python3
"""
Modular E2E Test Runner Script
==============================

This script sets up the environment and runs the modular end-to-end tests.
The new modular approach consists of three sequential scripts:
1. Create prompts and versions
2. Create test runs  
3. Perform A/B testing evaluations
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
    print("🚀 Starting Modular E2E Tests for PromptVs API")
    print("=" * 50)
    
    # Check dependencies
    if not check_python_dependencies():
        sys.exit(1)
    
    # Start API server
    if not start_api_server():
        logger.error("Cannot proceed without API server")
        sys.exit(1)    # Run the modular end-to-end test
    try:
        print("\n🔄 Running modular E2E tests...")
        
        # Run the modular test runner script
        modular_script_path = os.path.join(os.path.dirname(__file__), 'modular_scripts', 'run_all_scripts.py')
        
        result = subprocess.run([
            sys.executable, modular_script_path
        ], cwd=os.path.dirname(modular_script_path), capture_output=True, text=True)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✅ All tests completed successfully!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
