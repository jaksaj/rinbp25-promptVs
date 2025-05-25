#!/usr/bin/env python3
"""
Quick API Test Script
====================

This script performs a quick test of the API endpoints to verify everything is working.
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api_health():
    """Test if API is healthy"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data.get('status')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_models():
    """Test model endpoints"""
    print("\n🔍 Testing model endpoints...")
    
    # List available models
    try:
        response = requests.get(f"{API_BASE}/api/models")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ Available models: {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"   - {model}")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    # List running models
    try:
        response = requests.get(f"{API_BASE}/api/models/running")
        if response.status_code == 200:
            running = response.json()['running_models']
            print(f"✅ Running models: {len(running)} models")
            for model in running:
                print(f"   - {model['name']}")
        else:
            print(f"❌ Failed to list running models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error listing running models: {e}")

def test_prompt_group_creation():
    """Test creating a prompt group"""
    print("\n🔍 Testing prompt group creation...")
    
    group_data = {
        "name": "Quick Test Group",
        "description": "Test group for API verification",
        "tags": ["test"]
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/prompt-groups", json=group_data)
        if response.status_code == 200:
            group_id = response.json()['id']
            print(f"✅ Created prompt group: {group_id}")
            return group_id
        else:
            print(f"❌ Failed to create prompt group: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error creating prompt group: {e}")
    
    return None

def test_prompt_creation(group_id):
    """Test creating a prompt"""
    print("\n🔍 Testing prompt creation...")
    
    prompt_data = {
        "prompt_group_id": group_id,
        "content": "What is 2 + 2?",
        "name": "Simple Math",
        "description": "Basic addition test",
        "expected_solution": "4",
        "tags": ["math", "test"]
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/prompts", json=prompt_data)
        if response.status_code == 200:
            prompt_id = response.json()['id']
            print(f"✅ Created prompt: {prompt_id}")
            return prompt_id
        else:
            print(f"❌ Failed to create prompt: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error creating prompt: {e}")
    
    return None

def test_prompting_techniques():
    """Test prompting techniques endpoint"""
    print("\n🔍 Testing prompting techniques...")
    
    try:
        response = requests.get(f"{API_BASE}/api/techniques/techniques")
        if response.status_code == 200:
            techniques = response.json()['techniques']
            print(f"✅ Available techniques: {len(techniques)}")
            for name, info in techniques.items():
                print(f"   - {name}: {info['description']}")
        else:
            print(f"❌ Failed to get techniques: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting techniques: {e}")

def main():
    """Run quick API tests"""
    print("🚀 Quick API Test")
    print("=" * 40)
    
    # Test API health
    if not test_api_health():
        print("\n❌ API is not healthy. Please start the server with 'python run.py'")
        return
    
    # Test models
    test_models()
    
    # Test prompting techniques
    test_prompting_techniques()
    
    # Test prompt group and prompt creation
    group_id = test_prompt_group_creation()
    if group_id:
        prompt_id = test_prompt_creation(group_id)
        if prompt_id:
            print(f"\n✅ Basic API functionality verified!")
            print(f"Group ID: {group_id}")
            print(f"Prompt ID: {prompt_id}")
    
    print("\n🎉 Quick test completed!")
    print("You can now run the full E2E test with: python e2e_test.py")

if __name__ == "__main__":
    main()
