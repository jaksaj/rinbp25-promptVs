"""
Script for testing and evaluating different prompt variations across multiple models.

This script:
1. Imports questions and expected answers from a CSV file
2. Imports prompt variation templates from a CSV file
3. Creates a new prompt group and prompts
4. Generates prompt variations for all prompts using the templates
5. Tests all prompt variations with specified models
6. Evaluates the results using an evaluation model

Usage: python prompt_variation_evaluation.py [args]
"""

import argparse
import csv
import json
import os
import requests
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tabulate import tabulate

# API URL
BASE_URL = "http://localhost:8000/api"

def load_questions_from_csv(file_path: str) -> List[Dict[str, str]]:
    """Load questions and expected answers from a CSV file."""
    qa_pairs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Verify that required columns exist
            if 'question' not in header or 'answer' not in header:
                print(f"Error: CSV file must contain 'question' and 'answer' columns.")
                sys.exit(1)
                
            question_idx = header.index('question')
            answer_idx = header.index('answer')
            
            for row in reader:
                if len(row) > max(question_idx, answer_idx):
                    qa_pairs.append({
                        "question": row[question_idx],
                        "expected_answer": row[answer_idx]
                    })
    except Exception as e:
        print(f"Error loading questions from CSV: {str(e)}")
        sys.exit(1)
        
    print(f"Loaded {len(qa_pairs)} question-answer pairs from {file_path}")
    return qa_pairs

def load_prompt_variations_from_csv(file_path: str) -> List[Dict[str, str]]:
    """Load prompt variation instructions from a CSV file."""
    variations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Verify that required columns exist
            if 'name' not in header or 'instruction' not in header:
                print(f"Error: CSV file must contain 'name' and 'instruction' columns.")
                sys.exit(1)
                
            name_idx = header.index('name')
            instruction_idx = header.index('instruction')
            
            for row in reader:
                if len(row) > max(name_idx, instruction_idx):
                    variations.append({
                        "variation_type": row[name_idx],
                        "instruction": row[instruction_idx]
                    })
    except Exception as e:
        print(f"Error loading prompt variations from CSV: {str(e)}")
        sys.exit(1)
        
    print(f"Loaded {len(variations)} prompt variations from {file_path}")
    return variations

def create_prompt_group(group_name: str, description: str = "") -> str:
    """Create a new prompt group and return its ID."""
    try:
        response = requests.post(
            f"{BASE_URL}/prompt-groups",
            json={
                "name": group_name,
                "description": description or f"Created on {time.strftime('%Y-%m-%d')}",
                "tags": ["prompt-variations", "evaluation"]
            }
        )
        response.raise_for_status()
        group_id = response.json()["id"]
        print(f"Created prompt group: {group_name} (ID: {group_id})")
        return group_id
    except Exception as e:
        print(f"Error creating prompt group: {str(e)}")
        sys.exit(1)

def create_prompts(group_id: str, qa_pairs: List[Dict[str, str]]) -> List[str]:
    """Create prompts from question-answer pairs and return their IDs."""
    prompt_ids = []
    
    try:
        # Import the questions as a batch using the batch operations API
        response = requests.post(
            f"{BASE_URL}/batch/import-questions",
            json={
                "questions": [
                    {
                        "question": qa["question"],
                        "answer": qa["expected_answer"],
                        "metadata": {}
                    } for qa in qa_pairs
                ],
                "prompt_group_name": f"Group {group_id}",
                "prompt_group_description": f"Questions imported for group {group_id}",
                "tags": ["prompt-variations", "evaluation"]
            }
        )
        response.raise_for_status()
        result = response.json()
        prompt_ids = result["prompt_ids"]
        print(f"Created {len(prompt_ids)} prompts in group {group_id}")
        return prompt_ids
    except Exception as e:
        print(f"Error creating prompts: {str(e)}")
        sys.exit(1)

def save_variation_types(variations: List[Dict[str, str]]) -> None:
    """Save the variation types to the system."""
    try:
        for variation in variations:
            response = requests.post(
                f"{BASE_URL}/generation/variation-types",
                json=variation
            )
            response.raise_for_status()
            print(f"Saved variation type: {variation['variation_type']}")
    except Exception as e:
        print(f"Error saving variation types: {str(e)}")
        sys.exit(1)

def generate_prompt_variations(prompt_ids: List[str], variation_types: List[str], generator_model: str) -> List[str]:
    """Generate prompt variations for all prompts and return their version IDs."""
    version_ids = []
    
    try:
        for prompt_id in prompt_ids:
            # Get the prompt to verify it exists
            response = requests.get(f"{BASE_URL}/prompts/{prompt_id}")
            response.raise_for_status()
            original_prompt = response.json()
            
            # Generate variations for this prompt
            response = requests.post(
                f"{BASE_URL}/generation/generate",
                json={
                    "prompt_uuid": prompt_id,
                    "variation_types": variation_types,
                    "save": True,
                    "generator_model": generator_model
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Collect all version IDs
            for variation in result["variations"]:
                if variation.get("uuid"):
                    version_ids.append(variation["uuid"])
                    
            print(f"Generated {len(result['variations'])} variations for prompt: {original_prompt.get('name', prompt_id)}")
    except Exception as e:
        print(f"Error generating prompt variations: {str(e)}")
        sys.exit(1)
    
    print(f"Generated {len(version_ids)} total prompt variations")
    return version_ids

def run_batch_tests(version_ids: List[str], models: List[str]) -> str:
    """Run batch tests for all prompt versions with all models and return the batch ID."""
    try:
        response = requests.post(
            f"{BASE_URL}/batch/run-tests",
            json={
                "prompt_version_ids": version_ids,
                "models": models
            }
        )
        response.raise_for_status()
        
        # Get batch ID from response headers or body
        batch_id = response.headers.get('X-Batch-ID')
        if not batch_id:
            batch_id = response.json().get('batch_id')
            
        if not batch_id:
            # Try to extract from Location header
            location = response.headers.get('Location', '')
            if location:
                batch_id = location.split('/')[-1]
                
        # Still no batch ID? Generate one (fallback)
        if not batch_id:
            import uuid
            batch_id = str(uuid.uuid4())
            print(f"Warning: No batch ID received from server, using generated ID: {batch_id}")
        
        total_tests = len(version_ids) * len(models)
        print(f"Started batch test with ID: {batch_id} ({total_tests} total tests)")
        
        return batch_id
    except Exception as e:
        print(f"Error running batch tests: {str(e)}")
        sys.exit(1)

def check_batch_status(batch_id: str) -> Dict[str, Any]:
    """Check the status of a batch test run."""
    try:
        response = requests.get(f"{BASE_URL}/batch/test-status/{batch_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error checking batch status: {str(e)}")
        return {"completed": 0}

def wait_for_batch_completion(batch_id: str, expected_total: int, timeout: int = 600) -> bool:
    """Wait for a batch test run to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_batch_status(batch_id)
        print(f"Progress: {status['completed']}/{expected_total} tests completed")
        
        if status['completed'] >= expected_total:
            return True
            
        # Wait before checking again
        time.sleep(5)
    
    print("Warning: Timeout reached while waiting for batch completion")
    return False

def get_batch_results(batch_id: str) -> Dict[str, Any]:
    """Get the results of a batch test run."""
    try:
        response = requests.get(f"{BASE_URL}/batch/results/{batch_id}")
        response.raise_for_status()
        results = response.json()
        
        print("\n===== BATCH TEST RESULTS =====")
        print(f"Best performing model: {results['best_performing_model']}")
        print(f"Best performing variation: {results['best_performing_version']}")
        
        # Print performance metrics for each model and version
        summaries = sorted(results['summaries'], key=lambda x: x['avg_latency_ms'])
        table_data = []
        for summary in summaries:
            table_data.append([
                summary['model'],
                summary['prompt_version'],
                f"{summary['avg_latency_ms']:.2f}ms",
                f"{summary['avg_token_count']:.1f}",
                f"{summary['avg_tokens_per_second']:.2f}",
                summary['total_tests']
            ])
        
        headers = ["Model", "Variation", "Avg Latency", "Avg Tokens", "Tokens/sec", "Tests"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return results
    except Exception as e:
        print(f"Error getting batch results: {str(e)}")
        return {}

def export_batch_results(batch_id: str, output_file: Optional[str] = None) -> None:
    """Export the results of a batch test run to a CSV file."""
    try:
        response = requests.get(f"{BASE_URL}/batch/export-results/{batch_id}?format=csv")
        response.raise_for_status()
        
        # Determine output filename
        if not output_file:
            output_file = f"batch_results_{batch_id}.csv"
            
        with open(output_file, 'wb') as f:
            f.write(response.content)
            
        print(f"Exported results to {output_file}")
    except Exception as e:
        print(f"Error exporting batch results: {str(e)}")

def evaluate_test_results(batch_id: str, evaluator_model: str) -> List[Dict[str, Any]]:
    """Evaluate model answers against expected answers using an evaluation model."""
    try:
        # First, get all the test results
        response = requests.get(f"{BASE_URL}/batch/export-results/{batch_id}?format=json")
        response.raise_for_status()
        test_results = response.json()
        
        # Prepare evaluations
        evaluations = []
        for result in test_results:
            if result.get("expected_answer"):
                evaluations.append({
                    "question": result["question"],
                    "expected_answer": result["expected_answer"],
                    "actual_answer": result["response"],
                    "model": result["model"],
                    "version": result["version"]
                })
        
        if not evaluations:
            print("No evaluations to perform (no test results with expected answers)")
            return []
            
        print(f"\nEvaluating {len(evaluations)} test results using model: {evaluator_model}")
        
        # Call the evaluation API in batches to avoid overwhelming the server
        batch_size = 10
        all_evaluation_results = []
        
        for i in range(0, len(evaluations), batch_size):
            batch = evaluations[i:i+batch_size]
            print(f"Evaluating batch {i//batch_size + 1}/{(len(evaluations) + batch_size - 1)//batch_size}")
            
            try:
                eval_response = requests.post(
                    f"{BASE_URL}/evaluation/batch",
                    json={
                        "evaluations": batch,
                        "evaluation_model": evaluator_model,
                        "detailed_metrics": True
                    }
                )
                eval_response.raise_for_status()
                batch_results = eval_response.json()
                
                # Ensure model and version information is preserved
                for j, result in enumerate(batch_results):
                    if j < len(batch):  # Safety check
                        # Copy model and version info from the request to the result
                        result["model"] = batch[j]["model"]
                        result["version"] = batch[j]["version"]
                
                all_evaluation_results.extend(batch_results)
            except Exception as e:
                print(f"Error evaluating batch {i//batch_size + 1}: {str(e)}")
                # Continue with next batch rather than failing completely
                continue
                
        return all_evaluation_results
    except Exception as e:
        print(f"Error evaluating test results: {str(e)}")
        return []

def print_evaluation_summary(evaluated_results: List[Dict[str, Any]]) -> None:
    """Print a summary of evaluation results grouped by model and variation."""
    if not evaluated_results:
        print("No evaluation results to summarize.")
        return
    
    # Group results by model and variation
    grouped_results = {}
    for result in evaluated_results:
        model = result.get("model", "unknown")
        version = result.get("version", "unknown")
        key = f"{model}_{version}"
        
        if key not in grouped_results:
            grouped_results[key] = {
                "model": model,
                "version": version,
                "accuracy": [],
                "relevance": [],
                "completeness": [],
                "conciseness": [],
                "overall_score": [],
                "count": 0
            }
            
        # Add metrics if available
        for metric in ["accuracy", "relevance", "completeness", "conciseness", "overall_score"]:
            if metric in result:
                grouped_results[key][metric].append(result[metric])
                
        grouped_results[key]["count"] += 1
    
    # Calculate averages and prepare table data
    table_data = []
    for key, data in grouped_results.items():
        row = [
            data["model"],
            data["version"],
            sum(data["accuracy"]) / len(data["accuracy"]) if data["accuracy"] else "N/A",
            sum(data["relevance"]) / len(data["relevance"]) if data["relevance"] else "N/A",
            sum(data["completeness"]) / len(data["completeness"]) if data["completeness"] else "N/A",
            sum(data["conciseness"]) / len(data["conciseness"]) if data["conciseness"] else "N/A",
            sum(data["overall_score"]) / len(data["overall_score"]) if data["overall_score"] else "N/A",
            data["count"]
        ]
        table_data.append(row)
    
    # Sort by overall score
    table_data.sort(key=lambda x: x[6] if x[6] != "N/A" else 0, reverse=True)
    
    # Print table
    headers = ["Model", "Variation", "Accuracy", "Relevance", "Completeness", "Conciseness", "Overall", "Tests"]
    print("\n===== EVALUATION RESULTS =====")
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))
    
    # Sample evaluations
    print("\n===== SAMPLE EVALUATIONS =====")
    samples = evaluated_results[:3]  # Just show first 3 for brevity
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Model: {sample.get('model', 'unknown')}")
        print(f"Variation: {sample.get('version', 'unknown')}")
        print(f"Question: {sample['question']}")
        print(f"Expected: {sample['expected_answer']}")
        print(f"Actual: {sample['actual_answer'][:100]}..." if len(sample['actual_answer']) > 100 else f"Actual: {sample['actual_answer']}")
        print(f"Accuracy: {sample.get('accuracy', 'N/A'):.3f}")
        print(f"Overall score: {sample.get('overall_score', 'N/A'):.3f}")
        print(f"Explanation: {sample.get('explanation', 'N/A')}")

def debug_batch_results(batch_id: str) -> None:
    """
    Helper function to debug batch results and test run information.
    This helps identify potential issues with the model and version tracking.
    """
    try:
        # Get detailed test run information
        response = requests.get(f"{BASE_URL}/batch/export-results/{batch_id}?format=json")
        response.raise_for_status()
        results = response.json()
        
        # Print key information for debugging
        print("\n===== DEBUG: TEST RUN DETAILS =====")
        for i, result in enumerate(results[:3]):  # Just show first 3 for brevity
            print(f"\nTest Run {i+1}:")
            print(f"  Run ID: {result.get('run_id', 'N/A')}")
            print(f"  Model: {result.get('model', 'unknown')}")
            print(f"  Version: {result.get('version', 'unknown')}")
            print(f"  Question: {result.get('question', 'N/A')}")
            print(f"  Response: {result.get('response', 'N/A')[:50]}..." if len(result.get('response', '')) > 50 else f"  Response: {result.get('response', 'N/A')}")
            
            # Print all available keys for debugging
            print(f"  Available fields: {', '.join(result.keys())}")
            
    except Exception as e:
        print(f"Error debugging batch results: {str(e)}")

def ensure_model_running(model_name: str) -> bool:
    """Make sure the model is running, try to start it if not."""
    try:
        # Check if model is running
        response = requests.get(f"{BASE_URL}/models/running")
        response.raise_for_status()
        
        running_models = response.json().get("running_models", [])
        if any(model["name"] == model_name for model in running_models):
            print(f"Model {model_name} is already running.")
            return True
            
        print(f"Model {model_name} is not running. Attempting to start it...")
        
        # Try to start the model
        response = requests.post(
            f"{BASE_URL}/models/start",
            json={"model_name": model_name}
        )
        
        if response.status_code == 200:
            print(f"Successfully started model: {model_name}")
            return True
        else:
            print(f"Failed to start model: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking/starting model: {str(e)}")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test and evaluate different prompt variations across multiple models')
    parser.add_argument('--questions-csv', type=str, required=True, help='CSV file with questions and answers')
    parser.add_argument('--variations-csv', type=str, required=True, help='CSV file with prompt variation instructions')
    parser.add_argument('--model1', type=str, required=True, help='First model to test')
    parser.add_argument('--model2', type=str, required=True, help='Second model to test')
    parser.add_argument('--evaluator', type=str, required=True, help='Model to use for evaluation and variation generation')
    parser.add_argument('--group-name', type=str, required=True, help='Name for the prompt group')
    parser.add_argument('--output', type=str, help='Output file for results (CSV)')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for batch completion')
    
    args = parser.parse_args()
    
    # Handle relative paths
    for path_arg in ['questions_csv', 'variations_csv']:
        path = getattr(args, path_arg)
        if not os.path.isabs(path):
            # Try to find the file in the current directory
            abs_path = os.path.join(os.path.abspath('.'), path)
            if os.path.exists(abs_path):
                setattr(args, path_arg, abs_path)
    
    return args

def main():
    args = parse_arguments()
    
    # Check if files exist
    if not os.path.exists(args.questions_csv):
        print(f"Error: Questions CSV file not found at {args.questions_csv}")
        sys.exit(1)
        
    if not os.path.exists(args.variations_csv):
        print(f"Error: Variations CSV file not found at {args.variations_csv}")
        sys.exit(1)
    
    # Ensure models are running
    for model in [args.model1, args.model2, args.evaluator]:
        if not ensure_model_running(model):
            print(f"Cannot proceed without model {model} running.")
            sys.exit(1)
    
    # Load questions and variations
    qa_pairs = load_questions_from_csv(args.questions_csv)
    variations = load_prompt_variations_from_csv(args.variations_csv)
    
    # Step 1: Create prompt group
    group_id = create_prompt_group(args.group_name)
    
    # Step 2: Create prompts from questions
    prompt_ids = create_prompts(group_id, qa_pairs)
    
    # Step 3: Save variation types
    save_variation_types(variations)
    
    # Step 4: Generate prompt variations using the API
    variation_types = [v["variation_type"] for v in variations]
    version_ids = generate_prompt_variations(prompt_ids, variation_types, args.evaluator)
    
    # Step 5: Run batch tests
    batch_id = run_batch_tests(version_ids, [args.model1, args.model2])
    
    # Total expected tests
    expected_total = len(version_ids) * 2  # Two models
    
    # Step 6: Wait for completion
    completed = wait_for_batch_completion(batch_id, expected_total, timeout=args.timeout)
    
    if completed:
        print("\nAll tests completed successfully!")
    else:
        print("\nWARNING: Not all tests completed within the timeout period.")
    
    # Step 7: Get results
    batch_results = get_batch_results(batch_id)
    
    # Step 8: Export results
    if args.output:
        export_batch_results(batch_id, args.output)
    
    # Step 9: Evaluate results
    evaluation_results = evaluate_test_results(batch_id, args.evaluator)
    
    # Step 10: Print evaluation summary
    print_evaluation_summary(evaluation_results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)