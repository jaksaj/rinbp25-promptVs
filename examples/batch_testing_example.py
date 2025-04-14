"""
Example script showing how to use the batch operations API to test multiple questions across different models.

This script demonstrates the complete workflow:
1. Import questions from a CSV file
2. Create multiple versions of each prompt using templates
3. Run tests with different models
4. Get results and export them for analysis

Usage: python batch_testing_example.py
"""

import requests
import json
import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# API URL
BASE_URL = "http://localhost:8000/api"

def import_questions_from_csv(file_path: str, group_name: str) -> Dict[str, Any]:
    """
    Import questions from a CSV file.
    Expected CSV format: question,answer
    """
    questions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                questions.append({
                    "question": row[0],
                    "answer": row[1] if len(row) > 1 and row[1] else None,
                    "metadata": {}
                })
    
    # Create the request payload
    payload = {
        "questions": questions,
        "prompt_group_name": group_name,
        "prompt_group_description": f"Imported from {file_path}",
        "tags": ["batch-import", "example"]
    }
    
    # Send the request
    response = requests.post(f"{BASE_URL}/batch/import-questions", json=payload)
    response.raise_for_status()
    
    return response.json()

def create_prompt_versions(prompt_ids: List[str]) -> Dict[str, Any]:
    """
    Create multiple versions of each prompt using templates.
    """
    # Define the templates for different versions
    templates = [
        {
            "template": "{question}",
            "version_name": "v1-plain",
            "notes": "Original question with no modifications"
        },
        {
            "template": "Please answer the following question thoroughly and accurately: {question}",
            "version_name": "v2-formal",
            "notes": "Formal prompt with instructions"
        },
        {
            "template": "I want you to think step-by-step about this question: {question}",
            "version_name": "v3-stepbystep",
            "notes": "Encouraging step-by-step reasoning"
        }
    ]
    
    # Create the request payload
    payload = {
        "prompt_ids": prompt_ids,
        "templates": templates
    }
    
    # Send the request
    response = requests.post(f"{BASE_URL}/batch/create-versions", json=payload)
    response.raise_for_status()
    
    return response.json()

def run_batch_tests(prompt_version_ids: List[str], models: List[str]) -> str:
    """
    Run batch tests for all prompt versions with all models.
    Returns the batch_id for tracking.
    """
    # Create the request payload
    payload = {
        "prompt_version_ids": prompt_version_ids,
        "models": models
    }
    
    # Send the request
    response = requests.post(f"{BASE_URL}/batch/run-tests", json=payload)
    response.raise_for_status()
    
    result = response.json()
    print(f"Started batch test with {result['total_tests']} total tests")
    
    # First try to get batch_id from headers
    batch_id = response.headers.get('X-Batch-ID')
    
    # If not in headers, try to get from the response body
    if not batch_id and 'batch_id' in result:
        batch_id = result['batch_id']
        
    # If still no batch ID, try to get from the Location header
    if not batch_id:
        location = response.headers.get('Location', '')
        if location:
            batch_id = location.split('/')[-1]
    
    # Generate a unique ID if none provided (fallback)
    if not batch_id:
        import uuid
        batch_id = str(uuid.uuid4())
        print(f"Warning: No batch ID received from server, using generated ID: {batch_id}")
    
    print(f"Batch ID: {batch_id}")
    return batch_id

def check_batch_status(batch_id: str) -> Dict[str, Any]:
    """
    Check the status of a batch test run.
    """
    response = requests.get(f"{BASE_URL}/batch/test-status/{batch_id}")
    response.raise_for_status()
    
    return response.json()

def wait_for_batch_completion(batch_id: str, expected_total: int, timeout: int = 600) -> bool:
    """
    Wait for a batch test run to complete.
    Returns True if completed within timeout, False otherwise.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_batch_status(batch_id)
        print(f"Progress: {status['completed']}/{expected_total} tests completed")
        
        if status['completed'] >= expected_total:
            return True
            
        # Wait before checking again
        time.sleep(5)
    
    return False

def get_batch_results(batch_id: str) -> Dict[str, Any]:
    """
    Get aggregated results for a batch test run.
    """
    response = requests.get(f"{BASE_URL}/batch/results/{batch_id}")
    response.raise_for_status()
    
    return response.json()

def export_results(batch_id: str, format: str = "csv", output_file: Optional[str] = None) -> None:
    """
    Export batch test results to a file.
    """
    response = requests.get(f"{BASE_URL}/batch/export-results/{batch_id}?format={format}")
    response.raise_for_status()
    
    if format == "csv":
        # Get filename from Content-Disposition header or use default
        content_disposition = response.headers.get('Content-Disposition', '')
        filename = content_disposition.split('filename=')[-1].strip('"') if 'filename=' in content_disposition else f"batch_results_{batch_id}.csv"
        
        output_path = Path(output_file) if output_file else Path(filename)
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Exported CSV results to {output_path}")
    else:
        # JSON format
        results = response.json()
        output_path = Path(output_file) if output_file else Path(f"batch_results_{batch_id}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"Exported JSON results to {output_path}")

def format_batch_results(results: Dict[str, Any]) -> None:
    """
    Format and print batch results in a readable format.
    """
    print("\n===== BATCH TEST RESULTS =====")
    print(f"Best performing model: {results['best_performing_model']}")
    print(f"Best performing version: {results['best_performing_version']}")
    
    print("\nPerformance by model and version:")
    for summary in results['summaries']:
        print(f"  - {summary['model']} / {summary['prompt_version']}:")
        print(f"    Avg latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"    Avg tokens: {summary['avg_token_count']:.1f}")
        print(f"    Avg tokens/sec: {summary['avg_tokens_per_second']:.2f}")
        print(f"    Tests: {summary['total_tests']}")
        print()

def evaluate_test_runs(batch_id: str, evaluation_model: str = "gemma3:4b") -> Dict[str, Any]:
    """
    Evaluate the test runs using an evaluation model.
    
    Args:
        batch_id: The ID of the batch test run
        evaluation_model: The model to use for evaluation
        
    Returns:
        Evaluation results summary
    """
    # First get all test runs from the batch
    test_runs_response = requests.get(f"{BASE_URL}/batch/export-results/{batch_id}?format=json")
    test_runs_response.raise_for_status()
    test_runs = test_runs_response.json()
    
    # Extract test run IDs
    test_run_ids = []
    for test_run in test_runs:
        if "run_id" in test_run:
            test_run_ids.append(test_run["run_id"])
    
    if not test_run_ids:
        print("No test run IDs found in batch results. Cannot perform evaluation.")
        return {}
    
    print(f"\nEvaluating {len(test_run_ids)} test runs using model: {evaluation_model}")
    
    # Use batch evaluation endpoint to process all test runs
    evaluation_response = requests.post(
        f"{BASE_URL}/evaluation/batch",
        json={
            "test_run_ids": test_run_ids,
            "evaluation_model": evaluation_model,
            "detailed_metrics": True
        }
    )
    evaluation_response.raise_for_status()
    evaluation_results = evaluation_response.json()
    
    # Parse and return evaluation results
    results = evaluation_results.get("results", [])
    avg_score = evaluation_results.get("average_score", 0.0)
    
    print(f"Evaluation complete. Average score: {avg_score:.3f}")
    
    # Group results by model and prompt version for better analysis
    grouped_results = {}
    for result in results:
        # Get the model and prompt version from the test run
        test_run_id = result.get("test_run_id")
        if not test_run_id:
            continue
            
        # Find the corresponding test run
        for test_run in test_runs:
            if test_run.get("run_id") == test_run_id:
                model = test_run.get("model", "unknown")
                prompt_version = test_run.get("version", "unknown")
                
                key = f"{model}_{prompt_version}"
                if key not in grouped_results:
                    grouped_results[key] = {
                        "model": model,
                        "prompt_version": prompt_version,
                        "scores": [],
                        "accuracy": [],
                        "relevance": [],
                        "completeness": [],
                        "conciseness": [],
                        "count": 0
                    }
                
                # Add metrics
                grouped_results[key]["scores"].append(result.get("overall_score", 0.0))
                grouped_results[key]["accuracy"].append(result.get("accuracy", 0.0))
                
                # Add optional detailed metrics if available
                for metric in ["relevance", "completeness", "conciseness"]:
                    if result.get(metric) is not None:
                        grouped_results[key][metric].append(result.get(metric, 0.0))
                
                grouped_results[key]["count"] += 1
                break
    
    # Calculate averages for each group
    summary = []
    for key, data in grouped_results.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        avg_accuracy = sum(data["accuracy"]) / len(data["accuracy"]) if data["accuracy"] else 0.0
        
        metrics = {
            "model": data["model"],
            "prompt_version": data["prompt_version"],
            "avg_score": avg_score,
            "avg_accuracy": avg_accuracy,
            "count": data["count"]
        }
        
        # Add detailed metrics if available
        for metric in ["relevance", "completeness", "conciseness"]:
            if data[metric]:
                metrics[f"avg_{metric}"] = sum(data[metric]) / len(data[metric])
        
        summary.append(metrics)
    
    # Sort by average score (descending)
    summary.sort(key=lambda x: x["avg_score"], reverse=True)
    
    # Print summary table
    if summary:
        print("\n===== EVALUATION SUMMARY =====")
        headers = ["Model", "Prompt Version", "Avg Score", "Avg Accuracy", "Count"]
        table_data = [
            [
                item["model"],
                item["prompt_version"],
                f"{item['avg_score']:.3f}",
                f"{item['avg_accuracy']:.3f}",
                item["count"]
            ]
            for item in summary
        ]
        
        print("\n" + "\n".join(
            ["\t".join(headers)] + 
            ["\t".join(row) for row in table_data]
        ))
        
        # Identify best performing combination
        best = summary[0]
        print(f"\nBest performing combination: {best['model']} with {best['prompt_version']} (Score: {best['avg_score']:.3f})")
    
    return {
        "summary": summary,
        "average_score": avg_score,
        "total_evaluations": len(results)
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run batch testing for multiple questions with different models')
    parser.add_argument('--csv', type=str, help='CSV file with questions and answers')
    parser.add_argument('--group-name', type=str, default='Batch Test Questions', help='Name for the prompt group')
    parser.add_argument('--models', type=str, nargs='+', default=['deepseek-r1:1.5b', 'llama3.2:1b'], help='Models to test with')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--format', choices=['json', 'csv'], default='csv', help='Output format')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Step 1: Import questions
    if args.csv:
        print(f"Importing questions from {args.csv}...")
        import_result = import_questions_from_csv(args.csv, args.group_name)
        prompt_ids = import_result['prompt_ids']
        print(f"Imported {len(prompt_ids)} questions into group {import_result['group_id']}")
    else:
        # For example purposes, create a simple test group with a few questions
        print("No CSV file provided. Creating example questions...")
        example_questions = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"}
        ]
        
        payload = {
            "questions": example_questions,
            "prompt_group_name": args.group_name,
            "prompt_group_description": "Example questions for batch testing",
            "tags": ["example"]
        }
        
        response = requests.post(f"{BASE_URL}/batch/import-questions", json=payload)
        import_result = response.json()
        prompt_ids = import_result['prompt_ids']
        print(f"Created {len(prompt_ids)} example questions in group {import_result['group_id']}")
    
    # Step 2: Create versions for each question
    print("\nCreating prompt versions...")
    versions_result = create_prompt_versions(prompt_ids)
    print(f"Created {versions_result['total_created']} prompt versions")
    
    # Extract all version IDs
    version_ids = [v["version_id"] for v in versions_result["created_versions"]]
    
    # Step 3: Run batch tests
    print(f"\nRunning batch tests with models: {', '.join(args.models)}")
    batch_id = run_batch_tests(version_ids, args.models)
    print(f"Batch test started with ID: {batch_id}")
    
    # Total expected tests
    expected_total = len(version_ids) * len(args.models)
    
    # Step 4: Wait for completion (with timeout)
    print("\nWaiting for tests to complete...")
    completed = wait_for_batch_completion(batch_id, expected_total, timeout=600)
    
    if completed:
        print("\nAll tests completed successfully!")
    else:
        print("\nWARNING: Not all tests completed within the timeout period.")
    
    # Step 5: Get results
    print("\nFetching test results...")
    results = get_batch_results(batch_id)
    
    # Print formatted results
    format_batch_results(results)
    
    # Step 6: Export results
    export_results(batch_id, format=args.format, output_file=args.output)
    
    # Step 7: Evaluate test runs
    print("\nEvaluating test runs...")
    evaluation_results = evaluate_test_runs(batch_id)
    print(f"Evaluation summary: {evaluation_results}")

if __name__ == "__main__":
    main()