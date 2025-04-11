"""
Example script showing how to use the evaluation API to assess model answers against expected answers.

This script:
1. Reads questions and expected answers from a CSV file
2. Processes each question through a model
3. Evaluates the model's answers against expected answers using the evaluation API
4. Produces a summary of accuracy results

Usage: python evaluate_sample_questions.py [--csv path/to/file.csv] [--model model_name] [--evaluator evaluator_model]
"""

import argparse
import csv
import requests
import json
import sys
import os
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

def get_model_answers(questions: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """Process questions through the specified model to get answers."""
    results = []
    
    print(f"Processing {len(questions)} questions with model: {model_name}")
    
    for i, qa in enumerate(questions):
        try:
            # Call the API to process the prompt
            response = requests.post(
                f"{BASE_URL}/single/{model_name}",
                json={"prompt": qa["question"]}
            )
            response.raise_for_status()
            
            # Add model answer to results
            actual_answer = response.json()["result"]
            results.append({
                "question": qa["question"],
                "expected_answer": qa["expected_answer"],
                "actual_answer": actual_answer
            })
            
            # Print progress
            print(f"Processed question {i+1}/{len(questions)}", end="\r")
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            results.append({
                "question": qa["question"],
                "expected_answer": qa["expected_answer"],
                "error": str(e)
            })
    
    print("\nCompleted processing all questions")
    return results

def evaluate_answers(qa_results: List[Dict[str, str]], evaluator_model: str, detailed: bool = False) -> List[Dict[str, Any]]:
    """Evaluate model answers against expected answers using the evaluator API."""
    print(f"Evaluating answers using {evaluator_model}...")
    
    # Prepare batch request
    evaluations = []
    for qa in qa_results:
        # Skip items with errors
        if "error" in qa:
            evaluations.append({
                "question": qa["question"],
                "expected_answer": qa["expected_answer"],
                "actual_answer": "ERROR",
                "error": qa["error"]
            })
            continue
            
        evaluations.append({
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "actual_answer": qa["actual_answer"]
        })
    
    # Call the evaluation API
    try:
        response = requests.post(
            f"{BASE_URL}/evaluation/batch",
            json={
                "evaluations": evaluations,
                "evaluation_model": evaluator_model,
                "detailed_metrics": detailed
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error evaluating answers: {str(e)}")
        return evaluations  # Return original data without evaluation

def print_evaluation_summary(evaluated_results: List[Dict[str, Any]], detailed: bool = False):
    """Print a summary of evaluation results."""
    if not evaluated_results:
        print("No evaluation results to summarize.")
        return
        
    # Check if we have proper evaluation results
    if "error" in evaluated_results[0] and "score" not in evaluated_results[0]:
        print("Evaluation failed. No results to summarize.")
        return
    
    # Calculate success metrics
    total_questions = len(evaluated_results)
    
    if detailed:
        # For detailed metrics
        metrics = {
            "accuracy": 0,
            "relevance": 0,
            "completeness": 0,
            "conciseness": 0,
            "overall_score": 0
        }
        
        valid_results = [r for r in evaluated_results if all(k in r for k in metrics.keys())]
        
        if not valid_results:
            print("No valid detailed metrics found in evaluation results.")
            return
            
        for metric in metrics.keys():
            metrics[metric] = sum(r[metric] for r in valid_results) / len(valid_results)
            
        # Print summary table for detailed metrics
        print("\n=== EVALUATION SUMMARY (DETAILED METRICS) ===")
        print(f"Total questions: {total_questions}")
        print(f"Valid evaluations: {len(valid_results)}")
        print("\nAverage Scores:")
        
        metrics_table = []
        for metric, value in metrics.items():
            metrics_table.append([metric.capitalize(), f"{value:.2f}"])
            
        print(tabulate(metrics_table, headers=["Metric", "Score (0-1)"]))
        
    else:
        # For basic correctness metrics
        correct_answers = [r for r in evaluated_results if r.get("is_correct", False)]
        avg_score = sum(r.get("score", 0) for r in evaluated_results) / total_questions
        
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {len(correct_answers)} ({len(correct_answers)/total_questions*100:.1f}%)")
        print(f"Average score: {avg_score:.2f}/1.0")
    
    # Print example evaluations
    print("\n=== SAMPLE EVALUATIONS ===")
    samples = evaluated_results[:3]  # Just show first 3 for brevity
    
    for i, sample in enumerate(samples):
        print(f"\nEvaluation {i+1}:")
        print(f"Q: {sample['question']}")
        print(f"Expected: {sample['expected_answer']}")
        print(f"Actual: {sample['actual_answer'][:100]}..." if len(sample['actual_answer']) > 100 else f"Actual: {sample['actual_answer']}")
        
        if detailed:
            print(f"Accuracy: {sample.get('accuracy', 'N/A'):.2f}")
            print(f"Overall score: {sample.get('overall_score', 'N/A'):.2f}")
        else:
            print(f"Correct: {sample.get('is_correct', 'N/A')}")
            print(f"Score: {sample.get('score', 'N/A'):.2f}")
        
        print(f"Explanation: {sample.get('explanation', 'N/A')}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model answers against expected answers')
    parser.add_argument('--csv', type=str, default='sample_questions.csv', help='Path to CSV file with questions and answers')
    parser.add_argument('--model', type=str, default='deepseek-r1:1.5b', help='Model to test')
    parser.add_argument('--evaluator', type=str, default='gemma3:4b', help='Model to use for evaluation (should be larger than test model)')
    parser.add_argument('--detailed', action='store_true', help='Use detailed metrics evaluation')
    
    args = parser.parse_args()
    
    # Handle relative paths
    if not os.path.isabs(args.csv):
        # Try to find the file in the current directory or examples directory
        current_dir = os.path.abspath('.')
        example_dir = os.path.join(current_dir, 'examples')
        
        if os.path.exists(os.path.join(current_dir, args.csv)):
            args.csv = os.path.join(current_dir, args.csv)
        elif os.path.exists(os.path.join(example_dir, args.csv)):
            args.csv = os.path.join(example_dir, args.csv)
    
    return args

def check_model_status(model_name: str) -> bool:
    """Check if a model is running."""
    try:
        response = requests.get(f"{BASE_URL}/models/running")
        response.raise_for_status()
        
        running_models = response.json().get("running_models", [])
        return any(model["name"] == model_name for model in running_models)
    except Exception as e:
        print(f"Error checking model status: {str(e)}")
        return False

def ensure_model_running(model_name: str) -> bool:
    """Make sure the model is running, try to start it if not."""
    if check_model_status(model_name):
        print(f"Model {model_name} is already running.")
        return True
        
    print(f"Model {model_name} is not running. Attempting to start it...")
    
    try:
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
        print(f"Error starting model: {str(e)}")
        return False

def main():
    args = parse_arguments()
    
    # Verify CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        sys.exit(1)
    
    # Ensure models are running
    if not ensure_model_running(args.model):
        print(f"Cannot proceed without model {args.model} running.")
        sys.exit(1)
        
    if not ensure_model_running(args.evaluator):
        print(f"Cannot proceed without evaluator model {args.evaluator} running.")
        sys.exit(1)
    
    # Load questions
    qa_pairs = load_questions_from_csv(args.csv)
    
    # Get model answers
    model_answers = get_model_answers(qa_pairs, args.model)
    
    # Evaluate answers
    evaluation_results = evaluate_answers(model_answers, args.evaluator, args.detailed)
    
    # Print summary
    print_evaluation_summary(evaluation_results, args.detailed)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)