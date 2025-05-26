#!/usr/bin/env python3
"""
Script 3: A/B Testing Evaluations (Refactored)
==============================================

This script handles the evaluation phase using a modular architecture:
1. Load test runs from previous script
2. Start API server and Ollama
3. Start evaluation model
4. Perform A/B testing comparisons
5. Generate comprehensive report

Usage:
    python 03_ab_testing_evaluations.py

Input:
    - test_runs.json: Data from script 2

Output:
    - evaluation_report.json: Comprehensive test report
"""

import logging
import sys
import argparse
from datetime import datetime
from evaluator import ABTestingEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'../logs/03_evaluations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Perform A/B testing evaluations')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--input', default='../output/test_runs.json', help='Input file from script 2')
    parser.add_argument('--max-comparisons', type=int, default=3, 
                       help='Maximum comparisons per version pair (default: 3)')
    parser.add_argument('--disable-model-comparison', action='store_true', 
                       help='Disable model comparison within versions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = ABTestingEvaluator(
        api_base_url=args.api_url, 
        input_file=args.input,
        max_comparisons_per_version_pair=args.max_comparisons,
        enable_model_comparison=not args.disable_model_comparison
    )
    
    try:
        summary = evaluator.run()
        print("\n✅ A/B testing evaluations completed successfully!")
        print(f"Performed {summary['total_comparisons']} comparisons")
        print(f"- Version comparisons: {summary['total_version_comparisons']}")
        print(f"- Model comparisons: {summary['total_model_comparisons']}")
        print(f"Report saved to: {summary['report_file']}")
        print(f"Execution time: {summary['execution_time']:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ A/B testing evaluations failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
