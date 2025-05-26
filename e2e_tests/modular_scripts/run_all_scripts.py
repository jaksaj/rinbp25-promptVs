#!/usr/bin/env python3
"""
Run All Scripts
===============

Convenience script to run all testing scripts in sequence:
1. Create prompts and versions
2. Create test runs  
3. Perform A/B testing evaluations
4. Analyze ELO ratings and generate insights (optional)

Usage:
    python run_all_scripts.py [options]
"""

import argparse
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, args=None):
    """Run a script and return success status"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}")
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        print(f"\n✅ {script_name} completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n❌ {script_name} failed after {end_time - start_time:.2f} seconds")
        print(f"Error code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all testing scripts in sequence')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--config', default='../config/test_prompts_config.json', help='Test configuration file')
    parser.add_argument('--runs-per-version', type=int, default=2, help='Number of test runs per version')
    parser.add_argument('--include-analysis', action='store_true', help='Include ELO analysis (Script 4) in the workflow')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Build common arguments
    common_args = ['--api-url', args.api_url]
    if args.verbose:
        common_args.append('--verbose')
    
    print(f"Starting complete testing workflow at {datetime.now()}")
    print(f"API URL: {args.api_url}")
    print(f"Config file: {args.config}")
    print(f"Runs per version: {args.runs_per_version}")
    print(f"Include ELO analysis: {args.include_analysis}")
    
    overall_start = time.time()
    
    # Script 1: Create prompts and versions
    script1_args = common_args + ['--config', args.config]
    if not run_script('01_create_prompts_and_versions.py', script1_args):
        print("\n❌ Workflow failed at script 1")
        sys.exit(1)
    
    # Script 2: Create test runs
    script2_args = common_args + ['--runs-per-version', str(args.runs_per_version)]
    if not run_script('02_create_test_runs.py', script2_args):
        print("\n❌ Workflow failed at script 2")
        sys.exit(1)
    
    # Script 3: A/B testing evaluations
    script3_args = common_args.copy()
    if not run_script('03_ab_testing_evaluations.py', script3_args):
        print("\n❌ Workflow failed at script 3")
        sys.exit(1)    # Script 4: ELO rating analysis (optional)
    if args.include_analysis:
        # Use the latest evaluation report from script 3
        import glob
        evaluation_reports = glob.glob('../output/evaluation_report_*.json')
        if evaluation_reports:
            latest_report = max(evaluation_reports)
            script4_args = common_args + ['--input', latest_report]
        else:
            print("\n❌ No evaluation report found for ELO analysis")
            sys.exit(1)
        
        if not run_script('04_elo_rating_analysis.py', script4_args):
            print("\n❌ Workflow failed at script 4")
            sys.exit(1)
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Completed at: {datetime.now()}")
    print("\nOutput files (in ../output/ directory):")
    print("- prompts_and_versions.json")
    print("- test_runs.json")
    print("- evaluation_report_*.json")
    if args.include_analysis:
        print("- elo_analysis_report_*.json")
        print("- elo_insights_*.md")
        print("- elo_recommendations_*.json")
    print("\nLog files are in ../logs/ directory")
    if args.include_analysis:
        print("Check the evaluation report and ELO analysis for detailed results!")
    else:
        print("Check the evaluation report for detailed results!")
        print("Run with --include-analysis to generate ELO insights!")

if __name__ == "__main__":
    main()
