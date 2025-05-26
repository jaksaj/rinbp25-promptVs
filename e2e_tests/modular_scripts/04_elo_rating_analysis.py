#!/usr/bin/env python3
"""
Script 4: ELO Rating Analysis and Insights
==========================================

Comprehensive analysis of ELO ratings from A/B testing evaluations.
Provides statistical analysis, insights, and actionable recommendations.

Usage:
    python 04_elo_rating_analysis.py --test-runs run1,run2,run3 [options]
    python 04_elo_rating_analysis.py --config config.json [options]
    python 04_elo_rating_analysis.py --help

Features:
    - Comprehensive ELO rating analysis (elo_score, version_elo, global_elo)
    - Statistical analysis with confidence intervals and effect sizes
    - Technique and model performance comparisons
    - Actionable insights and recommendations
    - Multiple report formats (JSON, Markdown, visualization data)
    - Risk assessment and optimization opportunities
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add the parent directory to the path to import from elo_analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elo_analyzer import EloAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('elo_analysis.log')
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive ELO rating analysis and insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze specific test runs
    python 04_elo_rating_analysis.py --test-runs "run1,run2,run3"
    
    # Use configuration file
    python 04_elo_rating_analysis.py --config analysis_config.json
    
    # Analyze with custom API URL and output directory
    python 04_elo_rating_analysis.py --test-runs "run1,run2" --api-url "http://localhost:8000" --output-dir "custom_reports"
    
    # Quick analysis with minimal output
    python 04_elo_rating_analysis.py --test-runs "run1,run2" --quiet
    
    # Generate only specific report types
    python 04_elo_rating_analysis.py --test-runs "run1,run2" --reports "summary,insights"
        """
    )      # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--test-runs',
        type=str,
        help='Comma-separated list of test run IDs to analyze'
    )
    input_group.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file with test run IDs and settings'
    )
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to evaluation_report.json file from script 3 (automatically extracts test run IDs)'
    )
    
    # API configuration
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='Base URL for the API (default: http://localhost:8000)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='elo_analysis_reports',
        help='Directory for output reports (default: elo_analysis_reports)'
    )
    
    # Report options
    parser.add_argument(
        '--reports',
        type=str,
        default='all',
        help='Comma-separated list of report types to generate: all, json, summary, insights, recommendations, visualization (default: all)'
    )
    
    # Analysis options
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for statistical analysis (default: 0.95)'
    )
    
    parser.add_argument(
        '--min-sample-size',
        type=int,
        default=5,
        help='Minimum sample size for technique/model analysis (default: 5)'
    )
    
    # Output options
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Increase output verbosity'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs without running analysis'
    )
    
    # Utility options
    parser.add_argument(
        '--version',
        action='version',
        version='ELO Rating Analysis Script v1.0'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def validate_test_runs(test_runs: List[str]) -> bool:
    """Validate test run IDs"""
    if not test_runs:
        logger.error("No test run IDs provided")
        return False
    
    if len(test_runs) > 100:
        logger.warning(f"Large number of test runs ({len(test_runs)}). Analysis may take significant time.")
    
    # Basic validation - check for valid format
    invalid_runs = [run for run in test_runs if not run or not isinstance(run, str)]
    if invalid_runs:
        logger.error(f"Invalid test run IDs: {invalid_runs}")
        return False
    
    logger.info(f"Validated {len(test_runs)} test run IDs")
    return True


def setup_logging(args: argparse.Namespace):
    """Setup logging based on arguments"""
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Add file handler for detailed logs
    log_file = os.path.join(args.output_dir, 'elo_analysis_detailed.log')
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)


def print_analysis_summary(results: Dict[str, Any]):
    """Print analysis summary to console"""
    summary = results.get('summary', {})
    
    print("\n" + "="*60)
    print("ELO RATING ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall performance
    overall_perf = summary.get('overall_performance', {})
    print("\nOVERALL PERFORMANCE:")
    for score_type, perf_data in overall_perf.items():
        score_name = score_type.replace('_', ' ').title()
        mean_score = perf_data.get('mean', 0)
        level = perf_data.get('performance_level', 'unknown')
        above_baseline = perf_data.get('above_baseline', False)
        
        status = "✓" if above_baseline else "✗"
        print(f"  {status} {score_name}: {mean_score:.1f} ({level})")
    
    # Key metrics
    key_metrics = summary.get('key_metrics', {})
    print(f"\nKEY METRICS:")
    print(f"  • Total Recommendations: {key_metrics.get('total_recommendations', 0)}")
    print(f"  • High Priority: {key_metrics.get('high_priority_recommendations', 0)}")
    print(f"  • Risk Level: {key_metrics.get('overall_risk_level', 'unknown').title()}")
    print(f"  • Analysis Confidence: {key_metrics.get('analysis_confidence', 'unknown').title()}")
    
    # Top findings
    top_findings = summary.get('top_findings', [])
    if top_findings:
        print(f"\nTOP FINDINGS:")
        for i, finding in enumerate(top_findings[:5], 1):
            print(f"  {i}. {finding}")
    
    # Next steps
    next_steps = summary.get('next_steps', [])
    if next_steps:
        print(f"\nNEXT STEPS:")
        for i, step in enumerate(next_steps[:3], 1):
            print(f"  {i}. {step}")
    
    print("\n" + "="*60)


def print_data_quality_info(results: Dict[str, Any]):
    """Print data quality information"""
    data_collection = results.get('data_collection', {})
    data_summary = data_collection.get('data_summary', {})
    
    print("\nDATA QUALITY:")
    print(f"  • Total Test Runs: {data_summary.get('total_test_runs', 0)}")
    print(f"  • Valid Test Runs: {data_summary.get('valid_test_runs', 0)}")
    print(f"  • Missing Data: {data_summary.get('missing_data_runs', 0)}")
    print(f"  • Completeness: {data_summary.get('data_completeness', 0):.1%}")


def print_report_info(report_files: Dict[str, str]):
    """Print information about generated reports"""
    print("\nGENERATED REPORTS:")
    for report_type, file_path in report_files.items():
        report_name = report_type.replace('_', ' ').title()
        print(f"  • {report_name}: {file_path}")


def load_test_runs_from_input_file(input_path: str) -> List[str]:
    """Load test run IDs from evaluation_report.json file from script 3"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        
        # Extract test run IDs from the evaluation report structure
        test_runs = []
        
        # Check if it's an evaluation report with detailed_results
        if isinstance(evaluation_data, dict) and 'detailed_results' in evaluation_data:
            detailed_results = evaluation_data['detailed_results']
            
            # Extract from test_runs_by_version (most reliable source)
            if 'test_runs_by_version' in detailed_results:
                test_runs_by_version = detailed_results['test_runs_by_version']
                for version_id, run_ids in test_runs_by_version.items():
                    if isinstance(run_ids, list):
                        test_runs.extend(run_ids)
                    elif isinstance(run_ids, str):
                        test_runs.append(run_ids)
            
            # Also extract from version_comparisons and model_comparisons as backup
            if not test_runs:
                for comparison_type in ['version_comparisons', 'model_comparisons']:
                    comparisons = detailed_results.get(comparison_type, [])
                    for comparison in comparisons:
                        if isinstance(comparison, dict):
                            # Extract test run IDs from comparison
                            for key in ['test_run_id1', 'test_run_id2', 'winner_test_run_id']:
                                if key in comparison and comparison[key]:
                                    test_runs.append(comparison[key])
        
        # Fallback: check if it's a test_runs file structure (backward compatibility)
        elif isinstance(evaluation_data, list):
            for run in evaluation_data:
                if isinstance(run, dict) and 'test_run_id' in run:
                    test_runs.append(run['test_run_id'])
                elif isinstance(run, str):
                    test_runs.append(run)
        elif isinstance(evaluation_data, dict) and 'test_runs' in evaluation_data:
            runs_list = evaluation_data['test_runs']
            if isinstance(runs_list, list):
                for run in runs_list:
                    if isinstance(run, dict) and 'test_run_id' in run:
                        test_runs.append(run['test_run_id'])
                    elif isinstance(run, str):
                        test_runs.append(run)
        
        # Remove duplicates and filter out empty values
        test_runs = list(set([run for run in test_runs if run and isinstance(run, str)]))
        
        if not test_runs:
            logger.error(f"No test run IDs found in input file: {input_path}")
            return []
        
        logger.info(f"Extracted {len(test_runs)} unique test run IDs from {input_path}")
        return test_runs
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading test runs from input file: {e}")
        return []


def run_analysis(args: argparse.Namespace) -> bool:
    """Run the ELO rating analysis"""
    try:
        # Determine test runs
        if args.config:
            config = load_config(args.config)
            test_runs = config.get('test_runs', [])
            
            # Override args with config values if not specified
            if 'api_url' in config and args.api_url == 'http://localhost:8000':
                args.api_url = config['api_url']
            if 'output_dir' in config and args.output_dir == 'elo_analysis_reports':
                args.output_dir = config['output_dir']
        elif args.input:
            test_runs = load_test_runs_from_input_file(args.input)
            if not test_runs:
                logger.error("Failed to extract test run IDs from input file")
                return False
        else:
            test_runs = [run.strip() for run in args.test_runs.split(',') if run.strip()]
        
        # Validate inputs
        if not validate_test_runs(test_runs):
            return False
        
        # Setup logging
        setup_logging(args)
        
        # Dry run check
        if args.dry_run:
            print(f"✓ Dry run successful")
            print(f"  - Test runs: {len(test_runs)}")
            print(f"  - API URL: {args.api_url}")
            print(f"  - Output directory: {args.output_dir}")
            return True
        
        # Initialize analyzer
        logger.info(f"Initializing ELO analyzer with API: {args.api_url}")
        analyzer = EloAnalyzer(args.api_url, args.output_dir)
        
        # Run comprehensive analysis
        print(f"\nStarting comprehensive ELO analysis for {len(test_runs)} test runs...")
        print(f"API URL: {args.api_url}")
        print(f"Output directory: {args.output_dir}")
        
        results = analyzer.run_comprehensive_analysis(test_runs)
        
        # Print results
        if not args.quiet:
            print_data_quality_info(results)
            print_analysis_summary(results)
            
            report_files = results.get('reports', {})
            if report_files:
                print_report_info(report_files)
        
        # Analysis metadata
        metadata = results.get('metadata', {})
        duration = metadata.get('analysis_duration_seconds', 0)
        
        print(f"\n✓ Analysis completed successfully in {duration:.2f} seconds")
        
        # Quick summary for programmatic use
        quick_summary = analyzer.get_quick_summary()
        logger.debug(f"Quick summary: {json.dumps(quick_summary, indent=2, default=str)}")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        print("\n⚠ Analysis interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n✗ Analysis failed: {e}")
        return False


def main():
    """Main entry point"""
    args = parse_arguments()
    
    print("ELO Rating Analysis and Insights")
    print("=" * 40)
    
    success = run_analysis(args)
    
    if success:
        print("\n🎉 ELO rating analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 ELO rating analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
