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
    """Load test run IDs from evaluation_report.json file from script 3 (custom structure support)"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)

        test_runs = set()
        # Try to extract from the custom structure
        # Look for runs under best_versions_per_prompt or versions_by_prompt
        if 'best_versions_per_prompt' in evaluation_data:
            bvp = evaluation_data['best_versions_per_prompt']
            # Try to find all version IDs
            all_version_ids = set()
            for prompt_data in bvp.values():
                if 'all_version_wins' in prompt_data:
                    all_version_ids.update(prompt_data['all_version_wins'].keys())
                if 'best_version_id' in prompt_data:
                    all_version_ids.add(prompt_data['best_version_id'])
            # Now look for runs for each version
            for version_id in all_version_ids:
                runs = evaluation_data.get('runs_by_version', {}).get(version_id, [])
                for run in runs:
                    if isinstance(run, dict) and 'run_id' in run:
                        test_runs.add(run['run_id'])
        # Also check for versions_by_prompt (for completeness)
        if 'versions_by_prompt' in evaluation_data:
            for version_list in evaluation_data['versions_by_prompt'].values():
                for version_id in version_list:
                    runs = evaluation_data.get('runs_by_version', {}).get(version_id, [])
                    for run in runs:
                        if isinstance(run, dict) and 'run_id' in run:
                            test_runs.add(run['run_id'])
        # Fallback: try to find any run_id in the whole file
        if not test_runs:
            def find_run_ids(obj):
                found = set()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == 'run_id' and isinstance(v, str):
                            found.add(v)
                        else:
                            found.update(find_run_ids(v))
                elif isinstance(obj, list):
                    for item in obj:
                        found.update(find_run_ids(item))
                return found
            test_runs = find_run_ids(evaluation_data)
        test_runs = list(test_runs)
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
        # Ensure API URL uses /api as base path
        if args.api_url.rstrip("/").endswith(":8000"):
            api_url = args.api_url.rstrip("/") + "/api"
        elif not args.api_url.rstrip("/").endswith("/api"):
            api_url = args.api_url.rstrip("/") + "/api"
        else:
            api_url = args.api_url.rstrip("/")
        analyzer = EloAnalyzer(api_url, args.output_dir)
        
        # Run comprehensive analysis
        print(f"\nStarting comprehensive ELO analysis for {len(test_runs)} test runs...")
        print(f"API URL: {api_url}")
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

    print("ELO Rating Analysis and Insights (Custom Analysis)")
    print("=" * 40)

    # Step 1: Load test run IDs from evaluation_report.json
    if not args.input:
        print("Error: --input evaluation_report.json is required.")
        sys.exit(1)
    test_run_ids = load_test_runs_from_input_file(args.input)
    if not test_run_ids:
        print("No test run IDs found in input file.")
        sys.exit(1)

    # Step 2: Collect all needed data using bulk endpoints
    from elo_analyzer.elo_data_collector import EloDataCollector
    # Ensure API URL uses /api as base path
    api_url = args.api_url.rstrip("/")
    if not api_url.endswith("/api"):
        api_url = api_url + "/api"
    collector = EloDataCollector(api_url)

    print(f"Fetching ELO ratings, test run metadata, and prompt version metadata for {len(test_run_ids)} test runs...")
    elo_ratings = collector.collect_all_elo_ratings(test_run_ids)
    test_run_metadata = collector.collect_test_run_metadata(test_run_ids)
    version_ids = list({tr.get('prompt_version_id') for tr in test_run_metadata.values() if tr.get('prompt_version_id')})
    prompt_version_metadata = collector.collect_prompt_version_metadata(version_ids)

    # Log a sample of each entity for debugging
    import json  # Ensure json is imported before logging samples
    if elo_ratings:
        sample_elo = next(iter(elo_ratings.values()))
        logger.info(f"Sample ELO rating: {json.dumps(sample_elo, indent=2)}")
    else:
        logger.warning("No ELO ratings fetched.")
    if test_run_metadata:
        sample_tr = next(iter(test_run_metadata.values()))
        logger.info(f"Sample test run metadata: {json.dumps(sample_tr, indent=2)}")
    else:
        logger.warning("No test run metadata fetched.")
    if prompt_version_metadata:
        sample_pv = next(iter(prompt_version_metadata.values()))
        logger.info(f"Sample prompt version metadata: {json.dumps(sample_pv, indent=2)}")
    else:
        logger.warning("No prompt version metadata fetched.")
    logger.info(f"All version_ids: {version_ids}")

    # --- NEW: Load runs_by_version from evaluation report for model mapping ---
    def find_runs_by_version(obj):
        if isinstance(obj, dict):
            if 'runs_by_version' in obj:
                return obj['runs_by_version']
            for v in obj.values():
                found = find_runs_by_version(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = find_runs_by_version(item)
                if found:
                    return found
        return None

    runs_by_version = {}
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        logger.info(f"Top-level keys in evaluation report: {list(evaluation_data.keys())}")
        if 'runs_by_version' in evaluation_data:
            runs_by_version = evaluation_data['runs_by_version']
            logger.info("Found runs_by_version at top level.")
        elif 'detailed_results' in evaluation_data and 'runs_by_version' in evaluation_data['detailed_results']:
            runs_by_version = evaluation_data['detailed_results']['runs_by_version']
            logger.info("Found runs_by_version in detailed_results.")
        else:
            runs_by_version = find_runs_by_version(evaluation_data) or {}
            if runs_by_version:
                logger.info("Found runs_by_version recursively.")
        # Log a sample of runs_by_version
        if runs_by_version:
            sample_ver, sample_runs = next(iter(runs_by_version.items()))
            logger.info(f"Sample runs_by_version: version_id={sample_ver}, runs={json.dumps(sample_runs, indent=2)}")
        else:
            logger.warning("No runs_by_version found in evaluation report.")
            # Print first 200 chars of file for debugging
            with open(args.input, 'r', encoding='utf-8') as f2:
                logger.warning(f"First 200 chars of file: {f2.read(200)}")
    except Exception as e:
        logger.warning(f"Could not load runs_by_version from evaluation report: {e}")

    # Build reverse mappings for model and technique
    model_to_runs = {}
    technique_to_versions = {}
    prompt_to_versions = {}
    version_to_prompt = {}
    version_to_model = {}
    version_to_technique = {}
    for tr in test_run_metadata.values():
        version = tr.get('prompt_version_id')
        # Fallback: get prompt from test run metadata or from prompt_version_metadata
        prompt = tr.get('prompt_id')
        if not prompt and version and version in prompt_version_metadata:
            prompt = prompt_version_metadata[version].get('prompt_id')
        # Try to get model from test run metadata, else from runs_by_version
        model = tr.get('model_used') or tr.get('model') or tr.get('model_id')
        if not model and version and version in runs_by_version:
            runs = runs_by_version[version]
            if runs and isinstance(runs, list) and 'model_name' in runs[0]:
                model = runs[0]['model_name']
        # Fallback: get technique from prompt_version_metadata, or use version/notes as fallback
        technique = None
        if version and version in prompt_version_metadata:
            technique = prompt_version_metadata[version].get('technique')
            if not technique:
                # Try to infer from 'version' or 'notes' fields
                technique = prompt_version_metadata[version].get('version') or prompt_version_metadata[version].get('notes')
        if model and version:
            model_to_runs.setdefault(model, []).append(version)
            version_to_model[version] = model
        if technique and version:
            technique_to_versions.setdefault(technique, []).append(version)
            version_to_technique[version] = technique
        if prompt and version:
            prompt_to_versions.setdefault(prompt, []).append(version)
            version_to_prompt[version] = prompt

    # Build ELO per version (both version_elo and global_elo)
    version_elo = {}
    version_global_elo = {}
    for tr in test_run_metadata.values():
        version = tr.get('prompt_version_id')
        tr_id = tr.get('id')
        if version and tr_id and tr_id in elo_ratings:
            version_elo.setdefault(version, []).append(elo_ratings[tr_id].get('version_elo_score', 1000))
            version_global_elo.setdefault(version, []).append(elo_ratings[tr_id].get('global_elo_score', 1000))
    avg_version_elo_per_version = {v: sum(scores)/len(scores) for v, scores in version_elo.items() if scores}
    avg_global_elo_per_version = {v: sum(scores)/len(scores) for v, scores in version_global_elo.items() if scores}

    # 1. Model comparison (use version elo)
    model_comparison = {}
    for model, versions in model_to_runs.items():
        elos = [avg_version_elo_per_version.get(v, 1000) for v in versions]
        model_comparison[model] = {
            'avg_elo': sum(elos)/len(elos) if elos else 1000,
            'num_prompt_versions': len(versions)
        }
    # Number of prompt versions where the model's test run had highest version elo
    prompt_best_model = {}
    for prompt, versions in prompt_to_versions.items():
        best_v = max(versions, key=lambda v: avg_version_elo_per_version.get(v, 1000))
        best_model = version_to_model.get(best_v)
        if best_model:
            prompt_best_model.setdefault(best_model, 0)
            prompt_best_model[best_model] += 1
    for model in model_comparison:
        model_comparison[model]['num_prompts_with_highest_elo'] = prompt_best_model.get(model, 0)

    # 2. Technique comparison (use global elo)
    technique_comparison = {}
    for technique, versions in technique_to_versions.items():
        elos = [avg_global_elo_per_version.get(v, 1000) for v in versions]
        technique_comparison[technique] = {
            'avg_elo': sum(elos)/len(elos) if elos else 1000,
            'num_prompt_versions': len(versions)
        }
    # Number of prompts where a prompt version using the technique had highest global elo
    prompt_best_technique = {}
    for prompt, versions in prompt_to_versions.items():
        best_v = max(versions, key=lambda v: avg_global_elo_per_version.get(v, 1000))
        best_tech = version_to_technique.get(best_v)
        if best_tech:
            prompt_best_technique.setdefault(best_tech, 0)
            prompt_best_technique[best_tech] += 1
    for technique in technique_comparison:
        technique_comparison[technique]['num_prompts_with_highest_elo'] = prompt_best_technique.get(technique, 0)

    # 3. Overall results
    # Best prompt version per prompt (use avg global elo)
    best_version_per_prompt = {}
    best_version_per_prompt_technique = {}
    for prompt, versions in prompt_to_versions.items():
        best_v = max(versions, key=lambda v: avg_global_elo_per_version.get(v, 1000))
        best_version_per_prompt[prompt] = best_v
        best_version_per_prompt_technique[prompt] = version_to_technique.get(best_v)
    # Best model per prompt (find avg version elo for each test run that uses specific model, return highest per model)
    best_model_per_prompt = {}
    for prompt, versions in prompt_to_versions.items():
        model_scores = {}
        for v in versions:
            model = version_to_model.get(v)
            if not model:
                continue
            score = avg_version_elo_per_version.get(v, 1000)
            model_scores.setdefault(model, []).append(score)
        # Find best model for this prompt
        best_model = None
        best_score = float('-inf')
        for model, scores in model_scores.items():
            avg_score = sum(scores)/len(scores) if scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        if best_model:
            best_model_per_prompt[prompt] = {'model': best_model, 'avg_version_elo': best_score}

    # Prepare output
    results = {
        'model_comparison': model_comparison,
        'technique_comparison': technique_comparison,
        'best_version_per_prompt': best_version_per_prompt,
        'best_version_per_prompt_technique': best_version_per_prompt_technique,
        'best_model_per_prompt': best_model_per_prompt,
        'avg_version_elo_per_version': avg_version_elo_per_version,
        'avg_global_elo_per_version': avg_global_elo_per_version
    }

    # Print concise summary
    print("\nMODEL COMPARISON:")
    for model, data in model_comparison.items():
        print(f"  {model}: avg_version_elo={data['avg_elo']:.2f}, #prompt_versions={data['num_prompt_versions']}, #prompts_with_highest_elo={data['num_prompts_with_highest_elo']}")
    print("\nTECHNIQUE COMPARISON:")
    for tech, data in technique_comparison.items():
        print(f"  {tech}: avg_global_elo={data['avg_elo']:.2f}, #prompt_versions={data['num_prompt_versions']}, #prompts_with_highest_elo={data['num_prompts_with_highest_elo']}")
    print("\nBEST PROMPT VERSION PER PROMPT:")
    for prompt, version in best_version_per_prompt.items():
        tech = best_version_per_prompt_technique.get(prompt)
        print(f"  Prompt {prompt}: Version {version} (technique: {tech})")
    print("\nBEST MODEL PER PROMPT:")
    for prompt, info in best_model_per_prompt.items():
        print(f"  Prompt {prompt}: Model {info['model']} (avg_version_elo={info['avg_version_elo']:.2f})")

    # Save to JSON and Markdown
    import datetime
    import json
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(output_dir, f'elo_custom_analysis_{timestamp}.json')
    md_path = os.path.join(output_dir, f'elo_custom_analysis_{timestamp}.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# ELO Custom Analysis ({timestamp})\n\n")
        f.write("## Model Comparison\n")
        for model, data in model_comparison.items():
            f.write(f"- **{model}**: avg_version_elo={data['avg_elo']:.2f}, #prompt_versions={data['num_prompt_versions']}, #prompts_with_highest_elo={data['num_prompts_with_highest_elo']}\n")
        f.write("\n## Technique Comparison\n")
        for tech, data in technique_comparison.items():
            f.write(f"- **{tech}**: avg_global_elo={data['avg_elo']:.2f}, #prompt_versions={data['num_prompt_versions']}, #prompts_with_highest_elo={data['num_prompts_with_highest_elo']}\n")
        f.write("\n## Best Prompt Version Per Prompt\n")
        for prompt, version in best_version_per_prompt.items():
            tech = best_version_per_prompt_technique.get(prompt)
            f.write(f"- Prompt {prompt}: Version {version} (technique: {tech})\n")
        f.write("\n## Best Model Per Prompt\n")
        for prompt, info in best_model_per_prompt.items():
            f.write(f"- Prompt {prompt}: Model {info['model']} (avg_version_elo={info['avg_version_elo']:.2f})\n")
    print(f"\nResults saved to {json_path} and {md_path}")
    print("\n🎉 Custom ELO analysis completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
