#!/usr/bin/env python3
"""
Analysis Engine
==============

Handles analysis, report generation, and result presentation.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List
from .sampling_strategies import SamplingStrategies

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Handles analysis and report generation for A/B testing results"""
    
    def __init__(self, api_manager):
        self.api_manager = api_manager
    
    def find_best_versions(self, prompt_ids: List[str], prompt_versions: Dict, 
                          test_runs: Dict, comparison_results: List[Dict]) -> Dict[str, Dict]:
        """Find the best performing version for each prompt based on A/B testing results"""
        logger.info("Finding best performing versions based on comparison results...")
        
        best_versions = {}
        
        for prompt_id in prompt_ids:
            # Count wins for each version in this prompt
            version_wins = {}
            version_ids = prompt_versions.get(prompt_id, [])
            
            for version_id in version_ids:
                version_wins[version_id] = 0
            
            # Count wins from comparison results
            prompt_comparisons = [
                comp for comp in comparison_results 
                if comp.get('prompt_id') == prompt_id
            ]
            
            for comparison in prompt_comparisons:
                winner_test_run_id = comparison.get('winner_test_run_id')
                version_id1 = comparison.get('version_id1')
                version_id2 = comparison.get('version_id2')
                
                # Find which version the winning test run belongs to
                if winner_test_run_id:
                    for version_id in [version_id1, version_id2]:
                        for run in test_runs.get(version_id, []):
                            run_id = run['run_id'] if isinstance(run, dict) and 'run_id' in run else run
                            if winner_test_run_id == run_id:
                                version_wins[version_id] = version_wins.get(version_id, 0) + 1
                                break
            
            # Find the version with most wins
            if version_wins:
                best_version_id = max(version_wins.keys(), key=lambda v: version_wins[v])
                best_versions[prompt_id] = {
                    'best_version_id': best_version_id,
                    'wins': version_wins[best_version_id],
                    'total_comparisons': len(prompt_comparisons),
                    'win_rate': version_wins[best_version_id] / len(prompt_comparisons) if prompt_comparisons else 0,
                    'all_version_wins': version_wins
                }
                logger.info(f"Best version for prompt {prompt_id}: {best_version_id} (wins: {version_wins[best_version_id]}/{len(prompt_comparisons)})")
        
        return best_versions
    
    def find_best_models_per_version(self, test_runs: Dict, model_comparison_results: List[Dict]) -> Dict[str, Dict]:
        """Find the best performing model for each version based on model comparison results"""
        logger.info("Finding best performing models for each version based on comparison results...")
        
        best_models = {}
        
        for version_id in test_runs.keys():
            # Count wins for each model in this version
            model_wins = {}
            runs_by_model = SamplingStrategies.group_test_runs_by_model(
                test_runs[version_id], self.api_manager.get_test_run_model
            )
            
            for model in runs_by_model.keys():
                model_wins[model] = 0
            
            # Count wins from model comparison results
            version_comparisons = [
                comp for comp in model_comparison_results 
                if comp.get('version_id') == version_id
            ]
            
            for comparison in version_comparisons:
                winner_test_run_id = comparison.get('winner_test_run_id')
                model1 = comparison.get('model1')
                model2 = comparison.get('model2')
                
                # Find which model the winning test run belongs to
                if winner_test_run_id:
                    for model in [model1, model2]:
                        for run_id in runs_by_model.get(model, []):
                            if winner_test_run_id == run_id:
                                model_wins[model] = model_wins.get(model, 0) + 1
                                break
            
            # Find the model with most wins
            if model_wins and any(wins > 0 for wins in model_wins.values()):
                best_model = max(model_wins.keys(), key=lambda m: model_wins[m])
                best_models[version_id] = {
                    'best_model': best_model,
                    'wins': model_wins[best_model],
                    'total_comparisons': len(version_comparisons),
                    'win_rate': model_wins[best_model] / len(version_comparisons) if version_comparisons else 0,
                    'all_model_wins': model_wins
                }
                logger.info(f"Best model for version {version_id}: {best_model} (wins: {model_wins[best_model]}/{len(version_comparisons)})")
        
        return best_models
    
    def generate_report(self, prompt_ids: List[str], prompt_versions: Dict, test_runs: Dict, 
                       test_models: List[str], comparison_results: List[Dict], 
                       model_comparison_results: List[Dict], max_comparisons_per_version_pair: int) -> Dict:
        """Generate comprehensive test report based on A/B testing and model comparisons"""
        logger.info("Generating comprehensive test report...")
        
        best_versions = self.find_best_versions(prompt_ids, prompt_versions, test_runs, comparison_results)
        best_models_per_version = self.find_best_models_per_version(test_runs, model_comparison_results)
        
        # Calculate summary statistics
        total_versions = sum(len(versions) for versions in prompt_versions.values())
        total_test_runs = sum(len(runs) for runs in test_runs.values())
        total_version_comparisons = len(comparison_results)
        total_model_comparisons = len(model_comparison_results)
        
        # Technique performance based on version comparison wins
        technique_wins = {}
        technique_totals = {}
        
        for comparison in comparison_results:
            prompt_id = comparison.get('prompt_id')
            version_id1 = comparison.get('version_id1')
            version_id2 = comparison.get('version_id2')
            winner_test_run_id = comparison.get('winner_test_run_id')
            
            if prompt_id and version_id1 and version_id2 and winner_test_run_id:
                # Determine which version won
                winner_version_id = None
                for version_id in [version_id1, version_id2]:
                    for run in test_runs.get(version_id, []):
                        run_id = run['run_id'] if isinstance(run, dict) and 'run_id' in run else run
                        if winner_test_run_id == run_id:
                            winner_version_id = version_id
                            break
                
                # Map version to technique
                version_ids = prompt_versions.get(prompt_id, [])
                for version_id in [version_id1, version_id2]:
                    if version_id in version_ids:
                        technique_idx = version_ids.index(version_id)
                        technique = ['cot_simple', 'cot_reasoning'][technique_idx] if technique_idx < 2 else f'technique_{technique_idx}'
                        
                        if technique not in technique_totals:
                            technique_totals[technique] = 0
                            technique_wins[technique] = 0
                        
                        technique_totals[technique] += 1
                        if version_id == winner_version_id:
                            technique_wins[technique] += 1
        
        # Model performance analysis
        model_wins = {}
        model_totals = {}
        
        for comparison in model_comparison_results:
            model1 = comparison.get('model1')
            model2 = comparison.get('model2')
            winner_test_run_id = comparison.get('winner_test_run_id')
            version_id = comparison.get('version_id')
            
            if model1 and model2 and winner_test_run_id and version_id:
                runs_by_model = SamplingStrategies.group_test_runs_by_model(
                    test_runs.get(version_id, []), self.api_manager.get_test_run_model
                )
                
                for model in [model1, model2]:
                    if model not in model_totals:
                        model_totals[model] = 0
                        model_wins[model] = 0
                    
                    model_totals[model] += 1
                    for run_id in runs_by_model.get(model, []):
                        if winner_test_run_id == run_id:
                            model_wins[model] += 1
                            break
        
        # Calculate win rates
        technique_performance = {}
        for technique in technique_totals:
            win_rate = technique_wins[technique] / technique_totals[technique] if technique_totals[technique] > 0 else 0
            technique_performance[technique] = {
                'wins': technique_wins[technique],
                'total_comparisons': technique_totals[technique],
                'win_rate': win_rate
            }
        
        model_performance = {}
        for model in model_totals:
            win_rate = model_wins[model] / model_totals[model] if model_totals[model] > 0 else 0
            model_performance[model] = {
                'wins': model_wins[model],
                'total_comparisons': model_totals[model],
                'win_rate': win_rate
            }
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_prompts': len(prompt_ids),
                'total_versions': total_versions,
                'total_test_runs': total_test_runs,
                'total_version_comparisons': total_version_comparisons,
                'total_model_comparisons': total_model_comparisons,
                'total_comparisons': total_version_comparisons + total_model_comparisons,
                'models_tested': test_models,
                'evaluation_model': self.api_manager.evaluation_model,
                'evaluation_method': 'Two-Tier A/B Testing (Model + Version Comparisons)',
                'max_comparisons_per_pair': max_comparisons_per_version_pair
            },
            'technique_performance': technique_performance,
            'model_performance': model_performance,
            'best_versions_per_prompt': best_versions,
            'best_models_per_version': best_models_per_version,
            'detailed_results': {
                'version_comparisons': comparison_results,
                'model_comparisons': model_comparison_results,
                'test_runs_by_version': test_runs,
                'versions_by_prompt': prompt_versions
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """Save evaluation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../output/evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filename}")
        return filename
    
    def print_summary(self, report: Dict):
        """Print a human-readable summary of the results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        summary = report['test_summary']
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Prompts: {summary['total_prompts']}")
        print(f"Total Versions: {summary['total_versions']}")
        print(f"Total Test Runs: {summary['total_test_runs']}")
        print(f"Version Comparisons: {summary['total_version_comparisons']}")
        print(f"Model Comparisons: {summary['total_model_comparisons']}")
        print(f"Total Comparisons: {summary['total_comparisons']}")
        print(f"Max Comparisons per Pair: {summary['max_comparisons_per_pair']}")
        print(f"Models Tested: {', '.join(summary['models_tested'])}")
        print(f"Evaluation Model: {summary['evaluation_model']}")
        print(f"Evaluation Method: {summary['evaluation_method']}")
        
        print("\nTECHNIQUE PERFORMANCE (VERSION COMPARISONS):")
        print("-" * 50)
        technique_perf = report['technique_performance']
        if technique_perf:
            for technique, perf in technique_perf.items():
                win_rate = perf['win_rate'] * 100
                print(f"{technique:15}: {perf['wins']:3}/{perf['total_comparisons']:3} wins ({win_rate:5.1f}%)")
        else:
            print("No technique performance data available")
        
        print("\nMODEL PERFORMANCE (MODEL COMPARISONS):")
        print("-" * 50)
        model_perf = report['model_performance']
        if model_perf:
            for model, perf in model_perf.items():
                win_rate = perf['win_rate'] * 100
                print(f"{model:15}: {perf['wins']:3}/{perf['total_comparisons']:3} wins ({win_rate:5.1f}%)")
        else:
            print("No model performance data available")
        
        print("\nBEST VERSIONS PER PROMPT:")
        print("-" * 40)
        best_versions = report['best_versions_per_prompt']
        if best_versions:
            for prompt_id, best_info in best_versions.items():
                win_rate = best_info['win_rate'] * 100
                print(f"Prompt {prompt_id}: Version {best_info['best_version_id']} ({best_info['wins']}/{best_info['total_comparisons']} wins, {win_rate:.1f}%)")
        else:
            print("No best version data available")
        
        print("\nBEST MODELS PER VERSION:")
        print("-" * 40)
        best_models = report['best_models_per_version']
        if best_models:
            for version_id, best_info in best_models.items():
                win_rate = best_info['win_rate'] * 100
                print(f"Version {version_id}: {best_info['best_model']} ({best_info['wins']}/{best_info['total_comparisons']} wins, {win_rate:.1f}%)")
        else:
            print("No best model data available")
        
        print("="*80)
