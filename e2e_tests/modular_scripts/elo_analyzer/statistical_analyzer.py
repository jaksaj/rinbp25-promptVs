#!/usr/bin/env python3
"""
Statistical Analyzer
====================

Performs statistical analysis on ELO rating data.
"""

import logging
import statistics
import math
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on ELO rating data"""
    
    def __init__(self):
        self.confidence_level = 0.95  # 95% confidence interval
    
    def calculate_elo_statistics(self, elo_ratings: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for ELO ratings"""
        logger.info("Calculating ELO rating statistics...")
        
        # Extract ELO scores by type
        elo_scores = []
        version_elo_scores = []
        global_elo_scores = []
        
        for test_run_id, elo_data in elo_ratings.items():
            if not elo_data.get('missing_data', False):
                elo_scores.append(elo_data.get('elo_score', 1000))
                version_elo_scores.append(elo_data.get('version_elo_score', 1000))
                global_elo_scores.append(elo_data.get('global_elo_score', 1000))
        
        stats = {
            'elo_score': self._calculate_score_statistics(elo_scores, "ELO Score"),
            'version_elo_score': self._calculate_score_statistics(version_elo_scores, "Version ELO Score"),
            'global_elo_score': self._calculate_score_statistics(global_elo_scores, "Global ELO Score"),
            'data_quality': {
                'total_ratings': len(elo_ratings),
                'valid_ratings': len(elo_scores),
                'missing_ratings': len(elo_ratings) - len(elo_scores),
                'data_completeness': len(elo_scores) / len(elo_ratings) if elo_ratings else 0
            }
        }
        
        return stats
    
    def _calculate_score_statistics(self, scores: List[float], score_type: str) -> Dict[str, Any]:
        """Calculate detailed statistics for a list of scores"""
        if not scores:
            return {'error': f'No {score_type} data available'}
        
        scores_sorted = sorted(scores)
        n = len(scores)
        
        # Basic statistics
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        mode_score = statistics.mode(scores) if n > 0 else None
        std_dev = statistics.stdev(scores) if n > 1 else 0
        variance = statistics.variance(scores) if n > 1 else 0
        
        # Quartiles and percentiles
        q1 = self._percentile(scores_sorted, 25)
        q3 = self._percentile(scores_sorted, 75)
        iqr = q3 - q1
        
        # Confidence interval for mean
        confidence_interval = self._calculate_confidence_interval(scores, self.confidence_level)
        
        # Range and outliers
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        # Outlier detection (using IQR method)
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [score for score in scores if score < lower_fence or score > upper_fence]
        
        # Distribution shape
        skewness = self._calculate_skewness(scores, mean_score, std_dev)
        kurtosis = self._calculate_kurtosis(scores, mean_score, std_dev)
        
        return {
            'count': n,
            'mean': round(mean_score, 2),
            'median': round(median_score, 2),
            'mode': round(mode_score, 2) if mode_score is not None else None,
            'std_dev': round(std_dev, 2),
            'variance': round(variance, 2),
            'min': round(min_score, 2),
            'max': round(max_score, 2),
            'range': round(score_range, 2),
            'quartiles': {
                'q1': round(q1, 2),
                'q3': round(q3, 2),
                'iqr': round(iqr, 2)
            },
            'percentiles': {
                '5th': round(self._percentile(scores_sorted, 5), 2),
                '10th': round(self._percentile(scores_sorted, 10), 2),
                '25th': round(q1, 2),
                '50th': round(median_score, 2),
                '75th': round(q3, 2),
                '90th': round(self._percentile(scores_sorted, 90), 2),
                '95th': round(self._percentile(scores_sorted, 95), 2)
            },
            'confidence_interval': {
                'level': self.confidence_level,
                'lower': round(confidence_interval[0], 2),
                'upper': round(confidence_interval[1], 2),
                'margin_of_error': round(confidence_interval[1] - mean_score, 2)
            },
            'outliers': {
                'count': len(outliers),
                'values': [round(x, 2) for x in outliers],
                'percentage': round(len(outliers) / n * 100, 1)
            },
            'distribution': {
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'is_normal': abs(skewness) < 0.5 and abs(kurtosis) < 3
            }
        }
    
    def analyze_elo_trends_by_technique(self, elo_ratings: Dict[str, Dict], 
                                      test_runs: Dict[str, Dict], 
                                      prompt_versions: Dict[str, Dict],
                                      input_data: Dict) -> Dict[str, Any]:
        """Analyze ELO trends by prompting technique"""
        logger.info("Analyzing ELO trends by technique...")
        
        technique_elo_data = defaultdict(lambda: {'elo_scores': [], 'version_elo_scores': [], 'global_elo_scores': []})
        
        # Map test runs to techniques
        for test_run_id, elo_data in elo_ratings.items():
            if elo_data.get('missing_data', False):
                continue
                
            # Get test run metadata
            test_run = test_runs.get(test_run_id, {})
            version_id = test_run.get('prompt_version_id')
            
            if version_id:
                # Determine technique from version
                technique = self._get_technique_from_version(version_id, input_data)
                
                if technique:
                    technique_elo_data[technique]['elo_scores'].append(elo_data.get('elo_score', 1000))
                    technique_elo_data[technique]['version_elo_scores'].append(elo_data.get('version_elo_score', 1000))
                    technique_elo_data[technique]['global_elo_scores'].append(elo_data.get('global_elo_score', 1000))
        
        # Calculate statistics for each technique
        technique_analysis = {}
        for technique, data in technique_elo_data.items():
            technique_analysis[technique] = {
                'elo_score': self._calculate_score_statistics(data['elo_scores'], f"{technique} ELO Score"),
                'version_elo_score': self._calculate_score_statistics(data['version_elo_scores'], f"{technique} Version ELO Score"),
                'global_elo_score': self._calculate_score_statistics(data['global_elo_scores'], f"{technique} Global ELO Score"),
                'sample_size': len(data['elo_scores'])
            }
        
        # Compare techniques
        technique_comparison = self._compare_techniques(technique_analysis)
        
        return {
            'technique_analysis': technique_analysis,
            'technique_comparison': technique_comparison,
            'summary': {
                'techniques_analyzed': list(technique_analysis.keys()),
                'total_techniques': len(technique_analysis)
            }
        }
    
    def analyze_elo_trends_by_model(self, elo_ratings: Dict[str, Dict], 
                                  test_runs: Dict[str, Dict],
                                  test_models: List[str]) -> Dict[str, Any]:
        """Analyze ELO trends by model"""
        logger.info("Analyzing ELO trends by model...")
        
        model_elo_data = defaultdict(lambda: {'elo_scores': [], 'version_elo_scores': [], 'global_elo_scores': []})
        
        # Map test runs to models
        for test_run_id, elo_data in elo_ratings.items():
            if elo_data.get('missing_data', False):
                continue
                
            # Get test run metadata
            test_run = test_runs.get(test_run_id, {})
            model = test_run.get('model')
            
            if model:
                model_elo_data[model]['elo_scores'].append(elo_data.get('elo_score', 1000))
                model_elo_data[model]['version_elo_scores'].append(elo_data.get('version_elo_score', 1000))
                model_elo_data[model]['global_elo_scores'].append(elo_data.get('global_elo_score', 1000))
        
        # Calculate statistics for each model
        model_analysis = {}
        for model, data in model_elo_data.items():
            model_analysis[model] = {
                'elo_score': self._calculate_score_statistics(data['elo_scores'], f"{model} ELO Score"),
                'version_elo_score': self._calculate_score_statistics(data['version_elo_scores'], f"{model} Version ELO Score"),
                'global_elo_score': self._calculate_score_statistics(data['global_elo_scores'], f"{model} Global ELO Score"),
                'sample_size': len(data['elo_scores'])
            }
        
        # Compare models
        model_comparison = self._compare_models(model_analysis)
        
        return {
            'model_analysis': model_analysis,
            'model_comparison': model_comparison,
            'summary': {
                'models_analyzed': list(model_analysis.keys()),
                'total_models': len(model_analysis),
                'expected_models': test_models
            }
        }
    
    def _get_technique_from_version(self, version_id: str, input_data: Dict) -> Optional[str]:
        """Determine technique from version ID"""
        if not isinstance(input_data, dict):
            return None
        for prompt_id, version_list in input_data.get('prompt_versions', {}).items():
            if version_id in version_list:
                version_index = version_list.index(version_id)
                # Map index to technique (based on script 1 pattern)
                techniques = ['cot_simple', 'cot_reasoning']
                if version_index < len(techniques):
                    return techniques[version_index]
                else:
                    return f'technique_{version_index}'
        return None
    
    def _compare_techniques(self, technique_analysis: Dict) -> Dict[str, Any]:
        """Compare techniques statistically"""
        if len(technique_analysis) < 2:
            return {'error': 'Need at least 2 techniques for comparison'}
        
        comparisons = {}
        techniques = list(technique_analysis.keys())
        
        # Compare each pair of techniques
        for i in range(len(techniques)):
            for j in range(i + 1, len(techniques)):
                tech1, tech2 = techniques[i], techniques[j]
                
                comparison_key = f"{tech1}_vs_{tech2}"
                comparisons[comparison_key] = {
                    'elo_score_comparison': self._compare_score_distributions(
                        technique_analysis[tech1]['elo_score'],
                        technique_analysis[tech2]['elo_score']
                    ),
                    'version_elo_comparison': self._compare_score_distributions(
                        technique_analysis[tech1]['version_elo_score'],
                        technique_analysis[tech2]['version_elo_score']
                    ),
                    'global_elo_comparison': self._compare_score_distributions(
                        technique_analysis[tech1]['global_elo_score'],
                        technique_analysis[tech2]['global_elo_score']
                    )
                }
        
        return comparisons
    
    def _compare_models(self, model_analysis: Dict) -> Dict[str, Any]:
        """Compare models statistically"""
        if len(model_analysis) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        comparisons = {}
        models = list(model_analysis.keys())
        
        # Compare each pair of models
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {
                    'elo_score_comparison': self._compare_score_distributions(
                        model_analysis[model1]['elo_score'],
                        model_analysis[model2]['elo_score']
                    ),
                    'version_elo_comparison': self._compare_score_distributions(
                        model_analysis[model1]['version_elo_score'],
                        model_analysis[model2]['version_elo_score']
                    ),
                    'global_elo_comparison': self._compare_score_distributions(
                        model_analysis[model1]['global_elo_score'],
                        model_analysis[model2]['global_elo_score']
                    )
                }
        
        return comparisons
    
    def _compare_score_distributions(self, stats1: Dict, stats2: Dict) -> Dict[str, Any]:
        """Compare two score distributions"""
        if 'error' in stats1 or 'error' in stats2:
            return {'error': 'Cannot compare distributions with missing data'}
        
        mean_diff = stats1['mean'] - stats2['mean']
        
        # Effect size (Cohen's d approximation)
        pooled_std = math.sqrt((stats1['variance'] + stats2['variance']) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'mean_difference': round(mean_diff, 2),
            'relative_improvement': round((mean_diff / stats2['mean']) * 100, 1) if stats2['mean'] != 0 else 0,
            'effect_size': round(effect_size, 3),
            'effect_magnitude': self._interpret_effect_size(effect_size),
            'confidence_intervals_overlap': self._check_confidence_interval_overlap(
                stats1['confidence_interval'], stats2['confidence_interval']
            ),
            'practical_significance': abs(mean_diff) > 10  # ELO difference > 10 points
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _check_confidence_interval_overlap(self, ci1: Dict, ci2: Dict) -> bool:
        """Check if confidence intervals overlap"""
        return not (ci1['upper'] < ci2['lower'] or ci2['upper'] < ci1['lower'])
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile of sorted data"""
        if not sorted_data:
            return 0
        
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        floor_k = int(k)
        ceil_k = floor_k + 1
        
        if ceil_k >= len(sorted_data):
            return sorted_data[-1]
        
        if floor_k == k:
            return sorted_data[floor_k]
        
        return sorted_data[floor_k] + (k - floor_k) * (sorted_data[ceil_k] - sorted_data[floor_k])
    
    def _calculate_confidence_interval(self, data: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for the mean"""
        if len(data) < 2:
            mean_val = statistics.mean(data) if data else 0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(data)
        std_err = statistics.stdev(data) / math.sqrt(len(data))
        
        # Using t-distribution approximation (assuming normal distribution for large samples)
        # For 95% confidence level, t-value ≈ 1.96 for large samples
        t_value = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence
        
        margin_error = t_value * std_err
        return (mean_val - margin_error, mean_val + margin_error)
    
    def _calculate_skewness(self, data: List[float], mean_val: float, std_dev: float) -> float:
        """Calculate skewness of the distribution"""
        if std_dev == 0 or len(data) < 3:
            return 0
        
        n = len(data)
        skew_sum = sum(((x - mean_val) / std_dev) ** 3 for x in data)
        return (n / ((n - 1) * (n - 2))) * skew_sum
    
    def _calculate_kurtosis(self, data: List[float], mean_val: float, std_dev: float) -> float:
        """Calculate kurtosis of the distribution"""
        if std_dev == 0 or len(data) < 4:
            return 0
        
        n = len(data)
        kurt_sum = sum(((x - mean_val) / std_dev) ** 4 for x in data)
        
        # Excess kurtosis (normal distribution has kurtosis of 3)
        kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurt_sum
        kurtosis -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        
        return kurtosis

    def calculate_effect_sizes(self, technique_analysis: Dict[str, Any], 
                              model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effect sizes for technique and model comparisons"""
        logger.info("Calculating effect sizes for technique and model comparisons...")
        
        effect_sizes = {
            'technique_effects': {},
            'model_effects': {},
            'summary': {
                'large_effects': [],
                'medium_effects': [],
                'small_effects': []
            }
        }
        
        # Calculate technique effect sizes
        for score_type, techniques in technique_analysis.items():
            if score_type in ['elo_score', 'version_elo_score', 'global_elo_score'] and len(techniques) >= 2:
                technique_effects = {}
                technique_list = list(techniques.items())
                
                # Compare each pair of techniques
                for i in range(len(technique_list)):
                    for j in range(i + 1, len(technique_list)):
                        name1, stats1 = technique_list[i]
                        name2, stats2 = technique_list[j]
                        
                        if stats1.get('count', 0) > 0 and stats2.get('count', 0) > 0:
                            effect_size = self._calculate_cohens_d(stats1, stats2)
                            comparison_key = f"{name1}_vs_{name2}"
                            technique_effects[comparison_key] = {
                                'effect_size': effect_size,
                                'interpretation': self._interpret_effect_size(effect_size),
                                'technique1': name1,
                                'technique2': name2,
                                'mean_difference': stats1.get('mean', 0) - stats2.get('mean', 0)
                            }
                
                effect_sizes['technique_effects'][score_type] = technique_effects
        
        # Calculate model effect sizes
        for score_type, models in model_analysis.items():
            if score_type in ['elo_score', 'version_elo_score', 'global_elo_score'] and len(models) >= 2:
                model_effects = {}
                model_list = list(models.items())
                
                # Compare each pair of models
                for i in range(len(model_list)):
                    for j in range(i + 1, len(model_list)):
                        name1, stats1 = model_list[i]
                        name2, stats2 = model_list[j]
                        
                        if stats1.get('count', 0) > 0 and stats2.get('count', 0) > 0:
                            effect_size = self._calculate_cohens_d(stats1, stats2)
                            comparison_key = f"{name1}_vs_{name2}"
                            model_effects[comparison_key] = {
                                'effect_size': effect_size,
                                'interpretation': self._interpret_effect_size(effect_size),
                                'model1': name1,
                                'model2': name2,
                                'mean_difference': stats1.get('mean', 0) - stats2.get('mean', 0)
                            }
                
                effect_sizes['model_effects'][score_type] = model_effects
        
        # Categorize effect sizes
        all_effects = []
        for score_type, effects in effect_sizes['technique_effects'].items():
            all_effects.extend([(comp, data, 'technique', score_type) for comp, data in effects.items()])
        for score_type, effects in effect_sizes['model_effects'].items():
            all_effects.extend([(comp, data, 'model', score_type) for comp, data in effects.items()])
        
        for comp_name, effect_data, effect_type, score_type in all_effects:
            effect_size = abs(effect_data['effect_size'])
            if effect_size >= 0.8:
                effect_sizes['summary']['large_effects'].append({
                    'comparison': comp_name,
                    'type': effect_type,
                    'score_type': score_type,
                    'effect_size': effect_data['effect_size'],
                    'interpretation': effect_data['interpretation']
                })
            elif effect_size >= 0.5:
                effect_sizes['summary']['medium_effects'].append({
                    'comparison': comp_name,
                    'type': effect_type,
                    'score_type': score_type,
                    'effect_size': effect_data['effect_size'],
                    'interpretation': effect_data['interpretation']
                })
            elif effect_size >= 0.2:
                effect_sizes['summary']['small_effects'].append({
                    'comparison': comp_name,
                    'type': effect_type,
                    'score_type': score_type,
                    'effect_size': effect_data['effect_size'],
                    'interpretation': effect_data['interpretation']
                })
        
        return effect_sizes
    
    def analyze_distributions(self, elo_ratings: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze ELO score distributions"""
        logger.info("Analyzing ELO score distributions...")
        
        # Extract scores by type
        elo_scores = []
        version_elo_scores = []
        global_elo_scores = []
        
        for test_run_id, elo_data in elo_ratings.items():
            if not elo_data.get('missing_data', False):
                elo_scores.append(elo_data.get('elo_score', 1000))
                version_elo_scores.append(elo_data.get('version_elo_score', 1000))
                global_elo_scores.append(elo_data.get('global_elo_score', 1000))
        
        distribution_analysis = {}
        
        for score_type, scores in [
            ('elo_score', elo_scores),
            ('version_elo_score', version_elo_scores),
            ('global_elo_score', global_elo_scores)
        ]:
            if scores:
                analysis = self._analyze_single_distribution(scores, score_type)
                distribution_analysis[score_type] = analysis
        
        # Cross-distribution comparisons
        distribution_analysis['cross_comparisons'] = self._compare_distributions(
            elo_scores, version_elo_scores, global_elo_scores
        )
        
        return distribution_analysis
    
    def _calculate_cohens_d(self, stats1: Dict[str, Any], stats2: Dict[str, Any]) -> float:
        """Calculate Cohen's d effect size between two groups"""
        mean1 = stats1.get('mean', 0)
        mean2 = stats2.get('mean', 0)
        std1 = stats1.get('std_dev', 0)
        std2 = stats2.get('std_dev', 0)
        n1 = stats1.get('count', 0)
        n2 = stats2.get('count', 0)
        
        if n1 == 0 or n2 == 0 or (std1 == 0 and std2 == 0):
            return 0
        
        # Pooled standard deviation
        if n1 == 1 and n2 == 1:
            pooled_std = (std1 + std2) / 2 if (std1 + std2) > 0 else 1
        else:
            pooled_variance = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
            pooled_std = math.sqrt(pooled_variance) if pooled_variance > 0 else 1
        
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    def _analyze_single_distribution(self, scores: List[float], score_type: str) -> Dict[str, Any]:
        """Analyze a single score distribution"""
        if not scores:
            return {'error': f'No {score_type} data available'}
        
        scores_sorted = sorted(scores)
        n = len(scores)
        
        # Basic statistics (reuse existing method)
        basic_stats = self._calculate_score_statistics(scores, score_type)
        
        # Distribution shape analysis
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if n > 1 else 0
        
        # Normality assessment (basic checks)
        skewness = self._calculate_skewness(scores, mean_score, std_dev)
        kurtosis = self._calculate_kurtosis(scores, mean_score, std_dev)
        
        # Distribution classification
        distribution_shape = self._classify_distribution_shape(skewness, kurtosis)
        
        # Percentile analysis
        percentiles = {
            '10th': self._percentile(scores_sorted, 10),
            '25th': self._percentile(scores_sorted, 25),
            '50th': self._percentile(scores_sorted, 50),
            '75th': self._percentile(scores_sorted, 75),
            '90th': self._percentile(scores_sorted, 90),
            '95th': self._percentile(scores_sorted, 95)
        }
        
        # Outlier analysis
        q1 = percentiles['25th']
        q3 = percentiles['75th']
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        outliers = [score for score in scores if score < lower_fence or score > upper_fence]
        outlier_analysis = {
            'count': len(outliers),
            'percentage': (len(outliers) / n) * 100,
            'values': outliers[:10],  # First 10 outliers
            'lower_fence': lower_fence,
            'upper_fence': upper_fence
        }
        
        return {
            'basic_statistics': basic_stats,
            'distribution_shape': distribution_shape,
            'percentiles': percentiles,
            'outlier_analysis': outlier_analysis,
            'normality_indicators': {
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'is_approximately_normal': abs(skewness) < 1 and abs(kurtosis) < 1
            }
        }
    
    def _classify_distribution_shape(self, skewness: float, kurtosis: float) -> Dict[str, Any]:
        """Classify the shape of the distribution"""
        # Skewness classification
        if skewness > 1:
            skew_description = 'highly_right_skewed'
        elif skewness > 0.5:
            skew_description = 'moderately_right_skewed'
        elif skewness > -0.5:
            skew_description = 'approximately_symmetric'
        elif skewness > -1:
            skew_description = 'moderately_left_skewed'
        else:
            skew_description = 'highly_left_skewed'
        
        # Kurtosis classification
        if kurtosis > 2:
            kurt_description = 'highly_peaked'
        elif kurtosis > 0:
            kurt_description = 'peaked'
        elif kurtosis > -1:
            kurt_description = 'normal_peakedness'
        else:
            kurt_description = 'flat'
        
        return {
            'skewness_description': skew_description,
            'kurtosis_description': kurt_description,
            'overall_shape': f"{skew_description}_{kurt_description}",
            'is_normal_like': abs(skewness) < 0.5 and abs(kurtosis) < 1
        }
    
    def _compare_distributions(self, elo_scores: List[float], 
                              version_elo_scores: List[float],
                              global_elo_scores: List[float]) -> Dict[str, Any]:
        """Compare distributions across different ELO score types"""
        comparisons = {}
        
        score_sets = [
            ('elo_score', elo_scores),
            ('version_elo_score', version_elo_scores),
            ('global_elo_score', global_elo_scores)
        ]
        
        # Compare each pair of distributions
        for i in range(len(score_sets)):
            for j in range(i + 1, len(score_sets)):
                name1, scores1 = score_sets[i]
                name2, scores2 = score_sets[j]
                
                if scores1 and scores2:
                    comparison_key = f"{name1}_vs_{name2}"
                    
                    # Basic statistical comparison
                    mean1 = statistics.mean(scores1)
                    mean2 = statistics.mean(scores2)
                    std1 = statistics.stdev(scores1) if len(scores1) > 1 else 0
                    std2 = statistics.stdev(scores2) if len(scores2) > 1 else 0
                    
                    # Effect size
                    stats1 = {'mean': mean1, 'std_dev': std1, 'count': len(scores1)}
                    stats2 = {'mean': mean2, 'std_dev': std2, 'count': len(scores2)}
                    effect_size = self._calculate_cohens_d(stats1, stats2)
                    
                    comparisons[comparison_key] = {
                        'mean_difference': mean1 - mean2,
                        'std_dev_difference': std1 - std2,
                        'effect_size': effect_size,
                        'effect_interpretation': self._interpret_effect_size(effect_size),
                        'correlation': self._calculate_correlation(scores1, scores2) if len(scores1) == len(scores2) else None
                    }
        
        return comparisons
    
    def _calculate_correlation(self, scores1: List[float], scores2: List[float]) -> float:
        """Calculate correlation between two score lists"""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0
        
        try:
            # Calculate Pearson correlation coefficient
            mean1 = statistics.mean(scores1)
            mean2 = statistics.mean(scores2)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2))
            sum_sq1 = sum((x - mean1) ** 2 for x in scores1)
            sum_sq2 = sum((y - mean2) ** 2 for y in scores2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                return 0
            
            return numerator / denominator
        except:
            return 0
