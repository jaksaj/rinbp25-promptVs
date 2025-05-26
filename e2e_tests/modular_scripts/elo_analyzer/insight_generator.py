#!/usr/bin/env python3
"""
Insight Generator
=================

Generates actionable insights from ELO rating analysis.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generates business intelligence insights from ELO analysis"""
    
    def __init__(self):
        self.elo_baseline = 1000  # Standard ELO baseline
        self.performance_thresholds = {
            'excellent': 1200,
            'good': 1100,
            'average': 1000,
            'poor': 900,
            'very_poor': 800
        }
        self.effect_size_thresholds = {
            'large': 0.8,
            'medium': 0.5,
            'small': 0.2
        }
    
    def generate_comprehensive_insights(self, stats: Dict[str, Any], 
                                      technique_analysis: Dict[str, Any],
                                      model_analysis: Dict[str, Any],
                                      elo_ratings: Dict[str, Dict],
                                      metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive insights from all analysis components"""
        logger.info("Generating comprehensive insights...")
        
        insights = {
            'performance_insights': self._generate_performance_insights(stats),
            'technique_insights': self._generate_technique_insights(technique_analysis),
            'model_insights': self._generate_model_insights(model_analysis),
            'comparative_insights': self._generate_comparative_insights(stats, technique_analysis, model_analysis),
            'actionable_recommendations': self._generate_recommendations(stats, technique_analysis, model_analysis),
            'risk_assessment': self._generate_risk_assessment(stats, elo_ratings),
            'optimization_opportunities': self._generate_optimization_opportunities(technique_analysis, model_analysis),
            'data_quality_insights': self._generate_data_quality_insights(stats, metadata)
        }
        
        return insights
    
    def _generate_performance_insights(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about overall performance"""
        insights = {
            'overall_performance': {},
            'score_distributions': {},
            'performance_stability': {},
            'outlier_analysis': {}
        }
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            if not score_stats:
                continue
                
            mean_score = score_stats.get('mean', self.elo_baseline)
            std_dev = score_stats.get('std_dev', 0)
            
            # Performance level classification
            performance_level = self._classify_performance(mean_score)
            
            # Consistency analysis
            consistency = self._analyze_consistency(std_dev, mean_score)
            
            insights['overall_performance'][score_type] = {
                'performance_level': performance_level,
                'score_above_baseline': mean_score > self.elo_baseline,
                'baseline_difference': mean_score - self.elo_baseline,
                'performance_description': self._get_performance_description(performance_level, mean_score)
            }
            
            insights['performance_stability'][score_type] = {
                'consistency_level': consistency,
                'coefficient_of_variation': (std_dev / mean_score) * 100 if mean_score != 0 else 0,
                'stability_description': self._get_stability_description(consistency)
            }
            
            # Distribution insights
            quartiles = score_stats.get('quartiles', {})
            insights['score_distributions'][score_type] = {
                'distribution_shape': self._analyze_distribution_shape(score_stats),
                'quartile_analysis': self._analyze_quartiles(quartiles),
                'range_analysis': {
                    'range': score_stats.get('max', 0) - score_stats.get('min', 0),
                    'interquartile_range': quartiles.get('q3', 0) - quartiles.get('q1', 0),
                    'range_interpretation': self._interpret_range(score_stats.get('max', 0) - score_stats.get('min', 0))
                }
            }
            
            # Outlier insights
            outliers = score_stats.get('outliers', {})
            insights['outlier_analysis'][score_type] = {
                'outlier_count': outliers.get('count', 0),
                'outlier_percentage': outliers.get('percentage', 0),
                'outlier_impact': self._assess_outlier_impact(outliers.get('percentage', 0)),
                'outlier_recommendations': self._get_outlier_recommendations(outliers.get('percentage', 0))
            }
        
        return insights
    
    def _generate_technique_insights(self, technique_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about technique performance"""
        insights = {
            'technique_rankings': {},
            'technique_effectiveness': {},
            'technique_recommendations': {},
            'technique_stability': {}
        }
        
        if not technique_analysis:
            return insights
        
        # Rank techniques by each ELO score type
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            technique_scores = technique_analysis.get(score_type, {})
            if not technique_scores:
                continue
                
            # Sort techniques by mean score
            sorted_techniques = sorted(
                technique_scores.items(),
                key=lambda x: x[1].get('mean', 0),
                reverse=True
            )
            
            insights['technique_rankings'][score_type] = {
                'top_performers': sorted_techniques[:3],
                'bottom_performers': sorted_techniques[-3:],
                'performance_gap': self._calculate_performance_gap(sorted_techniques)
            }
            
            # Analyze effectiveness
            for technique, stats in technique_scores.items():
                mean_score = stats.get('mean', self.elo_baseline)
                sample_size = stats.get('count', 0)
                
                effectiveness = self._assess_technique_effectiveness(mean_score, sample_size)
                
                insights['technique_effectiveness'][technique] = {
                    score_type: {
                        'effectiveness_level': effectiveness,
                        'reliability': self._assess_reliability(sample_size),
                        'recommendation': self._get_technique_recommendation(effectiveness, sample_size)
                    }
                }
        
        return insights
    
    def _generate_model_insights(self, model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about model performance"""
        insights = {
            'model_rankings': {},
            'model_suitability': {},
            'model_recommendations': {},
            'cross_model_analysis': {}
        }
        
        if not model_analysis:
            return insights
        
        # Analyze each ELO score type
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            model_scores = model_analysis.get(score_type, {})
            if not model_scores:
                continue
                
            # Rank models
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1].get('mean', 0),
                reverse=True
            )
            
            insights['model_rankings'][score_type] = {
                'best_model': sorted_models[0] if sorted_models else None,
                'worst_model': sorted_models[-1] if sorted_models else None,
                'model_performance_range': self._calculate_model_range(sorted_models)
            }
            
            # Assess model suitability
            for model, stats in model_scores.items():
                mean_score = stats.get('mean', self.elo_baseline)
                consistency = stats.get('std_dev', 0)
                sample_size = stats.get('count', 0)
                
                suitability = self._assess_model_suitability(mean_score, consistency, sample_size)
                
                insights['model_suitability'][model] = {
                    score_type: suitability
                }
        
        # Cross-model analysis
        insights['cross_model_analysis'] = self._perform_cross_model_analysis(model_analysis)
        
        return insights
    
    def _generate_comparative_insights(self, stats: Dict[str, Any], 
                                     technique_analysis: Dict[str, Any],
                                     model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative insights across different dimensions"""
        insights = {
            'score_type_comparison': {},
            'technique_vs_model_impact': {},
            'performance_correlations': {},
            'optimization_priorities': {}
        }
        
        # Compare ELO score types
        score_means = {}
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            score_means[score_type] = score_stats.get('mean', self.elo_baseline)
        
        insights['score_type_comparison'] = {
            'highest_performing_metric': max(score_means, key=score_means.get) if score_means else None,
            'metric_consistency': self._analyze_metric_consistency(score_means),
            'metric_recommendations': self._get_metric_recommendations(score_means)
        }
        
        # Analyze technique vs model impact
        insights['technique_vs_model_impact'] = self._compare_technique_vs_model_impact(
            technique_analysis, model_analysis
        )
        
        return insights
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                technique_analysis: Dict[str, Any],
                                model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            mean_score = score_stats.get('mean', self.elo_baseline)
            
            if mean_score < self.performance_thresholds['average']:
                recommendations.append({
                    'type': 'performance_improvement',
                    'priority': 'high',
                    'metric': score_type,
                    'recommendation': f"Focus on improving {score_type.replace('_', ' ')} - currently below average",
                    'current_score': mean_score,
                    'target_score': self.performance_thresholds['good'],
                    'actions': self._get_performance_improvement_actions(score_type)
                })
        
        # Technique recommendations
        if technique_analysis:
            technique_recs = self._generate_technique_recommendations(technique_analysis)
            recommendations.extend(technique_recs)
        
        # Model recommendations
        if model_analysis:
            model_recs = self._generate_model_recommendations(model_analysis)
            recommendations.extend(model_recs)
        
        # Data quality recommendations
        data_quality = stats.get('data_quality', {})
        if data_quality.get('data_completeness', 1.0) < 0.9:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'recommendation': "Improve data collection completeness",
                'current_completeness': data_quality.get('data_completeness', 0),
                'target_completeness': 0.95,
                'actions': [
                    "Review data collection processes",
                    "Implement better error handling",
                    "Add data validation checks"
                ]
            })
        
        return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    def _generate_risk_assessment(self, stats: Dict[str, Any], elo_ratings: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate risk assessment based on performance patterns"""
        risks = {
            'performance_risks': [],
            'consistency_risks': [],
            'data_risks': [],
            'overall_risk_level': 'low'
        }
        
        # Analyze performance risks
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            mean_score = score_stats.get('mean', self.elo_baseline)
            std_dev = score_stats.get('std_dev', 0)
            
            if mean_score < self.performance_thresholds['poor']:
                risks['performance_risks'].append({
                    'metric': score_type,
                    'risk_level': 'high',
                    'description': f"Poor performance in {score_type.replace('_', ' ')}",
                    'impact': "May indicate fundamental issues with prompts or evaluation"
                })
            
            # High variability risk
            cv = (std_dev / mean_score) * 100 if mean_score != 0 else 0
            if cv > 20:  # High coefficient of variation
                risks['consistency_risks'].append({
                    'metric': score_type,
                    'risk_level': 'medium',
                    'description': f"High variability in {score_type.replace('_', ' ')}",
                    'coefficient_of_variation': cv,
                    'impact': "Inconsistent performance may affect reliability"
                })
        
        # Data quality risks
        data_quality = stats.get('data_quality', {})
        if data_quality.get('data_completeness', 1.0) < 0.8:
            risks['data_risks'].append({
                'risk_level': 'high',
                'description': "Low data completeness",
                'completeness': data_quality.get('data_completeness', 0),
                'impact': "Insufficient data may lead to unreliable insights"
            })
        
        # Determine overall risk level
        high_risks = sum(1 for risk_category in risks.values() 
                        if isinstance(risk_category, list) 
                        for risk in risk_category 
                        if risk.get('risk_level') == 'high')
        
        if high_risks > 0:
            risks['overall_risk_level'] = 'high'
        elif any(risk.get('risk_level') == 'medium' for risk_category in risks.values() 
                if isinstance(risk_category, list) for risk in risk_category):
            risks['overall_risk_level'] = 'medium'
        
        return risks
    
    def _generate_optimization_opportunities(self, technique_analysis: Dict[str, Any],
                                           model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Technique optimization
        if technique_analysis:
            for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                technique_scores = technique_analysis.get(score_type, {})
                if technique_scores:
                    best_technique = max(technique_scores.items(), key=lambda x: x[1].get('mean', 0))
                    worst_technique = min(technique_scores.items(), key=lambda x: x[1].get('mean', 0))
                    
                    performance_gap = best_technique[1].get('mean', 0) - worst_technique[1].get('mean', 0)
                    
                    if performance_gap > 50:  # Significant gap
                        opportunities.append({
                            'type': 'technique_optimization',
                            'opportunity': f"Standardize on best-performing technique for {score_type}",
                            'best_technique': best_technique[0],
                            'potential_improvement': performance_gap,
                            'impact': 'high' if performance_gap > 100 else 'medium'
                        })
        
        # Model optimization
        if model_analysis:
            for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                model_scores = model_analysis.get(score_type, {})
                if model_scores:
                    best_model = max(model_scores.items(), key=lambda x: x[1].get('mean', 0))
                    
                    opportunities.append({
                        'type': 'model_optimization',
                        'opportunity': f"Optimize for best-performing model in {score_type}",
                        'best_model': best_model[0],
                        'performance_score': best_model[1].get('mean', 0),
                        'impact': 'high' if best_model[1].get('mean', 0) > 1100 else 'medium'
                    })
        
        return opportunities
    
    def _generate_data_quality_insights(self, stats: Dict[str, Any], metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate insights about data quality"""
        data_quality = stats.get('data_quality', {})
        
        insights = {
            'completeness_analysis': {
                'data_completeness': data_quality.get('data_completeness', 0),
                'missing_data_impact': self._assess_missing_data_impact(data_quality.get('data_completeness', 0)),
                'completeness_recommendations': self._get_completeness_recommendations(data_quality.get('data_completeness', 0))
            },
            'sample_size_analysis': {
                'total_samples': data_quality.get('total_ratings', 0),
                'valid_samples': data_quality.get('valid_ratings', 0),
                'sample_adequacy': self._assess_sample_adequacy(data_quality.get('valid_ratings', 0)),
                'sample_recommendations': self._get_sample_recommendations(data_quality.get('valid_ratings', 0))
            },
            'data_reliability': {
                'reliability_score': self._calculate_reliability_score(data_quality),
                'reliability_level': self._classify_reliability(data_quality),
                'reliability_factors': self._identify_reliability_factors(data_quality)
            }
        }
        
        return insights
    
    # Helper methods for classification and analysis
    
    def _classify_performance(self, score: float) -> str:
        """Classify performance level based on score"""
        if score >= self.performance_thresholds['excellent']:
            return 'excellent'
        elif score >= self.performance_thresholds['good']:
            return 'good'
        elif score >= self.performance_thresholds['average']:
            return 'average'
        elif score >= self.performance_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    def _analyze_consistency(self, std_dev: float, mean: float) -> str:
        """Analyze consistency based on coefficient of variation"""
        if mean == 0:
            return 'unknown'
        
        cv = (std_dev / mean) * 100
        if cv < 5:
            return 'very_consistent'
        elif cv < 10:
            return 'consistent'
        elif cv < 20:
            return 'moderately_consistent'
        else:
            return 'inconsistent'
    
    def _get_performance_description(self, level: str, score: float) -> str:
        """Get descriptive text for performance level"""
        descriptions = {
            'excellent': f"Exceptional performance (score: {score:.1f}). Well above industry standards.",
            'good': f"Good performance (score: {score:.1f}). Above average with room for optimization.",
            'average': f"Average performance (score: {score:.1f}). Meets baseline expectations.",
            'poor': f"Below average performance (score: {score:.1f}). Requires immediate attention.",
            'very_poor': f"Poor performance (score: {score:.1f}). Critical improvements needed."
        }
        return descriptions.get(level, f"Performance score: {score:.1f}")
    
    def _get_stability_description(self, consistency: str) -> str:
        """Get descriptive text for stability level"""
        descriptions = {
            'very_consistent': "Very stable performance with minimal variation.",
            'consistent': "Stable performance with low variation.",
            'moderately_consistent': "Moderate stability with some variation.",
            'inconsistent': "Unstable performance with high variation."
        }
        return descriptions.get(consistency, "Stability unknown")
    
    def _analyze_distribution_shape(self, stats: Dict[str, Any]) -> str:
        """Analyze the shape of the distribution"""
        mean = stats.get('mean', 0)
        median = stats.get('median', 0)
        
        if abs(mean - median) < 5:  # Small difference
            return 'normal'
        elif mean > median:
            return 'right_skewed'
        else:
            return 'left_skewed'
    
    def _analyze_quartiles(self, quartiles: Dict[str, float]) -> Dict[str, Any]:
        """Analyze quartile distribution"""
        if not quartiles:
            return {}
        
        q1, q2, q3 = quartiles.get('q1', 0), quartiles.get('q2', 0), quartiles.get('q3', 0)
        iqr = q3 - q1
        
        return {
            'interquartile_range': iqr,
            'lower_half_spread': q2 - q1,
            'upper_half_spread': q3 - q2,
            'symmetry': 'symmetric' if abs((q2 - q1) - (q3 - q2)) < 10 else 'asymmetric'
        }
    
    def _interpret_range(self, range_value: float) -> str:
        """Interpret the range of scores"""
        if range_value < 50:
            return 'narrow - consistent performance'
        elif range_value < 150:
            return 'moderate - some variation in performance'
        else:
            return 'wide - high variation in performance'
    
    def _assess_outlier_impact(self, outlier_percentage: float) -> str:
        """Assess the impact of outliers"""
        if outlier_percentage < 2:
            return 'minimal'
        elif outlier_percentage < 5:
            return 'low'
        elif outlier_percentage < 10:
            return 'moderate'
        else:
            return 'high'
    
    def _get_outlier_recommendations(self, outlier_percentage: float) -> List[str]:
        """Get recommendations for handling outliers"""
        if outlier_percentage < 2:
            return ["Monitor outliers but no immediate action needed"]
        elif outlier_percentage < 5:
            return ["Review outlier cases to understand causes"]
        elif outlier_percentage < 10:
            return [
                "Investigate outlier patterns",
                "Consider if outliers indicate data quality issues",
                "Review evaluation criteria consistency"
            ]
        else:
            return [
                "High outlier rate indicates potential issues",
                "Comprehensive review of evaluation process needed",
                "Check for systematic biases or errors",
                "Consider data cleaning procedures"
            ]
    
    def _calculate_performance_gap(self, sorted_techniques: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Calculate performance gap between best and worst techniques"""
        if len(sorted_techniques) < 2:
            return {'gap': 0, 'interpretation': 'insufficient_data'}
        
        best_score = sorted_techniques[0][1].get('mean', 0)
        worst_score = sorted_techniques[-1][1].get('mean', 0)
        gap = best_score - worst_score
        
        return {
            'gap': gap,
            'percentage_improvement': (gap / worst_score) * 100 if worst_score != 0 else 0,
            'interpretation': self._interpret_performance_gap(gap)
        }
    
    def _interpret_performance_gap(self, gap: float) -> str:
        """Interpret the performance gap"""
        if gap < 25:
            return 'small - techniques perform similarly'
        elif gap < 75:
            return 'moderate - noticeable difference between techniques'
        else:
            return 'large - significant technique effectiveness variation'
    
    def _assess_technique_effectiveness(self, mean_score: float, sample_size: int) -> str:
        """Assess technique effectiveness"""
        performance_level = self._classify_performance(mean_score)
        reliability = self._assess_reliability(sample_size)
        
        if performance_level in ['excellent', 'good'] and reliability in ['high', 'medium']:
            return 'highly_effective'
        elif performance_level == 'average' and reliability in ['high', 'medium']:
            return 'moderately_effective'
        elif performance_level in ['poor', 'very_poor']:
            return 'ineffective'
        else:
            return 'uncertain'
    
    def _assess_reliability(self, sample_size: int) -> str:
        """Assess reliability based on sample size"""
        if sample_size >= 30:
            return 'high'
        elif sample_size >= 10:
            return 'medium'
        elif sample_size >= 5:
            return 'low'
        else:
            return 'very_low'
    
    def _get_technique_recommendation(self, effectiveness: str, sample_size: int) -> str:
        """Get recommendation for technique usage"""
        if effectiveness == 'highly_effective':
            return "Recommend widespread adoption"
        elif effectiveness == 'moderately_effective':
            return "Consider for specific use cases"
        elif effectiveness == 'ineffective':
            return "Avoid or significantly modify"
        elif sample_size < 5:
            return "Collect more data before making decisions"
        else:
            return "Monitor performance closely"
    
    def _calculate_model_range(self, sorted_models: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Calculate performance range across models"""
        if len(sorted_models) < 2:
            return {'range': 0, 'interpretation': 'insufficient_data'}
        
        best_score = sorted_models[0][1].get('mean', 0)
        worst_score = sorted_models[-1][1].get('mean', 0)
        range_value = best_score - worst_score
        
        return {
            'range': range_value,
            'best_model_score': best_score,
            'worst_model_score': worst_score,
            'interpretation': self._interpret_model_range(range_value)
        }
    
    def _interpret_model_range(self, range_value: float) -> str:
        """Interpret model performance range"""
        if range_value < 30:
            return 'small - models perform similarly'
        elif range_value < 80:
            return 'moderate - some model preferences evident'
        else:
            return 'large - significant model performance differences'
    
    def _assess_model_suitability(self, mean_score: float, consistency: float, sample_size: int) -> Dict[str, Any]:
        """Assess model suitability"""
        performance_level = self._classify_performance(mean_score)
        consistency_level = self._analyze_consistency(consistency, mean_score)
        reliability = self._assess_reliability(sample_size)
        
        # Calculate suitability score
        score_points = {'excellent': 5, 'good': 4, 'average': 3, 'poor': 2, 'very_poor': 1}
        consistency_points = {'very_consistent': 3, 'consistent': 2, 'moderately_consistent': 1, 'inconsistent': 0}
        reliability_points = {'high': 3, 'medium': 2, 'low': 1, 'very_low': 0}
        
        total_points = (
            score_points.get(performance_level, 0) +
            consistency_points.get(consistency_level, 0) +
            reliability_points.get(reliability, 0)
        )
        
        max_points = 11
        suitability_percentage = (total_points / max_points) * 100
        
        if suitability_percentage >= 80:
            suitability = 'highly_suitable'
        elif suitability_percentage >= 60:
            suitability = 'suitable'
        elif suitability_percentage >= 40:
            suitability = 'moderately_suitable'
        else:
            suitability = 'unsuitable'
        
        return {
            'suitability_level': suitability,
            'suitability_score': suitability_percentage,
            'performance_level': performance_level,
            'consistency_level': consistency_level,
            'reliability_level': reliability,
            'recommendation': self._get_model_recommendation(suitability, performance_level)
        }
    
    def _get_model_recommendation(self, suitability: str, performance: str) -> str:
        """Get recommendation for model usage"""
        if suitability == 'highly_suitable':
            return "Strongly recommend for production use"
        elif suitability == 'suitable':
            return "Recommend with monitoring"
        elif suitability == 'moderately_suitable':
            return "Use with caution and extensive testing"
        else:
            return "Not recommended for production use"
    
    def _perform_cross_model_analysis(self, model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-model analysis"""
        analysis = {
            'model_consistency': {},
            'model_specialization': {},
            'optimal_model_selection': {}
        }
        
        # Analyze consistency across score types for each model
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            model_scores = model_analysis.get(score_type, {})
            for model, stats in model_scores.items():
                if model not in analysis['model_consistency']:
                    analysis['model_consistency'][model] = {}
                
                analysis['model_consistency'][model][score_type] = {
                    'mean': stats.get('mean', 0),
                    'std_dev': stats.get('std_dev', 0),
                    'performance_level': self._classify_performance(stats.get('mean', 0))
                }
        
        return analysis
    
    def _analyze_metric_consistency(self, score_means: Dict[str, float]) -> Dict[str, Any]:
        """Analyze consistency between different ELO metrics"""
        if len(score_means) < 2:
            return {'consistency': 'insufficient_data'}
        
        scores = list(score_means.values())
        max_score = max(scores)
        min_score = min(scores)
        range_value = max_score - min_score
        mean_score = sum(scores) / len(scores)
        
        cv = (range_value / mean_score) * 100 if mean_score != 0 else 0
        
        if cv < 5:
            consistency = 'highly_consistent'
        elif cv < 10:
            consistency = 'consistent'
        elif cv < 20:
            consistency = 'moderately_consistent'
        else:
            consistency = 'inconsistent'
        
        return {
            'consistency': consistency,
            'coefficient_of_variation': cv,
            'range': range_value,
            'interpretation': self._interpret_metric_consistency(consistency, range_value)
        }
    
    def _interpret_metric_consistency(self, consistency: str, range_value: float) -> str:
        """Interpret metric consistency"""
        if consistency == 'highly_consistent':
            return "All ELO metrics show similar performance patterns"
        elif consistency == 'consistent':
            return "ELO metrics are generally aligned with minor variations"
        elif consistency == 'moderately_consistent':
            return "Some differences between ELO metrics - review evaluation criteria"
        else:
            return "Significant inconsistencies between ELO metrics - investigate evaluation process"
    
    def _get_metric_recommendations(self, score_means: Dict[str, float]) -> List[str]:
        """Get recommendations for metric usage"""
        recommendations = []
        
        if not score_means:
            return ["Insufficient metric data for recommendations"]
        
        best_metric = max(score_means, key=score_means.get)
        worst_metric = min(score_means, key=score_means.get)
        
        recommendations.append(f"Best performing metric: {best_metric.replace('_', ' ')}")
        
        if score_means[best_metric] - score_means[worst_metric] > 50:
            recommendations.append(f"Consider focusing on {best_metric.replace('_', ' ')} for optimization")
            recommendations.append(f"Investigate why {worst_metric.replace('_', ' ')} underperforms")
        
        return recommendations
    
    def _compare_technique_vs_model_impact(self, technique_analysis: Dict[str, Any],
                                         model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the impact of techniques vs models"""
        comparison = {
            'technique_impact': {},
            'model_impact': {},
            'relative_importance': {}
        }
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            # Calculate technique impact
            technique_scores = technique_analysis.get(score_type, {})
            if technique_scores:
                technique_means = [stats.get('mean', 0) for stats in technique_scores.values()]
                technique_range = max(technique_means) - min(technique_means) if technique_means else 0
                comparison['technique_impact'][score_type] = technique_range
            
            # Calculate model impact
            model_scores = model_analysis.get(score_type, {})
            if model_scores:
                model_means = [stats.get('mean', 0) for stats in model_scores.values()]
                model_range = max(model_means) - min(model_means) if model_means else 0
                comparison['model_impact'][score_type] = model_range
            
            # Determine relative importance
            technique_impact = comparison['technique_impact'].get(score_type, 0)
            model_impact = comparison['model_impact'].get(score_type, 0)
            
            if technique_impact > model_impact * 1.5:
                importance = 'technique_dominant'
            elif model_impact > technique_impact * 1.5:
                importance = 'model_dominant'
            else:
                importance = 'balanced'
            
            comparison['relative_importance'][score_type] = {
                'importance': importance,
                'technique_impact': technique_impact,
                'model_impact': model_impact,
                'recommendation': self._get_impact_recommendation(importance)
            }
        
        return comparison
    
    def _get_impact_recommendation(self, importance: str) -> str:
        """Get recommendation based on technique vs model importance"""
        if importance == 'technique_dominant':
            return "Focus optimization efforts on technique selection and refinement"
        elif importance == 'model_dominant':
            return "Focus optimization efforts on model selection and configuration"
        else:
            return "Balance optimization efforts between techniques and models"
    
    def _generate_technique_recommendations(self, technique_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate technique-specific recommendations"""
        recommendations = []
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            technique_scores = technique_analysis.get(score_type, {})
            if not technique_scores:
                continue
            
            # Find best and worst techniques
            sorted_techniques = sorted(
                technique_scores.items(),
                key=lambda x: x[1].get('mean', 0),
                reverse=True
            )
            
            if len(sorted_techniques) >= 2:
                best_technique = sorted_techniques[0]
                worst_technique = sorted_techniques[-1]
                
                performance_gap = best_technique[1].get('mean', 0) - worst_technique[1].get('mean', 0)
                
                if performance_gap > 50:  # Significant gap
                    recommendations.append({
                        'type': 'technique_standardization',
                        'priority': 'high',
                        'metric': score_type,
                        'recommendation': f"Standardize on {best_technique[0]} technique",
                        'best_technique': best_technique[0],
                        'best_score': best_technique[1].get('mean', 0),
                        'worst_technique': worst_technique[0],
                        'worst_score': worst_technique[1].get('mean', 0),
                        'potential_improvement': performance_gap,
                        'actions': [
                            f"Phase out {worst_technique[0]} technique",
                            f"Train team on {best_technique[0]} best practices",
                            "Update prompt templates and guidelines"
                        ]
                    })
        
        return recommendations
    
    def _generate_model_recommendations(self, model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            model_scores = model_analysis.get(score_type, {})
            if not model_scores:
                continue
            
            # Find best performing model
            best_model = max(model_scores.items(), key=lambda x: x[1].get('mean', 0))
            best_score = best_model[1].get('mean', 0)
            
            if best_score > self.performance_thresholds['good']:
                recommendations.append({
                    'type': 'model_optimization',
                    'priority': 'medium',
                    'metric': score_type,
                    'recommendation': f"Optimize prompts for {best_model[0]} model",
                    'best_model': best_model[0],
                    'performance_score': best_score,
                    'actions': [
                        f"Focus prompt development on {best_model[0]} capabilities",
                        "Analyze successful prompt patterns for this model",
                        "Consider model-specific optimization techniques"
                    ]
                })
        
        return recommendations
    
    def _get_performance_improvement_actions(self, score_type: str) -> List[str]:
        """Get specific actions for improving performance"""
        actions = {
            'elo_score': [
                "Review and improve prompt quality",
                "Analyze losing comparisons for patterns",
                "Enhance prompt clarity and specificity",
                "Test alternative prompt structures"
            ],
            'version_elo_score': [
                "Compare prompt versions systematically",
                "Identify best-performing version patterns",
                "Implement version control best practices",
                "Focus on iterative improvements"
            ],
            'global_elo_score': [
                "Benchmark against broader dataset",
                "Review global performance patterns",
                "Align with industry best practices",
                "Consider cross-domain optimization"
            ]
        }
        return actions.get(score_type, ["Review and improve prompt performance"])
    
    def _assess_missing_data_impact(self, completeness: float) -> str:
        """Assess the impact of missing data"""
        if completeness >= 0.95:
            return 'minimal'
        elif completeness >= 0.90:
            return 'low'
        elif completeness >= 0.80:
            return 'moderate'
        else:
            return 'high'
    
    def _get_completeness_recommendations(self, completeness: float) -> List[str]:
        """Get recommendations for improving data completeness"""
        if completeness >= 0.95:
            return ["Maintain current data collection standards"]
        elif completeness >= 0.90:
            return ["Monitor for data collection issues", "Review error handling"]
        elif completeness >= 0.80:
            return [
                "Improve error handling in evaluation pipeline",
                "Add retry mechanisms for failed evaluations",
                "Review data validation processes"
            ]
        else:
            return [
                "Critical: Comprehensive review of data collection",
                "Implement robust error handling",
                "Add monitoring and alerting for data issues",
                "Consider data quality checkpoints"
            ]
    
    def _assess_sample_adequacy(self, sample_size: int) -> str:
        """Assess whether sample size is adequate for analysis"""
        if sample_size >= 100:
            return 'excellent'
        elif sample_size >= 50:
            return 'good'
        elif sample_size >= 30:
            return 'adequate'
        elif sample_size >= 10:
            return 'limited'
        else:
            return 'insufficient'
    
    def _get_sample_recommendations(self, sample_size: int) -> List[str]:
        """Get recommendations for sample size"""
        if sample_size >= 100:
            return ["Sample size is excellent for reliable analysis"]
        elif sample_size >= 50:
            return ["Good sample size, continue current collection rate"]
        elif sample_size >= 30:
            return ["Adequate sample size, consider increasing for better precision"]
        elif sample_size >= 10:
            return [
                "Limited sample size affects reliability",
                "Increase data collection efforts",
                "Interpret results with caution"
            ]
        else:
            return [
                "Insufficient sample size for reliable analysis",
                "Significantly increase data collection",
                "Results may not be statistically meaningful"
            ]
    
    def _calculate_reliability_score(self, data_quality: Dict[str, Any]) -> float:
        """Calculate overall data reliability score"""
        completeness = data_quality.get('data_completeness', 0)
        total_samples = data_quality.get('total_ratings', 0)
        
        # Completeness factor (0-0.5)
        completeness_factor = completeness * 0.5
        
        # Sample size factor (0-0.5)
        if total_samples >= 100:
            sample_factor = 0.5
        elif total_samples >= 50:
            sample_factor = 0.4
        elif total_samples >= 30:
            sample_factor = 0.3
        elif total_samples >= 10:
            sample_factor = 0.2
        else:
            sample_factor = 0.1
        
        return completeness_factor + sample_factor
    
    def _classify_reliability(self, data_quality: Dict[str, Any]) -> str:
        """Classify overall data reliability"""
        reliability_score = self._calculate_reliability_score(data_quality)
        
        if reliability_score >= 0.9:
            return 'very_high'
        elif reliability_score >= 0.8:
            return 'high'
        elif reliability_score >= 0.6:
            return 'moderate'
        elif reliability_score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _identify_reliability_factors(self, data_quality: Dict[str, Any]) -> List[str]:
        """Identify factors affecting data reliability"""
        factors = []
        
        completeness = data_quality.get('data_completeness', 0)
        total_samples = data_quality.get('total_ratings', 0)
        
        if completeness < 0.9:
            factors.append("Data completeness below optimal level")
        
        if total_samples < 30:
            factors.append("Small sample size affects statistical power")
        
        if total_samples < 10:
            factors.append("Very small sample size - results unreliable")
        
        if not factors:
            factors.append("Data quality factors within acceptable ranges")
        
        return factors
