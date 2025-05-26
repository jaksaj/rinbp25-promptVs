#!/usr/bin/env python3
"""
ELO Analyzer Core
=================

Core orchestrator for comprehensive ELO rating analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from .elo_data_collector import EloDataCollector
from .statistical_analyzer import StatisticalAnalyzer
from .insight_generator import InsightGenerator
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class EloAnalyzer:
    """Core ELO analyzer that orchestrates the complete analysis pipeline"""
    
    def __init__(self, api_base_url: str, output_dir: str = "elo_analysis_reports"):
        self.api_base_url = api_base_url
        self.output_dir = output_dir
        
        # Initialize components
        self.data_collector = EloDataCollector(api_base_url)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.insight_generator = InsightGenerator()
        self.report_generator = ReportGenerator(output_dir)
        
        # Analysis state
        self.analysis_start_time = None
        self.analysis_results = {}
    
    def run_comprehensive_analysis(self, test_run_ids: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive ELO rating analysis
        
        Args:
            test_run_ids: List of test run IDs to analyze
            
        Returns:
            Dictionary containing complete analysis results and report file paths
        """
        logger.info(f"Starting comprehensive ELO analysis for {len(test_run_ids)} test runs...")
        self.analysis_start_time = time.time()
        
        try:
            # Phase 1: Data Collection
            logger.info("Phase 1: Collecting ELO rating data...")
            elo_data = self._collect_analysis_data(test_run_ids)
            
            # Phase 2: Statistical Analysis
            logger.info("Phase 2: Performing statistical analysis...")
            statistical_results = self._perform_statistical_analysis(elo_data)
            
            # Phase 3: Insight Generation
            logger.info("Phase 3: Generating insights...")
            insights = self._generate_insights(statistical_results, elo_data)
            
            # Phase 4: Report Generation
            logger.info("Phase 4: Generating reports...")
            reports = self._generate_reports(statistical_results, insights, test_run_ids)
            
            # Compile final results
            analysis_duration = time.time() - self.analysis_start_time
            
            self.analysis_results = {
                'metadata': {
                    'test_run_ids': test_run_ids,
                    'analysis_duration_seconds': analysis_duration,
                    'analysis_completed_at': time.time(),
                    'total_test_runs': len(test_run_ids),
                    'api_base_url': self.api_base_url
                },
                'data_collection': elo_data,
                'statistical_analysis': statistical_results,
                'insights': insights,
                'reports': reports,
                'summary': self._create_analysis_summary(statistical_results, insights)
            }
            
            logger.info(f"Comprehensive ELO analysis completed in {analysis_duration:.2f} seconds")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def _collect_analysis_data(self, test_run_ids: List[str]) -> Dict[str, Any]:
        """Collect all necessary data for analysis"""
        logger.info("Collecting ELO ratings and metadata...")
        
        # Collect ELO ratings
        elo_ratings = self.data_collector.collect_all_elo_ratings(test_run_ids)
          # Collect metadata for valid test runs
        valid_test_runs = [tid for tid, data in elo_ratings.items() if not data.get('missing_data', False)]
        metadata = self.data_collector.collect_test_run_metadata(valid_test_runs)
        
        # Collect prompt version data
        prompt_versions = self.data_collector.collect_prompt_version_metadata(metadata)
        
        # Collect comparison results for context
        comparison_results = self.data_collector.collect_comparison_results(valid_test_runs)
        
        return {
            'elo_ratings': elo_ratings,
            'metadata': metadata,
            'prompt_versions': prompt_versions,
            'comparison_results': comparison_results,
            'data_summary': {
                'total_test_runs': len(test_run_ids),
                'valid_test_runs': len(valid_test_runs),
                'missing_data_runs': len(test_run_ids) - len(valid_test_runs),
                'data_completeness': len(valid_test_runs) / len(test_run_ids) if test_run_ids else 0
            }
        }
    
    def _perform_statistical_analysis(self, elo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        logger.info("Performing statistical analysis on ELO data...")
        
        elo_ratings = elo_data['elo_ratings']
        metadata = elo_data['metadata']
        prompt_versions = elo_data['prompt_versions']
        comparison_results = elo_data['comparison_results']
          # Basic ELO statistics
        elo_statistics = self.statistical_analyzer.calculate_elo_statistics(elo_ratings)
        
        # Technique-based analysis
        technique_analysis = self.statistical_analyzer.analyze_elo_trends_by_technique(
            elo_ratings, metadata, prompt_versions, comparison_results
        )
        
        # Extract models from metadata for model analysis
        models = []
        for test_run_data in metadata.values():
            model_name = test_run_data.get('model_name')
            if model_name and model_name not in models:
                models.append(model_name)
        
        # Model-based analysis
        model_analysis = self.statistical_analyzer.analyze_elo_trends_by_model(
            elo_ratings, metadata, models
        )
        
        # Advanced statistical analysis
        effect_sizes = self.statistical_analyzer.calculate_effect_sizes(technique_analysis, model_analysis)
        
        # Distribution analysis
        distribution_analysis = self.statistical_analyzer.analyze_distributions(elo_ratings)
        
        return {
            'elo_statistics': elo_statistics,
            'technique_analysis': technique_analysis,
            'model_analysis': model_analysis,
            'effect_sizes': effect_sizes,
            'distribution_analysis': distribution_analysis,
            'statistical_summary': self._create_statistical_summary(
                elo_statistics, technique_analysis, model_analysis
            )
        }
    
    def _generate_insights(self, statistical_results: Dict[str, Any], 
                          elo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive insights from statistical analysis"""
        logger.info("Generating business insights from analysis...")
        
        # Extract components for insight generation
        elo_statistics = statistical_results['elo_statistics']
        technique_analysis = statistical_results['technique_analysis']
        model_analysis = statistical_results['model_analysis']
        elo_ratings = elo_data['elo_ratings']
        metadata = elo_data['metadata']
        
        # Generate comprehensive insights
        insights = self.insight_generator.generate_comprehensive_insights(
            elo_statistics, technique_analysis, model_analysis, elo_ratings, metadata
        )
        
        # Add context and meta-insights
        insights['meta_analysis'] = {
            'analysis_confidence': self._assess_analysis_confidence(elo_statistics, elo_data),
            'key_trends': self._identify_key_trends(statistical_results),
            'critical_findings': self._identify_critical_findings(insights),
            'business_impact': self._assess_business_impact(insights)
        }
        
        return insights
    
    def _generate_reports(self, statistical_results: Dict[str, Any],
                         insights: Dict[str, Any], test_run_ids: List[str]) -> Dict[str, str]:
        """Generate comprehensive reports"""
        logger.info("Generating analysis reports...")
        
        # Combine all analysis results for reporting
        complete_analysis = {
            'statistics': statistical_results['elo_statistics'],
            'technique_analysis': statistical_results['technique_analysis'],
            'model_analysis': statistical_results['model_analysis'],
            'insights': insights,
            'elo_ratings': self.analysis_results.get('data_collection', {}).get('elo_ratings', {}),
            'metadata': self.analysis_results.get('data_collection', {}).get('metadata', {})
        }
        
        # Generate reports
        report_files = self.report_generator.generate_comprehensive_report(
            complete_analysis, test_run_ids, self.api_base_url
        )
        
        return report_files
    
    def _create_analysis_summary(self, statistical_results: Dict[str, Any],
                               insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level analysis summary"""
        elo_stats = statistical_results['elo_statistics']
        
        # Overall performance summary
        performance_summary = {}
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            stats = elo_stats.get(score_type, {})
            performance_summary[score_type] = {
                'mean': stats.get('mean', 0),
                'performance_level': self._classify_performance(stats.get('mean', 0)),
                'sample_size': stats.get('count', 0),
                'above_baseline': stats.get('mean', 0) > 1000
            }
        
        # Key insights summary
        recommendations = insights.get('actionable_recommendations', [])
        high_priority_count = len([r for r in recommendations if r.get('priority') == 'high'])
        
        # Risk summary
        risk_assessment = insights.get('risk_assessment', {})
        overall_risk = risk_assessment.get('overall_risk_level', 'unknown')
        
        return {
            'overall_performance': performance_summary,
            'data_quality': elo_stats.get('data_quality', {}),
            'key_metrics': {
                'total_recommendations': len(recommendations),
                'high_priority_recommendations': high_priority_count,
                'overall_risk_level': overall_risk,
                'analysis_confidence': self._assess_overall_confidence(elo_stats)
            },
            'top_findings': self._extract_top_findings(insights),
            'next_steps': self._generate_next_steps(recommendations, overall_risk)
        }
    
    def _create_statistical_summary(self, elo_statistics: Dict[str, Any],
                                  technique_analysis: Dict[str, Any],
                                  model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of statistical findings"""
        summary = {
            'data_overview': {},
            'performance_overview': {},
            'variability_overview': {},
            'technique_overview': {},
            'model_overview': {}
        }
        
        # Data overview
        data_quality = elo_statistics.get('data_quality', {})
        summary['data_overview'] = {
            'total_ratings': data_quality.get('total_ratings', 0),
            'valid_ratings': data_quality.get('valid_ratings', 0),
            'completeness': data_quality.get('data_completeness', 0),
            'quality_level': self._assess_data_quality(data_quality)
        }
        
        # Performance overview
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            stats = elo_statistics.get(score_type, {})
            summary['performance_overview'][score_type] = {
                'mean': stats.get('mean', 0),
                'median': stats.get('median', 0),
                'range': stats.get('max', 0) - stats.get('min', 0),
                'performance_level': self._classify_performance(stats.get('mean', 0))
            }
        
        # Variability overview
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            stats = elo_statistics.get(score_type, {})
            mean = stats.get('mean', 0)
            std_dev = stats.get('std_dev', 0)
            cv = (std_dev / mean * 100) if mean > 0 else 0
            
            summary['variability_overview'][score_type] = {
                'std_dev': std_dev,
                'coefficient_of_variation': cv,
                'consistency_level': self._classify_consistency(cv)
            }
        
        # Technique overview
        if technique_analysis:
            technique_counts = {}
            for score_type, techniques in technique_analysis.items():
                if score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                    technique_counts[score_type] = len(techniques)
            
            summary['technique_overview'] = {
                'techniques_analyzed': technique_counts,
                'has_technique_data': bool(technique_counts)
            }
        
        # Model overview
        if model_analysis:
            model_counts = {}
            for score_type, models in model_analysis.items():
                if score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                    model_counts[score_type] = len(models)
            
            summary['model_overview'] = {
                'models_analyzed': model_counts,
                'has_model_data': bool(model_counts)
            }
        
        return summary
    
    def _assess_analysis_confidence(self, elo_statistics: Dict[str, Any], 
                                  elo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence in analysis results"""
        data_quality = elo_statistics.get('data_quality', {})
        completeness = data_quality.get('data_completeness', 0)
        sample_size = data_quality.get('valid_ratings', 0)
        
        # Calculate confidence factors
        completeness_score = min(completeness, 1.0)
        
        if sample_size >= 100:
            sample_score = 1.0
        elif sample_size >= 50:
            sample_score = 0.8
        elif sample_size >= 30:
            sample_score = 0.6
        elif sample_size >= 10:
            sample_score = 0.4
        else:
            sample_score = 0.2
        
        # Overall confidence
        overall_confidence = (completeness_score + sample_score) / 2
        
        if overall_confidence >= 0.9:
            confidence_level = 'very_high'
        elif overall_confidence >= 0.8:
            confidence_level = 'high'
        elif overall_confidence >= 0.6:
            confidence_level = 'moderate'
        elif overall_confidence >= 0.4:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': overall_confidence,
            'factors': {
                'data_completeness': completeness_score,
                'sample_size_adequacy': sample_score,
                'total_sample_size': sample_size
            },
            'recommendations': self._get_confidence_recommendations(confidence_level)
        }
    
    def _identify_key_trends(self, statistical_results: Dict[str, Any]) -> List[str]:
        """Identify key trends from statistical analysis"""
        trends = []
        
        elo_stats = statistical_results['elo_statistics']
        technique_analysis = statistical_results.get('technique_analysis', {})
        model_analysis = statistical_results.get('model_analysis', {})
        
        # Performance trends
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            stats = elo_stats.get(score_type, {})
            mean_score = stats.get('mean', 0)
            
            if mean_score > 1100:
                trends.append(f"{score_type.replace('_', ' ').title()} shows strong performance above baseline")
            elif mean_score < 950:
                trends.append(f"{score_type.replace('_', ' ').title()} shows concerning performance below baseline")
        
        # Technique trends
        if technique_analysis:
            for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                techniques = technique_analysis.get(score_type, {})
                if techniques:
                    technique_scores = [(name, data.get('mean', 0)) for name, data in techniques.items()]
                    if technique_scores:
                        best_technique = max(technique_scores, key=lambda x: x[1])
                        worst_technique = min(technique_scores, key=lambda x: x[1])
                        
                        if best_technique[1] - worst_technique[1] > 100:
                            trends.append(f"Significant technique performance gap in {score_type}")
        
        # Model trends
        if model_analysis:
            for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                models = model_analysis.get(score_type, {})
                if models:
                    model_scores = [(name, data.get('mean', 0)) for name, data in models.items()]
                    if model_scores:
                        best_model = max(model_scores, key=lambda x: x[1])
                        worst_model = min(model_scores, key=lambda x: x[1])
                        
                        if best_model[1] - worst_model[1] > 75:
                            trends.append(f"Notable model performance differences in {score_type}")
        
        return trends[:10]  # Top 10 trends
    
    def _identify_critical_findings(self, insights: Dict[str, Any]) -> List[str]:
        """Identify critical findings requiring immediate attention"""
        critical_findings = []
        
        # High-priority recommendations indicate critical issues
        recommendations = insights.get('actionable_recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        
        for rec in high_priority:
            critical_findings.append(f"Critical: {rec.get('recommendation', 'Unknown issue')}")
        
        # Risk assessment findings
        risk_assessment = insights.get('risk_assessment', {})
        if risk_assessment.get('overall_risk_level') == 'high':
            critical_findings.append("High overall risk level identified")
        
        # Performance risks
        perf_risks = risk_assessment.get('performance_risks', [])
        for risk in perf_risks:
            if risk.get('risk_level') == 'high':
                critical_findings.append(f"Performance risk: {risk.get('description', 'Unknown')}")
        
        # Data quality issues
        data_risks = risk_assessment.get('data_risks', [])
        for risk in data_risks:
            if risk.get('risk_level') == 'high':
                critical_findings.append(f"Data quality issue: {risk.get('description', 'Unknown')}")
        
        return critical_findings[:5]  # Top 5 critical findings
    
    def _assess_business_impact(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of findings"""
        impact_assessment = {
            'performance_impact': 'unknown',
            'cost_impact': 'unknown',
            'quality_impact': 'unknown',
            'operational_impact': 'unknown',
            'overall_impact': 'unknown'
        }
        
        # Assess performance impact
        perf_insights = insights.get('performance_insights', {}).get('overall_performance', {})
        poor_performers = [k for k, v in perf_insights.items() 
                          if not v.get('score_above_baseline', False)]
        
        if len(poor_performers) == 0:
            impact_assessment['performance_impact'] = 'positive'
        elif len(poor_performers) <= 1:
            impact_assessment['performance_impact'] = 'minimal'
        else:
            impact_assessment['performance_impact'] = 'negative'
        
        # Assess operational impact based on recommendations
        recommendations = insights.get('actionable_recommendations', [])
        high_priority_count = len([r for r in recommendations if r.get('priority') == 'high'])
        
        if high_priority_count == 0:
            impact_assessment['operational_impact'] = 'minimal'
        elif high_priority_count <= 2:
            impact_assessment['operational_impact'] = 'moderate'
        else:
            impact_assessment['operational_impact'] = 'significant'
        
        # Overall impact assessment
        negative_impacts = sum(1 for impact in impact_assessment.values() 
                             if impact in ['negative', 'significant'])
        
        if negative_impacts == 0:
            impact_assessment['overall_impact'] = 'positive'
        elif negative_impacts <= 1:
            impact_assessment['overall_impact'] = 'neutral'
        else:
            impact_assessment['overall_impact'] = 'concerning'
        
        return impact_assessment
    
    def _classify_performance(self, score: float) -> str:
        """Classify performance level"""
        if score >= 1200:
            return 'excellent'
        elif score >= 1100:
            return 'good'
        elif score >= 1000:
            return 'average'
        elif score >= 900:
            return 'poor'
        else:
            return 'very_poor'
    
    def _classify_consistency(self, cv: float) -> str:
        """Classify consistency level"""
        if cv < 5:
            return 'very_consistent'
        elif cv < 10:
            return 'consistent'
        elif cv < 20:
            return 'moderately_consistent'
        else:
            return 'inconsistent'
    
    def _assess_data_quality(self, data_quality: Dict[str, Any]) -> str:
        """Assess overall data quality"""
        completeness = data_quality.get('data_completeness', 0)
        total_ratings = data_quality.get('total_ratings', 0)
        
        if completeness >= 0.95 and total_ratings >= 50:
            return 'excellent'
        elif completeness >= 0.90 and total_ratings >= 30:
            return 'good'
        elif completeness >= 0.80 and total_ratings >= 10:
            return 'adequate'
        else:
            return 'poor'
    
    def _assess_overall_confidence(self, elo_statistics: Dict[str, Any]) -> str:
        """Assess overall analysis confidence"""
        data_quality = elo_statistics.get('data_quality', {})
        completeness = data_quality.get('data_completeness', 0)
        sample_size = data_quality.get('valid_ratings', 0)
        
        if completeness >= 0.95 and sample_size >= 50:
            return 'high'
        elif completeness >= 0.90 and sample_size >= 30:
            return 'moderate'
        else:
            return 'low'
    
    def _extract_top_findings(self, insights: Dict[str, Any]) -> List[str]:
        """Extract top findings from insights"""
        findings = []
        
        # Performance findings
        perf_insights = insights.get('performance_insights', {}).get('overall_performance', {})
        for score_type, perf_data in perf_insights.items():
            level = perf_data.get('performance_level', 'unknown')
            if level in ['excellent', 'very_poor']:
                findings.append(f"{score_type.replace('_', ' ').title()}: {level} performance")
        
        # Top techniques
        tech_rankings = insights.get('technique_insights', {}).get('technique_rankings', {})
        for score_type, rankings in tech_rankings.items():
            top_performers = rankings.get('top_performers', [])
            if top_performers:
                findings.append(f"Best technique for {score_type}: {top_performers[0][0]}")
        
        # Top models
        model_rankings = insights.get('model_insights', {}).get('model_rankings', {})
        for score_type, rankings in model_rankings.items():
            best_model = rankings.get('best_model')
            if best_model:
                findings.append(f"Best model for {score_type}: {best_model[0]}")
        
        return findings[:8]  # Top 8 findings
    
    def _generate_next_steps(self, recommendations: List[Dict[str, Any]], 
                           risk_level: str) -> List[str]:
        """Generate next steps based on analysis"""
        next_steps = []
        
        # Prioritize based on risk level
        if risk_level == 'high':
            next_steps.append("Immediate: Address high-risk performance issues")
            next_steps.append("Urgent: Implement high-priority recommendations")
        elif risk_level == 'medium':
            next_steps.append("Short-term: Review and plan implementation of recommendations")
        
        # Add specific next steps based on recommendations
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        if high_priority:
            next_steps.append("Focus on high-priority optimization opportunities")
        
        # Standard next steps
        next_steps.extend([
            "Monitor ELO rating trends for improvements",
            "Schedule follow-up analysis after implementing changes",
            "Review and update evaluation criteria if needed"
        ])
        
        return next_steps[:6]  # Top 6 next steps
    
    def _get_confidence_recommendations(self, confidence_level: str) -> List[str]:
        """Get recommendations for improving analysis confidence"""
        if confidence_level == 'very_low':
            return [
                "Increase sample size significantly",
                "Improve data collection completeness",
                "Consider extending evaluation period",
                "Review data quality processes"
            ]
        elif confidence_level == 'low':
            return [
                "Increase sample size for better reliability",
                "Improve data collection processes",
                "Monitor data quality metrics"
            ]
        elif confidence_level == 'moderate':
            return [
                "Continue current data collection efforts",
                "Monitor for any data quality issues"
            ]
        else:
            return [
                "Maintain current data quality standards",
                "Analysis confidence is sufficient for decision-making"
            ]
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        if not self.analysis_start_time:
            return {'status': 'not_started'}
        
        current_time = time.time()
        elapsed_time = current_time - self.analysis_start_time
        
        status = {
            'status': 'completed' if self.analysis_results else 'in_progress',
            'elapsed_time_seconds': elapsed_time,
            'has_results': bool(self.analysis_results)
        }
        
        if self.analysis_results:
            status['summary'] = self.analysis_results.get('summary', {})
        
        return status
    
    def get_quick_summary(self) -> Dict[str, Any]:
        """Get quick summary of analysis results"""
        if not self.analysis_results:
            return {'error': 'No analysis results available'}
        
        summary = self.analysis_results.get('summary', {})
        return {
            'overall_performance': summary.get('overall_performance', {}),
            'key_metrics': summary.get('key_metrics', {}),
            'top_findings': summary.get('top_findings', []),
            'reports_generated': list(self.analysis_results.get('reports', {}).keys())
        }
