#!/usr/bin/env python3
"""
Report Generator
================

Generates comprehensive reports from ELO rating analysis.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive analysis reports in various formats"""
    
    def __init__(self, output_dir: str = "elo_analysis_reports"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    analysis_results: Dict[str, Any],
                                    test_run_ids: List[str],
                                    api_base_url: str) -> Dict[str, str]:
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive ELO analysis report...")
        
        # Generate different report formats
        report_files = {}
        
        # JSON report (detailed data)
        json_file = self._generate_json_report(analysis_results, test_run_ids, api_base_url)
        report_files['json'] = json_file
        
        # Summary report (executive summary)
        summary_file = self._generate_summary_report(analysis_results)
        report_files['summary'] = summary_file
        
        # Insights report (actionable insights)
        insights_file = self._generate_insights_report(analysis_results.get('insights', {}))
        report_files['insights'] = insights_file
        
        # Recommendations report (action items)
        recommendations_file = self._generate_recommendations_report(
            analysis_results.get('insights', {}).get('actionable_recommendations', [])
        )
        report_files['recommendations'] = recommendations_file
        
        # Data visualization prep
        viz_file = self._generate_visualization_data(analysis_results)
        report_files['visualization_data'] = viz_file
        
        logger.info(f"Generated {len(report_files)} report files in {self.output_dir}")
        return report_files
    
    def _generate_json_report(self, analysis_results: Dict[str, Any], 
                             test_run_ids: List[str], api_base_url: str) -> str:
        """Generate comprehensive JSON report with all analysis data"""
        filename = f"elo_analysis_report_{self.timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '1.0',
                'test_run_ids': test_run_ids,
                'api_base_url': api_base_url,
                'total_test_runs': len(test_run_ids)
            },
            'executive_summary': self._create_executive_summary(analysis_results),
            'statistical_analysis': analysis_results.get('statistics', {}),
            'technique_analysis': analysis_results.get('technique_analysis', {}),
            'model_analysis': analysis_results.get('model_analysis', {}),
            'insights': analysis_results.get('insights', {}),
            'raw_data': {
                'elo_ratings': analysis_results.get('elo_ratings', {}),
                'metadata': analysis_results.get('metadata', {}),
                'data_quality': analysis_results.get('statistics', {}).get('data_quality', {})
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Generated JSON report: {filepath}")
        return filepath
    
    def _generate_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary report"""
        filename = f"elo_analysis_summary_{self.timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        summary_content = self._create_markdown_summary(analysis_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"Generated summary report: {filepath}")
        return filepath
    
    def _generate_insights_report(self, insights: Dict[str, Any]) -> str:
        """Generate insights-focused report"""
        filename = f"elo_insights_report_{self.timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        insights_content = self._create_insights_markdown(insights)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(insights_content)
        
        logger.info(f"Generated insights report: {filepath}")
        return filepath
    
    def _generate_recommendations_report(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate actionable recommendations report"""
        filename = f"elo_recommendations_{self.timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        recommendations_content = self._create_recommendations_markdown(recommendations)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(recommendations_content)
        
        logger.info(f"Generated recommendations report: {filepath}")
        return filepath
    
    def _generate_visualization_data(self, analysis_results: Dict[str, Any]) -> str:
        """Generate data prepared for visualization"""
        filename = f"elo_visualization_data_{self.timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        viz_data = self._prepare_visualization_data(analysis_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Generated visualization data: {filepath}")
        return filepath
    
    def _create_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary from analysis results"""
        stats = analysis_results.get('statistics', {})
        insights = analysis_results.get('insights', {})
        
        # Overall performance summary
        overall_scores = {}
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            overall_scores[score_type] = {
                'mean': score_stats.get('mean', 0),
                'performance_level': self._classify_performance_level(score_stats.get('mean', 0)),
                'consistency': self._assess_consistency(score_stats)
            }
        
        # Key findings
        key_findings = []
        
        # Performance findings
        performance_insights = insights.get('performance_insights', {}).get('overall_performance', {})
        for score_type, perf_data in performance_insights.items():
            if perf_data.get('score_above_baseline'):
                key_findings.append(f"{score_type.replace('_', ' ').title()} performs above baseline")
            else:
                key_findings.append(f"{score_type.replace('_', ' ').title()} needs improvement")
        
        # Technique findings
        technique_insights = insights.get('technique_insights', {}).get('technique_rankings', {})
        for score_type, rankings in technique_insights.items():
            top_performers = rankings.get('top_performers', [])
            if top_performers:
                key_findings.append(f"Best technique for {score_type}: {top_performers[0][0]}")
        
        # Model findings  
        model_insights = insights.get('model_insights', {}).get('model_rankings', {})
        for score_type, rankings in model_insights.items():
            best_model = rankings.get('best_model')
            if best_model:
                key_findings.append(f"Best model for {score_type}: {best_model[0]}")
        
        # Risk assessment
        risk_assessment = insights.get('risk_assessment', {})
        overall_risk = risk_assessment.get('overall_risk_level', 'low')
        
        # Recommendations count
        recommendations = insights.get('actionable_recommendations', [])
        high_priority_recs = [r for r in recommendations if r.get('priority') == 'high']
        
        return {
            'overall_performance': overall_scores,
            'key_findings': key_findings[:10],  # Top 10 findings
            'risk_level': overall_risk,
            'total_recommendations': len(recommendations),
            'high_priority_recommendations': len(high_priority_recs),
            'data_quality': stats.get('data_quality', {}),
            'analysis_confidence': self._calculate_analysis_confidence(stats)
        }
    
    def _create_markdown_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Create markdown summary report"""
        exec_summary = self._create_executive_summary(analysis_results)
        
        content = f"""# ELO Rating Analysis Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Overall Performance
"""
        
        # Performance overview
        for score_type, perf_data in exec_summary['overall_performance'].items():
            score_name = score_type.replace('_', ' ').title()
            mean_score = perf_data['mean']
            level = perf_data['performance_level']
            consistency = perf_data['consistency']
            
            content += f"""
#### {score_name}
- **Average Score:** {mean_score:.1f}
- **Performance Level:** {level.replace('_', ' ').title()}
- **Consistency:** {consistency.replace('_', ' ').title()}
"""
        
        # Key findings
        content += f"""
### Key Findings
"""
        for finding in exec_summary['key_findings']:
            content += f"- {finding}\n"
        
        # Risk and recommendations
        content += f"""
### Risk Assessment
- **Overall Risk Level:** {exec_summary['risk_level'].title()}

### Recommendations
- **Total Recommendations:** {exec_summary['total_recommendations']}
- **High Priority:** {exec_summary['high_priority_recommendations']}

### Data Quality
- **Total Ratings:** {exec_summary['data_quality'].get('total_ratings', 0)}
- **Valid Ratings:** {exec_summary['data_quality'].get('valid_ratings', 0)}
- **Data Completeness:** {exec_summary['data_quality'].get('data_completeness', 0):.1%}
- **Analysis Confidence:** {exec_summary['analysis_confidence']}
"""
        
        # Statistics overview
        stats = analysis_results.get('statistics', {})
        content += "\n## Statistical Overview\n"
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            if not score_stats:
                continue
                
            score_name = score_type.replace('_', ' ').title()
            content += f"""
### {score_name}
- **Mean:** {score_stats.get('mean', 0):.1f}
- **Median:** {score_stats.get('median', 0):.1f}
- **Standard Deviation:** {score_stats.get('std_dev', 0):.1f}
- **Range:** {score_stats.get('min', 0):.1f} - {score_stats.get('max', 0):.1f}
- **Sample Size:** {score_stats.get('count', 0)}
"""
        
        return content
    
    def _create_insights_markdown(self, insights: Dict[str, Any]) -> str:
        """Create markdown insights report"""
        content = f"""# ELO Rating Analysis Insights

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Performance insights
        perf_insights = insights.get('performance_insights', {})
        if perf_insights:
            content += "## Performance Insights\n\n"
            
            # Overall performance
            overall_perf = perf_insights.get('overall_performance', {})
            for score_type, perf_data in overall_perf.items():
                score_name = score_type.replace('_', ' ').title()
                content += f"""### {score_name}
- **Performance Level:** {perf_data.get('performance_level', 'unknown').replace('_', ' ').title()}
- **Above Baseline:** {'Yes' if perf_data.get('score_above_baseline') else 'No'}
- **Baseline Difference:** {perf_data.get('baseline_difference', 0):+.1f}
- **Description:** {perf_data.get('performance_description', 'N/A')}

"""
            
            # Outlier analysis
            outlier_analysis = perf_insights.get('outlier_analysis', {})
            if outlier_analysis:
                content += "### Outlier Analysis\n\n"
                for score_type, outlier_data in outlier_analysis.items():
                    score_name = score_type.replace('_', ' ').title()
                    content += f"""#### {score_name}
- **Outlier Count:** {outlier_data.get('outlier_count', 0)}
- **Outlier Percentage:** {outlier_data.get('outlier_percentage', 0):.1f}%
- **Impact:** {outlier_data.get('outlier_impact', 'unknown').title()}

**Recommendations:**
"""
                    for rec in outlier_data.get('outlier_recommendations', []):
                        content += f"- {rec}\n"
                    content += "\n"
        
        # Technique insights
        tech_insights = insights.get('technique_insights', {})
        if tech_insights:
            content += "## Technique Insights\n\n"
            
            # Rankings
            rankings = tech_insights.get('technique_rankings', {})
            for score_type, ranking_data in rankings.items():
                score_name = score_type.replace('_', ' ').title()
                content += f"### {score_name} - Technique Rankings\n\n"
                
                top_performers = ranking_data.get('top_performers', [])
                content += "**Top Performers:**\n"
                for i, (technique, stats) in enumerate(top_performers, 1):
                    content += f"{i}. {technique} (Score: {stats.get('mean', 0):.1f})\n"
                
                bottom_performers = ranking_data.get('bottom_performers', [])
                content += "\n**Bottom Performers:**\n"
                for i, (technique, stats) in enumerate(bottom_performers, 1):
                    content += f"{i}. {technique} (Score: {stats.get('mean', 0):.1f})\n"
                
                gap = ranking_data.get('performance_gap', {})
                content += f"\n**Performance Gap:** {gap.get('gap', 0):.1f} ({gap.get('interpretation', 'unknown')})\n\n"
        
        # Model insights
        model_insights = insights.get('model_insights', {})
        if model_insights:
            content += "## Model Insights\n\n"
            
            # Rankings
            rankings = model_insights.get('model_rankings', {})
            for score_type, ranking_data in rankings.items():
                score_name = score_type.replace('_', ' ').title()
                content += f"### {score_name} - Model Rankings\n\n"
                
                best_model = ranking_data.get('best_model')
                if best_model:
                    content += f"**Best Model:** {best_model[0]} (Score: {best_model[1].get('mean', 0):.1f})\n"
                
                worst_model = ranking_data.get('worst_model')
                if worst_model:
                    content += f"**Worst Model:** {worst_model[0]} (Score: {worst_model[1].get('mean', 0):.1f})\n"
                
                range_data = ranking_data.get('model_performance_range', {})
                content += f"**Performance Range:** {range_data.get('range', 0):.1f} ({range_data.get('interpretation', 'unknown')})\n\n"
        
        # Comparative insights
        comp_insights = insights.get('comparative_insights', {})
        if comp_insights:
            content += "## Comparative Insights\n\n"
            
            # Score type comparison
            score_comparison = comp_insights.get('score_type_comparison', {})
            if score_comparison:
                content += "### Score Type Comparison\n\n"
                
                highest_metric = score_comparison.get('highest_performing_metric')
                if highest_metric:
                    content += f"**Highest Performing Metric:** {highest_metric.replace('_', ' ').title()}\n"
                
                consistency = score_comparison.get('metric_consistency', {})
                content += f"**Metric Consistency:** {consistency.get('consistency', 'unknown').replace('_', ' ').title()}\n"
                content += f"**Consistency Interpretation:** {consistency.get('interpretation', 'N/A')}\n\n"
            
            # Technique vs model impact
            impact_comparison = comp_insights.get('technique_vs_model_impact', {})
            if impact_comparison:
                content += "### Technique vs Model Impact\n\n"
                
                relative_importance = impact_comparison.get('relative_importance', {})
                for score_type, importance_data in relative_importance.items():
                    score_name = score_type.replace('_', ' ').title()
                    importance = importance_data.get('importance', 'unknown')
                    recommendation = importance_data.get('recommendation', 'N/A')
                    
                    content += f"**{score_name}:**\n"
                    content += f"- Importance: {importance.replace('_', ' ').title()}\n"
                    content += f"- Technique Impact: {importance_data.get('technique_impact', 0):.1f}\n"
                    content += f"- Model Impact: {importance_data.get('model_impact', 0):.1f}\n"
                    content += f"- Recommendation: {recommendation}\n\n"
        
        # Risk assessment
        risk_assessment = insights.get('risk_assessment', {})
        if risk_assessment:
            content += "## Risk Assessment\n\n"
            
            content += f"**Overall Risk Level:** {risk_assessment.get('overall_risk_level', 'unknown').title()}\n\n"
            
            # Performance risks
            perf_risks = risk_assessment.get('performance_risks', [])
            if perf_risks:
                content += "### Performance Risks\n\n"
                for risk in perf_risks:
                    content += f"- **{risk.get('metric', 'Unknown').replace('_', ' ').title()}** ({risk.get('risk_level', 'unknown').title()}): {risk.get('description', 'N/A')}\n"
                    content += f"  - Impact: {risk.get('impact', 'N/A')}\n"
                content += "\n"
            
            # Consistency risks
            consistency_risks = risk_assessment.get('consistency_risks', [])
            if consistency_risks:
                content += "### Consistency Risks\n\n"
                for risk in consistency_risks:
                    content += f"- **{risk.get('metric', 'Unknown').replace('_', ' ').title()}** ({risk.get('risk_level', 'unknown').title()}): {risk.get('description', 'N/A')}\n"
                    content += f"  - CV: {risk.get('coefficient_of_variation', 0):.1f}%\n"
                    content += f"  - Impact: {risk.get('impact', 'N/A')}\n"
                content += "\n"
            
            # Data risks
            data_risks = risk_assessment.get('data_risks', [])
            if data_risks:
                content += "### Data Quality Risks\n\n"
                for risk in data_risks:
                    content += f"- **{risk.get('risk_level', 'unknown').title()}**: {risk.get('description', 'N/A')}\n"
                    content += f"  - Impact: {risk.get('impact', 'N/A')}\n"
                content += "\n"
        
        # Optimization opportunities
        optimization = insights.get('optimization_opportunities', [])
        if optimization:
            content += "## Optimization Opportunities\n\n"
            for opp in optimization:
                content += f"### {opp.get('type', 'Unknown').replace('_', ' ').title()}\n"
                content += f"- **Opportunity:** {opp.get('opportunity', 'N/A')}\n"
                content += f"- **Impact:** {opp.get('impact', 'unknown').title()}\n"
                
                if 'best_technique' in opp:
                    content += f"- **Best Technique:** {opp.get('best_technique')}\n"
                    content += f"- **Potential Improvement:** {opp.get('potential_improvement', 0):.1f}\n"
                
                if 'best_model' in opp:
                    content += f"- **Best Model:** {opp.get('best_model')}\n"
                    content += f"- **Performance Score:** {opp.get('performance_score', 0):.1f}\n"
                
                content += "\n"
        
        return content
    
    def _create_recommendations_markdown(self, recommendations: List[Dict[str, Any]]) -> str:
        """Create markdown recommendations report"""
        content = f"""# ELO Rating Analysis Recommendations

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if not recommendations:
            content += "No specific recommendations generated from the analysis.\n"
            return content
        
        # Group recommendations by priority
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
        low_priority = [r for r in recommendations if r.get('priority') == 'low']
        
        # High priority recommendations
        if high_priority:
            content += "## High Priority Recommendations\n\n"
            for i, rec in enumerate(high_priority, 1):
                content += f"### {i}. {rec.get('recommendation', 'No description')}\n\n"
                content += f"**Type:** {rec.get('type', 'unknown').replace('_', ' ').title()}\n\n"
                
                if 'metric' in rec:
                    content += f"**Metric:** {rec.get('metric').replace('_', ' ').title()}\n\n"
                
                if 'current_score' in rec and 'target_score' in rec:
                    content += f"**Current Score:** {rec.get('current_score'):.1f}\n\n"
                    content += f"**Target Score:** {rec.get('target_score'):.1f}\n\n"
                
                if 'potential_improvement' in rec:
                    content += f"**Potential Improvement:** {rec.get('potential_improvement'):.1f}\n\n"
                
                actions = rec.get('actions', [])
                if actions:
                    content += "**Actions:**\n"
                    for action in actions:
                        content += f"- {action}\n"
                    content += "\n"
                
                content += "---\n\n"
        
        # Medium priority recommendations
        if medium_priority:
            content += "## Medium Priority Recommendations\n\n"
            for i, rec in enumerate(medium_priority, 1):
                content += f"### {i}. {rec.get('recommendation', 'No description')}\n\n"
                content += f"**Type:** {rec.get('type', 'unknown').replace('_', ' ').title()}\n\n"
                
                if 'metric' in rec:
                    content += f"**Metric:** {rec.get('metric').replace('_', ' ').title()}\n\n"
                
                actions = rec.get('actions', [])
                if actions:
                    content += "**Actions:**\n"
                    for action in actions:
                        content += f"- {action}\n"
                    content += "\n"
                
                content += "---\n\n"
        
        # Low priority recommendations
        if low_priority:
            content += "## Low Priority Recommendations\n\n"
            for i, rec in enumerate(low_priority, 1):
                content += f"### {i}. {rec.get('recommendation', 'No description')}\n\n"
                content += f"**Type:** {rec.get('type', 'unknown').replace('_', ' ').title()}\n\n"
                
                actions = rec.get('actions', [])
                if actions:
                    content += "**Actions:**\n"
                    for action in actions:
                        content += f"- {action}\n"
                    content += "\n"
                
                content += "---\n\n"
        
        # Summary
        content += f"""## Summary

- **Total Recommendations:** {len(recommendations)}
- **High Priority:** {len(high_priority)}
- **Medium Priority:** {len(medium_priority)}
- **Low Priority:** {len(low_priority)}

**Next Steps:**
1. Review and prioritize high-priority recommendations
2. Develop implementation plan for critical improvements
3. Establish monitoring for recommended changes
4. Schedule follow-up analysis to measure impact
"""
        
        return content
    
    def _prepare_visualization_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualization"""
        viz_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'chart_types': [
                    'score_distribution',
                    'technique_comparison',
                    'model_comparison',
                    'performance_trends',
                    'outlier_analysis'
                ]
            },
            'score_distributions': {},
            'technique_comparisons': {},
            'model_comparisons': {},
            'performance_summary': {},
            'outlier_data': {},
            'correlation_data': {}
        }
        
        stats = analysis_results.get('statistics', {})
        
        # Score distributions
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            if score_stats:
                viz_data['score_distributions'][score_type] = {
                    'mean': score_stats.get('mean', 0),
                    'median': score_stats.get('median', 0),
                    'std_dev': score_stats.get('std_dev', 0),
                    'min': score_stats.get('min', 0),
                    'max': score_stats.get('max', 0),
                    'quartiles': score_stats.get('quartiles', {}),
                    'count': score_stats.get('count', 0)
                }
        
        # Technique comparisons
        technique_analysis = analysis_results.get('technique_analysis', {})
        for score_type, techniques in technique_analysis.items():
            if score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                viz_data['technique_comparisons'][score_type] = []
                for technique, stats in techniques.items():
                    viz_data['technique_comparisons'][score_type].append({
                        'technique': technique,
                        'mean': stats.get('mean', 0),
                        'std_dev': stats.get('std_dev', 0),
                        'count': stats.get('count', 0),
                        'confidence_interval': stats.get('confidence_interval', {})
                    })
        
        # Model comparisons
        model_analysis = analysis_results.get('model_analysis', {})
        for score_type, models in model_analysis.items():
            if score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
                viz_data['model_comparisons'][score_type] = []
                for model, stats in models.items():
                    viz_data['model_comparisons'][score_type].append({
                        'model': model,
                        'mean': stats.get('mean', 0),
                        'std_dev': stats.get('std_dev', 0),
                        'count': stats.get('count', 0),
                        'confidence_interval': stats.get('confidence_interval', {})
                    })
        
        # Performance summary for overview charts
        viz_data['performance_summary'] = {
            'overall_scores': {},
            'baseline': 1000
        }
        
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            viz_data['performance_summary']['overall_scores'][score_type] = {
                'mean': score_stats.get('mean', 0),
                'performance_level': self._classify_performance_level(score_stats.get('mean', 0))
            }
        
        # Outlier data
        for score_type in ['elo_score', 'version_elo_score', 'global_elo_score']:
            score_stats = stats.get(score_type, {})
            outliers = score_stats.get('outliers', {})
            if outliers:
                viz_data['outlier_data'][score_type] = {
                    'count': outliers.get('count', 0),
                    'percentage': outliers.get('percentage', 0),
                    'values': outliers.get('values', [])
                }
        
        return viz_data
    
    def _classify_performance_level(self, score: float) -> str:
        """Classify performance level for visualization"""
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
    
    def _assess_consistency(self, score_stats: Dict[str, Any]) -> str:
        """Assess consistency for summary"""
        mean = score_stats.get('mean', 0)
        std_dev = score_stats.get('std_dev', 0)
        
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
    
    def _calculate_analysis_confidence(self, stats: Dict[str, Any]) -> str:
        """Calculate overall confidence in analysis"""
        data_quality = stats.get('data_quality', {})
        completeness = data_quality.get('data_completeness', 0)
        total_ratings = data_quality.get('total_ratings', 0)
        
        # Factor in data completeness and sample size
        if completeness >= 0.95 and total_ratings >= 50:
            return 'high'
        elif completeness >= 0.90 and total_ratings >= 30:
            return 'moderate'
        elif completeness >= 0.80 and total_ratings >= 10:
            return 'low'
        else:
            return 'very_low'
