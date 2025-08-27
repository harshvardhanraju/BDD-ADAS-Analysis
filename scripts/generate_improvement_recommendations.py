#!/usr/bin/env python3
"""
Phase 6: Generate Targeted Improvement Recommendations

This script creates specific, actionable recommendations for model improvement
based on comprehensive analysis from all previous phases.
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class ImprovementRecommendationEngine:
    """Generate targeted, actionable improvement recommendations."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Define improvement strategies and their effectiveness
        self.improvement_strategies = {
            'architecture': {
                'efficientdet': {'complexity': 'high', 'impact': 'high', 'timeline': '3-4 months'},
                'vision_transformer': {'complexity': 'very_high', 'impact': 'high', 'timeline': '4-6 months'},
                'feature_pyramid_networks': {'complexity': 'medium', 'impact': 'medium', 'timeline': '4-6 weeks'},
                'attention_mechanisms': {'complexity': 'medium', 'impact': 'medium', 'timeline': '3-4 weeks'},
                'multi_scale_training': {'complexity': 'low', 'impact': 'medium', 'timeline': '2-3 weeks'}
            },
            'training': {
                'focal_loss': {'complexity': 'low', 'impact': 'medium', 'timeline': '1-2 weeks'},
                'class_balanced_sampling': {'complexity': 'low', 'impact': 'medium', 'timeline': '1-2 weeks'},
                'hard_negative_mining': {'complexity': 'medium', 'impact': 'high', 'timeline': '2-3 weeks'},
                'progressive_resizing': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'knowledge_distillation': {'complexity': 'medium', 'impact': 'medium', 'timeline': '3-4 weeks'},
                'mixup_cutmix': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'}
            },
            'data': {
                'targeted_data_collection': {'complexity': 'high', 'impact': 'high', 'timeline': '2-3 months'},
                'synthetic_data_generation': {'complexity': 'high', 'impact': 'medium', 'timeline': '1-2 months'},
                'advanced_augmentation': {'complexity': 'medium', 'impact': 'medium', 'timeline': '2-3 weeks'},
                'domain_adaptation': {'complexity': 'high', 'impact': 'high', 'timeline': '2-3 months'},
                'active_learning': {'complexity': 'high', 'impact': 'medium', 'timeline': '1-2 months'}
            },
            'optimization': {
                'learning_rate_scheduling': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'gradient_accumulation': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'mixed_precision_training': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'optimizer_tuning': {'complexity': 'medium', 'impact': 'medium', 'timeline': '2-3 weeks'},
                'hyperparameter_optimization': {'complexity': 'medium', 'impact': 'medium', 'timeline': '3-4 weeks'}
            },
            'post_processing': {
                'nms_optimization': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'test_time_augmentation': {'complexity': 'low', 'impact': 'low', 'timeline': '1 week'},
                'model_ensembling': {'complexity': 'medium', 'impact': 'medium', 'timeline': '2-3 weeks'},
                'confidence_calibration': {'complexity': 'medium', 'impact': 'medium', 'timeline': '2-3 weeks'}
            }
        }
    
    def generate_comprehensive_recommendations(self, all_results: Dict, output_dir: Path) -> Dict:
        """Generate comprehensive improvement recommendations."""
        
        print("🎯 Generating Targeted Improvement Recommendations...")
        print("=" * 60)
        
        # Analyze current state
        current_state = self._analyze_current_state(all_results)
        
        # Generate targeted recommendations
        recommendations = {
            'executive_summary': self._create_executive_recommendation_summary(current_state, all_results),
            'immediate_actions': self._generate_immediate_actions(current_state, all_results),
            'technical_strategies': self._generate_technical_strategies(current_state, all_results),
            'data_strategies': self._generate_data_strategies(current_state, all_results),
            'architecture_recommendations': self._generate_architecture_recommendations(current_state, all_results),
            'training_optimizations': self._generate_training_optimizations(current_state, all_results),
            'safety_specific_improvements': self._generate_safety_improvements(current_state, all_results),
            'environmental_robustness': self._generate_environmental_improvements(current_state, all_results),
            'implementation_roadmap': self._create_implementation_roadmap(current_state, all_results),
            'cost_benefit_analysis': self._perform_cost_benefit_analysis(current_state, all_results),
            'risk_mitigation': self._identify_risk_mitigation_strategies(current_state, all_results)
        }
        
        # Create detailed recommendation documents
        print("Creating detailed recommendation documents...")
        
        # 1. Executive Recommendations
        exec_rec_file = self._create_executive_recommendations(recommendations, output_dir)
        print(f"✅ Executive recommendations: {Path(exec_rec_file).name}")
        
        # 2. Technical Implementation Guide
        tech_guide_file = self._create_technical_implementation_guide(recommendations, output_dir)
        print(f"✅ Technical guide: {Path(tech_guide_file).name}")
        
        # 3. Data Strategy Document
        data_strategy_file = self._create_data_strategy_document(recommendations, output_dir)
        print(f"✅ Data strategy: {Path(data_strategy_file).name}")
        
        # 4. Safety Enhancement Plan
        safety_plan_file = self._create_safety_enhancement_plan(recommendations, output_dir)
        print(f"✅ Safety plan: {Path(safety_plan_file).name}")
        
        # 5. Visual Improvement Roadmap
        roadmap_viz_file = self._create_improvement_roadmap_visualization(recommendations, output_dir)
        print(f"✅ Visual roadmap: {Path(roadmap_viz_file).name}")
        
        # Save comprehensive recommendations
        comprehensive_file = output_dir / 'comprehensive_improvement_recommendations.json'
        with open(comprehensive_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        recommendations['generated_files'] = {
            'executive_recommendations': exec_rec_file,
            'technical_guide': tech_guide_file,
            'data_strategy': data_strategy_file,
            'safety_plan': safety_plan_file,
            'roadmap_visualization': roadmap_viz_file,
            'comprehensive_json': str(comprehensive_file)
        }
        
        return recommendations
    
    def _analyze_current_state(self, all_results: Dict) -> Dict:
        """Analyze current model state to determine improvement priorities."""
        
        # Extract key metrics
        coco_metrics = all_results.get('evaluation_metrics', {}).get('coco_metrics', {})
        safety_metrics = all_results.get('evaluation_metrics', {}).get('safety_metrics', {})
        failure_analysis = all_results.get('failure_analysis', {})
        
        current_state = {
            'overall_performance': {
                'map': coco_metrics.get('mAP', 0.0),
                'map_50': coco_metrics.get('mAP@0.5', 0.0),
                'map_small': coco_metrics.get('mAP_small', 0.0),
                'map_medium': coco_metrics.get('mAP_medium', 0.0),
                'map_large': coco_metrics.get('mAP_large', 0.0),
                'tier': self._classify_performance_tier(coco_metrics.get('mAP', 0.0))
            },
            'safety_performance': {
                'safety_map': coco_metrics.get('safety_critical_mAP', 0.0),
                'tier': self._classify_safety_tier(coco_metrics.get('safety_critical_mAP', 0.0))
            },
            'primary_weaknesses': self._identify_primary_weaknesses(all_results),
            'failure_patterns': self._analyze_failure_patterns(failure_analysis),
            'environmental_issues': self._analyze_environmental_weaknesses(all_results),
            'improvement_priority': self._determine_improvement_priority(all_results),
            'deployment_readiness': self._assess_deployment_readiness(all_results)
        }
        
        return current_state
    
    def _classify_performance_tier(self, map_score: float) -> str:
        """Classify performance into tiers."""
        if map_score >= 0.70:
            return 'excellent'
        elif map_score >= 0.55:
            return 'good'
        elif map_score >= 0.40:
            return 'fair'
        elif map_score >= 0.25:
            return 'poor'
        else:
            return 'critical'
    
    def _classify_safety_tier(self, safety_map: float) -> str:
        """Classify safety performance into tiers."""
        if safety_map >= 0.65:
            return 'acceptable'
        elif safety_map >= 0.50:
            return 'marginal'
        elif safety_map >= 0.35:
            return 'concerning'
        else:
            return 'unacceptable'
    
    def _identify_primary_weaknesses(self, all_results: Dict) -> List[Dict]:
        """Identify primary model weaknesses."""
        weaknesses = []
        
        coco_metrics = all_results.get('evaluation_metrics', {}).get('coco_metrics', {})
        per_class_ap = coco_metrics.get('per_class_AP', {})
        
        # Overall performance weakness
        overall_map = coco_metrics.get('mAP', 0.0)
        if overall_map < 0.45:
            weaknesses.append({
                'category': 'overall_performance',
                'severity': 'high' if overall_map < 0.30 else 'medium',
                'description': f'Overall mAP ({overall_map:.3f}) below acceptable threshold',
                'impact': 'Model not suitable for production deployment'
            })
        
        # Small object detection weakness
        small_map = coco_metrics.get('mAP_small', 0.0)
        if small_map < 0.30:
            weaknesses.append({
                'category': 'small_object_detection',
                'severity': 'high' if small_map < 0.15 else 'medium',
                'description': f'Small object detection mAP ({small_map:.3f}) critically low',
                'impact': 'Traffic signs and lights poorly detected - safety risk'
            })
        
        # Safety-critical performance weakness
        safety_map = coco_metrics.get('safety_critical_mAP', 0.0)
        if safety_map < 0.50:
            weaknesses.append({
                'category': 'safety_critical_performance',
                'severity': 'critical' if safety_map < 0.35 else 'high',
                'description': f'Safety-critical mAP ({safety_map:.3f}) below safety threshold',
                'impact': 'Unacceptable safety risk for autonomous vehicle deployment'
            })
        
        # Per-class weaknesses
        safety_critical_classes = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
        for class_name, ap in per_class_ap.items():
            if ap < 0.30:
                severity = 'critical' if class_name in safety_critical_classes else 'high'
                weaknesses.append({
                    'category': 'class_specific_performance',
                    'class': class_name,
                    'severity': severity,
                    'description': f'{class_name} class performance ({ap:.3f}) critically low',
                    'impact': f'Poor {class_name} detection affects system reliability'
                })
        
        return weaknesses
    
    def _analyze_failure_patterns(self, failure_analysis: Dict) -> Dict:
        """Analyze failure patterns to identify improvement opportunities."""
        if not failure_analysis or 'summary' not in failure_analysis:
            return {'error': 'No failure analysis available'}
        
        summary = failure_analysis['summary']
        patterns = failure_analysis.get('failure_patterns', {})
        
        failure_analysis_result = {
            'dominant_failure_mode': summary.get('most_common_failure_type', 'unknown'),
            'safety_failure_rate': summary.get('safety_critical_failures', 0) / max(summary.get('total_failures', 1), 1),
            'class_specific_issues': [],
            'environmental_issues': []
        }
        
        # Identify problematic classes
        if 'class_failure_rates' in patterns:
            class_failures = patterns['class_failure_rates']
            for class_name, failures in class_failures.items():
                total_class_failures = sum(failures.values())
                if total_class_failures > 5:  # Significant failure count
                    failure_analysis_result['class_specific_issues'].append({
                        'class': class_name,
                        'total_failures': total_class_failures,
                        'dominant_mode': max(failures.items(), key=lambda x: x[1])[0]
                    })
        
        return failure_analysis_result
    
    def _analyze_environmental_weaknesses(self, all_results: Dict) -> Dict:
        """Analyze environmental robustness weaknesses."""
        contextual = all_results.get('evaluation_metrics', {}).get('contextual_metrics', {})
        
        environmental_analysis = {
            'weather_variance': 0.0,
            'lighting_variance': 0.0,
            'scene_variance': 0.0,
            'most_challenging_conditions': [],
            'stability_assessment': 'unknown'
        }
        
        for env_type, env_data in contextual.items():
            if isinstance(env_data, dict):
                scores = []
                condition_scores = []
                
                for condition, data in env_data.items():
                    if isinstance(data, dict) and 'mean_ap' in data:
                        score = data['mean_ap']
                        scores.append(score)
                        condition_scores.append((condition, score))
                
                if scores:
                    variance = np.var(scores)
                    
                    if env_type == 'weather_performance':
                        environmental_analysis['weather_variance'] = variance
                    elif env_type == 'lighting_performance':
                        environmental_analysis['lighting_variance'] = variance
                    elif env_type == 'scene_performance':
                        environmental_analysis['scene_variance'] = variance
                    
                    # Find challenging conditions (bottom 25%)
                    condition_scores.sort(key=lambda x: x[1])
                    challenging = condition_scores[:len(condition_scores)//4 + 1]
                    environmental_analysis['most_challenging_conditions'].extend(challenging)
        
        # Overall stability assessment
        avg_variance = np.mean([
            environmental_analysis['weather_variance'],
            environmental_analysis['lighting_variance'], 
            environmental_analysis['scene_variance']
        ])
        
        if avg_variance < 0.01:
            environmental_analysis['stability_assessment'] = 'stable'
        elif avg_variance < 0.04:
            environmental_analysis['stability_assessment'] = 'moderately_stable'
        else:
            environmental_analysis['stability_assessment'] = 'unstable'
        
        return environmental_analysis
    
    def _determine_improvement_priority(self, all_results: Dict) -> str:
        """Determine overall improvement priority level."""
        coco_metrics = all_results.get('evaluation_metrics', {}).get('coco_metrics', {})
        overall_map = coco_metrics.get('mAP', 0.0)
        safety_map = coco_metrics.get('safety_critical_mAP', 0.0)
        
        if safety_map < 0.35 or overall_map < 0.25:
            return 'critical'
        elif safety_map < 0.50 or overall_map < 0.40:
            return 'high'
        elif overall_map < 0.60:
            return 'medium'
        else:
            return 'low'
    
    def _assess_deployment_readiness(self, all_results: Dict) -> Dict:
        """Assess deployment readiness and timeline."""
        coco_metrics = all_results.get('evaluation_metrics', {}).get('coco_metrics', {})
        overall_map = coco_metrics.get('mAP', 0.0)
        safety_map = coco_metrics.get('safety_critical_mAP', 0.0)
        
        if overall_map >= 0.70 and safety_map >= 0.60:
            return {
                'status': 'ready',
                'timeline': '2-4 weeks',
                'remaining_work': 'Final validation and deployment prep',
                'confidence': 'high'
            }
        elif overall_map >= 0.50 and safety_map >= 0.45:
            return {
                'status': 'near_ready',
                'timeline': '2-4 months',
                'remaining_work': 'Targeted performance improvements',
                'confidence': 'medium'
            }
        else:
            return {
                'status': 'not_ready',
                'timeline': '6+ months',
                'remaining_work': 'Major improvements required',
                'confidence': 'low'
            }
    
    def _create_executive_recommendation_summary(self, current_state: Dict, all_results: Dict) -> Dict:
        """Create executive summary of recommendations."""
        
        priority = current_state['improvement_priority']
        deployment = current_state['deployment_readiness']
        
        if priority == 'critical':
            investment_recommendation = "Major Investment Required"
            strategic_decision = "Consider alternative approaches or significant resource allocation"
            risk_level = "Very High"
        elif priority == 'high':
            investment_recommendation = "Significant Investment Recommended"
            strategic_decision = "Proceed with targeted improvements"
            risk_level = "High"
        elif priority == 'medium':
            investment_recommendation = "Moderate Investment"
            strategic_decision = "Optimize current approach"
            risk_level = "Medium"
        else:
            investment_recommendation = "Minimal Investment"
            strategic_decision = "Focus on deployment readiness"
            risk_level = "Low"
        
        return {
            'overall_recommendation': strategic_decision,
            'investment_level': investment_recommendation,
            'timeline_to_production': deployment['timeline'],
            'risk_level': risk_level,
            'confidence_in_success': deployment['confidence'],
            'key_focus_areas': self._identify_key_focus_areas(current_state),
            'expected_roi': self._estimate_roi(current_state, all_results)
        }
    
    def _identify_key_focus_areas(self, current_state: Dict) -> List[str]:
        """Identify key focus areas for improvement."""
        focus_areas = []
        
        weaknesses = current_state['primary_weaknesses']
        
        # Group weaknesses by category
        weakness_categories = {}
        for weakness in weaknesses:
            category = weakness['category']
            if category not in weakness_categories:
                weakness_categories[category] = []
            weakness_categories[category].append(weakness)
        
        # Prioritize focus areas
        if 'safety_critical_performance' in weakness_categories:
            focus_areas.append("Safety-Critical Class Performance")
        
        if 'overall_performance' in weakness_categories:
            focus_areas.append("Overall Model Performance")
        
        if 'small_object_detection' in weakness_categories:
            focus_areas.append("Small Object Detection")
        
        if current_state['environmental_issues']['stability_assessment'] == 'unstable':
            focus_areas.append("Environmental Robustness")
        
        if 'class_specific_performance' in weakness_categories:
            focus_areas.append("Class-Specific Improvements")
        
        return focus_areas[:4]  # Top 4 focus areas
    
    def _estimate_roi(self, current_state: Dict, all_results: Dict) -> Dict:
        """Estimate return on investment for improvements."""
        current_map = current_state['overall_performance']['map']
        
        if current_map >= 0.60:
            return {
                'timeline': 'Short-term (2-4 months)',
                'investment': 'Low-Medium ($50K-$150K)',
                'expected_return': 'High - Ready for market deployment',
                'payback_period': '6-12 months'
            }
        elif current_map >= 0.40:
            return {
                'timeline': 'Medium-term (4-8 months)', 
                'investment': 'Medium-High ($150K-$400K)',
                'expected_return': 'Good - Competitive market position',
                'payback_period': '12-18 months'
            }
        else:
            return {
                'timeline': 'Long-term (8+ months)',
                'investment': 'High ($400K+)',
                'expected_return': 'Uncertain - Technology validation needed',
                'payback_period': '18+ months'
            }
    
    def _generate_immediate_actions(self, current_state: Dict, all_results: Dict) -> List[Dict]:
        """Generate immediate actions (0-4 weeks)."""
        actions = []
        
        # Based on primary weaknesses, suggest immediate actions
        weaknesses = current_state['primary_weaknesses']
        
        for weakness in weaknesses:
            if weakness['severity'] == 'critical':
                if weakness['category'] == 'safety_critical_performance':
                    actions.append({
                        'action': 'Implement Safety-Focused Loss Function',
                        'description': 'Apply focal loss with higher weights for safety-critical classes',
                        'timeline': '1-2 weeks',
                        'effort': 'Low',
                        'expected_impact': 'Medium-High',
                        'implementation': 'Modify loss function in training script'
                    })
                
                elif weakness['category'] == 'small_object_detection':
                    actions.append({
                        'action': 'Enable Multi-Scale Training',
                        'description': 'Implement multi-scale training with different input resolutions',
                        'timeline': '2-3 weeks',
                        'effort': 'Medium',
                        'expected_impact': 'High',
                        'implementation': 'Update data pipeline for multi-scale inputs'
                    })
        
        # Always include basic optimizations
        actions.extend([
            {
                'action': 'Optimize Learning Rate Schedule',
                'description': 'Implement cosine annealing with warm restarts',
                'timeline': '1 week',
                'effort': 'Low',
                'expected_impact': 'Low-Medium',
                'implementation': 'Update optimizer configuration'
            },
            {
                'action': 'Enable Mixed Precision Training',
                'description': 'Use automatic mixed precision for faster training',
                'timeline': '1 week',
                'effort': 'Low',
                'expected_impact': 'Low',
                'implementation': 'Add AMP to training loop'
            }
        ])
        
        return actions[:5]  # Top 5 immediate actions
    
    def _generate_technical_strategies(self, current_state: Dict, all_results: Dict) -> Dict:
        """Generate technical improvement strategies."""
        
        strategies = {
            'architecture_improvements': [],
            'training_enhancements': [],
            'data_optimizations': [],
            'post_processing_improvements': []
        }
        
        performance_tier = current_state['overall_performance']['tier']
        primary_weaknesses = current_state['primary_weaknesses']
        
        # Architecture improvements based on performance tier
        if performance_tier in ['critical', 'poor']:
            strategies['architecture_improvements'].extend([
                {
                    'strategy': 'EfficientDet Architecture',
                    'rationale': 'Proven superior performance for object detection',
                    'complexity': 'High',
                    'timeline': '3-4 months',
                    'expected_improvement': '+15-25% mAP'
                },
                {
                    'strategy': 'Feature Pyramid Networks (FPN)',
                    'rationale': 'Improves multi-scale object detection',
                    'complexity': 'Medium',
                    'timeline': '4-6 weeks',
                    'expected_improvement': '+10-15% mAP'
                }
            ])
        
        # Small object detection improvements
        small_object_issues = [w for w in primary_weaknesses if w['category'] == 'small_object_detection']
        if small_object_issues:
            strategies['architecture_improvements'].append({
                'strategy': 'Feature Pyramid Network with P2 Level',
                'rationale': 'Higher resolution features for small objects',
                'complexity': 'Medium',
                'timeline': '3-4 weeks',
                'expected_improvement': '+20-30% small object mAP'
            })
            
            strategies['training_enhancements'].append({
                'strategy': 'Scale-Aware Loss Function',
                'rationale': 'Specialized loss for different object scales',
                'complexity': 'Medium',
                'timeline': '2-3 weeks',
                'expected_improvement': '+15-20% small object mAP'
            })
        
        # Safety-critical improvements
        safety_issues = [w for w in primary_weaknesses if w['category'] == 'safety_critical_performance']
        if safety_issues:
            strategies['training_enhancements'].extend([
                {
                    'strategy': 'Hard Negative Mining',
                    'rationale': 'Focus training on difficult safety-critical examples',
                    'complexity': 'Medium',
                    'timeline': '3-4 weeks',
                    'expected_improvement': '+10-15% safety class recall'
                },
                {
                    'strategy': 'Class-Balanced Sampling',
                    'rationale': 'Ensure adequate representation of safety-critical classes',
                    'complexity': 'Low',
                    'timeline': '1-2 weeks',
                    'expected_improvement': '+5-10% safety class performance'
                }
            ])
        
        return strategies
    
    def _generate_data_strategies(self, current_state: Dict, all_results: Dict) -> Dict:
        """Generate data improvement strategies."""
        
        data_strategies = {
            'collection_priorities': [],
            'augmentation_strategies': [],
            'quality_improvements': [],
            'annotation_enhancements': []
        }
        
        environmental_issues = current_state['environmental_issues']
        challenging_conditions = environmental_issues.get('most_challenging_conditions', [])
        
        # Data collection priorities based on challenging conditions
        for condition, score in challenging_conditions[:3]:  # Top 3 challenging
            data_strategies['collection_priorities'].append({
                'condition': condition,
                'current_performance': score,
                'collection_target': '2x current dataset size',
                'priority': 'High' if score < 0.30 else 'Medium',
                'timeline': '2-3 months'
            })
        
        # Augmentation strategies
        if environmental_issues['stability_assessment'] == 'unstable':
            data_strategies['augmentation_strategies'].extend([
                {
                    'strategy': 'Weather-Specific Augmentation',
                    'description': 'Simulate rain, fog, snow effects',
                    'implementation': 'Use albumentations with weather transforms',
                    'expected_impact': '+10-15% environmental robustness'
                },
                {
                    'strategy': 'Lighting Variation Augmentation', 
                    'description': 'Simulate different lighting conditions',
                    'implementation': 'Brightness, contrast, gamma adjustments',
                    'expected_impact': '+8-12% lighting robustness'
                }
            ])
        
        # Small object focus
        small_object_issues = [w for w in current_state['primary_weaknesses'] 
                             if w['category'] == 'small_object_detection']
        if small_object_issues:
            data_strategies['augmentation_strategies'].append({
                'strategy': 'Small Object Augmentation Pipeline',
                'description': 'Copy-paste small objects, mosaic augmentation',
                'implementation': 'Custom augmentation pipeline',
                'expected_impact': '+20-25% small object detection'
            })
        
        return data_strategies
    
    def _generate_architecture_recommendations(self, current_state: Dict, all_results: Dict) -> List[Dict]:
        """Generate specific architecture recommendations."""
        
        recommendations = []
        performance_tier = current_state['overall_performance']['tier']
        
        if performance_tier in ['critical', 'poor']:
            recommendations.append({
                'recommendation': 'Migrate to EfficientDet-D4 Architecture',
                'rationale': 'Current architecture likely insufficient for performance requirements',
                'implementation_steps': [
                    'Evaluate EfficientDet pre-trained models on validation set',
                    'Implement EfficientDet-D4 with BiFPN',
                    'Transfer learning from COCO pre-trained weights',
                    'Fine-tune with BDD100K-specific hyperparameters'
                ],
                'timeline': '3-4 months',
                'resource_requirements': '2-3 ML engineers, 4-8 A100 GPUs',
                'risk_level': 'Medium',
                'expected_outcome': '+20-30% overall mAP improvement'
            })
        
        elif performance_tier == 'fair':
            recommendations.append({
                'recommendation': 'Enhance Current Architecture with Advanced Features',
                'rationale': 'Current architecture has potential with targeted enhancements',
                'implementation_steps': [
                    'Add Feature Pyramid Network (FPN) if not present',
                    'Implement attention mechanisms in backbone',
                    'Add multi-scale training capability',
                    'Optimize anchor generation and matching'
                ],
                'timeline': '6-8 weeks',
                'resource_requirements': '1-2 ML engineers, 2-4 GPUs',
                'risk_level': 'Low',
                'expected_outcome': '+10-15% overall mAP improvement'
            })
        
        # Small object specific recommendations
        small_map = current_state['overall_performance']['map_small']
        if small_map < 0.30:
            recommendations.append({
                'recommendation': 'Implement Small Object Detection Specialization',
                'rationale': f'Small object mAP ({small_map:.3f}) critically low for traffic safety',
                'implementation_steps': [
                    'Add P2 feature level to FPN (higher resolution)',
                    'Implement scale-aware loss function',
                    'Add small object specific data augmentation',
                    'Optimize anchor scales for small objects'
                ],
                'timeline': '4-6 weeks',
                'resource_requirements': '1 ML engineer, 2 GPUs',
                'risk_level': 'Low',
                'expected_outcome': '+25-35% small object mAP improvement'
            })
        
        return recommendations
    
    def _generate_training_optimizations(self, current_state: Dict, all_results: Dict) -> List[Dict]:
        """Generate training optimization recommendations."""
        
        optimizations = []
        
        # Safety-critical focused training
        safety_tier = current_state['safety_performance']['tier']
        if safety_tier in ['unacceptable', 'concerning']:
            optimizations.append({
                'optimization': 'Safety-Critical Focused Training',
                'techniques': [
                    'Implement focal loss with α=2, γ=2 for safety classes',
                    'Use class weights: pedestrian=3.0, rider=2.5, bicycle=2.5, motorcycle=2.0',
                    'Hard negative mining with 3:1 negative to positive ratio',
                    'Safety-class specific data augmentation'
                ],
                'timeline': '2-3 weeks',
                'complexity': 'Medium',
                'expected_impact': '+15-25% safety-critical recall'
            })
        
        # General performance optimizations
        optimizations.extend([
            {
                'optimization': 'Advanced Learning Rate Scheduling',
                'techniques': [
                    'Cosine annealing with warm restarts',
                    'Learning rate finder for optimal initial LR',
                    'Different LR for backbone vs head',
                    'Gradual unfreezing strategy'
                ],
                'timeline': '1-2 weeks',
                'complexity': 'Low',
                'expected_impact': '+3-8% overall performance'
            },
            {
                'optimization': 'Enhanced Data Pipeline',
                'techniques': [
                    'Multi-scale training (480-832 pixels)',
                    'Mixup and CutMix augmentation',
                    'AutoAugment policies',
                    'Improved batch sampling strategy'
                ],
                'timeline': '2-3 weeks', 
                'complexity': 'Medium',
                'expected_impact': '+5-12% overall performance'
            }
        ])
        
        return optimizations
    
    def _generate_safety_improvements(self, current_state: Dict, all_results: Dict) -> Dict:
        """Generate safety-specific improvements."""
        
        safety_improvements = {
            'immediate_actions': [],
            'technical_enhancements': [],
            'validation_requirements': [],
            'monitoring_recommendations': []
        }
        
        safety_tier = current_state['safety_performance']['tier']
        
        if safety_tier in ['unacceptable', 'concerning']:
            safety_improvements['immediate_actions'].extend([
                {
                    'action': 'Implement Safety-First Training Protocol',
                    'description': 'Prioritize safety-critical class performance over overall mAP',
                    'timeline': '1-2 weeks',
                    'success_criteria': 'Safety-critical recall > 80%'
                },
                {
                    'action': 'Deploy Safety-Specific Validation Pipeline',
                    'description': 'Create dedicated validation set for safety scenarios',
                    'timeline': '2-3 weeks',
                    'success_criteria': 'False negative rate < 15% for safety classes'
                }
            ])
        
        safety_improvements['technical_enhancements'] = [
            {
                'enhancement': 'Multi-Stage Safety Validation',
                'components': [
                    'Stage 1: Basic detection accuracy',
                    'Stage 2: Edge case performance',
                    'Stage 3: Adversarial robustness',
                    'Stage 4: Real-world scenario testing'
                ]
            },
            {
                'enhancement': 'Safety-Aware Model Architecture',
                'components': [
                    'Dedicated safety-critical detection head',
                    'Uncertainty quantification for predictions',
                    'Multi-modal fusion (camera + other sensors)',
                    'Fail-safe detection mechanisms'
                ]
            }
        ]
        
        return safety_improvements
    
    def _generate_environmental_improvements(self, current_state: Dict, all_results: Dict) -> Dict:
        """Generate environmental robustness improvements."""
        
        env_improvements = {
            'data_collection_strategy': [],
            'augmentation_pipeline': [],
            'domain_adaptation': [],
            'robustness_testing': []
        }
        
        environmental_issues = current_state['environmental_issues']
        stability = environmental_issues['stability_assessment']
        
        if stability in ['unstable', 'moderately_stable']:
            # Data collection for challenging conditions
            challenging_conditions = environmental_issues.get('most_challenging_conditions', [])
            for condition, score in challenging_conditions:
                env_improvements['data_collection_strategy'].append({
                    'condition': condition,
                    'current_performance': score,
                    'target_improvement': '+20-30% performance',
                    'data_needed': f'5K-10K additional {condition} images',
                    'timeline': '2-3 months'
                })
            
            # Advanced augmentation
            env_improvements['augmentation_pipeline'] = [
                {
                    'technique': 'Weather Simulation Pipeline',
                    'implementation': 'Physically-based weather effects (rain, fog, snow)',
                    'expected_impact': '+15-25% weather robustness'
                },
                {
                    'technique': 'Lighting Adaptation Pipeline',
                    'implementation': 'HDR simulation, time-of-day variations',
                    'expected_impact': '+10-20% lighting robustness'
                },
                {
                    'technique': 'Scene Diversification',
                    'implementation': 'Background replacement, scene mixing',
                    'expected_impact': '+8-15% scene robustness'
                }
            ]
        
        return env_improvements
    
    def _create_implementation_roadmap(self, current_state: Dict, all_results: Dict) -> Dict:
        """Create detailed implementation roadmap."""
        
        priority = current_state['improvement_priority']
        deployment_status = current_state['deployment_readiness']['status']
        
        roadmap = {
            'phases': [],
            'critical_path': [],
            'parallel_workstreams': [],
            'milestone_gates': []
        }
        
        # Phase 1: Immediate Improvements (0-1 month)
        phase1_actions = []
        if priority in ['critical', 'high']:
            phase1_actions.extend([
                'Implement safety-focused loss function',
                'Enable multi-scale training',
                'Optimize learning rate schedule',
                'Deploy advanced data augmentation'
            ])
        
        roadmap['phases'].append({
            'phase': 'Phase 1 - Immediate Improvements',
            'timeline': '0-4 weeks',
            'actions': phase1_actions,
            'success_criteria': '+5-15% performance improvement',
            'resources': '1-2 ML engineers'
        })
        
        # Phase 2: Architecture Enhancements (1-3 months)
        phase2_actions = []
        if current_state['overall_performance']['tier'] in ['critical', 'poor']:
            phase2_actions.extend([
                'Evaluate and implement EfficientDet architecture',
                'Implement Feature Pyramid Network',
                'Add attention mechanisms'
            ])
        else:
            phase2_actions.extend([
                'Enhance current architecture',
                'Implement specialized heads for safety classes',
                'Add uncertainty quantification'
            ])
        
        roadmap['phases'].append({
            'phase': 'Phase 2 - Architecture Enhancement',
            'timeline': '4-16 weeks',
            'actions': phase2_actions,
            'success_criteria': '+10-25% performance improvement',
            'resources': '2-3 ML engineers, specialized compute'
        })
        
        # Phase 3: Advanced Optimizations (3-6 months)
        roadmap['phases'].append({
            'phase': 'Phase 3 - Advanced Optimization',
            'timeline': '16-24 weeks',
            'actions': [
                'Deploy advanced training strategies',
                'Implement domain adaptation techniques',
                'Create comprehensive validation pipeline',
                'Optimize for deployment'
            ],
            'success_criteria': 'Production readiness achieved',
            'resources': '2-3 ML engineers, MLOps support'
        })
        
        return roadmap
    
    def _perform_cost_benefit_analysis(self, current_state: Dict, all_results: Dict) -> Dict:
        """Perform cost-benefit analysis of improvements."""
        
        current_map = current_state['overall_performance']['map']
        priority = current_state['improvement_priority']
        
        # Cost estimates based on improvement priority
        cost_analysis = {
            'development_costs': {},
            'infrastructure_costs': {},
            'timeline_costs': {},
            'expected_benefits': {},
            'roi_analysis': {}
        }
        
        if priority == 'critical':
            cost_analysis['development_costs'] = {
                'engineering_resources': '$400K-$600K (6-9 person-months)',
                'research_and_development': '$100K-$200K',
                'data_collection': '$50K-$100K',
                'compute_infrastructure': '$50K-$100K'
            }
            cost_analysis['timeline_costs'] = {
                'time_to_market_delay': '6-9 months',
                'opportunity_cost': '$500K-$1M (delayed market entry)',
                'ongoing_development': '$50K/month'
            }
        elif priority == 'high':
            cost_analysis['development_costs'] = {
                'engineering_resources': '$200K-$400K (3-6 person-months)',
                'research_and_development': '$50K-$100K',
                'data_collection': '$25K-$50K',
                'compute_infrastructure': '$25K-$50K'
            }
            cost_analysis['timeline_costs'] = {
                'time_to_market_delay': '3-6 months',
                'opportunity_cost': '$200K-$500K',
                'ongoing_development': '$30K/month'
            }
        else:
            cost_analysis['development_costs'] = {
                'engineering_resources': '$100K-$200K (2-3 person-months)',
                'research_and_development': '$25K-$50K',
                'data_collection': '$10K-$25K',
                'compute_infrastructure': '$10K-$25K'
            }
        
        # Expected benefits
        if current_map < 0.35:
            cost_analysis['expected_benefits'] = {
                'performance_improvement': '+25-40% mAP improvement possible',
                'market_viability': 'Path to competitive product',
                'safety_compliance': 'Achieve minimum safety standards',
                'deployment_readiness': '12-18 months to production'
            }
        elif current_map < 0.55:
            cost_analysis['expected_benefits'] = {
                'performance_improvement': '+15-25% mAP improvement',
                'market_viability': 'Strong competitive position',
                'safety_compliance': 'Meet industry safety standards',
                'deployment_readiness': '6-12 months to production'
            }
        else:
            cost_analysis['expected_benefits'] = {
                'performance_improvement': '+5-15% mAP improvement',
                'market_viability': 'Industry-leading performance',
                'safety_compliance': 'Exceed safety requirements',
                'deployment_readiness': '2-6 months to production'
            }
        
        return cost_analysis
    
    def _identify_risk_mitigation_strategies(self, current_state: Dict, all_results: Dict) -> Dict:
        """Identify risk mitigation strategies."""
        
        risk_mitigation = {
            'technical_risks': [],
            'timeline_risks': [],
            'resource_risks': [],
            'market_risks': [],
            'mitigation_strategies': []
        }
        
        priority = current_state['improvement_priority']
        
        # Technical risks
        if priority == 'critical':
            risk_mitigation['technical_risks'].extend([
                {
                    'risk': 'Architecture changes may not achieve target performance',
                    'probability': 'Medium',
                    'impact': 'High',
                    'mitigation': 'Prototype and validate architecture changes early'
                },
                {
                    'risk': 'Safety improvements may conflict with overall performance',
                    'probability': 'Medium',
                    'impact': 'High',
                    'mitigation': 'Multi-objective optimization approach'
                }
            ])
        
        # Timeline risks
        risk_mitigation['timeline_risks'] = [
            {
                'risk': 'Data collection takes longer than expected',
                'probability': 'High',
                'impact': 'Medium',
                'mitigation': 'Parallel synthetic data generation pipeline'
            },
            {
                'risk': 'Training iterations require more time than estimated',
                'probability': 'Medium',
                'impact': 'Medium',
                'mitigation': 'Incremental improvement approach with regular checkpoints'
            }
        ]
        
        # Mitigation strategies
        risk_mitigation['mitigation_strategies'] = [
            {
                'strategy': 'Incremental Development Approach',
                'description': 'Implement improvements in small, testable increments',
                'timeline': 'Ongoing',
                'cost': 'Low'
            },
            {
                'strategy': 'Multiple Architecture Prototyping',
                'description': 'Develop 2-3 architecture candidates in parallel',
                'timeline': '4-6 weeks',
                'cost': 'Medium'
            },
            {
                'strategy': 'Continuous Validation Pipeline',
                'description': 'Automated testing of each improvement increment',
                'timeline': '2-3 weeks setup',
                'cost': 'Low'
            }
        ]
        
        return risk_mitigation
    
    # Document generation methods
    def _create_executive_recommendations(self, recommendations: Dict, output_dir: Path) -> str:
        """Create executive recommendations document."""
        
        lines = []
        exec_summary = recommendations['executive_summary']
        
        lines.extend([
            "# Executive Model Improvement Recommendations",
            "",
            f"**Generated:** {self.timestamp}  ",
            f"**Document Type:** Strategic Decision Support  ",
            f"**Audience:** Executive Leadership, Product Management  ",
            "",
            "## Executive Summary",
            "",
            f"**Overall Recommendation:** {exec_summary['overall_recommendation']}  ",
            f"**Investment Level Required:** {exec_summary['investment_level']}  ",
            f"**Timeline to Production:** {exec_summary['timeline_to_production']}  ",
            f"**Business Risk Level:** {exec_summary['risk_level']}  ",
            f"**Success Confidence:** {exec_summary['confidence_in_success']}  ",
            "",
            "## Key Strategic Focus Areas",
            "",
        ])
        
        for i, area in enumerate(exec_summary['key_focus_areas'], 1):
            lines.append(f"{i}. **{area}**")
        
        lines.extend([
            "",
            "## Investment Analysis",
            "",
        ])
        
        roi = exec_summary['expected_roi']
        lines.extend([
            f"- **Timeline:** {roi['timeline']}",
            f"- **Investment Required:** {roi['investment']}",
            f"- **Expected Return:** {roi['expected_return']}",
            f"- **Payback Period:** {roi['payback_period']}",
            "",
        ])
        
        # Cost-benefit analysis
        if 'cost_benefit_analysis' in recommendations:
            cost_benefit = recommendations['cost_benefit_analysis']
            lines.extend([
                "## Cost-Benefit Analysis",
                "",
                "### Development Costs",
                "",
            ])
            
            for cost_type, cost_value in cost_benefit.get('development_costs', {}).items():
                lines.append(f"- **{cost_type.replace('_', ' ').title()}:** {cost_value}")
            
            lines.extend([
                "",
                "### Expected Benefits",
                "",
            ])
            
            for benefit_type, benefit_value in cost_benefit.get('expected_benefits', {}).items():
                lines.append(f"- **{benefit_type.replace('_', ' ').title()}:** {benefit_value}")
        
        # Immediate actions
        if 'immediate_actions' in recommendations:
            lines.extend([
                "",
                "## Recommended Immediate Actions (Next 4 Weeks)",
                "",
            ])
            
            for i, action in enumerate(recommendations['immediate_actions'][:3], 1):
                lines.extend([
                    f"### {i}. {action['action']}",
                    f"**Timeline:** {action['timeline']}  ",
                    f"**Effort:** {action['effort']}  ",
                    f"**Expected Impact:** {action['expected_impact']}  ",
                    "",
                    action['description'],
                    "",
                ])
        
        # Risk assessment
        if 'risk_mitigation' in recommendations:
            risk_data = recommendations['risk_mitigation']
            lines.extend([
                "## Risk Assessment & Mitigation",
                "",
                "### Key Risks",
                "",
            ])
            
            all_risks = (risk_data.get('technical_risks', []) + 
                        risk_data.get('timeline_risks', []) +
                        risk_data.get('resource_risks', []))
            
            for risk in all_risks[:3]:  # Top 3 risks
                lines.extend([
                    f"- **Risk:** {risk['risk']}",
                    f"  - Probability: {risk['probability']}, Impact: {risk['impact']}",
                    f"  - Mitigation: {risk['mitigation']}",
                    "",
                ])
        
        # Conclusion
        lines.extend([
            "## Conclusion & Next Steps",
            "",
        ])
        
        if exec_summary['risk_level'] == 'Low':
            conclusion = "The model shows strong performance with clear path to production deployment. Recommended to proceed with optimization phase."
        elif exec_summary['risk_level'] == 'Medium':
            conclusion = "The model demonstrates good potential with moderate investment required. Recommended to proceed with targeted improvements."
        elif exec_summary['risk_level'] == 'High':
            conclusion = "The model requires significant improvement but has viable path forward. Substantial investment recommended with careful milestone management."
        else:
            conclusion = "The model faces critical performance challenges. Consider alternative approaches or major architectural changes."
        
        lines.extend([
            conclusion,
            "",
            "**Recommended Decision:** Approve recommended improvement roadmap with appropriate resource allocation.",
            "",
            "---",
            f"*This report was generated by the comprehensive BDD100K evaluation framework on {self.timestamp}*"
        ])
        
        # Save executive recommendations
        exec_file_path = output_dir / "executive_improvement_recommendations.md"
        with open(exec_file_path, 'w') as f:
            f.write("\\n".join(lines))
        
        return str(exec_file_path)
    
    def _create_technical_implementation_guide(self, recommendations: Dict, output_dir: Path) -> str:
        """Create technical implementation guide."""
        
        lines = []
        
        lines.extend([
            "# Technical Implementation Guide",
            "",
            f"**Generated:** {self.timestamp}  ",
            f"**Document Type:** Technical Implementation Roadmap  ",
            f"**Audience:** ML Engineers, Technical Team  ",
            "",
            "## Overview",
            "",
            "This guide provides detailed technical implementation steps for model improvements",
            "based on comprehensive performance analysis and identified optimization opportunities.",
            "",
        ])
        
        # Technical strategies
        if 'technical_strategies' in recommendations:
            tech_strategies = recommendations['technical_strategies']
            
            lines.extend([
                "## Architecture Improvements",
                "",
            ])
            
            for strategy in tech_strategies.get('architecture_improvements', []):
                lines.extend([
                    f"### {strategy['strategy']}",
                    "",
                    f"**Rationale:** {strategy['rationale']}  ",
                    f"**Complexity:** {strategy['complexity']}  ",
                    f"**Timeline:** {strategy['timeline']}  ",
                    f"**Expected Improvement:** {strategy['expected_improvement']}  ",
                    "",
                ])
            
            lines.extend([
                "## Training Enhancements",
                "",
            ])
            
            for strategy in tech_strategies.get('training_enhancements', []):
                lines.extend([
                    f"### {strategy['strategy']}",
                    "",
                    f"**Rationale:** {strategy['rationale']}  ",
                    f"**Complexity:** {strategy['complexity']}  ",
                    f"**Timeline:** {strategy['timeline']}  ",
                    f"**Expected Improvement:** {strategy['expected_improvement']}  ",
                    "",
                ])
        
        # Architecture recommendations
        if 'architecture_recommendations' in recommendations:
            arch_recs = recommendations['architecture_recommendations']
            
            lines.extend([
                "## Detailed Architecture Recommendations",
                "",
            ])
            
            for i, rec in enumerate(arch_recs, 1):
                lines.extend([
                    f"### {i}. {rec['recommendation']}",
                    "",
                    f"**Rationale:** {rec['rationale']}",
                    "",
                    "**Implementation Steps:**",
                ])
                
                for step in rec['implementation_steps']:
                    lines.append(f"1. {step}")
                
                lines.extend([
                    "",
                    f"**Timeline:** {rec['timeline']}  ",
                    f"**Resources:** {rec['resource_requirements']}  ",
                    f"**Risk Level:** {rec['risk_level']}  ",
                    f"**Expected Outcome:** {rec['expected_outcome']}  ",
                    "",
                ])
        
        # Training optimizations
        if 'training_optimizations' in recommendations:
            train_opts = recommendations['training_optimizations']
            
            lines.extend([
                "## Training Optimization Strategies",
                "",
            ])
            
            for opt in train_opts:
                lines.extend([
                    f"### {opt['optimization']}",
                    "",
                    "**Techniques:**",
                ])
                
                for technique in opt['techniques']:
                    lines.append(f"- {technique}")
                
                lines.extend([
                    "",
                    f"**Timeline:** {opt['timeline']}  ",
                    f"**Complexity:** {opt['complexity']}  ",
                    f"**Expected Impact:** {opt['expected_impact']}  ",
                    "",
                ])
        
        # Implementation roadmap
        if 'implementation_roadmap' in recommendations:
            roadmap = recommendations['implementation_roadmap']
            
            lines.extend([
                "## Implementation Roadmap",
                "",
            ])
            
            for phase in roadmap['phases']:
                lines.extend([
                    f"### {phase['phase']}",
                    f"**Timeline:** {phase['timeline']}  ",
                    f"**Resources:** {phase['resources']}  ",
                    f"**Success Criteria:** {phase['success_criteria']}  ",
                    "",
                    "**Actions:**",
                ])
                
                for action in phase['actions']:
                    lines.append(f"- {action}")
                
                lines.append("")
        
        # Save technical guide
        tech_file_path = output_dir / "technical_implementation_guide.md"
        with open(tech_file_path, 'w') as f:
            f.write("\\n".join(lines))
        
        return str(tech_file_path)
    
    def _create_data_strategy_document(self, recommendations: Dict, output_dir: Path) -> str:
        """Create data strategy document."""
        
        lines = []
        
        lines.extend([
            "# Data Strategy for Model Improvement",
            "",
            f"**Generated:** {self.timestamp}  ",
            f"**Document Type:** Data Collection & Enhancement Strategy  ",
            f"**Audience:** Data Engineering, ML Engineering Teams  ",
            "",
            "## Executive Summary",
            "",
            "This document outlines the data strategy required to achieve target model performance",
            "improvements, including collection priorities, augmentation strategies, and quality enhancements.",
            "",
        ])
        
        # Data strategies
        if 'data_strategies' in recommendations:
            data_strat = recommendations['data_strategies']
            
            # Collection priorities
            if 'collection_priorities' in data_strat and data_strat['collection_priorities']:
                lines.extend([
                    "## Data Collection Priorities",
                    "",
                ])
                
                for priority in data_strat['collection_priorities']:
                    lines.extend([
                        f"### {priority['condition'].title()} Conditions",
                        f"**Current Performance:** {priority['current_performance']:.3f}  ",
                        f"**Collection Target:** {priority['collection_target']}  ",
                        f"**Priority:** {priority['priority']}  ",
                        f"**Timeline:** {priority['timeline']}  ",
                        "",
                    ])
            
            # Augmentation strategies
            if 'augmentation_strategies' in data_strat and data_strat['augmentation_strategies']:
                lines.extend([
                    "## Advanced Augmentation Pipeline",
                    "",
                ])
                
                for aug in data_strat['augmentation_strategies']:
                    lines.extend([
                        f"### {aug['strategy']}",
                        f"**Description:** {aug['description']}  ",
                        f"**Implementation:** {aug['implementation']}  ",
                        f"**Expected Impact:** {aug['expected_impact']}  ",
                        "",
                    ])
        
        # Environmental improvements
        if 'environmental_robustness' in recommendations:
            env_improvements = recommendations['environmental_robustness']
            
            if 'augmentation_pipeline' in env_improvements and env_improvements['augmentation_pipeline']:
                lines.extend([
                    "## Environmental Robustness Enhancement",
                    "",
                ])
                
                for technique in env_improvements['augmentation_pipeline']:
                    lines.extend([
                        f"### {technique['technique']}",
                        f"**Implementation:** {technique['implementation']}  ",
                        f"**Expected Impact:** {technique['expected_impact']}  ",
                        "",
                    ])
        
        # Save data strategy
        data_file_path = output_dir / "data_improvement_strategy.md"
        with open(data_file_path, 'w') as f:
            f.write("\\n".join(lines))
        
        return str(data_file_path)
    
    def _create_safety_enhancement_plan(self, recommendations: Dict, output_dir: Path) -> str:
        """Create safety enhancement plan."""
        
        lines = []
        
        lines.extend([
            "# Safety Enhancement Plan",
            "",
            f"**Generated:** {self.timestamp}  ",
            f"**Document Type:** Safety-Critical Performance Improvement Plan  ",
            f"**Audience:** Safety Team, ML Engineering, Product Management  ",
            "",
            "## Executive Summary",
            "",
            "This plan addresses safety-critical performance improvements required for",
            "autonomous vehicle deployment readiness.",
            "",
        ])
        
        # Safety improvements
        if 'safety_specific_improvements' in recommendations:
            safety_improvements = recommendations['safety_specific_improvements']
            
            # Immediate actions
            if 'immediate_actions' in safety_improvements and safety_improvements['immediate_actions']:
                lines.extend([
                    "## Immediate Safety Actions",
                    "",
                ])
                
                for action in safety_improvements['immediate_actions']:
                    lines.extend([
                        f"### {action['action']}",
                        f"**Description:** {action['description']}  ",
                        f"**Timeline:** {action['timeline']}  ",
                        f"**Success Criteria:** {action['success_criteria']}  ",
                        "",
                    ])
            
            # Technical enhancements
            if 'technical_enhancements' in safety_improvements and safety_improvements['technical_enhancements']:
                lines.extend([
                    "## Technical Safety Enhancements",
                    "",
                ])
                
                for enhancement in safety_improvements['technical_enhancements']:
                    lines.extend([
                        f"### {enhancement['enhancement']}",
                        "",
                        "**Components:**",
                    ])
                    
                    for component in enhancement['components']:
                        lines.append(f"- {component}")
                    
                    lines.append("")
        
        # Save safety plan
        safety_file_path = output_dir / "safety_enhancement_plan.md"
        with open(safety_file_path, 'w') as f:
            f.write("\\n".join(lines))
        
        return str(safety_file_path)
    
    def _create_improvement_roadmap_visualization(self, recommendations: Dict, output_dir: Path) -> str:
        """Create visual improvement roadmap."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Timeline visualization
        if 'implementation_roadmap' in recommendations:
            roadmap = recommendations['implementation_roadmap']
            phases = roadmap['phases']
            
            # Extract phase data
            phase_names = []
            start_weeks = []
            durations = []
            
            current_week = 0
            for phase in phases:
                phase_names.append(phase['phase'].replace(' - ', '\\n'))
                start_weeks.append(current_week)
                
                # Parse timeline to get duration
                timeline = phase['timeline']
                if 'week' in timeline:
                    if '-' in timeline:
                        duration = int(timeline.split('-')[1].split()[0])
                    else:
                        duration = int(timeline.split()[0])
                else:
                    duration = 12  # Default
                
                durations.append(duration)
                current_week += duration
            
            # Create Gantt chart
            colors = ['lightblue', 'lightgreen', 'lightsalmon']
            for i, (name, start, duration) in enumerate(zip(phase_names, start_weeks, durations)):
                ax1.barh(i, duration, left=start, height=0.6, 
                        color=colors[i % len(colors)], alpha=0.8)
                ax1.text(start + duration/2, i, f'{duration}w', 
                        ha='center', va='center', fontweight='bold')
            
            ax1.set_yticks(range(len(phase_names)))
            ax1.set_yticklabels(phase_names)
            ax1.set_xlabel('Timeline (Weeks)')
            ax1.set_title('Model Improvement Implementation Timeline', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
        
        # Cost-benefit visualization
        if 'cost_benefit_analysis' in recommendations:
            cost_benefit = recommendations['cost_benefit_analysis']
            
            # Extract development costs (simplified)
            costs = cost_benefit.get('development_costs', {})
            cost_categories = []
            cost_values = []
            
            for category, cost_str in costs.items():
                cost_categories.append(category.replace('_', '\\n').title())
                # Extract numeric value (simplified)
                import re
                numbers = re.findall(r'\\$([0-9]+)K', cost_str)
                if numbers:
                    cost_values.append(int(numbers[0]))
                else:
                    cost_values.append(50)  # Default
            
            if cost_categories and cost_values:
                bars = ax2.bar(cost_categories, cost_values, alpha=0.7, color='orange')
                ax2.set_ylabel('Cost (K$)')
                ax2.set_title('Estimated Development Costs', fontweight='bold', fontsize=14)
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, cost_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            f'${value}K', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        viz_file_path = output_dir / "improvement_roadmap_visualization.png"
        plt.savefig(viz_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(viz_file_path)


def main():
    """Generate comprehensive improvement recommendations."""
    parser = argparse.ArgumentParser(description='Generate Comprehensive Improvement Recommendations')
    parser.add_argument('--comprehensive-results', type=str,
                       default='evaluation_results/comprehensive_reports',
                       help='Path to comprehensive evaluation results directory')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_results/improvement_recommendations',
                       help='Output directory for improvement recommendations')
    
    args = parser.parse_args()
    
    print("🎯 Phase 6: Generating Comprehensive Improvement Recommendations")
    print("=" * 70)
    
    # Load comprehensive results from all previous phases
    comprehensive_dir = Path(args.comprehensive_results)
    
    # Try to load all available results
    all_results = {}
    
    # Load from phase reports if available
    phase_files = [
        'evaluation_results/evaluation_results.json',
        'evaluation_results/failure_analysis_tests/comprehensive_failure_analysis_results.json',
        'evaluation_results/phase4_clustering/phase4_clustering_results.json'
    ]
    
    for file_path in phase_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_results.update(data)
                print(f"✅ Loaded: {Path(file_path).name}")
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if not all_results:
        print("❌ No evaluation results found. Please run previous phases first.")
        return False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate comprehensive recommendations
        print("\\nGenerating comprehensive improvement recommendations...")
        
        recommendation_engine = ImprovementRecommendationEngine()
        recommendations = recommendation_engine.generate_comprehensive_recommendations(all_results, output_dir)
        
        print("\\n" + "=" * 70)
        print("✅ Phase 6: Improvement Recommendations COMPLETED!")
        print(f"\\n📁 All recommendations saved to: {output_dir}")
        
        print("\\n📋 Generated Recommendation Documents:")
        for doc_type, file_path in recommendations.get('generated_files', {}).items():
            if file_path:
                print(f"  - {Path(file_path).name} ({doc_type.replace('_', ' ').title()})")
        
        # Display key recommendations summary
        if 'executive_summary' in recommendations:
            exec_summary = recommendations['executive_summary']
            print("\\n🎯 Key Recommendations:")
            print(f"  - Overall Strategy: {exec_summary['overall_recommendation']}")
            print(f"  - Investment Level: {exec_summary['investment_level']}")
            print(f"  - Timeline: {exec_summary['timeline_to_production']}")
            
            if 'key_focus_areas' in exec_summary:
                print("  - Priority Focus Areas:")
                for area in exec_summary['key_focus_areas'][:3]:
                    print(f"    * {area}")
        
        print("\\n🎯 Phase 6 Status: COMPLETED")
        print("✅ All 6 phases of comprehensive BDD100K evaluation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Phase 6 improvement recommendations failed: {e}")
        raise


if __name__ == "__main__":
    main()