#!/usr/bin/env python3
"""
Performance Pattern Detection for BDD100K Object Detection

This module identifies patterns in model performance across different conditions,
object characteristics, and failure modes to provide actionable insights.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd


class PerformancePatternDetector:
    """Advanced pattern detection for model performance analysis."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize pattern detector.
        
        Args:
            class_names: List of class names for BDD100K dataset
        """
        self.class_names = class_names or [
            'pedestrian', 'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
        ]
        
        # Define safety-critical classes
        self.safety_critical = {'pedestrian', 'rider', 'bicycle', 'motorcycle'}
        
        # Performance thresholds for pattern detection
        self.performance_thresholds = {
            'excellent': 0.80,
            'good': 0.65,
            'fair': 0.45,
            'poor': 0.30
        }
    
    def detect_performance_patterns(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Detect comprehensive performance patterns.
        
        Args:
            evaluation_results: Complete evaluation results from all metrics
            
        Returns:
            Dictionary containing detected patterns and insights
        """
        patterns = {
            'class_performance_clusters': self._cluster_class_performance(evaluation_results),
            'environmental_impact': self._analyze_environmental_patterns(evaluation_results),
            'object_size_patterns': self._analyze_size_patterns(evaluation_results),
            'spatial_bias_patterns': self._detect_spatial_bias(evaluation_results),
            'confidence_calibration': self._analyze_confidence_patterns(evaluation_results),
            'failure_correlations': self._find_failure_correlations(evaluation_results),
            'performance_degradation': self._identify_degradation_patterns(evaluation_results),
            'safety_risk_patterns': self._analyze_safety_patterns(evaluation_results)
        }
        
        # Generate actionable insights
        patterns['actionable_insights'] = self._generate_insights(patterns, evaluation_results)
        
        return patterns
    
    def _cluster_class_performance(self, results: Dict) -> Dict[str, Any]:
        """Cluster classes based on performance characteristics."""
        if 'coco_metrics' not in results or 'per_class_AP' not in results['coco_metrics']:
            return {'error': 'Missing per-class AP data'}
        
        per_class_ap = results['coco_metrics']['per_class_AP']
        
        # Create feature matrix for clustering
        features = []
        class_list = []
        
        for class_name in self.class_names:
            if class_name in per_class_ap:
                # Multi-dimensional performance features
                ap = per_class_ap[class_name]
                
                # Get additional metrics if available
                recall = 0.0
                precision = 0.0
                
                if 'safety_metrics' in results and 'per_class_safety' in results['safety_metrics']:
                    safety_data = results['safety_metrics']['per_class_safety'].get(class_name, {})
                    recall = safety_data.get('recall', 0.0)
                    precision = safety_data.get('precision', 0.0)
                
                # Feature vector: [AP, Recall, Precision, IsSafetyCritical]
                is_safety_critical = 1.0 if class_name in self.safety_critical else 0.0
                features.append([ap, recall, precision, is_safety_critical])
                class_list.append(class_name)
        
        if len(features) < 3:
            return {'error': 'Insufficient data for clustering'}
        
        # Perform clustering
        features_array = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Determine optimal number of clusters (2-4)
        n_clusters = min(4, max(2, len(features) // 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_classes = [class_list[j] for j in range(len(class_list)) if cluster_mask[j]]
            cluster_features = features_array[cluster_mask]
            
            cluster_analysis[f'cluster_{i}'] = {
                'classes': cluster_classes,
                'avg_ap': float(np.mean(cluster_features[:, 0])),
                'avg_recall': float(np.mean(cluster_features[:, 1])),
                'avg_precision': float(np.mean(cluster_features[:, 2])),
                'safety_critical_count': int(np.sum(cluster_features[:, 3])),
                'performance_tier': self._classify_performance_tier(np.mean(cluster_features[:, 0]))
            }
        
        return {
            'clusters': cluster_analysis,
            'n_clusters': n_clusters,
            'total_classes': len(class_list)
        }
    
    def _analyze_environmental_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze performance patterns across environmental conditions."""
        if 'contextual_metrics' not in results:
            return {'error': 'Missing contextual metrics data'}
        
        contextual = results['contextual_metrics']
        patterns = {}
        
        # Weather impact analysis
        if 'weather_performance' in contextual:
            weather_data = contextual['weather_performance']
            patterns['weather_impact'] = self._analyze_condition_impact(weather_data, 'weather')
        
        # Lighting impact analysis
        if 'lighting_performance' in contextual:
            lighting_data = contextual['lighting_performance']
            patterns['lighting_impact'] = self._analyze_condition_impact(lighting_data, 'lighting')
        
        # Scene type impact
        if 'scene_performance' in contextual:
            scene_data = contextual['scene_performance']
            patterns['scene_impact'] = self._analyze_condition_impact(scene_data, 'scene_type')
        
        return patterns
    
    def _analyze_condition_impact(self, condition_data: Dict, condition_type: str) -> Dict[str, Any]:
        """Analyze impact of specific environmental condition."""
        condition_scores = []
        condition_names = []
        
        for condition, metrics in condition_data.items():
            if isinstance(metrics, dict) and 'mean_ap' in metrics:
                condition_scores.append(metrics['mean_ap'])
                condition_names.append(condition)
        
        if len(condition_scores) < 2:
            return {'error': f'Insufficient {condition_type} data'}
        
        # Statistical analysis
        mean_score = np.mean(condition_scores)
        std_score = np.std(condition_scores)
        cv = std_score / mean_score if mean_score > 0 else 0
        
        # Find best and worst conditions
        best_idx = np.argmax(condition_scores)
        worst_idx = np.argmin(condition_scores)
        
        # Performance gap analysis
        performance_gap = condition_scores[best_idx] - condition_scores[worst_idx]
        
        return {
            'mean_performance': float(mean_score),
            'std_performance': float(std_score),
            'coefficient_of_variation': float(cv),
            'best_condition': {
                'name': condition_names[best_idx],
                'score': float(condition_scores[best_idx])
            },
            'worst_condition': {
                'name': condition_names[worst_idx],
                'score': float(condition_scores[worst_idx])
            },
            'performance_gap': float(performance_gap),
            'stability_assessment': 'stable' if cv < 0.2 else 'variable' if cv < 0.4 else 'unstable'
        }
    
    def _analyze_size_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze performance patterns by object size."""
        if 'coco_metrics' not in results:
            return {'error': 'Missing COCO metrics data'}
        
        metrics = results['coco_metrics']
        size_metrics = {
            'small': metrics.get('mAP_small', 0.0),
            'medium': metrics.get('mAP_medium', 0.0),
            'large': metrics.get('mAP_large', 0.0)
        }
        
        # Size bias analysis
        max_performance = max(size_metrics.values())
        min_performance = min(size_metrics.values())
        size_gap = max_performance - min_performance
        
        # Determine bias direction
        best_size = max(size_metrics.keys(), key=lambda k: size_metrics[k])
        worst_size = min(size_metrics.keys(), key=lambda k: size_metrics[k])
        
        return {
            'size_performance': size_metrics,
            'size_gap': float(size_gap),
            'best_performing_size': best_size,
            'worst_performing_size': worst_size,
            'bias_severity': 'low' if size_gap < 0.1 else 'moderate' if size_gap < 0.2 else 'high',
            'small_object_challenge': size_metrics['small'] < 0.3
        }
    
    def _detect_spatial_bias(self, results: Dict) -> Dict[str, Any]:
        """Detect spatial bias patterns in object detection."""
        if 'contextual_metrics' not in results or 'position_analysis' not in results['contextual_metrics']:
            return {'error': 'Missing position analysis data'}
        
        position_data = results['contextual_metrics']['position_analysis']
        
        # Analyze center vs edge performance
        center_performance = position_data.get('center', {}).get('mean_ap', 0.0)
        edge_performance = position_data.get('edge', {}).get('mean_ap', 0.0)
        
        spatial_bias = center_performance - edge_performance
        
        return {
            'center_performance': float(center_performance),
            'edge_performance': float(edge_performance),
            'spatial_bias': float(spatial_bias),
            'bias_direction': 'center' if spatial_bias > 0.05 else 'edge' if spatial_bias < -0.05 else 'balanced',
            'bias_severity': abs(spatial_bias)
        }
    
    def _analyze_confidence_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze confidence calibration patterns."""
        # This would typically analyze prediction confidence vs actual accuracy
        # For now, return basic analysis structure
        return {
            'calibration_quality': 'unknown',
            'overconfidence_detected': False,
            'underconfidence_detected': False,
            'confidence_distribution': 'analysis_needed'
        }
    
    def _find_failure_correlations(self, results: Dict) -> Dict[str, Any]:
        """Find correlations between different failure modes."""
        if 'failure_analysis' not in results:
            return {'error': 'Missing failure analysis data'}
        
        # Analyze failure patterns from failure analyzer results
        failure_data = results['failure_analysis']
        
        correlations = {}
        
        # Environmental correlation with failures
        if 'environmental_failures' in failure_data:
            env_failures = failure_data['environmental_failures']
            correlations['environmental'] = self._calculate_failure_correlations(env_failures)
        
        return correlations
    
    def _calculate_failure_correlations(self, failure_data: Dict) -> Dict[str, float]:
        """Calculate correlations between environmental conditions and failures."""
        correlations = {}
        
        for condition, data in failure_data.items():
            if isinstance(data, dict) and 'failure_rate' in data:
                correlations[condition] = data['failure_rate']
        
        return correlations
    
    def _identify_degradation_patterns(self, results: Dict) -> Dict[str, Any]:
        """Identify performance degradation patterns."""
        # This would analyze performance trends over time or conditions
        # Currently return structure for future implementation
        return {
            'degradation_detected': False,
            'degradation_factors': [],
            'stability_score': 0.8
        }
    
    def _analyze_safety_patterns(self, results: Dict) -> Dict[str, Any]:
        """Analyze safety-related performance patterns."""
        if 'safety_metrics' not in results:
            return {'error': 'Missing safety metrics data'}
        
        safety_data = results['safety_metrics']
        
        # Overall safety assessment
        overall_safety = safety_data.get('overall_safety_score', {})
        safety_score = overall_safety.get('overall_safety_score', 0.0)
        
        # Safety-critical class analysis
        safety_classes_performance = {}
        if 'per_class_safety' in safety_data:
            for class_name in self.safety_critical:
                if class_name in safety_data['per_class_safety']:
                    class_data = safety_data['per_class_safety'][class_name]
                    safety_classes_performance[class_name] = {
                        'recall': class_data.get('recall', 0.0),
                        'precision': class_data.get('precision', 0.0),
                        'meets_threshold': class_data.get('recall', 0.0) >= 0.80
                    }
        
        # Safety risk assessment
        high_risk_classes = [
            cls for cls, metrics in safety_classes_performance.items()
            if not metrics['meets_threshold']
        ]
        
        return {
            'overall_safety_score': float(safety_score),
            'safety_classes_performance': safety_classes_performance,
            'high_risk_classes': high_risk_classes,
            'safety_compliance': len(high_risk_classes) == 0,
            'critical_safety_gaps': len(high_risk_classes)
        }
    
    def _classify_performance_tier(self, score: float) -> str:
        """Classify performance score into tier."""
        if score >= self.performance_thresholds['excellent']:
            return 'excellent'
        elif score >= self.performance_thresholds['good']:
            return 'good'
        elif score >= self.performance_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_insights(self, patterns: Dict, results: Dict) -> List[Dict[str, str]]:
        """Generate actionable insights from detected patterns."""
        insights = []
        
        # Class performance insights
        if 'class_performance_clusters' in patterns and 'clusters' in patterns['class_performance_clusters']:
            clusters = patterns['class_performance_clusters']['clusters']
            
            for cluster_id, cluster_data in clusters.items():
                if cluster_data['performance_tier'] == 'poor':
                    insights.append({
                        'category': 'Class Performance',
                        'priority': 'high',
                        'insight': f"Classes {', '.join(cluster_data['classes'])} show poor performance (avg mAP: {cluster_data['avg_ap']:.3f})",
                        'recommendation': 'Focus training on these classes with data augmentation and class-specific optimizations'
                    })
        
        # Environmental insights
        if 'environmental_impact' in patterns:
            for condition_type, condition_data in patterns['environmental_impact'].items():
                if 'error' not in condition_data and condition_data.get('stability_assessment') == 'unstable':
                    worst_condition = condition_data['worst_condition']['name']
                    insights.append({
                        'category': 'Environmental Robustness',
                        'priority': 'medium',
                        'insight': f"Performance highly variable across {condition_type} conditions, worst in {worst_condition}",
                        'recommendation': f'Increase training data for {worst_condition} conditions and add domain adaptation techniques'
                    })
        
        # Size bias insights
        if 'object_size_patterns' in patterns:
            size_data = patterns['object_size_patterns']
            if 'error' not in size_data and size_data.get('small_object_challenge'):
                insights.append({
                    'category': 'Object Detection',
                    'priority': 'high',
                    'insight': f"Poor small object detection (mAP: {size_data['size_performance']['small']:.3f})",
                    'recommendation': 'Implement multi-scale training, feature pyramid networks, or specialized small object detection techniques'
                })
        
        # Safety insights
        if 'safety_risk_patterns' in patterns:
            safety_data = patterns['safety_risk_patterns']
            if 'error' not in safety_data and not safety_data['safety_compliance']:
                high_risk = ', '.join(safety_data['high_risk_classes'])
                insights.append({
                    'category': 'Safety Critical',
                    'priority': 'critical',
                    'insight': f"Safety-critical classes below threshold: {high_risk}",
                    'recommendation': 'Implement safety-focused training with weighted loss, hard negative mining, and extensive validation'
                })
        
        return insights
    
    def generate_pattern_visualizations(self, patterns: Dict, output_dir: Path) -> List[str]:
        """Generate visualizations for detected patterns."""
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        # 1. Class performance clustering visualization
        if 'class_performance_clusters' in patterns and 'clusters' in patterns['class_performance_clusters']:
            fig_path = self._visualize_class_clusters(patterns['class_performance_clusters'], output_dir)
            if fig_path:
                generated_files.append(fig_path)
        
        # 2. Environmental impact heatmap
        if 'environmental_impact' in patterns:
            fig_path = self._visualize_environmental_patterns(patterns['environmental_impact'], output_dir)
            if fig_path:
                generated_files.append(fig_path)
        
        # 3. Size performance comparison
        if 'object_size_patterns' in patterns:
            fig_path = self._visualize_size_patterns(patterns['object_size_patterns'], output_dir)
            if fig_path:
                generated_files.append(fig_path)
        
        # 4. Safety analysis dashboard
        if 'safety_risk_patterns' in patterns:
            fig_path = self._visualize_safety_patterns(patterns['safety_risk_patterns'], output_dir)
            if fig_path:
                generated_files.append(fig_path)
        
        return generated_files
    
    def _visualize_class_clusters(self, cluster_data: Dict, output_dir: Path) -> Optional[str]:
        """Visualize class performance clusters."""
        if 'clusters' not in cluster_data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cluster performance comparison
        cluster_ids = []
        cluster_aps = []
        cluster_sizes = []
        
        for cluster_id, data in cluster_data['clusters'].items():
            cluster_ids.append(cluster_id.replace('cluster_', 'Cluster '))
            cluster_aps.append(data['avg_ap'])
            cluster_sizes.append(len(data['classes']))
        
        bars = ax1.bar(cluster_ids, cluster_aps, alpha=0.7)
        ax1.set_title('Performance by Class Clusters', fontweight='bold')
        ax1.set_ylabel('Average Precision (mAP)')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by performance tier
        colors = ['red', 'orange', 'lightblue', 'green']
        for bar, ap in zip(bars, cluster_aps):
            if ap >= 0.8:
                bar.set_color('green')
            elif ap >= 0.65:
                bar.set_color('lightblue')
            elif ap >= 0.45:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for bar, ap, size in zip(bars, cluster_aps, cluster_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}\n({size} classes)', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Class distribution in clusters
        cluster_names = []
        class_counts = []
        for cluster_id, data in cluster_data['clusters'].items():
            cluster_names.append(cluster_id.replace('cluster_', 'C'))
            class_counts.append(len(data['classes']))
        
        ax2.pie(class_counts, labels=cluster_names, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Class Distribution Across Clusters', fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / 'class_performance_clusters.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_environmental_patterns(self, env_patterns: Dict, output_dir: Path) -> Optional[str]:
        """Visualize environmental impact patterns."""
        fig, axes = plt.subplots(1, len(env_patterns), figsize=(5 * len(env_patterns), 6))
        if len(env_patterns) == 1:
            axes = [axes]
        
        for idx, (condition_type, data) in enumerate(env_patterns.items()):
            if 'error' in data:
                continue
                
            ax = axes[idx] if idx < len(axes) else axes[0]
            
            # Create performance comparison
            best = data['best_condition']
            worst = data['worst_condition']
            mean_perf = data['mean_performance']
            
            conditions = ['Best\n' + best['name'], 'Average', 'Worst\n' + worst['name']]
            performances = [best['score'], mean_perf, worst['score']]
            
            bars = ax.bar(conditions, performances, 
                         color=['green', 'lightblue', 'red'], alpha=0.7)
            ax.set_title(f'{condition_type.replace("_", " ").title()} Impact', fontweight='bold')
            ax.set_ylabel('Performance Score')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, performances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / 'environmental_patterns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_size_patterns(self, size_data: Dict, output_dir: Path) -> Optional[str]:
        """Visualize object size performance patterns."""
        if 'error' in size_data:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sizes = list(size_data['size_performance'].keys())
        performances = list(size_data['size_performance'].values())
        
        colors = ['red' if size == size_data['worst_performing_size'] 
                 else 'green' if size == size_data['best_performing_size'] 
                 else 'lightblue' for size in sizes]
        
        bars = ax.bar(sizes, performances, color=colors, alpha=0.7)
        ax.set_title('Performance by Object Size', fontweight='bold')
        ax.set_ylabel('mAP Score')
        ax.grid(True, alpha=0.3)
        
        # Add bias severity indicator
        bias_text = f"Size Bias: {size_data['bias_severity']} ({size_data['size_gap']:.3f})"
        ax.text(0.02, 0.98, bias_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               verticalalignment='top')
        
        # Add value labels
        for bar, score in zip(bars, performances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / 'object_size_patterns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _visualize_safety_patterns(self, safety_data: Dict, output_dir: Path) -> Optional[str]:
        """Visualize safety-related patterns."""
        if 'error' in safety_data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Safety-critical class performance
        if 'safety_classes_performance' in safety_data:
            classes = list(safety_data['safety_classes_performance'].keys())
            recalls = [safety_data['safety_classes_performance'][cls]['recall'] for cls in classes]
            precisions = [safety_data['safety_classes_performance'][cls]['precision'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, recalls, width, label='Recall', alpha=0.7)
            bars2 = ax1.bar(x + width/2, precisions, width, label='Precision', alpha=0.7)
            
            ax1.set_title('Safety-Critical Class Performance', fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45, ha='right')
            ax1.axhline(y=0.8, color='red', linestyle='--', label='Safety Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Safety compliance overview
        compliance_data = {
            'Compliant': len(safety_data['safety_classes_performance']) - len(safety_data['high_risk_classes']),
            'High Risk': len(safety_data['high_risk_classes'])
        }
        
        colors = ['green', 'red']
        ax2.pie(compliance_data.values(), labels=compliance_data.keys(), 
               colors=colors, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Safety Compliance Status', fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / 'safety_patterns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)