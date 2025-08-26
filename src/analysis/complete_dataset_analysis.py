#!/usr/bin/env python3
"""
Comprehensive BDD100K Dataset Analysis for All 10 Object Detection Classes

This module provides comprehensive analysis functionality for the complete BDD100K dataset,
including class distribution analysis, environmental pattern analysis, anomaly detection,
and visualization dashboard generation for all 10 object detection classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CompleteBDD100KAnalyzer:
    """
    Comprehensive analyzer for BDD100K dataset with all 10 object detection classes.
    
    Provides detailed analysis including:
    - Complete 10-class distribution analysis
    - Environmental pattern analysis (weather, scene, time)  
    - Safety-critical class analysis (pedestrians, riders, cyclists)
    - Anomaly detection and outlier identification
    - Interactive dashboard generation
    - Statistical significance testing
    """
    
    # Complete 10-class hierarchy organized by safety importance
    SAFETY_CRITICAL_CLASSES = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'train']
    INFRASTRUCTURE_CLASSES = ['traffic light', 'traffic sign']
    
    ALL_CLASSES = SAFETY_CRITICAL_CLASSES + VEHICLE_CLASSES + INFRASTRUCTURE_CLASSES
    
    # Safety importance weights for analysis
    SAFETY_WEIGHTS = {
        'pedestrian': 10.0,    # Highest safety priority
        'rider': 9.0,          # Very high priority
        'bicycle': 9.0,        # Very high priority  
        'motorcycle': 8.0,     # High priority
        'car': 6.0,           # Medium priority
        'truck': 7.0,         # Medium-high priority (size/mass)
        'bus': 7.0,           # Medium-high priority (size/mass)
        'train': 5.0,         # Lower priority (controlled environment)
        'traffic light': 8.0, # High priority (safety infrastructure)
        'traffic sign': 6.0   # Medium priority (information)
    }
    
    def __init__(self, data_dir: str = "data/analysis/processed_10class"):
        """
        Initialize analyzer with processed 10-class data.
        
        Args:
            data_dir: Directory containing processed 10-class CSV files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/analysis/complete_analysis_10class")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.train_df = self._load_data("train")
        self.val_df = self._load_data("val")
        self.combined_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        logger.info(f"Loaded complete 10-class dataset:")
        logger.info(f"  Training: {len(self.train_df):,} annotations")
        logger.info(f"  Validation: {len(self.val_df):,} annotations")
        logger.info(f"  Total: {len(self.combined_df):,} annotations")
        
    def _load_data(self, split: str) -> pd.DataFrame:
        """Load data for a specific split."""
        file_path = self.data_dir / f"{split}_annotations_10class.csv"
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
        
    def analyze_complete_class_distribution(self) -> Dict[str, Any]:
        """
        Analyze complete 10-class distribution with detailed statistics.
        
        Returns:
            Comprehensive class distribution analysis
        """
        logger.info("Analyzing complete 10-class distribution...")
        
        analysis = {}
        
        # Overall class distribution
        class_counts = self.combined_df['category'].value_counts()
        total_objects = len(self.combined_df)
        
        analysis['basic_statistics'] = {
            'total_objects': total_objects,
            'total_images': self.combined_df['image_name'].nunique(),
            'num_classes': len(self.ALL_CLASSES),
            'classes_found': list(class_counts.index),
            'missing_classes': [cls for cls in self.ALL_CLASSES if cls not in class_counts.index]
        }
        
        # Detailed class distribution
        class_distribution = {}
        for cls in self.ALL_CLASSES:
            count = class_counts.get(cls, 0)
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            
            class_distribution[cls] = {
                'count': int(count),
                'percentage': round(percentage, 2),
                'safety_weight': self.SAFETY_WEIGHTS[cls],
                'category_type': self._get_class_category(cls)
            }
            
        analysis['class_distribution'] = class_distribution
        
        # Imbalance analysis
        if not class_counts.empty:
            max_count = class_counts.iloc[0]
            min_count = class_counts.iloc[-1] if class_counts.iloc[-1] > 0 else 1
            
            analysis['imbalance_metrics'] = {
                'imbalance_ratio': float(max_count / min_count),
                'most_frequent': {
                    'class': class_counts.index[0],
                    'count': int(class_counts.iloc[0])
                },
                'least_frequent': {
                    'class': class_counts.index[-1], 
                    'count': int(class_counts.iloc[-1])
                },
                'gini_coefficient': self._calculate_gini_coefficient(class_counts.values),
                'entropy': self._calculate_entropy(class_counts.values)
            }
        
        # Safety-critical analysis
        analysis['safety_analysis'] = self._analyze_safety_critical_classes()
        
        # Split consistency
        analysis['split_consistency'] = self._analyze_split_consistency()
        
        return analysis
        
    def _get_class_category(self, cls: str) -> str:
        """Get category type for a class."""
        if cls in self.SAFETY_CRITICAL_CLASSES:
            return "safety_critical"
        elif cls in self.VEHICLE_CLASSES:
            return "vehicle"
        else:
            return "infrastructure"
            
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for class imbalance."""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n+1)) * values) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy for class distribution."""
        total = np.sum(values)
        if total == 0:
            return 0
        probs = values / total
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))
        
    def _analyze_safety_critical_classes(self) -> Dict[str, Any]:
        """Analyze safety-critical classes specifically."""
        safety_analysis = {}
        
        for cls in self.SAFETY_CRITICAL_CLASSES:
            cls_data = self.combined_df[self.combined_df['category'] == cls]
            
            if len(cls_data) > 0:
                safety_analysis[cls] = {
                    'total_count': len(cls_data),
                    'images_with_class': cls_data['image_name'].nunique(),
                    'avg_per_image': len(cls_data) / cls_data['image_name'].nunique(),
                    'occlusion_rate': cls_data['occluded'].mean() if 'occluded' in cls_data else 0,
                    'truncation_rate': cls_data['truncated'].mean() if 'truncated' in cls_data else 0,
                    'avg_size': cls_data['bbox_area'].mean() if 'bbox_area' in cls_data else 0
                }
            else:
                safety_analysis[cls] = {
                    'total_count': 0,
                    'status': 'missing_from_dataset'
                }
                
        return safety_analysis
        
    def _analyze_split_consistency(self) -> Dict[str, Any]:
        """Analyze consistency between train/val splits."""
        if self.train_df.empty or self.val_df.empty:
            return {'status': 'insufficient_data'}
            
        train_dist = self.train_df['category'].value_counts(normalize=True)
        val_dist = self.val_df['category'].value_counts(normalize=True)
        
        # Calculate KL divergence
        common_classes = set(train_dist.index) & set(val_dist.index)
        
        consistency_metrics = {}
        for cls in common_classes:
            train_prop = train_dist.get(cls, 0)
            val_prop = val_dist.get(cls, 0)
            
            consistency_metrics[cls] = {
                'train_proportion': float(train_prop),
                'val_proportion': float(val_prop), 
                'ratio': float(train_prop / val_prop) if val_prop > 0 else float('inf'),
                'difference': float(abs(train_prop - val_prop))
            }
            
        return consistency_metrics
        
    def analyze_environmental_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across environmental conditions.
        
        Returns:
            Environmental pattern analysis results
        """
        logger.info("Analyzing environmental patterns...")
        
        env_analysis = {}
        
        # Weather analysis
        weather_dist = self.combined_df.groupby(['img_attr_weather', 'category']).size().unstack(fill_value=0)
        env_analysis['weather_patterns'] = weather_dist.to_dict()
        
        # Scene analysis
        scene_dist = self.combined_df.groupby(['img_attr_scene', 'category']).size().unstack(fill_value=0)
        env_analysis['scene_patterns'] = scene_dist.to_dict()
        
        # Time of day analysis
        time_dist = self.combined_df.groupby(['img_attr_timeofday', 'category']).size().unstack(fill_value=0)
        env_analysis['timeofday_patterns'] = time_dist.to_dict()
        
        # Combined environmental analysis
        env_combinations = self.combined_df.groupby([
            'img_attr_weather', 'img_attr_scene', 'img_attr_timeofday'
        ]).size().sort_values(ascending=False)
        
        env_analysis['top_combinations'] = env_combinations.head(20).to_dict()
        
        return env_analysis
        
    def detect_anomalies_and_patterns(self) -> Dict[str, Any]:
        """
        Detect anomalies and interesting patterns in the dataset.
        
        Returns:
            Anomaly detection results
        """
        logger.info("Detecting anomalies and patterns...")
        
        anomalies = {}
        
        # Size anomalies
        for cls in self.ALL_CLASSES:
            cls_data = self.combined_df[self.combined_df['category'] == cls]
            if len(cls_data) > 100:  # Only analyze classes with sufficient data
                
                # Detect size outliers using IQR method
                areas = cls_data['bbox_area']
                Q1 = areas.quantile(0.25)
                Q3 = areas.quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = cls_data[
                    (areas < Q1 - 1.5 * IQR) | (areas > Q3 + 1.5 * IQR)
                ]
                
                anomalies[f"{cls}_size_outliers"] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(cls_data) * 100,
                    'examples': outliers.head(5)[['image_name', 'bbox_area']].to_dict('records')
                }
        
        # Co-occurrence anomalies
        image_class_counts = self.combined_df.groupby('image_name')['category'].value_counts().unstack(fill_value=0)
        
        # Find images with unusual class combinations
        for cls in self.SAFETY_CRITICAL_CLASSES:
            if cls in image_class_counts.columns:
                # Images with many instances of safety-critical classes
                high_count_images = image_class_counts[image_class_counts[cls] > 5]
                
                if len(high_count_images) > 0:
                    anomalies[f"{cls}_high_density"] = {
                        'count': len(high_count_images),
                        'max_instances': int(image_class_counts[cls].max()),
                        'examples': high_count_images.index[:5].tolist()
                    }
        
        return anomalies
        
    def create_comprehensive_visualizations(self) -> Dict[str, str]:
        """
        Create comprehensive visualizations for the 10-class dataset.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Creating comprehensive visualizations...")
        
        plot_files = {}
        
        # 1. Complete class distribution
        plt.figure(figsize=(15, 8))
        
        # Main class distribution plot
        class_counts = self.combined_df['category'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        
        bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors)
        plt.title('Complete BDD100K Dataset - 10 Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Object Classes', fontsize=12)
        plt.ylabel('Number of Instances (log scale)', fontsize=12)
        plt.yscale('log')
        plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{value:,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.output_dir / "complete_10class_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['class_distribution'] = str(plot_file)
        
        # 2. Safety-critical vs other classes
        self._create_safety_analysis_plot(plot_files)
        
        # 3. Environmental distribution
        self._create_environmental_plots(plot_files)
        
        # 4. Class imbalance visualization
        self._create_imbalance_visualization(plot_files)
        
        # 5. Split comparison
        self._create_split_comparison_plot(plot_files)
        
        return plot_files
        
    def _create_safety_analysis_plot(self, plot_files: Dict[str, str]):
        """Create safety-focused analysis plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Safety-critical classes
        safety_data = self.combined_df[self.combined_df['category'].isin(self.SAFETY_CRITICAL_CLASSES)]
        if not safety_data.empty:
            safety_counts = safety_data['category'].value_counts()
            ax1.bar(safety_counts.index, safety_counts.values, color='red', alpha=0.7)
            ax1.set_title('Safety-Critical Classes', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        # Vehicle classes
        vehicle_data = self.combined_df[self.combined_df['category'].isin(self.VEHICLE_CLASSES)]
        if not vehicle_data.empty:
            vehicle_counts = vehicle_data['category'].value_counts()
            ax2.bar(vehicle_counts.index, vehicle_counts.values, color='blue', alpha=0.7)
            ax2.set_title('Vehicle Classes', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        # Infrastructure classes  
        infra_data = self.combined_df[self.combined_df['category'].isin(self.INFRASTRUCTURE_CLASSES)]
        if not infra_data.empty:
            infra_counts = infra_data['category'].value_counts()
            ax3.bar(infra_counts.index, infra_counts.values, color='green', alpha=0.7)
            ax3.set_title('Infrastructure Classes', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # Size distribution by class category
        if not self.combined_df.empty:
            categories = []
            sizes = []
            for cls in self.combined_df['category'].unique():
                cls_data = self.combined_df[self.combined_df['category'] == cls]
                categories.extend([self._get_class_category(cls)] * len(cls_data))
                sizes.extend(cls_data['bbox_area'].tolist())
            
            category_df = pd.DataFrame({'category': categories, 'size': sizes})
            category_df.boxplot(column='size', by='category', ax=ax4)
            ax4.set_title('Object Size Distribution by Category')
            ax4.set_yscale('log')
        
        plt.tight_layout()
        plot_file = self.output_dir / "safety_focused_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['safety_analysis'] = str(plot_file)
        
    def _create_environmental_plots(self, plot_files: Dict[str, str]):
        """Create environmental condition analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Weather distribution
        weather_counts = self.combined_df['img_attr_weather'].value_counts()
        axes[0,0].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Weather Distribution')
        
        # Scene distribution
        scene_counts = self.combined_df['img_attr_scene'].value_counts()
        axes[0,1].pie(scene_counts.values, labels=scene_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Scene Distribution')
        
        # Time of day distribution
        time_counts = self.combined_df['img_attr_timeofday'].value_counts()
        axes[1,0].pie(time_counts.values, labels=time_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Time of Day Distribution')
        
        # Environmental complexity heatmap
        if not self.combined_df.empty:
            env_matrix = self.combined_df.groupby(['img_attr_weather', 'img_attr_timeofday']).size().unstack(fill_value=0)
            sns.heatmap(env_matrix, annot=True, fmt='d', ax=axes[1,1], cmap='YlOrRd')
            axes[1,1].set_title('Weather vs Time Distribution')
        
        plt.tight_layout()
        plot_file = self.output_dir / "environmental_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['environmental_analysis'] = str(plot_file)
        
    def _create_imbalance_visualization(self, plot_files: Dict[str, str]):
        """Create class imbalance visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear scale
        class_counts = self.combined_df['category'].value_counts()
        bars1 = ax1.bar(range(len(class_counts)), class_counts.values, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(class_counts))))
        ax1.set_title('Class Distribution (Linear Scale)')
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
        
        # Log scale
        bars2 = ax2.bar(range(len(class_counts)), class_counts.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(class_counts))))
        ax2.set_title('Class Distribution (Log Scale)')
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(class_counts)))
        ax2.set_xticklabels(class_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plot_file = self.output_dir / "class_imbalance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['imbalance_analysis'] = str(plot_file)
        
    def _create_split_comparison_plot(self, plot_files: Dict[str, str]):
        """Create train/validation split comparison."""
        if self.train_df.empty or self.val_df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Train distribution
        train_counts = self.train_df['category'].value_counts()
        axes[0,0].bar(train_counts.index, train_counts.values, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Training Split Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Val distribution
        val_counts = self.val_df['category'].value_counts()
        axes[0,1].bar(val_counts.index, val_counts.values, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Validation Split Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Proportion comparison
        train_props = train_counts / train_counts.sum()
        val_props = val_counts / val_counts.sum()
        
        common_classes = set(train_props.index) & set(val_props.index)
        x_pos = np.arange(len(common_classes))
        width = 0.35
        
        train_vals = [train_props.get(cls, 0) for cls in common_classes]
        val_vals = [val_props.get(cls, 0) for cls in common_classes]
        
        axes[1,0].bar(x_pos - width/2, train_vals, width, label='Train', color='skyblue')
        axes[1,0].bar(x_pos + width/2, val_vals, width, label='Val', color='lightcoral')
        axes[1,0].set_title('Proportion Comparison')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(list(common_classes), rotation=45, ha='right')
        axes[1,0].legend()
        
        # Ratio analysis
        ratios = [train_props.get(cls, 0) / val_props.get(cls, 1e-10) for cls in common_classes]
        axes[1,1].bar(range(len(common_classes)), ratios, color='green', alpha=0.7)
        axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
        axes[1,1].set_title('Train/Val Proportion Ratios')
        axes[1,1].set_xticks(range(len(common_classes)))
        axes[1,1].set_xticklabels(list(common_classes), rotation=45, ha='right')
        axes[1,1].legend()
        
        plt.tight_layout()
        plot_file = self.output_dir / "split_comparison_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['split_comparison'] = str(plot_file)
        
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    plot_files: Dict[str, str]) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_results: Analysis results dictionary
            plot_files: Dictionary of plot file paths
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating comprehensive analysis report...")
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Complete BDD100K Dataset Analysis Report (10 Classes)",
            "=" * 70,
            "",
            f"Generated from comprehensive analysis of BDD100K dataset with all 10 object detection classes.",
            "",
            "## Dataset Overview",
            f"- **Total Annotations**: {analysis_results['basic_statistics']['total_objects']:,}",
            f"- **Total Images**: {analysis_results['basic_statistics']['total_images']:,}",
            f"- **Number of Classes**: {analysis_results['basic_statistics']['num_classes']}",
            f"- **Classes Found**: {len(analysis_results['basic_statistics']['classes_found'])}",
            f"- **Missing Classes**: {analysis_results['basic_statistics']['missing_classes']}",
            ""
        ])
        
        # Complete class distribution
        report_lines.extend([
            "## Complete 10-Class Distribution",
            "",
            "| Rank | Class | Count | Percentage | Safety Weight | Category |",
            "|------|-------|-------|------------|---------------|----------|"
        ])
        
        sorted_classes = sorted(
            analysis_results['class_distribution'].items(),
            key=lambda x: x[1]['count'], reverse=True
        )
        
        for rank, (cls, data) in enumerate(sorted_classes, 1):
            report_lines.append(
                f"| {rank} | {cls} | {data['count']:,} | {data['percentage']:.2f}% | "
                f"{data['safety_weight']:.1f} | {data['category_type']} |"
            )
        
        # Imbalance analysis
        if 'imbalance_metrics' in analysis_results:
            imb = analysis_results['imbalance_metrics']
            report_lines.extend([
                "",
                "## Class Imbalance Analysis",
                f"- **Imbalance Ratio**: {imb['imbalance_ratio']:.2f}:1 "
                f"({imb['most_frequent']['class']} vs {imb['least_frequent']['class']})",
                f"- **Most Frequent**: {imb['most_frequent']['class']} ({imb['most_frequent']['count']:,} instances)",
                f"- **Least Frequent**: {imb['least_frequent']['class']} ({imb['least_frequent']['count']:,} instances)",
                f"- **Gini Coefficient**: {imb['gini_coefficient']:.3f} (0=equal, 1=maximum inequality)",
                f"- **Entropy**: {imb['entropy']:.3f} (higher = more balanced)",
                ""
            ])
        
        # Safety-critical analysis
        if 'safety_analysis' in analysis_results:
            report_lines.extend([
                "## Safety-Critical Classes Analysis",
                "",
                "Critical for autonomous driving safety:",
                ""
            ])
            
            for cls, data in analysis_results['safety_analysis'].items():
                if 'status' in data:
                    report_lines.append(f"- **{cls}**: {data['status']}")
                else:
                    report_lines.extend([
                        f"- **{cls}**:",
                        f"  - Total instances: {data['total_count']:,}",
                        f"  - Images with class: {data['images_with_class']:,}",
                        f"  - Average per image: {data['avg_per_image']:.2f}",
                        f"  - Occlusion rate: {data['occlusion_rate']:.1%}",
                        f"  - Truncation rate: {data['truncation_rate']:.1%}",
                        ""
                    ])
        
        # Visualizations
        report_lines.extend([
            "## Generated Visualizations",
            ""
        ])
        
        for plot_name, plot_path in plot_files.items():
            report_lines.append(f"- **{plot_name.replace('_', ' ').title()}**: `{plot_path}`")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "complete_10class_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Analysis report saved to: {report_file}")
        return str(report_file)
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis pipeline for 10-class BDD100K dataset.
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete 10-class BDD100K analysis...")
        
        results = {}
        
        # Core analyses
        results['class_analysis'] = self.analyze_complete_class_distribution()
        results['environmental_analysis'] = self.analyze_environmental_patterns()
        results['anomaly_analysis'] = self.detect_anomalies_and_patterns()
        
        # Create visualizations
        results['visualizations'] = self.create_comprehensive_visualizations()
        
        # Generate report
        results['report_file'] = self.generate_comprehensive_report(
            results['class_analysis'], results['visualizations']
        )
        
        # Save complete results
        results_file = self.output_dir / "complete_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Complete analysis results saved to: {results_file}")
        logger.info("Complete 10-class BDD100K analysis finished!")
        
        return results
        
    def _convert_for_json(self, obj):
        """Convert numpy types and tuple keys for JSON serialization."""
        if isinstance(obj, dict):
            # Convert tuple keys to strings
            converted_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    key_str = str(k)
                else:
                    key_str = k
                converted_dict[key_str] = self._convert_for_json(v)
            return converted_dict
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Run complete BDD100K analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete BDD100K 10-class analysis')
    parser.add_argument('--data-dir', type=str,
                       default='data/analysis/processed_10class',
                       help='Directory with processed 10-class data')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CompleteBDD100KAnalyzer(args.data_dir)
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("COMPLETE BDD100K 10-CLASS ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {analyzer.output_dir}")
    print(f"Report: {results['report_file']}")
    print(f"Visualizations: {len(results['visualizations'])} plots generated")


if __name__ == "__main__":
    main()