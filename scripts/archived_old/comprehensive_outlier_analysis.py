#!/usr/bin/env python3
"""
Comprehensive Outlier Analysis for Complete BDD100K 10-Class Dataset

This script performs thorough outlier detection and analysis for all 10 BDD100K classes,
identifying anomalous patterns, extreme values, and data quality issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Comprehensive10ClassOutlierAnalyzer:
    """Comprehensive outlier analyzer for all 10 BDD100K classes."""
    
    def __init__(self, data_dir: str, output_dir: str = "data/analysis/outliers_10class"):
        """Initialize the outlier analyzer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load complete 10-class data
        self.train_df = pd.read_csv(self.data_dir / "train_annotations_10class.csv")
        self.val_df = pd.read_csv(self.data_dir / "val_annotations_10class.csv")
        self.combined_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        print(f"Loaded complete 10-class dataset:")
        print(f"  Training: {len(self.train_df):,} annotations")
        print(f"  Validation: {len(self.val_df):,} annotations")
        print(f"  Combined: {len(self.combined_df):,} annotations")
        print(f"  Classes: {sorted(self.combined_df['category'].unique())}")
        
        # Initialize results storage
        self.outlier_results = {}
        
    def detect_size_outliers(self) -> Dict[str, Any]:
        """Detect bounding box size outliers for each class."""
        print("🔍 Detecting size outliers...")
        
        size_outliers = {}
        extreme_outliers = []
        
        for class_name in self.combined_df['category'].unique():
            class_data = self.combined_df[self.combined_df['category'] == class_name].copy()
            
            # Calculate IQR-based outliers for area
            Q1 = class_data['bbox_area'].quantile(0.25)
            Q3 = class_data['bbox_area'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries (using 1.5 * IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            area_outliers = class_data[
                (class_data['bbox_area'] < lower_bound) | 
                (class_data['bbox_area'] > upper_bound)
            ].copy()
            
            # Z-score based outliers (for extreme cases)
            z_scores = np.abs(stats.zscore(class_data['bbox_area']))
            zscore_outliers = class_data[z_scores > 3].copy()
            
            # Aspect ratio outliers
            ar_outliers = class_data[
                (class_data['bbox_aspect_ratio'] > 10) | 
                (class_data['bbox_aspect_ratio'] < 0.1)
            ].copy()
            
            size_outliers[class_name] = {
                'iqr_outliers': len(area_outliers),
                'zscore_outliers': len(zscore_outliers),
                'aspect_ratio_outliers': len(ar_outliers),
                'iqr_percentage': len(area_outliers) / len(class_data) * 100,
                'mean_area': class_data['bbox_area'].mean(),
                'std_area': class_data['bbox_area'].std(),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            # Store extreme outliers for visualization
            if len(area_outliers) > 0:
                extreme_cases = area_outliers.nlargest(5, 'bbox_area')
                for _, row in extreme_cases.iterrows():
                    extreme_outliers.append({
                        'image_name': row['image_name'],
                        'split': row['split'],
                        'category': class_name,
                        'area': row['bbox_area'],
                        'width': row['bbox_width'],
                        'height': row['bbox_height'],
                        'aspect_ratio': row['bbox_aspect_ratio'],
                        'outlier_type': 'size_extreme'
                    })
        
        # Save results
        results = {
            'size_outliers_by_class': size_outliers,
            'extreme_cases': extreme_outliers
        }
        
        # Create detailed report
        self._create_size_outlier_report(results)
        
        return results
    
    def detect_spatial_outliers(self) -> Dict[str, Any]:
        """Detect spatial distribution outliers for each class."""
        print("🗺️ Detecting spatial outliers...")
        
        spatial_outliers = {}
        edge_cases = []
        
        for class_name in self.combined_df['category'].unique():
            class_data = self.combined_df[self.combined_df['category'] == class_name].copy()
            
            # Calculate distances to edges
            class_data['edge_distance'] = np.minimum.reduce([
                class_data['center_x'],  # Distance to left
                1 - class_data['center_x'],  # Distance to right
                class_data['center_y'],  # Distance to bottom
                1 - class_data['center_y']  # Distance to top
            ])
            
            # Find objects very close to edges (< 0.05 normalized distance)
            near_edge = class_data[class_data['edge_distance'] < 0.05].copy()
            
            # Find objects in unusual positions using 2D clustering
            features = class_data[['center_x', 'center_y']].values
            if len(features) > 10:  # Need enough samples for isolation forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                outlier_labels = iso_forest.fit_predict(features)
                position_outliers = class_data[outlier_labels == -1].copy()
            else:
                position_outliers = pd.DataFrame()
            
            spatial_outliers[class_name] = {
                'near_edge_count': len(near_edge),
                'position_outliers_count': len(position_outliers),
                'near_edge_percentage': len(near_edge) / len(class_data) * 100,
                'mean_edge_distance': class_data['edge_distance'].mean(),
                'std_edge_distance': class_data['edge_distance'].std()
            }
            
            # Store edge cases for visualization
            for _, row in near_edge.head(10).iterrows():
                edge_cases.append({
                    'image_name': row['image_name'],
                    'split': row['split'],
                    'category': class_name,
                    'center_x': row['center_x'],
                    'center_y': row['center_y'],
                    'edge_distance': row['edge_distance'],
                    'outlier_type': 'near_edge'
                })
        
        results = {
            'spatial_outliers_by_class': spatial_outliers,
            'edge_cases': edge_cases
        }
        
        # Create detailed report
        self._create_spatial_outlier_report(results)
        
        return results
    
    def detect_co_occurrence_outliers(self) -> Dict[str, Any]:
        """Detect unusual class co-occurrence patterns."""
        print("🔗 Detecting co-occurrence outliers...")
        
        # Group by image to analyze co-occurrences
        image_groups = self.combined_df.groupby('image_name')
        
        unusual_combinations = []
        class_counts = self.combined_df['category'].value_counts()
        
        for image_name, group in image_groups:
            classes_in_image = group['category'].unique()
            class_count = len(classes_in_image)
            
            # Find images with unusual number of classes
            if class_count > 8:  # Images with too many different classes
                unusual_combinations.append({
                    'image_name': image_name,
                    'split': group['split'].iloc[0],
                    'class_count': class_count,
                    'classes': list(classes_in_image),
                    'total_objects': len(group),
                    'outlier_type': 'too_many_classes'
                })
            
            # Find rare class combinations
            safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
            safety_count = sum(1 for cls in classes_in_image if cls in safety_critical)
            
            if safety_count >= 3:  # Unusual concentration of safety-critical classes
                unusual_combinations.append({
                    'image_name': image_name,
                    'split': group['split'].iloc[0],
                    'safety_critical_count': safety_count,
                    'safety_classes': [cls for cls in classes_in_image if cls in safety_critical],
                    'outlier_type': 'safety_critical_concentration'
                })
        
        # Analyze object count outliers per image
        objects_per_image = image_groups.size()
        q99 = objects_per_image.quantile(0.99)
        
        crowded_images = []
        for image_name, count in objects_per_image[objects_per_image > q99].items():
            group = image_groups.get_group(image_name)
            crowded_images.append({
                'image_name': image_name,
                'split': group['split'].iloc[0],
                'object_count': count,
                'class_distribution': group['category'].value_counts().to_dict(),
                'outlier_type': 'crowded_scene'
            })
        
        results = {
            'unusual_combinations': unusual_combinations,
            'crowded_images': crowded_images,
            'statistics': {
                'max_classes_per_image': image_groups.apply(lambda x: x['category'].nunique()).max(),
                'max_objects_per_image': objects_per_image.max(),
                'mean_objects_per_image': objects_per_image.mean(),
                'images_with_rare_classes': len([
                    img for img, group in image_groups 
                    if any(cls in ['train', 'motorcycle', 'bicycle'] for cls in group['category'].unique())
                ])
            }
        }
        
        self._create_cooccurrence_outlier_report(results)
        
        return results
    
    def detect_environmental_outliers(self) -> Dict[str, Any]:
        """Detect unusual environmental condition patterns."""
        print("🌤️ Detecting environmental outliers...")
        
        environmental_outliers = {}
        
        # Check if environmental attributes exist
        env_columns = ['img_attr_weather', 'img_attr_scene', 'img_attr_timeofday']
        available_env = [col for col in env_columns if col in self.combined_df.columns]
        
        if not available_env:
            print("⚠️ No environmental attributes found in dataset")
            return {'message': 'No environmental attributes available'}
        
        unusual_env_combinations = []
        
        for attr in available_env:
            # Find rare environmental conditions
            attr_counts = self.combined_df[attr].value_counts()
            rare_conditions = attr_counts[attr_counts < attr_counts.quantile(0.05)]
            
            environmental_outliers[attr] = {
                'total_unique_values': len(attr_counts),
                'rare_conditions': rare_conditions.to_dict(),
                'most_common': attr_counts.index[0] if len(attr_counts) > 0 else None,
                'least_common': attr_counts.index[-1] if len(attr_counts) > 0 else None
            }
            
            # Find images with rare conditions
            for condition in rare_conditions.index:
                rare_images = self.combined_df[self.combined_df[attr] == condition]
                for _, row in rare_images.head(5).iterrows():
                    unusual_env_combinations.append({
                        'image_name': row['image_name'],
                        'split': row['split'],
                        'attribute': attr,
                        'value': condition,
                        'count': rare_conditions[condition],
                        'outlier_type': 'rare_environmental_condition'
                    })
        
        results = {
            'environmental_outliers_by_attribute': environmental_outliers,
            'unusual_environmental_cases': unusual_env_combinations
        }
        
        self._create_environmental_outlier_report(results)
        
        return results
    
    def detect_class_specific_outliers(self) -> Dict[str, Any]:
        """Detect class-specific outliers based on domain knowledge."""
        print("🎯 Detecting class-specific outliers...")
        
        class_specific_outliers = {}
        
        # Define class-specific rules
        class_rules = {
            'pedestrian': {
                'min_height': 20,  # Very small pedestrians are suspicious
                'max_width': 200,  # Very wide pedestrians are suspicious
                'typical_aspect_ratio_range': (0.3, 3.0)
            },
            'car': {
                'min_area': 100,  # Very small cars
                'max_area': 200000,  # Extremely large cars
                'typical_aspect_ratio_range': (0.5, 4.0)
            },
            'truck': {
                'min_area': 500,  # Very small trucks
                'typical_aspect_ratio_range': (0.5, 5.0)
            },
            'bus': {
                'min_area': 800,  # Very small buses
                'typical_aspect_ratio_range': (0.3, 6.0)
            },
            'train': {
                'min_area': 1000,  # Very small trains
                'typical_aspect_ratio_range': (0.2, 10.0)  # Trains can be very long
            },
            'rider': {
                'min_height': 15,  # Very small riders
                'typical_aspect_ratio_range': (0.3, 3.0)
            },
            'motorcycle': {
                'min_area': 50,  # Very small motorcycles
                'typical_aspect_ratio_range': (0.5, 4.0)
            },
            'bicycle': {
                'min_area': 30,  # Very small bicycles
                'typical_aspect_ratio_range': (0.5, 4.0)
            },
            'traffic_light': {
                'max_width': 100,  # Very wide traffic lights
                'typical_aspect_ratio_range': (0.2, 3.0)
            },
            'traffic_sign': {
                'min_area': 20,  # Very small signs
                'max_area': 10000,  # Very large signs
                'typical_aspect_ratio_range': (0.3, 4.0)
            }
        }
        
        for class_name in self.combined_df['category'].unique():
            if class_name not in class_rules:
                continue
                
            class_data = self.combined_df[self.combined_df['category'] == class_name].copy()
            rules = class_rules[class_name]
            outliers = []
            
            # Apply class-specific rules
            for rule_name, threshold in rules.items():
                if rule_name == 'min_height' and 'bbox_height' in class_data.columns:
                    violations = class_data[class_data['bbox_height'] < threshold]
                elif rule_name == 'max_width' and 'bbox_width' in class_data.columns:
                    violations = class_data[class_data['bbox_width'] > threshold]
                elif rule_name == 'min_area':
                    violations = class_data[class_data['bbox_area'] < threshold]
                elif rule_name == 'max_area':
                    violations = class_data[class_data['bbox_area'] > threshold]
                elif rule_name == 'typical_aspect_ratio_range':
                    min_ar, max_ar = threshold
                    violations = class_data[
                        (class_data['bbox_aspect_ratio'] < min_ar) | 
                        (class_data['bbox_aspect_ratio'] > max_ar)
                    ]
                else:
                    continue
                
                for _, row in violations.iterrows():
                    outliers.append({
                        'image_name': row['image_name'],
                        'split': row['split'],
                        'rule_violated': rule_name,
                        'value': row.get(rule_name.split('_')[1] if '_' in rule_name else 'bbox_area'),
                        'threshold': threshold,
                        'bbox_info': {
                            'area': row['bbox_area'],
                            'width': row['bbox_width'],
                            'height': row['bbox_height'],
                            'aspect_ratio': row['bbox_aspect_ratio']
                        }
                    })
            
            class_specific_outliers[class_name] = {
                'total_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(class_data) * 100 if len(class_data) > 0 else 0,
                'outliers': outliers[:20]  # Keep top 20 for storage
            }
        
        self._create_class_specific_outlier_report(class_specific_outliers)
        
        return class_specific_outliers
    
    def _create_size_outlier_report(self, results: Dict[str, Any]):
        """Create detailed size outlier report."""
        report_path = self.output_dir / "size_outliers_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Size Outliers Analysis Report (10 Classes)\n\n")
            f.write("## Overview\n")
            f.write("Analysis of bounding box size outliers across all 10 BDD100K classes.\n\n")
            
            f.write("## Outliers by Class\n\n")
            f.write("| Class | IQR Outliers | Z-score Outliers | AR Outliers | IQR % | Mean Area | Std Area |\n")
            f.write("|-------|--------------|------------------|-------------|-------|-----------|----------|\n")
            
            for class_name, stats in results['size_outliers_by_class'].items():
                f.write(f"| {class_name} | {stats['iqr_outliers']} | {stats['zscore_outliers']} | "
                       f"{stats['aspect_ratio_outliers']} | {stats['iqr_percentage']:.2f}% | "
                       f"{stats['mean_area']:.0f} | {stats['std_area']:.0f} |\n")
            
            f.write("\n## Extreme Cases\n\n")
            for case in results['extreme_cases'][:10]:
                f.write(f"- **{case['image_name']}** ({case['split']}): {case['category']}\n")
                f.write(f"  - Area: {case['area']:,.0f} pixels²\n")
                f.write(f"  - Dimensions: {case['width']:.0f} x {case['height']:.0f}\n")
                f.write(f"  - Aspect Ratio: {case['aspect_ratio']:.2f}\n\n")
    
    def _create_spatial_outlier_report(self, results: Dict[str, Any]):
        """Create spatial outlier report."""
        report_path = self.output_dir / "spatial_outliers_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Spatial Outliers Analysis Report (10 Classes)\n\n")
            f.write("Analysis of spatial distribution outliers.\n\n")
            
            f.write("## Near-Edge Objects by Class\n\n")
            f.write("| Class | Near Edge | Position Outliers | Near Edge % | Mean Distance |\n")
            f.write("|-------|-----------|-------------------|-------------|---------------|\n")
            
            for class_name, stats in results['spatial_outliers_by_class'].items():
                f.write(f"| {class_name} | {stats['near_edge_count']} | {stats['position_outliers_count']} | "
                       f"{stats['near_edge_percentage']:.2f}% | {stats['mean_edge_distance']:.3f} |\n")
    
    def _create_cooccurrence_outlier_report(self, results: Dict[str, Any]):
        """Create co-occurrence outlier report."""
        report_path = self.output_dir / "cooccurrence_outliers_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Co-occurrence Outliers Analysis Report (10 Classes)\n\n")
            f.write("Analysis of unusual class co-occurrence patterns.\n\n")
            
            f.write("## Statistics\n\n")
            stats = results['statistics']
            f.write(f"- Max classes per image: {stats['max_classes_per_image']}\n")
            f.write(f"- Max objects per image: {stats['max_objects_per_image']}\n")
            f.write(f"- Mean objects per image: {stats['mean_objects_per_image']:.1f}\n")
            f.write(f"- Images with rare classes: {stats['images_with_rare_classes']}\n\n")
            
            f.write("## Unusual Combinations\n\n")
            for combo in results['unusual_combinations'][:10]:
                f.write(f"- **{combo['image_name']}** ({combo['split']})\n")
                if combo['outlier_type'] == 'too_many_classes':
                    f.write(f"  - Classes ({combo['class_count']}): {', '.join(combo['classes'])}\n")
                elif combo['outlier_type'] == 'safety_critical_concentration':
                    f.write(f"  - Safety-critical classes ({combo['safety_critical_count']}): {', '.join(combo['safety_classes'])}\n")
    
    def _create_environmental_outlier_report(self, results: Dict[str, Any]):
        """Create environmental outlier report."""
        report_path = self.output_dir / "environmental_outliers_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Environmental Outliers Analysis Report (10 Classes)\n\n")
            
            if 'message' in results:
                f.write(f"**{results['message']}**\n")
                return
            
            f.write("Analysis of unusual environmental conditions.\n\n")
            
            for attr, stats in results['environmental_outliers_by_attribute'].items():
                f.write(f"## {attr}\n\n")
                f.write(f"- Unique values: {stats['total_unique_values']}\n")
                f.write(f"- Most common: {stats['most_common']}\n")
                f.write(f"- Least common: {stats['least_common']}\n")
                f.write(f"- Rare conditions: {list(stats['rare_conditions'].keys())}\n\n")
    
    def _create_class_specific_outlier_report(self, results: Dict[str, Any]):
        """Create class-specific outlier report."""
        report_path = self.output_dir / "class_specific_outliers_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Class-Specific Outliers Analysis Report (10 Classes)\n\n")
            f.write("Analysis based on domain knowledge and class-specific rules.\n\n")
            
            f.write("| Class | Total Outliers | Percentage |\n")
            f.write("|-------|---------------|-----------|\n")
            
            for class_name, stats in results.items():
                f.write(f"| {class_name} | {stats['total_outliers']} | {stats['outlier_percentage']:.2f}% |\n")
    
    def create_outlier_visualizations(self):
        """Create comprehensive outlier visualizations."""
        print("📊 Creating outlier visualizations...")
        
        # Size outlier visualization
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Area distribution by class (log scale)
        plt.subplot(2, 3, 1)
        class_order = self.combined_df['category'].value_counts().index
        areas_by_class = [self.combined_df[self.combined_df['category'] == cls]['bbox_area'].values 
                         for cls in class_order]
        
        plt.boxplot(areas_by_class, labels=class_order)
        plt.yscale('log')
        plt.title('Bounding Box Area Distribution (Log Scale)')
        plt.xlabel('Class')
        plt.ylabel('Area (pixels²)')
        plt.xticks(rotation=45, ha='right')
        
        # Subplot 2: Aspect ratio outliers
        plt.subplot(2, 3, 2)
        for i, class_name in enumerate(class_order[:5]):  # Top 5 classes
            class_data = self.combined_df[self.combined_df['category'] == class_name]
            plt.hist(class_data['bbox_aspect_ratio'], bins=50, alpha=0.6, 
                    label=class_name, density=True)
        
        plt.title('Aspect Ratio Distributions')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(0, 5)
        
        # Subplot 3: Spatial outliers
        plt.subplot(2, 3, 3)
        safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
        
        for class_name in safety_critical:
            if class_name in self.combined_df['category'].unique():
                class_data = self.combined_df[self.combined_df['category'] == class_name].sample(
                    min(500, len(self.combined_df[self.combined_df['category'] == class_name])))
                plt.scatter(class_data['center_x'], class_data['center_y'], 
                           alpha=0.6, s=10, label=class_name)
        
        plt.title('Safety-Critical Classes Spatial Distribution')
        plt.xlabel('Center X')
        plt.ylabel('Center Y')
        plt.legend()
        
        # Subplot 4: Objects per image distribution
        plt.subplot(2, 3, 4)
        objects_per_image = self.combined_df.groupby('image_name').size()
        plt.hist(objects_per_image, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(objects_per_image.quantile(0.99), color='red', linestyle='--', 
                   label=f'99th percentile: {objects_per_image.quantile(0.99):.0f}')
        plt.title('Objects per Image Distribution')
        plt.xlabel('Number of Objects')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Subplot 5: Class co-occurrence matrix
        plt.subplot(2, 3, 5)
        classes = sorted(self.combined_df['category'].unique())
        co_occurrence = pd.DataFrame(0, index=classes, columns=classes)
        
        for image_name in self.combined_df['image_name'].unique():
            image_data = self.combined_df[self.combined_df['image_name'] == image_name]
            image_classes = image_data['category'].unique()
            
            for i, class1 in enumerate(image_classes):
                for j, class2 in enumerate(image_classes):
                    if i <= j:
                        co_occurrence.loc[class1, class2] += 1
                        if i != j:
                            co_occurrence.loc[class2, class1] += 1
        
        # Normalize by diagonal (self co-occurrence)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and co_occurrence.iloc[i, i] > 0:
                    co_occurrence.iloc[i, j] = co_occurrence.iloc[i, j] / co_occurrence.iloc[i, i]
        
        sns.heatmap(co_occurrence, annot=True, fmt='.2f', cmap='viridis', square=True)
        plt.title('Class Co-occurrence (Normalized)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Subplot 6: Edge proximity analysis
        plt.subplot(2, 3, 6)
        edge_distances = []
        class_names = []
        
        for class_name in class_order[:7]:  # Top 7 classes
            class_data = self.combined_df[self.combined_df['category'] == class_name]
            
            edge_dist = np.minimum.reduce([
                class_data['center_x'],
                1 - class_data['center_x'],
                class_data['center_y'],
                1 - class_data['center_y']
            ])
            
            edge_distances.extend(edge_dist.tolist())
            class_names.extend([class_name] * len(edge_dist))
        
        edge_df = pd.DataFrame({'class': class_names, 'edge_distance': edge_distances})
        sns.boxplot(data=edge_df, x='class', y='edge_distance')
        plt.title('Distance to Image Edge by Class')
        plt.xlabel('Class')
        plt.ylabel('Min Distance to Edge')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_outlier_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Outlier visualizations saved to {self.output_dir}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete outlier analysis for all 10 classes."""
        print("🚀 Running Complete 10-Class Outlier Analysis")
        print("=" * 60)
        
        results = {}
        
        # Run all outlier analyses
        results['size_outliers'] = self.detect_size_outliers()
        results['spatial_outliers'] = self.detect_spatial_outliers()
        results['cooccurrence_outliers'] = self.detect_co_occurrence_outliers()
        results['environmental_outliers'] = self.detect_environmental_outliers()
        results['class_specific_outliers'] = self.detect_class_specific_outliers()
        
        # Create visualizations
        self.create_outlier_visualizations()
        
        # Save complete results
        results_file = self.output_dir / "complete_outlier_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Create master report
        self._create_master_report(results)
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE 10-CLASS OUTLIER ANALYSIS FINISHED!")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print("📊 Generated comprehensive outlier visualizations")
        print("📝 Generated detailed analysis reports")
        print("🎯 Analyzed all 10 BDD100K classes for anomalies")
        
        return results
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _create_master_report(self, results: Dict[str, Any]):
        """Create master outlier analysis report."""
        report_path = self.output_dir / "master_outlier_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Complete BDD100K 10-Class Outlier Analysis Report\n\n")
            f.write("Comprehensive outlier detection and analysis across all 10 BDD100K object detection classes.\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Dataset**: {len(self.combined_df):,} annotations across {self.combined_df['image_name'].nunique():,} images\n")
            f.write(f"- **Classes Analyzed**: {len(self.combined_df['category'].unique())}\n")
            f.write(f"- **Analysis Types**: 5 comprehensive outlier detection methods\n\n")
            
            # Size outliers summary
            if 'size_outliers' in results:
                total_size_outliers = sum(
                    stats['iqr_outliers'] for stats in results['size_outliers']['size_outliers_by_class'].values()
                )
                f.write(f"### Size Outliers\n")
                f.write(f"- Total IQR-based size outliers: {total_size_outliers:,}\n")
                f.write(f"- Extreme cases identified: {len(results['size_outliers']['extreme_cases'])}\n\n")
            
            # Spatial outliers summary
            if 'spatial_outliers' in results:
                total_spatial_outliers = sum(
                    stats['near_edge_count'] for stats in results['spatial_outliers']['spatial_outliers_by_class'].values()
                )
                f.write(f"### Spatial Outliers\n")
                f.write(f"- Objects near image edges: {total_spatial_outliers:,}\n")
                f.write(f"- Edge cases identified: {len(results['spatial_outliers']['edge_cases'])}\n\n")
            
            # Co-occurrence outliers summary
            if 'cooccurrence_outliers' in results:
                f.write(f"### Co-occurrence Outliers\n")
                f.write(f"- Unusual class combinations: {len(results['cooccurrence_outliers']['unusual_combinations'])}\n")
                f.write(f"- Crowded scenes: {len(results['cooccurrence_outliers']['crowded_images'])}\n\n")
            
            f.write("## Detailed Analysis\n\n")
            f.write("See individual reports for detailed findings:\n")
            f.write("- `size_outliers_report.md`\n")
            f.write("- `spatial_outliers_report.md`\n")
            f.write("- `cooccurrence_outliers_report.md`\n")
            f.write("- `environmental_outliers_report.md`\n")
            f.write("- `class_specific_outliers_report.md`\n\n")


def main():
    """Main function to run complete outlier analysis."""
    data_dir = "data/analysis/processed_10class_corrected"
    output_dir = "data/analysis/outliers_10class"
    
    analyzer = Comprehensive10ClassOutlierAnalyzer(data_dir, output_dir)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()