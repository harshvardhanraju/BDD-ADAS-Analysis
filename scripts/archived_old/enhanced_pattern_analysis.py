#!/usr/bin/env python3
"""
Enhanced BDD100K Pattern Analysis

Performs deep analysis of patterns, relationships and insights in the 10-class BDD100K dataset
including environmental, temporal, spatial, and safety-critical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class EnhancedBDD100KPatternAnalysis:
    """Enhanced pattern analysis for BDD100K dataset with deep insights."""
    
    def __init__(self, data_dir: str, output_dir: str = "data/analysis/enhanced_patterns"):
        """
        Initialize enhanced pattern analyzer.
        
        Args:
            data_dir: Directory containing processed 10-class data
            output_dir: Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.train_df = pd.read_csv(self.data_dir / "train_annotations_10class.csv")
        self.val_df = pd.read_csv(self.data_dir / "val_annotations_10class.csv")
        self.combined_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        print(f"Loaded {len(self.combined_df)} annotations across {len(self.combined_df['image_name'].unique())} images")
        
        # Class definitions
        self.safety_critical_classes = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
        self.vehicle_classes = ['car', 'truck', 'bus', 'train']
        self.infrastructure_classes = ['traffic light', 'traffic sign']
        
        self.analysis_results = {}

    def analyze_environmental_patterns(self) -> Dict[str, Any]:
        """Analyze how object detection varies by weather and time conditions."""
        print("🌤️ Analyzing environmental patterns...")
        
        results = {}
        
        # Weather distribution by class
        if 'img_attr_weather' in self.combined_df.columns:
            weather_class = pd.crosstab(
                self.combined_df['category'], 
                self.combined_df['img_attr_weather'], 
                normalize='columns'
            ) * 100
            
            results['weather_class_distribution'] = weather_class.to_dict()
            
            # Time of day distribution by class
            if 'img_attr_timeofday' in self.combined_df.columns:
                time_class = pd.crosstab(
                    self.combined_df['category'], 
                    self.combined_df['img_attr_timeofday'], 
                    normalize='columns'
                ) * 100
                
                results['timeofday_class_distribution'] = time_class.to_dict()
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
                
                # Weather patterns
                weather_class.T.plot(kind='bar', ax=ax1, stacked=False)
                ax1.set_title('Class Distribution by Weather Condition', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Weather Condition')
                ax1.set_ylabel('Percentage of Objects')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.tick_params(axis='x', rotation=45)
                
                # Time of day patterns
                time_class.T.plot(kind='bar', ax=ax2, stacked=False)
                ax2.set_title('Class Distribution by Time of Day', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Time of Day')
                ax2.set_ylabel('Percentage of Objects')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'environmental_patterns.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        return results

    def analyze_cooccurrence_patterns(self) -> Dict[str, Any]:
        """Analyze which objects appear together and their spatial relationships."""
        print("🔗 Analyzing class co-occurrence patterns...")
        
        results = {}
        
        # Create co-occurrence matrix
        image_class_matrix = self.combined_df.groupby(['image_name', 'category']).size().unstack(fill_value=0)
        
        # Convert to binary presence/absence
        image_class_binary = (image_class_matrix > 0).astype(int)
        
        # Calculate co-occurrence probabilities
        cooccurrence_matrix = image_class_binary.T.dot(image_class_binary)
        
        # Normalize by total images to get co-occurrence probabilities
        total_images = len(image_class_binary)
        cooccurrence_probs = cooccurrence_matrix / total_images
        
        results['cooccurrence_matrix'] = cooccurrence_probs.to_dict()
        
        # Calculate PMI (Pointwise Mutual Information)
        class_probs = image_class_binary.mean()
        pmi_matrix = pd.DataFrame(index=class_probs.index, columns=class_probs.index)
        
        for class1 in class_probs.index:
            for class2 in class_probs.index:
                if class1 != class2:
                    joint_prob = cooccurrence_probs.loc[class1, class2]
                    individual_prob1 = class_probs[class1]
                    individual_prob2 = class_probs[class2]
                    
                    if joint_prob > 0 and individual_prob1 > 0 and individual_prob2 > 0:
                        pmi = np.log(joint_prob / (individual_prob1 * individual_prob2))
                    else:
                        pmi = 0
                    
                    pmi_matrix.loc[class1, class2] = pmi
                else:
                    pmi_matrix.loc[class1, class2] = 0
        
        results['pmi_matrix'] = pmi_matrix.astype(float).to_dict()
        
        # Visualize co-occurrence patterns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Co-occurrence heatmap
        sns.heatmap(cooccurrence_probs, annot=True, fmt='.3f', ax=ax1, cmap='Blues')
        ax1.set_title('Object Co-occurrence Probabilities', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Class')
        
        # PMI heatmap
        pmi_numeric = pmi_matrix.astype(float)
        sns.heatmap(pmi_numeric, annot=True, fmt='.2f', ax=ax2, cmap='RdBu_r', center=0)
        ax2.set_title('Pointwise Mutual Information (PMI)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Class')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooccurrence_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

    def analyze_safety_critical_patterns(self) -> Dict[str, Any]:
        """Deep analysis of safety-critical classes and their contexts."""
        print("🚨 Analyzing safety-critical patterns...")
        
        results = {}
        
        # Filter safety-critical objects
        safety_df = self.combined_df[self.combined_df['category'].isin(self.safety_critical_classes)]
        
        # Analyze safety-critical object contexts
        safety_contexts = {}
        
        for safety_class in self.safety_critical_classes:
            class_df = safety_df[safety_df['category'] == safety_class]
            
            context = {
                'count': len(class_df),
                'avg_size': class_df['bbox_area'].mean(),
                'size_std': class_df['bbox_area'].std(),
                'position_distribution': {
                    'center_x_mean': class_df['center_x'].mean(),
                    'center_y_mean': class_df['center_y'].mean(),
                    'center_x_std': class_df['center_x'].std(),
                    'center_y_std': class_df['center_y'].std()
                }
            }
            
            # Environmental context
            if 'img_attr_weather' in class_df.columns:
                context['weather_distribution'] = class_df['img_attr_weather'].value_counts(normalize=True).to_dict()
            
            if 'img_attr_timeofday' in class_df.columns:
                context['timeofday_distribution'] = class_df['img_attr_timeofday'].value_counts(normalize=True).to_dict()
            
            safety_contexts[safety_class] = context
        
        results['safety_contexts'] = safety_contexts
        
        # Analyze interactions between safety-critical classes and vehicles
        vehicle_safety_interactions = {}
        
        for safety_class in self.safety_critical_classes:
            interactions = {}
            
            # Find images with both safety class and vehicles
            safety_images = set(self.combined_df[self.combined_df['category'] == safety_class]['image_name'])
            
            for vehicle_class in self.vehicle_classes:
                vehicle_images = set(self.combined_df[self.combined_df['category'] == vehicle_class]['image_name'])
                
                # Co-occurrence
                cooccurrence = len(safety_images.intersection(vehicle_images))
                total_safety = len(safety_images)
                
                interactions[vehicle_class] = {
                    'cooccurrence_count': cooccurrence,
                    'cooccurrence_rate': cooccurrence / total_safety if total_safety > 0 else 0
                }
            
            vehicle_safety_interactions[safety_class] = interactions
        
        results['vehicle_safety_interactions'] = vehicle_safety_interactions
        
        return results

    def analyze_spatial_clustering(self) -> Dict[str, Any]:
        """Analyze spatial clustering patterns of different object classes."""
        print("📍 Analyzing spatial clustering patterns...")
        
        results = {}
        
        # Perform clustering analysis for each class
        clustering_results = {}
        
        for class_name in self.combined_df['category'].unique():
            class_df = self.combined_df[self.combined_df['category'] == class_name]
            
            if len(class_df) < 10:  # Skip classes with too few samples
                continue
            
            # Use center coordinates and size for clustering
            features = class_df[['center_x', 'center_y', 'bbox_area']].copy()
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal number of clusters (max 5)
            n_clusters = min(5, len(class_df) // 50, 5)
            if n_clusters < 2:
                n_clusters = 2
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Analyze cluster characteristics
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data = class_df[cluster_mask]
                
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'avg_x': cluster_data['center_x'].mean(),
                    'avg_y': cluster_data['center_y'].mean(),
                    'avg_area': cluster_data['bbox_area'].mean(),
                    'std_x': cluster_data['center_x'].std(),
                    'std_y': cluster_data['center_y'].std()
                }
            
            clustering_results[class_name] = cluster_stats
        
        results['spatial_clusters'] = clustering_results
        
        return results

    def analyze_scale_distance_relationships(self) -> Dict[str, Any]:
        """Analyze relationship between object scale and position in image."""
        print("📏 Analyzing scale-distance relationships...")
        
        results = {}
        
        # Analyze correlation between object size and vertical position
        # (objects lower in image are typically closer/larger)
        
        correlations = {}
        
        for class_name in self.combined_df['category'].unique():
            class_df = self.combined_df[self.combined_df['category'] == class_name]
            
            if len(class_df) < 20:  # Skip classes with insufficient data
                continue
            
            # Calculate correlations
            corr_area_y, p_area_y = spearmanr(class_df['bbox_area'], class_df['center_y'])
            corr_width_y, p_width_y = spearmanr(class_df['bbox_width'], class_df['center_y'])
            corr_height_y, p_height_y = spearmanr(class_df['bbox_height'], class_df['center_y'])
            
            correlations[class_name] = {
                'area_vs_y_position': {'correlation': corr_area_y, 'p_value': p_area_y},
                'width_vs_y_position': {'correlation': corr_width_y, 'p_value': p_width_y},
                'height_vs_y_position': {'correlation': corr_height_y, 'p_value': p_height_y}
            }
        
        results['scale_position_correlations'] = correlations
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Plot correlations for major classes
        major_classes = ['car', 'pedestrian', 'truck', 'traffic sign']
        
        for idx, class_name in enumerate(major_classes):
            if class_name in correlations:
                class_df = self.combined_df[self.combined_df['category'] == class_name]
                
                # Sample data for visualization (max 1000 points)
                if len(class_df) > 1000:
                    class_df = class_df.sample(1000, random_state=42)
                
                axes[idx].scatter(class_df['center_y'], class_df['bbox_area'], alpha=0.5)
                axes[idx].set_xlabel('Vertical Position (pixels)')
                axes[idx].set_ylabel('Bounding Box Area (pixels²)')
                axes[idx].set_title(f'{class_name.title()} - Size vs Position')
                
                # Add correlation info
                corr = correlations[class_name]['area_vs_y_position']['correlation']
                axes[idx].text(0.05, 0.95, f'ρ = {corr:.3f}', 
                              transform=axes[idx].transAxes, fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scale_position_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights combining all analyses."""
        print("💡 Generating comprehensive insights...")
        
        insights = {}
        
        # Key findings
        total_objects = len(self.combined_df)
        total_images = self.combined_df['image_name'].nunique()
        
        insights['dataset_overview'] = {
            'total_objects': total_objects,
            'total_images': total_images,
            'avg_objects_per_image': total_objects / total_images,
            'num_classes': len(self.combined_df['category'].unique())
        }
        
        # Class imbalance insights
        class_counts = self.combined_df['category'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        insights['class_balance'] = {
            'most_frequent_class': class_counts.index[0],
            'least_frequent_class': class_counts.index[-1],
            'imbalance_ratio': max_count / min_count,
            'gini_coefficient': self._calculate_gini_coefficient(class_counts.values)
        }
        
        # Safety analysis
        safety_count = self.combined_df[
            self.combined_df['category'].isin(self.safety_critical_classes)
        ]['category'].count()
        
        insights['safety_analysis'] = {
            'safety_critical_percentage': (safety_count / total_objects) * 100,
            'safety_critical_classes': self.safety_critical_classes,
            'safety_critical_count': safety_count
        }
        
        return insights

    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for class distribution."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1])) - 1 - (1/n)

    def run_complete_analysis(self):
        """Run complete enhanced pattern analysis."""
        print("🚀 Starting Enhanced BDD100K Pattern Analysis...")
        print("=" * 80)
        
        # Run all analyses
        self.analysis_results['environmental'] = self.analyze_environmental_patterns()
        self.analysis_results['cooccurrence'] = self.analyze_cooccurrence_patterns()
        self.analysis_results['safety_critical'] = self.analyze_safety_critical_patterns()
        self.analysis_results['spatial_clustering'] = self.analyze_spatial_clustering()
        self.analysis_results['scale_distance'] = self.analyze_scale_distance_relationships()
        self.analysis_results['comprehensive_insights'] = self.generate_comprehensive_insights()
        
        # Save results
        results_file = self.output_dir / 'enhanced_pattern_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"\n✅ Enhanced pattern analysis complete!")
        print(f"📊 Results saved to: {self.output_dir}")
        print(f"📈 Visualizations and detailed analysis available")
        print("=" * 80)
        
        return self.analysis_results


def main():
    """Run enhanced pattern analysis."""
    analyzer = EnhancedBDD100KPatternAnalysis(
        data_dir="data/analysis/processed_10class_corrected",
        output_dir="data/analysis/enhanced_patterns"
    )
    
    results = analyzer.run_complete_analysis()
    
    print("🎉 Enhanced BDD100K Pattern Analysis Complete!")
    return results


if __name__ == "__main__":
    main()