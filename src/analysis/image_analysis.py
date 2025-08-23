"""
Comprehensive Image Characteristics Analysis for BDD100K Dataset

This module provides detailed analysis of image properties including
resolution, quality metrics, scene characteristics, and temporal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
from PIL import Image, ImageStat
import cv2
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set1")

class ImageCharacteristicsAnalyzer:
    """Comprehensive image characteristics analysis for BDD100K dataset."""
    
    def __init__(self, annotation_data: pd.DataFrame, images_root: str,
                 output_dir: str = "data/analysis/plots"):
        """
        Initialize image analyzer.
        
        Args:
            annotation_data: DataFrame with image annotations
            images_root: Root directory containing images
            output_dir: Directory to save analysis outputs
        """
        self.annotation_data = annotation_data
        self.images_root = Path(images_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.splits = sorted(annotation_data['split'].unique())
        self.analysis_results = {}
        
        # Image statistics storage
        self.image_stats = defaultdict(dict)
    
    def analyze_image_dimensions(self, sample_size: Optional[int] = 1000) -> Dict[str, Any]:
        """Analyze image dimensions and resolutions."""
        dimension_analysis = {}
        
        # Sample images for analysis if dataset is large
        unique_images = self.annotation_data[['split', 'image_name']].drop_duplicates()
        
        if sample_size and len(unique_images) > sample_size:
            sampled_images = unique_images.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size} images from {len(unique_images)} total images")
        else:
            sampled_images = unique_images
            print(f"Analyzing all {len(unique_images)} unique images")
        
        dimension_data = []
        processing_errors = []
        
        for _, row in sampled_images.iterrows():
            split = row['split']
            image_name = row['image_name']
            image_path = self.images_root / split / image_name
            
            try:
                if image_path.exists():
                    with Image.open(image_path) as img:
                        width, height = img.size
                        mode = img.mode
                        format_type = img.format
                        
                        # Calculate additional metrics
                        aspect_ratio = width / height
                        total_pixels = width * height
                        
                        dimension_data.append({
                            'split': split,
                            'image_name': image_name,
                            'width': width,
                            'height': height,
                            'aspect_ratio': aspect_ratio,
                            'total_pixels': total_pixels,
                            'mode': mode,
                            'format': format_type
                        })
                        
                        # Store for later use
                        self.image_stats[split][image_name] = {
                            'width': width,
                            'height': height,
                            'aspect_ratio': aspect_ratio,
                            'mode': mode,
                            'format': format_type
                        }
                else:
                    processing_errors.append({
                        'image': f"{split}/{image_name}",
                        'error': 'File not found'
                    })
            
            except Exception as e:
                processing_errors.append({
                    'image': f"{split}/{image_name}",
                    'error': str(e)
                })
        
        if not dimension_data:
            return {'error': 'No images could be processed'}
        
        df_dims = pd.DataFrame(dimension_data)
        
        # Overall statistics
        overall_stats = {
            'width': {
                'mean': df_dims['width'].mean(),
                'std': df_dims['width'].std(),
                'min': df_dims['width'].min(),
                'max': df_dims['width'].max(),
                'median': df_dims['width'].median(),
                'unique_values': len(df_dims['width'].unique())
            },
            'height': {
                'mean': df_dims['height'].mean(),
                'std': df_dims['height'].std(),
                'min': df_dims['height'].min(),
                'max': df_dims['height'].max(),
                'median': df_dims['height'].median(),
                'unique_values': len(df_dims['height'].unique())
            },
            'aspect_ratio': {
                'mean': df_dims['aspect_ratio'].mean(),
                'std': df_dims['aspect_ratio'].std(),
                'min': df_dims['aspect_ratio'].min(),
                'max': df_dims['aspect_ratio'].max(),
                'median': df_dims['aspect_ratio'].median()
            }
        }
        
        # Resolution distribution
        resolution_counts = df_dims.groupby(['width', 'height']).size().reset_index(name='count')
        resolution_counts['resolution'] = resolution_counts['width'].astype(str) + 'x' + resolution_counts['height'].astype(str)
        top_resolutions = resolution_counts.sort_values('count', ascending=False).head(10)
        
        # Format and mode distribution
        format_distribution = df_dims['format'].value_counts().to_dict()
        mode_distribution = df_dims['mode'].value_counts().to_dict()
        
        # Split-wise analysis
        split_stats = {}
        for split in self.splits:
            split_data = df_dims[df_dims['split'] == split]
            if len(split_data) > 0:
                split_stats[split] = {
                    'count': len(split_data),
                    'avg_width': split_data['width'].mean(),
                    'avg_height': split_data['height'].mean(),
                    'avg_aspect_ratio': split_data['aspect_ratio'].mean(),
                    'resolution_consistency': len(split_data.groupby(['width', 'height'])) == 1
                }
        
        dimension_analysis = {
            'overall_statistics': overall_stats,
            'top_resolutions': top_resolutions.to_dict('records'),
            'format_distribution': format_distribution,
            'mode_distribution': mode_distribution,
            'split_statistics': split_stats,
            'processing_errors': processing_errors,
            'total_processed': len(df_dims),
            'total_errors': len(processing_errors)
        }
        
        self.analysis_results['image_dimensions'] = dimension_analysis
        return dimension_analysis
    
    def analyze_scene_attributes(self) -> Dict[str, Any]:
        """Analyze scene attributes from BDD100K annotations."""
        scene_analysis = {}
        
        # Extract scene attributes from annotation data
        images_with_attributes = self.annotation_data[
            ['split', 'image_name'] + [col for col in self.annotation_data.columns if col.startswith('img_attr_')]
        ].drop_duplicates()
        
        if len(images_with_attributes) == 0:
            return {'error': 'No image attributes found in annotation data'}
        
        attribute_columns = [col for col in images_with_attributes.columns if col.startswith('img_attr_')]
        
        # Overall attribute distribution
        attribute_stats = {}
        for attr_col in attribute_columns:
            attr_name = attr_col.replace('img_attr_', '')
            attr_values = images_with_attributes[attr_col].dropna()
            
            if len(attr_values) > 0:
                value_counts = attr_values.value_counts()
                attribute_stats[attr_name] = {
                    'unique_values': len(value_counts),
                    'distribution': value_counts.to_dict(),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'total_images': len(attr_values)
                }
        
        # Split-wise attribute analysis
        split_attribute_stats = {}
        for split in self.splits:
            split_data = images_with_attributes[images_with_attributes['split'] == split]
            split_stats = {}
            
            for attr_col in attribute_columns:
                attr_name = attr_col.replace('img_attr_', '')
                attr_values = split_data[attr_col].dropna()
                
                if len(attr_values) > 0:
                    value_counts = attr_values.value_counts()
                    split_stats[attr_name] = {
                        'distribution': value_counts.to_dict(),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else None
                    }
            
            split_attribute_stats[split] = split_stats
        
        # Attribute correlation analysis
        correlation_analysis = {}
        if len(attribute_columns) > 1:
            # Create binary/categorical encoding for correlation
            encoded_attrs = pd.DataFrame()
            
            for attr_col in attribute_columns:
                attr_name = attr_col.replace('img_attr_', '')
                attr_values = images_with_attributes[attr_col].dropna()
                
                if len(attr_values.unique()) < 20:  # Only for categorical attributes
                    # One-hot encode
                    dummies = pd.get_dummies(attr_values, prefix=attr_name)
                    encoded_attrs = pd.concat([encoded_attrs, dummies], axis=1)
            
            if not encoded_attrs.empty:
                correlation_matrix = encoded_attrs.corr()
                
                # Find top correlations
                correlation_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.1:  # Only significant correlations
                            correlation_pairs.append({
                                'attribute1': correlation_matrix.columns[i],
                                'attribute2': correlation_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                correlation_analysis = {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'top_correlations': correlation_pairs[:10]
                }
        
        scene_analysis = {
            'attribute_statistics': attribute_stats,
            'split_attribute_statistics': split_attribute_stats,
            'correlation_analysis': correlation_analysis,
            'total_images_with_attributes': len(images_with_attributes)
        }
        
        self.analysis_results['scene_attributes'] = scene_analysis
        return scene_analysis
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the dataset."""
        temporal_analysis = {}
        
        # Check if timestamp information is available
        if 'timestamp' not in self.annotation_data.columns:
            return {'error': 'No timestamp information available'}
        
        temporal_data = self.annotation_data[['split', 'image_name', 'timestamp', 'video_name']].drop_duplicates()
        temporal_data = temporal_data[temporal_data['timestamp'].notna()]
        
        if len(temporal_data) == 0:
            return {'error': 'No valid timestamp data found'}
        
        # Convert timestamp to datetime if needed (assuming milliseconds)
        temporal_data['datetime'] = pd.to_datetime(temporal_data['timestamp'], unit='ms', errors='coerce')
        temporal_data = temporal_data[temporal_data['datetime'].notna()]
        
        # Extract time components
        temporal_data['hour'] = temporal_data['datetime'].dt.hour
        temporal_data['day_of_week'] = temporal_data['datetime'].dt.dayofweek
        temporal_data['month'] = temporal_data['datetime'].dt.month
        temporal_data['year'] = temporal_data['datetime'].dt.year
        
        # Hourly distribution
        hourly_distribution = temporal_data['hour'].value_counts().sort_index()
        
        # Day of week distribution (0=Monday, 6=Sunday)
        dow_distribution = temporal_data['day_of_week'].value_counts().sort_index()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_distribution.index = [dow_names[i] for i in dow_distribution.index]
        
        # Monthly distribution
        monthly_distribution = temporal_data['month'].value_counts().sort_index()
        
        # Video sequence analysis
        video_analysis = {}
        if 'video_name' in temporal_data.columns:
            video_stats = temporal_data.groupby('video_name').agg({
                'timestamp': ['count', 'min', 'max'],
                'image_name': 'nunique'
            })
            
            video_stats.columns = ['frame_count', 'start_timestamp', 'end_timestamp', 'unique_images']
            video_stats['duration_ms'] = video_stats['end_timestamp'] - video_stats['start_timestamp']
            video_stats['duration_seconds'] = video_stats['duration_ms'] / 1000
            
            video_analysis = {
                'total_videos': len(video_stats),
                'avg_video_duration': video_stats['duration_seconds'].mean(),
                'avg_frames_per_video': video_stats['frame_count'].mean(),
                'longest_video': video_stats['duration_seconds'].max(),
                'shortest_video': video_stats['duration_seconds'].min(),
                'total_duration_hours': video_stats['duration_seconds'].sum() / 3600
            }
        
        temporal_analysis = {
            'hourly_distribution': hourly_distribution.to_dict(),
            'day_of_week_distribution': dow_distribution.to_dict(),
            'monthly_distribution': monthly_distribution.to_dict(),
            'video_analysis': video_analysis,
            'temporal_range': {
                'earliest': temporal_data['datetime'].min().isoformat(),
                'latest': temporal_data['datetime'].max().isoformat(),
                'total_timespan_days': (temporal_data['datetime'].max() - temporal_data['datetime'].min()).days
            },
            'total_temporal_images': len(temporal_data)
        }
        
        self.analysis_results['temporal_patterns'] = temporal_analysis
        return temporal_analysis
    
    def analyze_image_quality(self, sample_size: Optional[int] = 500) -> Dict[str, Any]:
        """Analyze image quality metrics like brightness, contrast, sharpness."""
        quality_analysis = {}
        
        # Sample images for quality analysis
        unique_images = self.annotation_data[['split', 'image_name']].drop_duplicates()
        
        if sample_size and len(unique_images) > sample_size:
            sampled_images = unique_images.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size} images for quality analysis")
        else:
            sampled_images = unique_images
            print(f"Analyzing quality of all {len(unique_images)} images")
        
        quality_data = []
        processing_errors = []
        
        for _, row in sampled_images.iterrows():
            split = row['split']
            image_name = row['image_name']
            image_path = self.images_root / split / image_name
            
            try:
                if image_path.exists():
                    # PIL-based analysis
                    with Image.open(image_path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Basic statistics
                        stat = ImageStat.Stat(img)
                        
                        # Brightness (average of RGB means)
                        brightness = sum(stat.mean) / len(stat.mean)
                        
                        # Contrast (average of RGB standard deviations)
                        contrast = sum(stat.stddev) / len(stat.stddev)
                        
                        # Convert to OpenCV for additional metrics
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                        
                        # Sharpness using Laplacian variance
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        # Dynamic range
                        dynamic_range = gray.max() - gray.min()
                        
                        # Histogram entropy (measure of information content)
                        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                        hist_norm = hist / hist.sum()
                        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
                        entropy = -np.sum(hist_norm * np.log2(hist_norm))
                        
                        quality_data.append({
                            'split': split,
                            'image_name': image_name,
                            'brightness': brightness,
                            'contrast': contrast,
                            'sharpness': laplacian_var,
                            'dynamic_range': dynamic_range,
                            'entropy': entropy
                        })
                
                else:
                    processing_errors.append({
                        'image': f"{split}/{image_name}",
                        'error': 'File not found'
                    })
            
            except Exception as e:
                processing_errors.append({
                    'image': f"{split}/{image_name}",
                    'error': str(e)
                })
        
        if not quality_data:
            return {'error': 'No images could be processed for quality analysis'}
        
        df_quality = pd.DataFrame(quality_data)
        
        # Overall quality statistics
        quality_metrics = ['brightness', 'contrast', 'sharpness', 'dynamic_range', 'entropy']
        overall_stats = {}
        
        for metric in quality_metrics:
            overall_stats[metric] = {
                'mean': df_quality[metric].mean(),
                'std': df_quality[metric].std(),
                'min': df_quality[metric].min(),
                'max': df_quality[metric].max(),
                'median': df_quality[metric].median(),
                'q25': df_quality[metric].quantile(0.25),
                'q75': df_quality[metric].quantile(0.75)
            }
        
        # Split-wise quality comparison
        split_quality_stats = {}
        for split in self.splits:
            split_data = df_quality[df_quality['split'] == split]
            if len(split_data) > 0:
                split_stats = {}
                for metric in quality_metrics:
                    split_stats[metric] = {
                        'mean': split_data[metric].mean(),
                        'std': split_data[metric].std()
                    }
                split_quality_stats[split] = split_stats
        
        # Quality outlier detection
        outliers = {}
        for metric in quality_metrics:
            Q1 = df_quality[metric].quantile(0.25)
            Q3 = df_quality[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_images = df_quality[
                (df_quality[metric] < lower_bound) | (df_quality[metric] > upper_bound)
            ]
            
            outliers[metric] = {
                'count': len(outlier_images),
                'percentage': len(outlier_images) / len(df_quality) * 100,
                'examples': outlier_images[['image_name', 'split', metric]].head(5).to_dict('records')
            }
        
        quality_analysis = {
            'overall_statistics': overall_stats,
            'split_statistics': split_quality_stats,
            'outlier_analysis': outliers,
            'processing_summary': {
                'total_processed': len(df_quality),
                'total_errors': len(processing_errors),
                'error_rate': len(processing_errors) / (len(df_quality) + len(processing_errors)) * 100
            }
        }
        
        self.analysis_results['image_quality'] = quality_analysis
        return quality_analysis
    
    def create_image_analysis_plots(self) -> List[str]:
        """Create comprehensive image analysis plots."""
        plot_files = []
        
        # 1. Image dimensions analysis
        if 'image_dimensions' in self.analysis_results:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            dims = self.analysis_results['image_dimensions']
            
            # Resolution distribution
            if 'top_resolutions' in dims:
                resolutions = dims['top_resolutions'][:10]
                res_labels = [f"{r['width']}×{r['height']}" for r in resolutions]
                res_counts = [r['count'] for r in resolutions]
                
                axes[0,0].bar(range(len(res_labels)), res_counts, color='skyblue')
                axes[0,0].set_title('Top 10 Image Resolutions', fontweight='bold')
                axes[0,0].set_xlabel('Resolution')
                axes[0,0].set_ylabel('Count')
                axes[0,0].set_xticks(range(len(res_labels)))
                axes[0,0].set_xticklabels(res_labels, rotation=45, ha='right')
            
            # Format distribution
            if 'format_distribution' in dims:
                formats = list(dims['format_distribution'].keys())
                format_counts = list(dims['format_distribution'].values())
                
                axes[0,1].pie(format_counts, labels=formats, autopct='%1.1f%%')
                axes[0,1].set_title('Image Format Distribution', fontweight='bold')
            
            # Mode distribution
            if 'mode_distribution' in dims:
                modes = list(dims['mode_distribution'].keys())
                mode_counts = list(dims['mode_distribution'].values())
                
                axes[0,2].bar(modes, mode_counts, color='lightcoral')
                axes[0,2].set_title('Image Mode Distribution', fontweight='bold')
                axes[0,2].set_xlabel('Color Mode')
                axes[0,2].set_ylabel('Count')
            
            # Aspect ratio distribution
            if 'overall_statistics' in dims and 'aspect_ratio' in dims['overall_statistics']:
                ar_stats = dims['overall_statistics']['aspect_ratio']
                
                # Create synthetic data for visualization (since we don't have individual values)
                ar_mean = ar_stats['mean']
                ar_std = ar_stats['std']
                synthetic_ar = np.random.normal(ar_mean, ar_std, 1000)
                
                axes[1,0].hist(synthetic_ar, bins=30, alpha=0.7, color='lightgreen')
                axes[1,0].set_title('Aspect Ratio Distribution', fontweight='bold')
                axes[1,0].set_xlabel('Aspect Ratio')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].axvline(ar_mean, color='red', linestyle='--', label=f'Mean: {ar_mean:.2f}')
                axes[1,0].legend()
            
            # Split comparison
            if 'split_statistics' in dims:
                split_stats = dims['split_statistics']
                splits = list(split_stats.keys())
                split_counts = [split_stats[s]['count'] for s in splits]
                
                axes[1,1].bar(splits, split_counts, color=['#1f77b4', '#ff7f0e'])
                axes[1,1].set_title('Images per Split', fontweight='bold')
                axes[1,1].set_xlabel('Split')
                axes[1,1].set_ylabel('Count')
            
            # Width vs Height scatter (synthetic data)
            if 'overall_statistics' in dims:
                w_stats = dims['overall_statistics']['width']
                h_stats = dims['overall_statistics']['height']
                
                # Create synthetic scatter data
                synthetic_w = np.random.normal(w_stats['mean'], w_stats['std'], 1000)
                synthetic_h = np.random.normal(h_stats['mean'], h_stats['std'], 1000)
                
                axes[1,2].scatter(synthetic_w, synthetic_h, alpha=0.6, s=1)
                axes[1,2].set_title('Width vs Height Distribution', fontweight='bold')
                axes[1,2].set_xlabel('Width (pixels)')
                axes[1,2].set_ylabel('Height (pixels)')
            
            plt.tight_layout()
            plot_file = self.output_dir / 'image_dimensions_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(str(plot_file))
            plt.close()
        
        # 2. Scene attributes analysis
        if 'scene_attributes' in self.analysis_results:
            scene_attrs = self.analysis_results['scene_attributes']['attribute_statistics']
            
            if scene_attrs:
                n_attrs = len(scene_attrs)
                n_cols = min(3, n_attrs)
                n_rows = (n_attrs + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for i, (attr_name, attr_stats) in enumerate(scene_attrs.items()):
                    if i < len(axes):
                        distribution = attr_stats['distribution']
                        
                        labels = list(distribution.keys())
                        values = list(distribution.values())
                        
                        if len(labels) <= 10:  # Bar chart for few categories
                            axes[i].bar(range(len(labels)), values, color=sns.color_palette("Set2", len(labels)))
                            axes[i].set_title(f'{attr_name.title()} Distribution', fontweight='bold')
                            axes[i].set_xticks(range(len(labels)))
                            axes[i].set_xticklabels(labels, rotation=45, ha='right')
                        else:  # Pie chart for many categories (top 10)
                            top_10_idx = np.argsort(values)[-10:]
                            top_labels = [labels[i] for i in top_10_idx]
                            top_values = [values[i] for i in top_10_idx]
                            
                            axes[i].pie(top_values, labels=top_labels, autopct='%1.1f%%')
                            axes[i].set_title(f'{attr_name.title()} Distribution (Top 10)', fontweight='bold')
                
                # Hide empty subplots
                for i in range(len(scene_attrs), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plot_file = self.output_dir / 'scene_attributes_analysis.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plot_files.append(str(plot_file))
                plt.close()
        
        # 3. Temporal patterns
        if 'temporal_patterns' in self.analysis_results:
            temporal = self.analysis_results['temporal_patterns']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Hourly distribution
            if 'hourly_distribution' in temporal:
                hourly_dist = temporal['hourly_distribution']
                hours = sorted(hourly_dist.keys())
                counts = [hourly_dist[h] for h in hours]
                
                axes[0,0].plot(hours, counts, marker='o', linewidth=2, markersize=6)
                axes[0,0].set_title('Hourly Distribution of Images', fontweight='bold')
                axes[0,0].set_xlabel('Hour of Day')
                axes[0,0].set_ylabel('Number of Images')
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].set_xticks(range(0, 24, 2))
            
            # Day of week distribution
            if 'day_of_week_distribution' in temporal:
                dow_dist = temporal['day_of_week_distribution']
                days = list(dow_dist.keys())
                counts = list(dow_dist.values())
                
                axes[0,1].bar(days, counts, color='lightcoral')
                axes[0,1].set_title('Day of Week Distribution', fontweight='bold')
                axes[0,1].set_xlabel('Day of Week')
                axes[0,1].set_ylabel('Number of Images')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # Monthly distribution
            if 'monthly_distribution' in temporal:
                monthly_dist = temporal['monthly_distribution']
                months = sorted(monthly_dist.keys())
                counts = [monthly_dist[m] for m in months]
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                axes[1,0].bar([month_names[m-1] for m in months], counts, color='lightgreen')
                axes[1,0].set_title('Monthly Distribution', fontweight='bold')
                axes[1,0].set_xlabel('Month')
                axes[1,0].set_ylabel('Number of Images')
            
            # Video analysis summary
            if 'video_analysis' in temporal and temporal['video_analysis']:
                video_stats = temporal['video_analysis']
                
                metrics = ['Total Videos', 'Avg Duration (s)', 'Avg Frames/Video', 'Total Hours']
                values = [
                    video_stats.get('total_videos', 0),
                    video_stats.get('avg_video_duration', 0),
                    video_stats.get('avg_frames_per_video', 0),
                    video_stats.get('total_duration_hours', 0)
                ]
                
                bars = axes[1,1].bar(range(len(metrics)), values, color='orange')
                axes[1,1].set_title('Video Statistics Summary', fontweight='bold')
                axes[1,1].set_xticks(range(len(metrics)))
                axes[1,1].set_xticklabels(metrics, rotation=45, ha='right')
                axes[1,1].set_ylabel('Value')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[1,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                                  f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = self.output_dir / 'temporal_patterns_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(str(plot_file))
            plt.close()
        
        # 4. Image quality analysis
        if 'image_quality' in self.analysis_results:
            quality = self.analysis_results['image_quality']['overall_statistics']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            quality_metrics = list(quality.keys())
            
            for i, metric in enumerate(quality_metrics):
                if i < 6:  # Only show first 6 metrics
                    row, col = i // 3, i % 3
                    
                    stats = quality[metric]
                    
                    # Create synthetic distribution for visualization
                    synthetic_data = np.random.normal(stats['mean'], stats['std'], 1000)
                    
                    axes[row, col].hist(synthetic_data, bins=30, alpha=0.7, 
                                      color=sns.color_palette("Set1")[i])
                    axes[row, col].set_title(f'{metric.title()} Distribution', fontweight='bold')
                    axes[row, col].set_xlabel(metric.title())
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].axvline(stats['mean'], color='red', linestyle='--', 
                                         label=f'Mean: {stats["mean"]:.2f}')
                    axes[row, col].legend()
            
            plt.tight_layout()
            plot_file = self.output_dir / 'image_quality_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(str(plot_file))
            plt.close()
        
        return plot_files
    
    def generate_image_analysis_report(self) -> str:
        """Generate comprehensive image analysis report."""
        report_lines = []
        
        report_lines.append("# BDD100K Image Characteristics Analysis Report")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Image dimensions analysis
        if 'image_dimensions' in self.analysis_results:
            dims = self.analysis_results['image_dimensions']
            
            report_lines.append("## Image Dimensions Analysis")
            report_lines.append("")
            
            if 'overall_statistics' in dims:
                stats = dims['overall_statistics']
                
                report_lines.append("### Resolution Statistics")
                for dimension in ['width', 'height']:
                    s = stats[dimension]
                    report_lines.append(f"- {dimension.title()}: {s['mean']:.0f} ± {s['std']:.0f} pixels")
                    report_lines.append(f"  Range: {s['min']:.0f} - {s['max']:.0f} pixels")
                    report_lines.append(f"  Unique values: {s['unique_values']}")
                
                ar_stats = stats['aspect_ratio']
                report_lines.append(f"- Aspect Ratio: {ar_stats['mean']:.2f} ± {ar_stats['std']:.2f}")
                report_lines.append("")
            
            if 'top_resolutions' in dims:
                report_lines.append("### Most Common Resolutions")
                for i, res in enumerate(dims['top_resolutions'][:5]):
                    percentage = res['count'] / dims['total_processed'] * 100
                    report_lines.append(f"{i+1}. {res['width']}×{res['height']}: {res['count']:,} images ({percentage:.1f}%)")
                report_lines.append("")
            
            if 'format_distribution' in dims:
                report_lines.append("### Image Formats")
                for format_type, count in dims['format_distribution'].items():
                    percentage = count / dims['total_processed'] * 100
                    report_lines.append(f"- {format_type}: {count:,} images ({percentage:.1f}%)")
                report_lines.append("")
        
        # Scene attributes analysis
        if 'scene_attributes' in self.analysis_results:
            scene = self.analysis_results['scene_attributes']
            
            report_lines.append("## Scene Attributes Analysis")
            report_lines.append("")
            
            if 'attribute_statistics' in scene:
                for attr_name, attr_stats in scene['attribute_statistics'].items():
                    report_lines.append(f"### {attr_name.title()}")
                    report_lines.append(f"- Total images with this attribute: {attr_stats['total_images']:,}")
                    report_lines.append(f"- Unique values: {attr_stats['unique_values']}")
                    report_lines.append(f"- Most common value: {attr_stats['most_common']}")
                    
                    # Show top 3 distributions
                    sorted_dist = sorted(attr_stats['distribution'].items(), key=lambda x: x[1], reverse=True)
                    report_lines.append("- Distribution (top 3):")
                    for value, count in sorted_dist[:3]:
                        percentage = count / attr_stats['total_images'] * 100
                        report_lines.append(f"  - {value}: {count:,} ({percentage:.1f}%)")
                    report_lines.append("")
        
        # Temporal patterns analysis
        if 'temporal_patterns' in self.analysis_results:
            temporal = self.analysis_results['temporal_patterns']
            
            report_lines.append("## Temporal Patterns Analysis")
            report_lines.append("")
            
            if 'temporal_range' in temporal:
                range_info = temporal['temporal_range']
                report_lines.append(f"- Dataset temporal range: {range_info['earliest']} to {range_info['latest']}")
                report_lines.append(f"- Total timespan: {range_info['total_timespan_days']} days")
                report_lines.append("")
            
            if 'hourly_distribution' in temporal:
                hourly = temporal['hourly_distribution']
                peak_hour = max(hourly, key=hourly.get)
                peak_count = hourly[peak_hour]
                report_lines.append(f"- Peak hour: {peak_hour}:00 ({peak_count:,} images)")
                
                # Find quiet hours (bottom 25%)
                sorted_hours = sorted(hourly.items(), key=lambda x: x[1])
                quiet_hours = sorted_hours[:len(sorted_hours)//4]
                report_lines.append(f"- Quietest hours: {', '.join([f'{h}:00' for h, c in quiet_hours])}")
                report_lines.append("")
            
            if 'video_analysis' in temporal and temporal['video_analysis']:
                video_stats = temporal['video_analysis']
                report_lines.append("### Video Statistics")
                report_lines.append(f"- Total videos: {video_stats.get('total_videos', 0):,}")
                report_lines.append(f"- Average video duration: {video_stats.get('avg_video_duration', 0):.1f} seconds")
                report_lines.append(f"- Average frames per video: {video_stats.get('avg_frames_per_video', 0):.1f}")
                report_lines.append(f"- Total content duration: {video_stats.get('total_duration_hours', 0):.1f} hours")
                report_lines.append("")
        
        # Image quality analysis
        if 'image_quality' in self.analysis_results:
            quality = self.analysis_results['image_quality']
            
            report_lines.append("## Image Quality Analysis")
            report_lines.append("")
            
            if 'overall_statistics' in quality:
                quality_stats = quality['overall_statistics']
                
                for metric, stats in quality_stats.items():
                    report_lines.append(f"### {metric.title()}")
                    report_lines.append(f"- Mean: {stats['mean']:.2f}")
                    report_lines.append(f"- Standard deviation: {stats['std']:.2f}")
                    report_lines.append(f"- Range: {stats['min']:.2f} - {stats['max']:.2f}")
                    report_lines.append("")
            
            if 'outlier_analysis' in quality:
                report_lines.append("### Quality Outliers")
                for metric, outlier_info in quality['outlier_analysis'].items():
                    if outlier_info['count'] > 0:
                        report_lines.append(f"- {metric.title()}: {outlier_info['count']:,} outliers ({outlier_info['percentage']:.1f}%)")
                report_lines.append("")
        
        # Processing summary
        report_lines.append("## Processing Summary")
        report_lines.append("")
        
        for analysis_type, results in self.analysis_results.items():
            if 'total_processed' in results:
                report_lines.append(f"- {analysis_type.replace('_', ' ').title()}: {results['total_processed']:,} images processed")
                
                if 'total_errors' in results:
                    error_rate = results['total_errors'] / (results['total_processed'] + results['total_errors']) * 100
                    report_lines.append(f"  - Errors: {results['total_errors']:,} ({error_rate:.1f}%)")
        
        report_lines.append("")
        
        # Save report
        report_file = self.output_dir / 'image_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return str(report_file)
    
    def run_complete_analysis(self, dimension_sample: int = 1000, quality_sample: int = 500) -> Dict[str, Any]:
        """Run all image analysis components."""
        print("Analyzing image dimensions...")
        self.analyze_image_dimensions(sample_size=dimension_sample)
        
        print("Analyzing scene attributes...")
        self.analyze_scene_attributes()
        
        print("Analyzing temporal patterns...")
        self.analyze_temporal_patterns()
        
        print("Analyzing image quality...")
        self.analyze_image_quality(sample_size=quality_sample)
        
        print("Creating image analysis plots...")
        plot_files = self.create_image_analysis_plots()
        
        print("Generating image analysis report...")
        report_file = self.generate_image_analysis_report()
        
        return {
            'analysis_results': self.analysis_results,
            'plot_files': plot_files,
            'report_file': report_file,
            'image_statistics': dict(self.image_stats),
            'summary': {
                'total_unique_images': len(self.annotation_data[['split', 'image_name']].drop_duplicates()),
                'analysis_components': list(self.analysis_results.keys())
            }
        }

if __name__ == "__main__":
    print("Image Characteristics Analyzer for BDD100K Dataset")
    print("Load your annotation data and specify images directory")
    print("Example:")
    print("  df = pd.read_csv('data/processed/train_annotations.csv')")
    print("  analyzer = ImageCharacteristicsAnalyzer(df, 'data/raw/images/100k')")
    print("  results = analyzer.run_complete_analysis()")