#!/usr/bin/env python3
"""
Generate Updated Plots for Complete 10-Class BDD100K Analysis

This script generates comprehensive visualizations for all 10 BDD100K classes,
updating the existing plots folder with new statistics and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_10class_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load complete 10-class train and validation data."""
    train_file = Path(data_dir) / "train_annotations_10class.csv"
    val_file = Path(data_dir) / "val_annotations_10class.csv"
    
    print(f"Loading 10-class data from {data_dir}...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    print(f"Loaded training data: {len(train_df):,} annotations")
    print(f"Loaded validation data: {len(val_df):,} annotations")
    
    return train_df, val_df

def create_class_distribution_plot(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create updated class distribution overview for all 10 classes."""
    
    # Combine data
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Get class counts
    class_counts = combined_df['category'].value_counts()
    
    # Calculate percentages
    total = len(combined_df)
    percentages = (class_counts / total * 100).round(2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BDD100K Complete 10-Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of counts
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(class_counts)), class_counts.values)
    ax1.set_title('Object Counts by Class', fontweight='bold')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # 2. Pie chart
    ax2 = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Distribution (Percentages)', fontweight='bold')
    
    # Improve pie chart readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')
    
    # 3. Log-scale bar plot to show extreme imbalance
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(class_counts)), class_counts.values)
    ax3.set_yscale('log')
    ax3.set_title('Class Counts (Log Scale) - Shows Extreme Imbalance', fontweight='bold')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Count (log scale)')
    ax3.set_xticks(range(len(class_counts)))
    ax3.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Safety-critical vs others comparison
    ax4 = axes[1, 1]
    safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    safety_counts = class_counts[class_counts.index.isin(safety_critical)].sum()
    other_counts = class_counts[~class_counts.index.isin(safety_critical)].sum()
    
    category_counts = pd.Series([safety_counts, other_counts], 
                               index=['Safety Critical', 'Other Classes'])
    
    bars = ax4.bar(category_counts.index, category_counts.values, 
                   color=['red', 'blue'], alpha=0.7)
    ax4.set_title('Safety-Critical vs Other Classes', fontweight='bold')
    ax4.set_ylabel('Total Count')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}\n({height/total*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated class distribution overview saved")
    plt.close()

def create_bbox_analysis_plot(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create detailed bounding box dimension analysis for all 10 classes."""
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BDD100K 10-Class Bounding Box Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot of areas by class
    ax1 = axes[0, 0]
    class_order = combined_df['category'].value_counts().index
    sns.boxplot(data=combined_df, x='category', y='bbox_area', ax=ax1, order=class_order)
    ax1.set_title('Bounding Box Areas by Class', fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Area (pixels²)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yscale('log')  # Log scale due to wide range
    
    # 2. Aspect ratio distribution
    ax2 = axes[0, 1]
    for i, class_name in enumerate(class_order):
        class_data = combined_df[combined_df['category'] == class_name]['bbox_aspect_ratio']
        ax2.hist(class_data, bins=30, alpha=0.6, label=class_name, density=True)
    
    ax2.set_title('Aspect Ratio Distributions by Class', fontweight='bold')
    ax2.set_xlabel('Aspect Ratio (width/height)')
    ax2.set_ylabel('Density')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(0, 5)  # Reasonable range for aspect ratios
    
    # 3. Width vs Height scatter
    ax3 = axes[1, 0]
    safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    
    for class_name in class_order:
        class_data = combined_df[combined_df['category'] == class_name]
        color = 'red' if class_name in safety_critical else 'blue'
        alpha = 0.7 if class_name in safety_critical else 0.3
        ax3.scatter(class_data['bbox_width'], class_data['bbox_height'], 
                   label=class_name, alpha=alpha, s=10)
    
    ax3.set_title('Width vs Height (Safety-Critical in Red)', fontweight='bold')
    ax3.set_xlabel('Width (pixels)')
    ax3.set_ylabel('Height (pixels)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Class size statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_data = []
    for class_name in class_order:
        class_data = combined_df[combined_df['category'] == class_name]
        stats_data.append([
            class_name,
            f"{class_data['bbox_area'].mean():.0f}",
            f"{class_data['bbox_area'].median():.0f}",
            f"{class_data['bbox_width'].mean():.0f}",
            f"{class_data['bbox_height'].mean():.0f}",
            f"{class_data['bbox_aspect_ratio'].mean():.2f}"
        ])
    
    # Create table
    table_data = [['Class', 'Mean Area', 'Median Area', 'Mean W', 'Mean H', 'Mean AR']] + stats_data
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax4.set_title('Bounding Box Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_dimension_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated bounding box analysis saved")
    plt.close()

def create_co_occurrence_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create class co-occurrence heatmap for all 10 classes."""
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Get unique classes
    classes = sorted(combined_df['category'].unique())
    
    # Create co-occurrence matrix
    co_occurrence = pd.DataFrame(0, index=classes, columns=classes)
    
    # Calculate co-occurrence for each image
    for image_name in combined_df['image_name'].unique():
        image_data = combined_df[combined_df['image_name'] == image_name]
        image_classes = image_data['category'].unique()
        
        # Update co-occurrence matrix
        for i, class1 in enumerate(image_classes):
            for j, class2 in enumerate(image_classes):
                if i <= j:  # Avoid double counting
                    co_occurrence.loc[class1, class2] += 1
                    if i != j:  # Make matrix symmetric
                        co_occurrence.loc[class2, class1] += 1
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Convert to correlation-like values (normalized by diagonal)
    normalized_co_occurrence = co_occurrence.copy()
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                diagonal_min = min(co_occurrence.iloc[i, i], co_occurrence.iloc[j, j])
                if diagonal_min > 0:
                    normalized_co_occurrence.iloc[i, j] = co_occurrence.iloc[i, j] / diagonal_min
    
    # Create heatmap
    mask = np.triu(np.ones_like(normalized_co_occurrence, dtype=bool), k=1)
    sns.heatmap(normalized_co_occurrence, mask=mask, annot=True, fmt='.2f', cmap='viridis',
                square=True, cbar_kws={'label': 'Co-occurrence Score'})
    
    plt.title('Class Co-occurrence Analysis (10 Classes)\nNormalized by Individual Class Frequency', 
              fontweight='bold', fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'co_occurrence_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated co-occurrence heatmap saved")
    plt.close()

def create_split_comparison_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create train/validation split comparison for all 10 classes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Train/Validation Split Comparison (10 Classes)', fontsize=16, fontweight='bold')
    
    # 1. Class distribution comparison
    ax1 = axes[0, 0]
    train_counts = train_df['category'].value_counts()
    val_counts = val_df['category'].value_counts()
    
    # Ensure both series have same classes
    all_classes = sorted(set(train_counts.index) | set(val_counts.index))
    train_counts = train_counts.reindex(all_classes, fill_value=0)
    val_counts = val_counts.reindex(all_classes, fill_value=0)
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_counts.values, width, label='Train', alpha=0.8)
    bars2 = ax1.bar(x + width/2, val_counts.values, width, label='Validation', alpha=0.8)
    
    ax1.set_title('Absolute Counts by Split', fontweight='bold')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_classes, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')  # Log scale due to wide range
    
    # 2. Percentage comparison
    ax2 = axes[0, 1]
    train_pct = (train_counts / train_counts.sum() * 100)
    val_pct = (val_counts / val_counts.sum() * 100)
    
    bars1 = ax2.bar(x - width/2, train_pct.values, width, label='Train', alpha=0.8)
    bars2 = ax2.bar(x + width/2, val_pct.values, width, label='Validation', alpha=0.8)
    
    ax2.set_title('Percentage Distribution by Split', fontweight='bold')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_classes, rotation=45, ha='right')
    ax2.legend()
    
    # 3. Ratio comparison (val/train)
    ax3 = axes[1, 0]
    ratios = val_counts / train_counts
    ratios = ratios.replace([np.inf, -np.inf], 0)  # Handle division by zero
    
    bars = ax3.bar(range(len(ratios)), ratios.values, alpha=0.8)
    ax3.set_title('Validation/Train Ratio by Class', fontweight='bold')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Val/Train Ratio')
    ax3.set_xticks(range(len(ratios)))
    ax3.set_xticklabels(ratios.index, rotation=45, ha='right')
    ax3.axhline(y=ratios.mean(), color='red', linestyle='--', 
                label=f'Mean Ratio: {ratios.mean():.3f}')
    ax3.legend()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Split consistency analysis
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    consistency_data = []
    total_train = train_counts.sum()
    total_val = val_counts.sum()
    overall_ratio = total_val / total_train
    
    consistency_data.append(['Overall Split Ratio', f"{overall_ratio:.3f}"])
    consistency_data.append(['Train Images', f"{train_df['image_name'].nunique():,}"])
    consistency_data.append(['Val Images', f"{val_df['image_name'].nunique():,}"])
    consistency_data.append(['Train Objects', f"{len(train_df):,}"])
    consistency_data.append(['Val Objects', f"{len(val_df):,}"])
    
    # Most/least consistent classes
    ratio_diff = np.abs(ratios - overall_ratio)
    most_consistent = ratio_diff.idxmin()
    least_consistent = ratio_diff.idxmax()
    
    consistency_data.append(['Most Consistent Class', f"{most_consistent}"])
    consistency_data.append(['Least Consistent Class', f"{least_consistent}"])
    
    # Create table
    table = ax4.table(cellText=consistency_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Split Consistency Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'split_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated split comparison analysis saved")
    plt.close()

def create_spatial_distribution_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create spatial distribution analysis for all 10 classes."""
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Distribution Analysis (10 Classes)', fontsize=16, fontweight='bold')
    
    # 1. Center position heatmap
    ax1 = axes[0, 0]
    
    # Sample data for performance (use every 10th point)
    sample_df = combined_df.iloc[::10].copy()
    
    # Create 2D histogram of center positions
    hist, xbins, ybins = np.histogram2d(sample_df['center_x'], sample_df['center_y'], 
                                       bins=50, density=True)
    
    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    im = ax1.imshow(hist.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    ax1.set_title('Object Center Position Heatmap', fontweight='bold')
    ax1.set_xlabel('Center X (normalized)')
    ax1.set_ylabel('Center Y (normalized)')
    plt.colorbar(im, ax=ax1, label='Density')
    
    # 2. Class-specific spatial patterns
    ax2 = axes[0, 1]
    safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    
    for class_name in safety_critical:
        class_data = combined_df[combined_df['category'] == class_name].sample(
            min(1000, len(combined_df[combined_df['category'] == class_name])))
        ax2.scatter(class_data['center_x'], class_data['center_y'], 
                   alpha=0.6, s=10, label=class_name)
    
    ax2.set_title('Safety-Critical Classes Spatial Distribution', fontweight='bold')
    ax2.set_xlabel('Center X (normalized)')
    ax2.set_ylabel('Center Y (normalized)')
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Edge proximity analysis
    ax3 = axes[1, 0]
    
    # Calculate edge distances
    edge_distances = []
    class_names = []
    
    for class_name in combined_df['category'].unique():
        class_data = combined_df[combined_df['category'] == class_name]
        
        # Distance to nearest edge
        edge_dist = np.minimum.reduce([
            class_data['center_x'],  # Left edge
            1 - class_data['center_x'],  # Right edge  
            class_data['center_y'],  # Bottom edge
            1 - class_data['center_y']   # Top edge
        ])
        
        edge_distances.extend(edge_dist.tolist())
        class_names.extend([class_name] * len(edge_dist))
    
    edge_df = pd.DataFrame({'class': class_names, 'edge_distance': edge_distances})
    
    # Box plot of edge distances
    class_order = combined_df['category'].value_counts().index
    sns.boxplot(data=edge_df, x='class', y='edge_distance', ax=ax3, order=class_order)
    ax3.set_title('Distance to Image Edge by Class', fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Minimum Distance to Edge')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Spatial statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate spatial statistics
    spatial_stats = []
    for class_name in class_order[:7]:  # Top 7 classes for space
        class_data = combined_df[combined_df['category'] == class_name]
        
        mean_x = class_data['center_x'].mean()
        mean_y = class_data['center_y'].mean()
        std_x = class_data['center_x'].std()
        std_y = class_data['center_y'].std()
        
        spatial_stats.append([
            class_name[:8],  # Truncate for table
            f"{mean_x:.3f}",
            f"{mean_y:.3f}",
            f"{std_x:.3f}",
            f"{std_y:.3f}"
        ])
    
    # Create table
    table_data = [['Class', 'Mean X', 'Mean Y', 'Std X', 'Std Y']] + spatial_stats
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax4.set_title('Spatial Distribution Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated spatial distribution analysis saved")
    plt.close()

def create_statistical_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive statistical summary for all 10 classes."""
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Summary (Complete 10 Classes)', fontsize=16, fontweight='bold')
    
    # 1. Class imbalance metrics
    ax1 = axes[0, 0]
    
    class_counts = combined_df['category'].value_counts()
    
    # Calculate imbalance metrics
    total_objects = len(combined_df)
    gini_coeff = 1 - sum((class_counts / total_objects) ** 2)
    entropy = -sum((class_counts / total_objects) * np.log2(class_counts / total_objects))
    max_min_ratio = class_counts.max() / class_counts.min()
    
    # Imbalance visualization
    normalized_counts = class_counts / class_counts.sum()
    cumulative = normalized_counts.cumsum()
    
    ax1.bar(range(len(class_counts)), normalized_counts.values, alpha=0.7)
    ax1.plot(range(len(class_counts)), cumulative.values, 'ro-', linewidth=2, markersize=6)
    ax1.set_title('Class Imbalance Analysis', fontweight='bold')
    ax1.set_xlabel('Classes (ordered by frequency)')
    ax1.set_ylabel('Proportion / Cumulative')
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
    
    # Add metrics text
    metrics_text = f'Gini Coefficient: {gini_coeff:.3f}\nEntropy: {entropy:.3f}\nMax/Min Ratio: {max_min_ratio:.1f}'
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Safety-critical analysis
    ax2 = axes[0, 1]
    
    safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    vehicle_classes = ['car', 'truck', 'bus', 'train']
    infrastructure = ['traffic_light', 'traffic_sign']
    
    category_counts = {
        'Safety Critical': class_counts[class_counts.index.isin(safety_critical)].sum(),
        'Vehicle': class_counts[class_counts.index.isin(vehicle_classes)].sum(), 
        'Infrastructure': class_counts[class_counts.index.isin(infrastructure)].sum()
    }
    
    colors = ['red', 'blue', 'green']
    wedges, texts, autotexts = ax2.pie(category_counts.values(), 
                                       labels=category_counts.keys(),
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90)
    ax2.set_title('Safety Category Distribution', fontweight='bold')
    
    # 3. Bounding box statistics
    ax3 = axes[1, 0]
    
    # Calculate statistics for each class
    bbox_stats = []
    for class_name in class_counts.index:
        class_data = combined_df[combined_df['category'] == class_name]
        
        stats = {
            'mean_area': class_data['bbox_area'].mean(),
            'median_area': class_data['bbox_area'].median(),
            'std_area': class_data['bbox_area'].std(),
            'mean_ar': class_data['bbox_aspect_ratio'].mean()
        }
        bbox_stats.append(stats)
    
    # Violin plot of areas by class (log scale)
    areas_by_class = [combined_df[combined_df['category'] == cls]['bbox_area'].values 
                     for cls in class_counts.index]
    
    parts = ax3.violinplot(areas_by_class, positions=range(len(class_counts)), 
                          showmeans=True, showmedians=True)
    ax3.set_yscale('log')
    ax3.set_title('Bounding Box Area Distributions', fontweight='bold')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Area (pixels², log scale)')
    ax3.set_xticks(range(len(class_counts)))
    ax3.set_xticklabels(class_counts.index, rotation=45, ha='right')
    
    # 4. Dataset overview metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate comprehensive metrics
    overview_metrics = [
        ['Total Objects', f"{len(combined_df):,}"],
        ['Total Images', f"{combined_df['image_name'].nunique():,}"],
        ['Number of Classes', f"{combined_df['category'].nunique()}"],
        ['Avg Objects/Image', f"{len(combined_df) / combined_df['image_name'].nunique():.1f}"],
        ['', ''],
        ['Most Frequent Class', f"{class_counts.index[0]} ({class_counts.iloc[0]:,})"],
        ['Least Frequent Class', f"{class_counts.index[-1]} ({class_counts.iloc[-1]:,})"],
        ['Imbalance Ratio', f"{max_min_ratio:.1f}:1"],
        ['', ''],
        ['Safety Critical Objects', f"{category_counts['Safety Critical']:,}"],
        ['Safety Critical %', f"{category_counts['Safety Critical']/len(combined_df)*100:.1f}%"],
        ['', ''],
        ['Train/Val Ratio', f"{len(train_df) / len(val_df):.2f}:1"],
    ]
    
    # Create table
    table = ax4.table(cellText=overview_metrics, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title('Dataset Overview Statistics', fontweight='bold', pad=20)
    
    # Style the table
    for i in range(len(overview_metrics)):
        if overview_metrics[i][0] == '':  # Empty rows for spacing
            table[(i+1, 0)].set_facecolor('#f0f0f0')
            table[(i+1, 1)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Updated statistical summary saved")
    plt.close()

def generate_analysis_report(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive text analysis report."""
    
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    class_counts = combined_df['category'].value_counts()
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BDD100K COMPLETE 10-CLASS ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 20)
    report_lines.append(f"Total Objects: {len(combined_df):,}")
    report_lines.append(f"Total Images: {combined_df['image_name'].nunique():,}")
    report_lines.append(f"Number of Classes: {combined_df['category'].nunique()}")
    report_lines.append(f"Average Objects per Image: {len(combined_df) / combined_df['image_name'].nunique():.2f}")
    report_lines.append("")
    
    # Class distribution
    report_lines.append("CLASS DISTRIBUTION")
    report_lines.append("-" * 20)
    total = len(combined_df)
    for i, (class_name, count) in enumerate(class_counts.items()):
        percentage = count / total * 100
        report_lines.append(f"{i+1:2d}. {class_name:15s}: {count:8,} ({percentage:5.2f}%)")
    
    report_lines.append("")
    
    # Imbalance analysis
    max_min_ratio = class_counts.max() / class_counts.min()
    gini_coeff = 1 - sum((class_counts / total) ** 2)
    entropy = -sum((class_counts / total) * np.log2(class_counts / total))
    
    report_lines.append("IMBALANCE ANALYSIS")
    report_lines.append("-" * 20)
    report_lines.append(f"Most Frequent: {class_counts.index[0]} ({class_counts.iloc[0]:,} instances)")
    report_lines.append(f"Least Frequent: {class_counts.index[-1]} ({class_counts.iloc[-1]} instances)")
    report_lines.append(f"Imbalance Ratio: {max_min_ratio:.1f}:1")
    report_lines.append(f"Gini Coefficient: {gini_coeff:.3f} (0=equal, 1=maximum inequality)")
    report_lines.append(f"Entropy: {entropy:.3f} (higher = more balanced)")
    report_lines.append("")
    
    # Safety analysis
    safety_critical = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
    vehicle_classes = ['car', 'truck', 'bus', 'train']
    infrastructure = ['traffic_light', 'traffic_sign']
    
    safety_count = class_counts[class_counts.index.isin(safety_critical)].sum()
    vehicle_count = class_counts[class_counts.index.isin(vehicle_classes)].sum()
    infra_count = class_counts[class_counts.index.isin(infrastructure)].sum()
    
    report_lines.append("SAFETY CATEGORY ANALYSIS")
    report_lines.append("-" * 25)
    report_lines.append(f"Safety-Critical Classes: {safety_count:,} ({safety_count/total*100:.1f}%)")
    for cls in safety_critical:
        if cls in class_counts.index:
            count = class_counts[cls]
            report_lines.append(f"  - {cls}: {count:,}")
    
    report_lines.append(f"\nVehicle Classes: {vehicle_count:,} ({vehicle_count/total*100:.1f}%)")
    report_lines.append(f"Infrastructure Classes: {infra_count:,} ({infra_count/total*100:.1f}%)")
    report_lines.append("")
    
    # Split analysis
    train_count = len(train_df)
    val_count = len(val_df)
    split_ratio = train_count / val_count
    
    report_lines.append("TRAIN/VALIDATION SPLIT ANALYSIS")
    report_lines.append("-" * 32)
    report_lines.append(f"Training Set: {train_count:,} objects ({train_df['image_name'].nunique():,} images)")
    report_lines.append(f"Validation Set: {val_count:,} objects ({val_df['image_name'].nunique():,} images)")
    report_lines.append(f"Split Ratio: {split_ratio:.2f}:1 (train:val)")
    report_lines.append("")
    
    # Bounding box analysis
    report_lines.append("BOUNDING BOX STATISTICS")
    report_lines.append("-" * 25)
    report_lines.append(f"{'Class':<15} {'Mean Area':<12} {'Median Area':<14} {'Mean Width':<12} {'Mean Height':<13} {'Mean AR':<8}")
    report_lines.append("-" * 80)
    
    for class_name in class_counts.index:
        class_data = combined_df[combined_df['category'] == class_name]
        mean_area = class_data['bbox_area'].mean()
        median_area = class_data['bbox_area'].median() 
        mean_width = class_data['bbox_width'].mean()
        mean_height = class_data['bbox_height'].mean()
        mean_ar = class_data['bbox_aspect_ratio'].mean()
        
        report_lines.append(f"{class_name:<15} {mean_area:<12.0f} {median_area:<14.0f} {mean_width:<12.0f} {mean_height:<13.0f} {mean_ar:<8.2f}")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Save report
    report_file = output_dir / 'class_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Updated analysis report saved to {report_file}")

def main():
    """Main function to generate all updated plots."""
    
    # Paths
    data_dir = "data/analysis/processed_10class_corrected"
    output_dir = Path("data/analysis/plots")
    
    print("🚀 Generating Updated BDD100K 10-Class Analysis Plots")
    print("=" * 60)
    
    # Load data
    train_df, val_df = load_10class_data(data_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\n📊 Generating visualizations...")
    create_class_distribution_plot(train_df, val_df, output_dir)
    create_bbox_analysis_plot(train_df, val_df, output_dir)
    create_co_occurrence_analysis(train_df, val_df, output_dir)
    create_split_comparison_analysis(train_df, val_df, output_dir)
    create_spatial_distribution_analysis(train_df, val_df, output_dir)
    create_statistical_summary(train_df, val_df, output_dir)
    
    # Generate text report
    print("\n📝 Generating analysis report...")
    generate_analysis_report(train_df, val_df, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ ALL 10-CLASS PLOTS AND ANALYSIS UPDATED!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print("📊 Generated 6 updated visualization plots")
    print("📝 Generated comprehensive analysis report")
    print("🎯 All analysis now includes complete 10-class data")

if __name__ == "__main__":
    main()