"""
Example Usage Script for BDD100K Analysis

This script demonstrates how to use individual components
of the BDD100K analysis toolkit.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.parsers.bdd_parser import BDDParser
from src.analysis.class_analysis import ClassDistributionAnalyzer
from src.analysis.spatial_analysis import SpatialAnalyzer
from src.analysis.image_analysis import ImageCharacteristicsAnalyzer

def example_parsing():
    """Example of BDD data parsing."""
    print("🔄 Example: BDD Data Parsing")
    print("-" * 40)
    
    # Initialize parser
    data_root = "data/raw"  # Adjust path as needed
    parser = BDDParser(data_root)
    
    # Validate dataset structure
    print("Validating dataset structure...")
    validation = parser.validate_dataset_structure()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}")
    
    # Load annotations (sample)
    if validation.get('train_labels_exist', False):
        print("\nLoading train annotations...")
        train_annotations = parser.load_split_annotations('train')
        print(f"Loaded {len(train_annotations)} training images")
        
        # Show sample annotation
        if train_annotations:
            sample = train_annotations[0]
            print(f"\nSample annotation:")
            print(f"- Image: {sample.name}")
            print(f"- Objects: {len(sample.objects)}")
            if sample.objects:
                obj = sample.objects[0]
                print(f"- First object: {obj.category} at ({obj.bbox.x1}, {obj.bbox.y1})")
    
    print("\n" + "="*50 + "\n")

def example_class_analysis():
    """Example of class distribution analysis."""
    print("📊 Example: Class Distribution Analysis")
    print("-" * 40)
    
    # Load processed data (you need to run parsing first)
    try:
        data = pd.read_csv("data/processed/train_annotations.csv")
        print(f"Loaded {len(data)} annotation records")
        
        # Initialize analyzer
        analyzer = ClassDistributionAnalyzer(data, "data/analysis/example_plots")
        
        # Run basic statistics
        print("\nComputing basic statistics...")
        basic_stats = analyzer.compute_basic_statistics()
        
        print(f"Total objects: {basic_stats['total_objects']}")
        print(f"Number of classes: {basic_stats['num_classes']}")
        
        # Show top 5 classes
        overall_dist = basic_stats['overall_distribution']
        print("\nTop 5 classes:")
        for i, (class_name, count) in enumerate(list(overall_dist.items())[:5]):
            percentage = count / basic_stats['total_objects'] * 100
            print(f"{i+1}. {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Imbalance metrics
        imb = basic_stats['imbalance_metrics']
        print(f"\nClass imbalance ratio: {imb['imbalance_ratio']:.2f}")
        print(f"Gini coefficient: {imb['gini_coefficient']:.3f}")
        
    except FileNotFoundError:
        print("❌ Processed data not found. Please run the parser first:")
        print("   python scripts/run_data_analysis.py --data-root data/raw")
    
    print("\n" + "="*50 + "\n")

def example_spatial_analysis():
    """Example of spatial analysis."""
    print("🌐 Example: Spatial Analysis")
    print("-" * 40)
    
    try:
        data = pd.read_csv("data/processed/train_annotations.csv")
        
        # Filter data with bounding boxes
        bbox_data = data.dropna(subset=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
        print(f"Found {len(bbox_data)} objects with bounding boxes")
        
        # Initialize analyzer
        analyzer = SpatialAnalyzer(bbox_data, output_dir="data/analysis/example_plots")
        
        # Analyze dimensions
        print("\nAnalyzing bounding box dimensions...")
        bbox_analysis = analyzer.analyze_bbox_dimensions()
        
        overall_stats = bbox_analysis['overall_statistics']
        print(f"Average width: {overall_stats['width']['mean']:.1f} pixels")
        print(f"Average height: {overall_stats['height']['mean']:.1f} pixels")
        print(f"Average area: {overall_stats['area']['mean']:.0f} pixels²")
        
        # Spatial distribution
        print("\nAnalyzing spatial distribution...")
        spatial_analysis = analyzer.analyze_spatial_distribution()
        
        if 'grid_distribution' in spatial_analysis:
            grid_dist = spatial_analysis['grid_distribution']
            print("Grid distribution (top 3 positions):")
            sorted_grid = sorted(grid_dist.items(), key=lambda x: x[1], reverse=True)
            for pos, count in sorted_grid[:3]:
                print(f"- {pos.replace('_', ' ')}: {count:,} objects")
        
    except FileNotFoundError:
        print("❌ Processed data not found. Please run the parser first.")
    
    print("\n" + "="*50 + "\n")

def example_image_analysis():
    """Example of image analysis."""
    print("🖼️  Example: Image Analysis")
    print("-" * 40)
    
    try:
        data = pd.read_csv("data/processed/train_annotations.csv")
        
        # Initialize analyzer
        images_root = "data/raw/images/100k"
        analyzer = ImageCharacteristicsAnalyzer(
            data, 
            images_root, 
            "data/analysis/example_plots"
        )
        
        # Analyze dimensions (small sample)
        print("Analyzing image dimensions (sample of 100 images)...")
        dimension_analysis = analyzer.analyze_image_dimensions(sample_size=100)
        
        if 'overall_statistics' in dimension_analysis:
            width_stats = dimension_analysis['overall_statistics']['width']
            height_stats = dimension_analysis['overall_statistics']['height']
            
            print(f"Average width: {width_stats['mean']:.0f} pixels")
            print(f"Average height: {height_stats['mean']:.0f} pixels")
            print(f"Resolution consistency: {width_stats['unique_values']} unique widths")
        
        # Scene attributes
        print("\nAnalyzing scene attributes...")
        scene_analysis = analyzer.analyze_scene_attributes()
        
        if 'attribute_statistics' in scene_analysis:
            attr_stats = scene_analysis['attribute_statistics']
            print(f"Found {len(attr_stats)} scene attributes")
            
            for attr_name, stats in list(attr_stats.items())[:3]:  # Show first 3
                print(f"- {attr_name}: {stats['unique_values']} unique values")
        
    except FileNotFoundError:
        print("❌ Processed data or images not found.")
        print("   Ensure BDD100K dataset is properly extracted.")
    
    print("\n" + "="*50 + "\n")

def example_dashboard_data():
    """Example of preparing data for dashboard."""
    print("🚀 Example: Dashboard Data Preparation")
    print("-" * 40)
    
    try:
        # Load both train and validation data
        train_data = pd.read_csv("data/processed/train_annotations.csv")
        val_data = pd.read_csv("data/processed/val_annotations.csv")
        
        print(f"Train data: {len(train_data)} records")
        print(f"Validation data: {len(val_data)} records")
        
        # Combine for dashboard
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        print(f"Combined: {len(combined_data)} records")
        
        # Basic dashboard statistics
        unique_images = combined_data['image_name'].nunique()
        unique_classes = combined_data['category'].nunique()
        total_objects = len(combined_data[combined_data['category'].notna()])
        
        print(f"\nDashboard ready with:")
        print(f"- {unique_images:,} unique images")  
        print(f"- {total_objects:,} objects")
        print(f"- {unique_classes} classes")
        
        print(f"\nTo start dashboard: streamlit run src/visualization/dashboard.py")
        
    except FileNotFoundError:
        print("❌ Processed data not found. Run analysis pipeline first.")
    
    print("\n" + "="*50 + "\n")

def main():
    """Run all examples."""
    print("🎯 BDD100K Analysis Toolkit - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_parsing,
        example_class_analysis, 
        example_spatial_analysis,
        example_image_analysis,
        example_dashboard_data
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print()
    
    print("✅ All examples completed!")
    print("\nNext steps:")
    print("1. Run full analysis: python scripts/run_data_analysis.py --data-root data/raw")
    print("2. Start dashboard: streamlit run src/visualization/dashboard.py")
    print("3. Review generated reports in data/analysis/reports/")

if __name__ == "__main__":
    main()