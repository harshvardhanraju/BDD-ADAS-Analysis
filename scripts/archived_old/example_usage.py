"""
Example Usage Script for BDD100K Analysis

This script demonstrates how to use individual components
of the BDD100K analysis toolkit.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analysis.class_analysis import ClassDistributionAnalyzer
from src.analysis.image_analysis import ImageCharacteristicsAnalyzer
from src.analysis.spatial_analysis import SpatialAnalyzer
from src.parsers.bdd_parser import BDDParser


def example_parsing():
    """
    Example of BDD data parsing.

    Demonstrates basic dataset validation and annotation loading.
    Shows how to access parsed annotation data and object properties.
    """
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
    if validation.get("train_labels_exist", False):
        print("\nLoading train annotations...")
        train_annotations = parser.load_split_annotations("train")
        print(f"Loaded {len(train_annotations)} training images")

        # Show sample annotation
        if train_annotations:
            sample = train_annotations[0]
            print("\nSample annotation:")
            print(f"- Image: {sample.name}")
            print(f"- Objects: {len(sample.objects)}")
            if sample.objects:
                obj = sample.objects[0]
                bbox_info = f"({obj.bbox.x1}, {obj.bbox.y1})"
                print(f"- First object: {obj.category} at {bbox_info}")

    print("\n" + "=" * 50 + "\n")


def example_class_analysis():
    """
    Example of class distribution analysis.

    Demonstrates statistical analysis of object class distribution,
    including imbalance metrics and visualization preparation.
    """
    print("📊 Example: Class Distribution Analysis")
    print("-" * 40)

    # Load processed data (you need to run parsing first)
    try:
        data = pd.read_csv("data/processed/train_annotations.csv")
        print(f"Loaded {len(data)} annotation records")

        # Initialize analyzer
        output_dir = "data/analysis/example_plots"
        analyzer = ClassDistributionAnalyzer(data, output_dir)

        # Run basic statistics
        print("\nComputing basic statistics...")
        basic_stats = analyzer.compute_basic_statistics()

        print(f"Total objects: {basic_stats['total_objects']}")
        print(f"Number of classes: {basic_stats['num_classes']}")

        # Show top 5 classes
        overall_dist = basic_stats["overall_distribution"]
        print("\nTop 5 classes:")
        for i, (class_name, count) in enumerate(list(overall_dist.items())[:5]):
            percentage = count / basic_stats["total_objects"] * 100
            print(f"{i+1}. {class_name}: {count:,} ({percentage:.1f}%)")

        # Imbalance metrics
        imb = basic_stats["imbalance_metrics"]
        print(f"\nClass imbalance ratio: {imb['imbalance_ratio']:.2f}")
        print(f"Gini coefficient: {imb['gini_coefficient']:.3f}")

    except FileNotFoundError:
        print("❌ Processed data not found. Please run the parser first:")
        print("   python scripts/run_data_analysis.py --data-root data/raw")

    print("\n" + "=" * 50 + "\n")


def example_spatial_analysis():
    """
    Example of spatial analysis.

    Demonstrates bounding box dimension analysis and spatial
    distribution patterns across the image grid.
    """
    print("🌐 Example: Spatial Analysis")
    print("-" * 40)

    try:
        data = pd.read_csv("data/processed/train_annotations.csv")

        # Filter data with bounding boxes
        bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        bbox_data = data.dropna(subset=bbox_cols)
        print(f"Found {len(bbox_data)} objects with bounding boxes")

        # Initialize analyzer
        output_dir = "data/analysis/example_plots"
        analyzer = SpatialAnalyzer(bbox_data, output_dir=output_dir)

        # Analyze dimensions
        print("\nAnalyzing bounding box dimensions...")
        bbox_analysis = analyzer.analyze_bbox_dimensions()

        overall_stats = bbox_analysis["overall_statistics"]
        width_mean = overall_stats["width"]["mean"]
        height_mean = overall_stats["height"]["mean"]
        area_mean = overall_stats["area"]["mean"]

        print(f"Average width: {width_mean:.1f} pixels")
        print(f"Average height: {height_mean:.1f} pixels")
        print(f"Average area: {area_mean:.0f} pixels²")

        # Spatial distribution
        print("\nAnalyzing spatial distribution...")
        spatial_analysis = analyzer.analyze_spatial_distribution()

        if "grid_distribution" in spatial_analysis:
            grid_dist = spatial_analysis["grid_distribution"]
            print("Grid distribution (top 3 positions):")
            sorted_grid = sorted(grid_dist.items(), key=lambda x: x[1], reverse=True)
            for pos, count in sorted_grid[:3]:
                pos_formatted = pos.replace("_", " ")
                print(f"- {pos_formatted}: {count:,} objects")

    except FileNotFoundError:
        print("❌ Processed data not found. Please run the parser first.")

    print("\n" + "=" * 50 + "\n")


def example_image_analysis():
    """
    Example of image analysis.

    Demonstrates image dimension analysis and scene attribute extraction
    from the BDD100K dataset metadata.
    """
    print("🖼️  Example: Image Analysis")
    print("-" * 40)

    try:
        data = pd.read_csv("data/processed/train_annotations.csv")

        # Initialize analyzer
        images_root = "data/raw/images/100k"
        output_dir = "data/analysis/example_plots"
        analyzer = ImageCharacteristicsAnalyzer(data, images_root, output_dir)

        # Analyze dimensions (small sample)
        print("Analyzing image dimensions (sample of 100 images)...")
        dimension_analysis = analyzer.analyze_image_dimensions(sample_size=100)

        if "overall_statistics" in dimension_analysis:
            stats = dimension_analysis["overall_statistics"]
            width_stats = stats["width"]
            height_stats = stats["height"]

            print(f"Average width: {width_stats['mean']:.0f} pixels")
            print(f"Average height: {height_stats['mean']:.0f} pixels")
            unique_widths = width_stats["unique_values"]
            print(f"Resolution consistency: {unique_widths} unique widths")

        # Scene attributes
        print("\nAnalyzing scene attributes...")
        scene_analysis = analyzer.analyze_scene_attributes()

        if "attribute_statistics" in scene_analysis:
            attr_stats = scene_analysis["attribute_statistics"]
            print(f"Found {len(attr_stats)} scene attributes")

            # Show first 3 attributes
            for attr_name, stats in list(attr_stats.items())[:3]:
                unique_vals = stats["unique_values"]
                print(f"- {attr_name}: {unique_vals} unique values")

    except FileNotFoundError:
        print("❌ Processed data or images not found.")
        print("   Ensure BDD100K dataset is properly extracted.")

    print("\n" + "=" * 50 + "\n")


def example_dashboard_data():
    """
    Example of preparing data for dashboard.

    Demonstrates how to combine train/validation data and prepare
    summary statistics for the interactive Streamlit dashboard.
    """
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
        unique_images = combined_data["image_name"].nunique()
        unique_classes = combined_data["category"].nunique()
        total_objects = len(combined_data[combined_data["category"].notna()])

        print("\nDashboard ready with:")
        print(f"- {unique_images:,} unique images")
        print(f"- {total_objects:,} objects")
        print(f"- {unique_classes} classes")

        dashboard_cmd = "streamlit run src/visualization/dashboard.py"
        print(f"\nTo start dashboard: {dashboard_cmd}")

    except FileNotFoundError:
        print("❌ Processed data not found. Run analysis pipeline first.")

    print("\n" + "=" * 50 + "\n")


def main():
    """
    Run all examples.

    Executes all example functions in sequence, demonstrating
    the complete BDD100K analysis toolkit functionality.
    """
    print("🎯 BDD100K Analysis Toolkit - Usage Examples")
    print("=" * 60)

    examples = [
        example_parsing,
        example_class_analysis,
        example_spatial_analysis,
        example_image_analysis,
        example_dashboard_data,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print()

    print("✅ All examples completed!")
    print("\nNext steps:")
    full_analysis_cmd = "python scripts/run_data_analysis.py --data-root data/raw"
    dashboard_cmd = "streamlit run src/visualization/dashboard.py"
    print(f"1. Run full analysis: {full_analysis_cmd}")
    print(f"2. Start dashboard: {dashboard_cmd}")
    print("3. Review generated reports in data/analysis/reports/")


if __name__ == "__main__":
    main()
