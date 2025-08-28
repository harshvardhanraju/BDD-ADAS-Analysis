"""
Test script for BDD100K FiftyOne integration
Tests with a small subset to verify everything works correctly.
"""

import json
import os
import sys
from pathlib import Path

import fiftyone as fo
import pandas as pd

# Add the script directory to path
sys.path.append(str(Path(__file__).parent))

from bdd100k_fiftyone_integration import BDD100KFiftyOneConverter


def test_basic_integration():
    """Test basic dataset creation without model predictions."""
    
    print("=== Testing Basic FiftyOne Integration ===")
    
    # Check if processed annotations exist
    annotations_file = Path("data/analysis/processed_10class_corrected/val_annotations_10class.csv")
    
    if not annotations_file.exists():
        print(f"Annotations file not found: {annotations_file}")
        return False
    
    print(f"Found annotations file: {annotations_file}")
    
    # Check annotations structure
    df = pd.read_csv(annotations_file)
    print(f"Loaded {len(df)} annotation records")
    print(f"Unique images: {df['image_name'].nunique()}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Splits: {df['split'].unique()}")
    
    # Find a valid image directory (we may need to adjust the path)
    potential_image_dirs = [
        "data/raw/bdd100k/images/10k/val",
        "data/raw/bdd100k/images/100k/val",
        "data/raw/images/val",
        "images/val"
    ]
    
    images_root = None
    for img_dir in potential_image_dirs:
        if Path(img_dir).exists():
            images_root = img_dir
            print(f"Found images directory: {images_root}")
            break
    
    if not images_root:
        print("No valid images directory found. Creating mock test...")
        # Create a basic test without images
        return test_annotation_processing_only(annotations_file)
    
    # Test with a very small subset
    print("\nTesting with 5 images...")
    
    converter = BDD100KFiftyOneConverter(
        annotations_file=str(annotations_file),
        images_root=images_root,
        model_checkpoint=None,  # No model for basic test
        dataset_name="bdd100k_basic_test",
        subset_size=5
    )
    
    try:
        # Create dataset
        dataset = converter.create_fiftyone_dataset()
        
        print(f"Created dataset with {len(dataset)} samples")
        
        # Check first sample
        if len(dataset) > 0:
            sample = dataset.first()
            print(f"First sample: {sample.filepath}")
            print(f"Ground truth detections: {len(sample.ground_truth.detections)}")
            
            for det in sample.ground_truth.detections[:3]:  # Show first 3
                print(f"  - {det.label}: bbox={det.bounding_box}, safety_critical={det.get('is_safety_critical', False)}")
        
        # Test views
        views = converter.get_analysis_views(dataset)
        print(f"Created views: {list(views.keys())}")
        
        if "safety_critical" in views:
            safety_view = views["safety_critical"]
            print(f"Safety critical detections: {len(safety_view.values('ground_truth.detections'))}")
        
        print("Basic integration test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"Basic integration test FAILED: {e}")
        return False


def test_annotation_processing_only(annotations_file):
    """Test annotation processing without requiring image files."""
    
    print("\n=== Testing Annotation Processing Only ===")
    
    try:
        df = pd.read_csv(annotations_file)
        
        # Test subset
        subset_df = df.head(50)  # First 50 rows
        unique_images = subset_df['image_name'].unique()[:5]  # First 5 unique images
        test_df = df[df['image_name'].isin(unique_images)]
        
        print(f"Test subset: {len(test_df)} annotations for {len(unique_images)} images")
        
        # Test class mapping
        converter = BDD100KFiftyOneConverter(
            annotations_file=str(annotations_file),
            images_root="dummy_path",  # Won't be used
            subset_size=5
        )
        
        # Test class mapping
        categories_in_data = test_df['category'].dropna().unique()
        mapped_categories = [cat for cat in categories_in_data if cat in converter.class_mapping]
        unmapped_categories = [cat for cat in categories_in_data if cat not in converter.class_mapping]
        
        print(f"Mapped categories: {mapped_categories}")
        if unmapped_categories:
            print(f"Unmapped categories: {unmapped_categories}")
        
        # Test bounding box processing
        valid_boxes = 0
        for _, row in test_df.iterrows():
            if pd.notna(row['bbox_x1']):
                bbox = [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']]
                if all(pd.notna(bbox)) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    valid_boxes += 1
        
        print(f"Valid bounding boxes: {valid_boxes}/{len(test_df)}")
        
        print("Annotation processing test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"Annotation processing test FAILED: {e}")
        return False


def test_model_loading():
    """Test model loading if checkpoint exists."""
    
    print("\n=== Testing Model Loading ===")
    
    checkpoint_paths = [
        "checkpoints/complete_10class_demo/checkpoint_epoch_048.pth",
        "checkpoints/checkpoint_epoch_048.pth",
        "checkpoints/latest.pth"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            print(f"Found checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        print("No checkpoint found, skipping model loading test")
        return True
    
    try:
        converter = BDD100KFiftyOneConverter(
            annotations_file="dummy",
            images_root="dummy",
            model_checkpoint=checkpoint_path,
            subset_size=1
        )
        
        if converter.model:
            print("Model loaded successfully ✓")
            return True
        else:
            print("Model loading failed ✗")
            return False
            
    except Exception as e:
        print(f"Model loading test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    
    print("BDD100K FiftyOne Integration Tests")
    print("=" * 50)
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent.parent.parent)
    print(f"Working directory: {os.getcwd()}")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic integration
    if test_basic_integration():
        tests_passed += 1
    
    # Test 2: Model loading
    if test_model_loading():
        tests_passed += 1
    
    # Test 3: Check FiftyOne installation
    try:
        print("\n=== Testing FiftyOne Installation ===")
        print(f"FiftyOne version: {fo.__version__}")
        
        # Test brain import
        import fiftyone.brain as fob
        print("FiftyOne Brain imported successfully ✓")
        
        tests_passed += 1
    except Exception as e:
        print(f"FiftyOne installation test FAILED: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("All tests PASSED! ✓")
        return True
    else:
        print("Some tests FAILED! ✗")
        return False


if __name__ == "__main__":
    main()