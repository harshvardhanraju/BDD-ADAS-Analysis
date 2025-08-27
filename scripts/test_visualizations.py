#!/usr/bin/env python3
"""
Test Visualization Framework

This script tests the visualization components with sample data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.visualization import DetectionVisualizer


def create_sample_data():
    """Create sample data for testing visualizations."""
    # Create a sample image (simple colored rectangle)
    image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    
    # Sample ground truth annotations
    ground_truth = [
        {
            'category_id': 0,  # pedestrian
            'bbox': [100, 200, 80, 150]  # [x, y, width, height]
        },
        {
            'category_id': 2,  # car
            'bbox': [400, 400, 200, 100]
        },
        {
            'category_id': 8,  # traffic_light
            'bbox': [600, 50, 30, 60]
        }
    ]
    
    # Sample predictions (some matching, some not)
    predictions = [
        {
            'category_id': 0,  # pedestrian - good match
            'bbox': [105, 205, 75, 145],
            'score': 0.85
        },
        {
            'category_id': 2,  # car - good match
            'bbox': [395, 405, 210, 95],
            'score': 0.92
        },
        {
            'category_id': 2,  # car - false positive
            'bbox': [800, 500, 150, 80],
            'score': 0.65
        },
        {
            'category_id': 1,  # rider - false positive
            'bbox': [300, 300, 40, 80],
            'score': 0.45
        }
    ]
    
    return image, ground_truth, predictions


def test_detection_visualization():
    """Test the detection visualization."""
    print("Testing Detection Visualization...")
    
    # Create visualizer
    visualizer = DetectionVisualizer()
    
    # Create sample data
    image, gt, preds = create_sample_data()
    
    # Test basic visualization
    comparison = visualizer.visualize_detections(
        image, gt, preds, confidence_threshold=0.4
    )
    
    # Save result
    output_dir = Path("evaluation_results/visualization_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    import cv2
    cv2.imwrite(str(output_dir / "detection_comparison.jpg"), 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Detection comparison saved to {output_dir / 'detection_comparison.jpg'}")
    
    # Test confidence distribution
    all_predictions = preds * 50  # Simulate more predictions
    for i, pred in enumerate(all_predictions):
        pred['score'] = np.random.beta(2, 2)  # Generate realistic confidence scores
        pred['category_id'] = i % 10  # Distribute across all classes
    
    fig = visualizer.visualize_confidence_distribution(
        all_predictions, save_path=str(output_dir / "confidence_distribution.png")
    )
    plt.close(fig)
    
    print(f"✅ Confidence distribution saved to {output_dir / 'confidence_distribution.png'}")
    
    # Test per-class performance chart
    sample_metrics = {
        'pedestrian': 0.65,
        'rider': 0.23,
        'car': 0.89,
        'truck': 0.71,
        'bus': 0.55,
        'train': 0.12,
        'motorcycle': 0.34,
        'bicycle': 0.42,
        'traffic_light': 0.78,
        'traffic_sign': 0.82
    }
    
    fig = visualizer.create_class_performance_chart(
        sample_metrics, metric_name='Average Precision',
        save_path=str(output_dir / "class_performance.png")
    )
    plt.close(fig)
    
    print(f"✅ Class performance chart saved to {output_dir / 'class_performance.png'}")
    
    # Generate legend
    legend = visualizer.generate_legend()
    cv2.imwrite(str(output_dir / "class_legend.jpg"),
                cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Class legend saved to {output_dir / 'class_legend.jpg'}")
    
    return True


def test_grid_visualization():
    """Test the grid visualization."""
    print("\\nTesting Grid Visualization...")
    
    visualizer = DetectionVisualizer()
    
    # Create multiple sample images
    images = []
    ground_truths = []
    predictions = []
    image_ids = []
    
    for i in range(4):
        image, gt, preds = create_sample_data()
        # Vary the data slightly
        image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        images.append(image)
        ground_truths.append(gt)
        predictions.append(preds)
        image_ids.append(f"test_img_{i:03d}")
    
    # Create grid
    grid = visualizer.create_comparison_grid(
        images, ground_truths, predictions, image_ids,
        grid_size=(2, 2), confidence_threshold=0.4
    )
    
    # Save grid
    output_dir = Path("evaluation_results/visualization_tests")
    cv2.imwrite(str(output_dir / "detection_grid.jpg"),
                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Detection grid saved to {output_dir / 'detection_grid.jpg'}")
    
    return True


def main():
    """Run all visualization tests."""
    print("🎨 Testing BDD100K Visualization Framework")
    print("=" * 50)
    
    try:
        # Test individual detection visualization
        test_detection_visualization()
        
        # Test grid visualization  
        test_grid_visualization()
        
        print("\\n" + "=" * 50)
        print("✅ All visualization tests completed successfully!")
        print("Check evaluation_results/visualization_tests/ for outputs")
        
    except Exception as e:
        print(f"\\n❌ Visualization test failed: {e}")
        raise


if __name__ == "__main__":
    main()