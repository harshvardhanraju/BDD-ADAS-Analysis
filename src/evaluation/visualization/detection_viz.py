"""
Detection Visualization Tools for BDD100K Object Detection

This module provides tools for visualizing model predictions vs ground truth,
with special emphasis on safety-critical classes and failure cases.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import colorsys


class DetectionVisualizer:
    """
    Comprehensive visualization tools for object detection evaluation.
    
    Provides ground truth vs prediction comparisons, confidence visualization,
    and safety-critical class highlighting for autonomous driving applications.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize detection visualizer.
        
        Args:
            class_names: List of class names. If None, uses BDD100K 10-class names.
        """
        if class_names is None:
            self.class_names = [
                'pedestrian', 'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
            ]
        else:
            self.class_names = class_names
        
        # Safety-critical classes for special highlighting
        self.safety_critical_classes = {'pedestrian', 'rider', 'bicycle', 'motorcycle'}
        
        # Generate distinct colors for each class
        self.class_colors = self._generate_class_colors()
        
        # Set default style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _generate_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        colors = {}
        n_classes = len(self.class_names)
        
        for i, class_name in enumerate(self.class_names):
            # Generate colors in HSV space for better distinction
            hue = i / n_classes
            saturation = 0.7 if class_name in self.safety_critical_classes else 0.5
            value = 0.9
            
            # Convert to RGB (0-255 range)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[class_name] = tuple(int(c * 255) for c in rgb)
        
        return colors
    
    def visualize_detections(self,
                           image: np.ndarray,
                           ground_truth: List[Dict],
                           predictions: List[Dict],
                           confidence_threshold: float = 0.5,
                           show_confidence: bool = True,
                           highlight_safety: bool = True) -> np.ndarray:
        """
        Create side-by-side visualization of ground truth and predictions.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            ground_truth: List of GT annotations
            predictions: List of prediction dictionaries
            confidence_threshold: Minimum confidence for showing predictions
            show_confidence: Whether to display confidence scores
            highlight_safety: Whether to highlight safety-critical classes
            
        Returns:
            Combined visualization image
        """
        # Create two copies of the image
        gt_image = image.copy()
        pred_image = image.copy()
        
        # Draw ground truth boxes
        gt_image = self._draw_boxes(
            gt_image, ground_truth, box_type='ground_truth',
            highlight_safety=highlight_safety
        )
        
        # Filter predictions by confidence
        filtered_predictions = [
            p for p in predictions 
            if p.get('score', 1.0) >= confidence_threshold
        ]
        
        # Draw prediction boxes
        pred_image = self._draw_boxes(
            pred_image, filtered_predictions, box_type='prediction',
            show_confidence=show_confidence, highlight_safety=highlight_safety
        )
        
        # Combine images side by side
        combined = self._create_side_by_side(gt_image, pred_image)
        
        return combined
    
    def _draw_boxes(self,
                   image: np.ndarray,
                   annotations: List[Dict],
                   box_type: str = 'ground_truth',
                   show_confidence: bool = False,
                   highlight_safety: bool = True) -> np.ndarray:
        """Draw bounding boxes on image."""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            # Try to load a font, fallback to default if not available
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            # Get class info
            class_id = ann['category_id']
            if class_id >= len(self.class_names):
                continue
                
            class_name = self.class_names[class_id]
            color = self.class_colors[class_name]
            
            # Adjust color intensity for safety-critical classes
            if highlight_safety and class_name in self.safety_critical_classes:
                color = tuple(min(255, int(c * 1.2)) for c in color)  # Brighten
                line_width = 4
            else:
                line_width = 2
            
            # Get bounding box coordinates
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=line_width
            )
            
            # Prepare label text
            label_text = class_name
            if show_confidence and 'score' in ann:
                label_text += f" {ann['score']:.2f}"
                
            if highlight_safety and class_name in self.safety_critical_classes:
                label_text += " ⚠️"
            
            # Draw label background
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position label above the box
            label_y = max(0, y - text_height - 5)
            
            draw.rectangle(
                [(x, label_y), (x + text_width + 4, label_y + text_height + 4)],
                fill=color
            )
            
            # Draw label text
            draw.text((x + 2, label_y + 2), label_text, fill='white', font=font)
        
        return np.array(img_pil)
    
    def _create_side_by_side(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Combine two images side by side with labels."""
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Ensure same height
        max_height = max(h1, h2)
        if h1 != max_height:
            image1 = cv2.resize(image1, (w1, max_height))
        if h2 != max_height:
            image2 = cv2.resize(image2, (w2, max_height))
        
        # Add spacing between images
        spacing = np.ones((max_height, 10, 3), dtype=np.uint8) * 255
        
        # Combine images
        combined = np.concatenate([image1, spacing, image2], axis=1)
        
        # Add titles
        combined = self._add_titles(combined, ['Ground Truth', 'Predictions'])
        
        return combined
    
    def _add_titles(self, image: np.ndarray, titles: List[str]) -> np.ndarray:
        """Add titles to the combined image."""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate positions
        img_width = image.shape[1]
        section_width = (img_width - 10) // 2  # Account for spacing
        
        for i, title in enumerate(titles):
            x_center = (i * (section_width + 10)) + section_width // 2
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            # Position title
            x_pos = x_center - text_width // 2
            y_pos = 10
            
            # Draw background
            draw.rectangle(
                [(x_pos - 5, y_pos - 2), (x_pos + text_width + 5, y_pos + 30)],
                fill='white', outline='black', width=2
            )
            
            # Draw title
            draw.text((x_pos, y_pos), title, fill='black', font=font)
        
        return np.array(img_pil)
    
    def create_comparison_grid(self,
                              images: List[np.ndarray],
                              ground_truths: List[List[Dict]],
                              predictions: List[List[Dict]],
                              image_ids: List[str],
                              grid_size: Tuple[int, int] = (2, 2),
                              confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Create a grid of detection comparison images.
        
        Args:
            images: List of input images
            ground_truths: List of ground truth annotations for each image
            predictions: List of predictions for each image
            image_ids: List of image identifiers
            grid_size: (rows, cols) for the grid
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Grid visualization as numpy array
        """
        rows, cols = grid_size
        max_images = min(len(images), rows * cols)
        
        # Create individual comparisons
        comparisons = []
        for i in range(max_images):
            comparison = self.visualize_detections(
                images[i], ground_truths[i], predictions[i],
                confidence_threshold=confidence_threshold
            )
            # Resize to standard size
            comparison = cv2.resize(comparison, (800, 400))
            
            # Add image ID
            comparison = self._add_image_id(comparison, image_ids[i])
            comparisons.append(comparison)
        
        # Pad with empty images if needed
        while len(comparisons) < rows * cols:
            empty = np.ones((400, 800, 3), dtype=np.uint8) * 240
            comparisons.append(empty)
        
        # Arrange in grid
        grid_rows = []
        for r in range(rows):
            row_images = comparisons[r * cols:(r + 1) * cols]
            if row_images:
                row = np.concatenate(row_images, axis=1)
                grid_rows.append(row)
        
        if grid_rows:
            grid = np.concatenate(grid_rows, axis=0)
        else:
            grid = np.ones((400, 800, 3), dtype=np.uint8) * 240
        
        return grid
    
    def _add_image_id(self, image: np.ndarray, image_id: str) -> np.ndarray:
        """Add image ID to the bottom of the image."""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Position at bottom center
        text = f"Image ID: {image_id}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        x_pos = (image.shape[1] - text_width) // 2
        y_pos = image.shape[0] - 25
        
        # Draw background
        draw.rectangle(
            [(x_pos - 5, y_pos - 2), (x_pos + text_width + 5, y_pos + 18)],
            fill='white', outline='gray'
        )
        
        # Draw text
        draw.text((x_pos, y_pos), text, fill='black', font=font)
        
        return np.array(img_pil)
    
    def visualize_confidence_distribution(self,
                                        predictions: List[Dict],
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize confidence score distributions by class.
        
        Args:
            predictions: List of all predictions
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Organize predictions by class
        class_confidences = {class_name: [] for class_name in self.class_names}
        
        for pred in predictions:
            class_id = pred['category_id']
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                confidence = pred.get('score', 0.0)
                class_confidences[class_name].append(confidence)
        
        # Create subplot for each class
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.suptitle('Confidence Score Distributions by Class', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            confidences = class_confidences[class_name]
            
            if confidences:
                # Create histogram
                color = tuple(c/255.0 for c in self.class_colors[class_name])
                ax.hist(confidences, bins=20, alpha=0.7, color=color, edgecolor='black')
                
                # Add statistics
                mean_conf = np.mean(confidences)
                ax.axvline(mean_conf, color='red', linestyle='--', 
                          label=f'Mean: {mean_conf:.3f}')
                ax.legend()
                
                # Highlight safety-critical classes
                if class_name in self.safety_critical_classes:
                    ax.set_title(f'{class_name} ⚠️', fontweight='bold', color='red')
                else:
                    ax.set_title(class_name)
                    
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, 'No Predictions', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(class_name)
        
        # Skip tight_layout to avoid sizing issues
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=80, bbox_inches=None)
        
        return fig
    
    def create_class_performance_chart(self,
                                     per_class_metrics: Dict[str, float],
                                     metric_name: str = 'Average Precision',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart showing per-class performance.
        
        Args:
            per_class_metrics: Dictionary mapping class names to metric values
            metric_name: Name of the metric being visualized
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Prepare data
        classes = []
        values = []
        colors = []
        
        for class_name in self.class_names:
            if class_name in per_class_metrics:
                classes.append(class_name)
                values.append(per_class_metrics[class_name])
                
                # Color coding
                if class_name in self.safety_critical_classes:
                    colors.append('red')  # Safety-critical classes in red
                else:
                    colors.append('steelblue')
        
        # Create bar chart
        bars = ax.bar(classes, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize chart
        ax.set_title(f'Per-Class {metric_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Object Classes', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        safety_patch = patches.Patch(color='red', alpha=0.7, label='Safety-Critical')
        regular_patch = patches.Patch(color='steelblue', alpha=0.7, label='Regular')
        ax.legend(handles=[safety_patch, regular_patch], loc='upper right')
        
        # Skip tight_layout to avoid sizing issues
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=80, bbox_inches=None)
        
        return fig
    
    def save_detection_samples(self,
                              images: List[np.ndarray],
                              ground_truths: List[List[Dict]],
                              predictions: List[List[Dict]],
                              image_ids: List[str],
                              output_dir: str,
                              max_samples: int = 20) -> None:
        """
        Save sample detection visualizations to files.
        
        Args:
            images: List of input images
            ground_truths: List of ground truth annotations
            predictions: List of predictions
            image_ids: List of image identifiers
            output_dir: Directory to save images
            max_samples: Maximum number of samples to save
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        num_samples = min(len(images), max_samples)
        
        for i in range(num_samples):
            # Create comparison
            comparison = self.visualize_detections(
                images[i], ground_truths[i], predictions[i],
                confidence_threshold=0.3  # Lower threshold for visualization
            )
            
            # Save image
            save_file = output_path / f"detection_sample_{image_ids[i]}.jpg"
            cv2.imwrite(str(save_file), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {num_samples} detection samples to {output_dir}")
    
    def generate_legend(self) -> np.ndarray:
        """Generate a legend showing class colors and labels."""
        # Create image for legend
        legend_height = 50 * len(self.class_names) + 100
        legend_width = 400
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        img_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        # Add title
        draw.text((10, 10), "Class Legend", fill='black', font=font)
        
        # Add class entries
        y_pos = 50
        for class_name in self.class_names:
            color = self.class_colors[class_name]
            
            # Draw color box
            draw.rectangle([(10, y_pos), (40, y_pos + 25)], fill=color, outline='black')
            
            # Add class name
            label = class_name
            if class_name in self.safety_critical_classes:
                label += " (Safety-Critical)"
            
            draw.text((50, y_pos + 3), label, fill='black', font=font)
            y_pos += 40
        
        return np.array(img_pil)