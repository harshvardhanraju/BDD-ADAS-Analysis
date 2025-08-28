"""
BDD100K FiftyOne Integration Script

This script converts BDD100K dataset and model predictions to FiftyOne format
for advanced analysis using FiftyOne Brain capabilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.detr_dataset import BDD100KDETRDataset
from models.detr_model import BDD100KDETR, BDD100KDetrConfig


class BDD100KFiftyOneConverter:
    """Converts BDD100K data and predictions to FiftyOne dataset format."""
    
    def __init__(
        self,
        annotations_file: str,
        images_root: str,
        model_checkpoint: Optional[str] = None,
        dataset_name: str = "bdd100k_analysis",
        subset_size: Optional[int] = None
    ):
        """
        Initialize the converter.
        
        Args:
            annotations_file: Path to processed annotations CSV
            images_root: Root directory containing images
            model_checkpoint: Path to trained model checkpoint for predictions
            dataset_name: Name for the FiftyOne dataset
            subset_size: Number of images to include (None for all)
        """
        self.annotations_file = annotations_file
        self.images_root = Path(images_root)
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.subset_size = subset_size
        
        # BDD100K class mapping
        self.class_mapping = {
            'pedestrian': 0,
            'rider': 1,
            'car': 2,
            'truck': 3,
            'bus': 4,
            'train': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic light': 8,
            'traffic sign': 9
        }
        
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Safety-critical classes
        self.safety_critical_classes = {'pedestrian', 'rider', 'bicycle', 'motorcycle'}
        
        # Load model if checkpoint provided
        self.model = None
        if model_checkpoint and os.path.exists(model_checkpoint):
            self._load_model()
    
    def _load_model(self):
        """Load trained DETR model for predictions."""
        try:
            print(f"Loading model from {self.model_checkpoint}")
            
            # Initialize model with config
            config = BDD100KDetrConfig()
            self.model = BDD100KDETR(config=config, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_checkpoint, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def create_fiftyone_dataset(self) -> fo.Dataset:
        """
        Create FiftyOne dataset with ground truth annotations.
        
        Returns:
            FiftyOne dataset with ground truth labels
        """
        print(f"Creating FiftyOne dataset: {self.dataset_name}")
        
        # Delete existing dataset if it exists
        if self.dataset_name in fo.list_datasets():
            fo.delete_dataset(self.dataset_name)
        
        # Create new dataset
        dataset = fo.Dataset(self.dataset_name)
        
        # Load annotations
        df = pd.read_csv(self.annotations_file)
        
        # Apply subset if specified
        if self.subset_size:
            unique_images = df['image_name'].unique()[:self.subset_size]
            df = df[df['image_name'].isin(unique_images)]
        
        # Group by image
        images_processed = 0
        for image_name, group in df.groupby('image_name'):
            image_path = self.images_root / image_name
            
            if not image_path.exists():
                continue
            
            # Create sample
            sample = fo.Sample(filepath=str(image_path))
            
            # Add metadata
            sample["split"] = group['split'].iloc[0]
            
            # Process ground truth detections
            detections = []
            objects = group[group['category'].notna()]
            
            for _, obj in objects.iterrows():
                category = obj['category']
                if category not in self.class_mapping:
                    continue
                
                # Convert bbox to relative coordinates [x, y, width, height]
                bbox = [
                    float(obj['bbox_x1']) / obj['image_width'],
                    float(obj['bbox_y1']) / obj['image_height'],
                    (float(obj['bbox_x2']) - float(obj['bbox_x1'])) / obj['image_width'],
                    (float(obj['bbox_y2']) - float(obj['bbox_y1'])) / obj['image_height']
                ]
                
                # Create detection
                detection = fo.Detection(
                    label=category,
                    bounding_box=bbox,
                    confidence=1.0,  # Ground truth has confidence 1.0
                    area=float(obj['bbox_area']),
                    iscrowd=0
                )
                
                # Add safety-critical flag
                detection["is_safety_critical"] = category in self.safety_critical_classes
                
                detections.append(detection)
            
            # Add detections to sample
            sample["ground_truth"] = fo.Detections(detections=detections)
            
            # Add sample to dataset
            dataset.add_sample(sample)
            images_processed += 1
            
            if images_processed % 100 == 0:
                print(f"Processed {images_processed} images")
        
        print(f"Created dataset with {len(dataset)} samples")
        return dataset
    
    def add_model_predictions(self, dataset: fo.Dataset, confidence_threshold: float = 0.05):
        """
        Add model predictions to the dataset.
        
        Args:
            dataset: FiftyOne dataset to add predictions to
            confidence_threshold: Minimum confidence for detections
        """
        if not self.model:
            print("No model loaded, skipping prediction generation")
            return
        
        print("Adding model predictions to dataset")
        
        # Setup transforms for model input
        transform = fo.transforms.ToTensor()
        
        predictions_added = 0
        for sample in dataset:
            try:
                # Load and preprocess image
                image = Image.open(sample.filepath).convert('RGB')
                original_size = image.size  # (width, height)
                
                # Resize to model input size (416x416 based on your config)
                image_tensor = transform(image.resize((416, 416)))
                
                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                
                # Process predictions
                predictions = self._process_model_outputs(
                    outputs, original_size, confidence_threshold
                )
                
                # Add predictions to sample
                sample["predictions"] = fo.Detections(detections=predictions)
                sample.save()
                
                predictions_added += 1
                
                if predictions_added % 50 == 0:
                    print(f"Added predictions for {predictions_added} images")
                
            except Exception as e:
                print(f"Error processing {sample.filepath}: {e}")
                continue
        
        print(f"Added predictions for {predictions_added} samples")
    
    def _process_model_outputs(
        self, 
        outputs: Dict, 
        original_size: Tuple[int, int],
        confidence_threshold: float
    ) -> List[fo.Detection]:
        """
        Process raw model outputs into FiftyOne detections.
        
        Args:
            outputs: Raw model outputs
            original_size: Original image size (width, height)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of FiftyOne Detection objects
        """
        detections = []
        
        try:
            # Extract predictions (assuming DETR format)
            pred_logits = outputs['pred_logits'][0]  # Remove batch dimension
            pred_boxes = outputs['pred_boxes'][0]    # Remove batch dimension
            
            # Get probabilities
            pred_probs = torch.softmax(pred_logits, dim=-1)
            
            # Get max probability and class for each prediction
            max_probs, pred_classes = torch.max(pred_probs, dim=-1)
            
            # Filter by confidence threshold
            confident_mask = max_probs > confidence_threshold
            
            # Process confident predictions
            for i in range(len(pred_classes)):
                if not confident_mask[i]:
                    continue
                
                class_id = pred_classes[i].item()
                confidence = max_probs[i].item()
                
                # Skip background class (assuming class 10 is background or no-object)
                if class_id >= len(self.id_to_class):
                    continue
                
                # Get class name
                class_name = self.id_to_class[class_id]
                
                # Convert box coordinates from [cx, cy, w, h] to [x, y, w, h]
                # DETR outputs normalized coordinates
                box = pred_boxes[i].tolist()
                cx, cy, w, h = box
                
                # Convert to top-left format
                x = cx - w / 2
                y = cy - h / 2
                
                # Ensure coordinates are within [0, 1]
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                # Create detection
                detection = fo.Detection(
                    label=class_name,
                    bounding_box=[x, y, w, h],
                    confidence=confidence
                )
                
                # Add safety-critical flag
                detection["is_safety_critical"] = class_name in self.safety_critical_classes
                
                detections.append(detection)
        
        except Exception as e:
            print(f"Error processing model outputs: {e}")
        
        return detections
    
    def compute_embeddings(self, dataset: fo.Dataset, patches_field: str = "ground_truth"):
        """
        Compute embeddings for ground truth or prediction patches.
        
        Args:
            dataset: FiftyOne dataset
            patches_field: Field containing detections ("ground_truth" or "predictions")
        """
        print(f"Computing embeddings for {patches_field}")
        
        try:
            # Compute visualization embeddings for object patches
            fob.compute_visualization(
                dataset,
                patches_field=patches_field,
                brain_key=f"{patches_field}_viz",
                method="umap",
                num_dims=2,
                batch_size=32
            )
            print(f"Embeddings computed successfully for {patches_field}")
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
    
    def compute_similarity(self, dataset: fo.Dataset, patches_field: str = "ground_truth"):
        """
        Compute similarity index for patches.
        
        Args:
            dataset: FiftyOne dataset
            patches_field: Field containing detections
        """
        print(f"Computing similarity for {patches_field}")
        
        try:
            fob.compute_similarity(
                dataset,
                patches_field=patches_field,
                brain_key=f"{patches_field}_similarity",
                batch_size=32
            )
            print(f"Similarity computed successfully for {patches_field}")
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
    
    def analyze_mistakes(self, dataset: fo.Dataset):
        """
        Analyze detection mistakes using FiftyOne Brain.
        
        Args:
            dataset: FiftyOne dataset with ground truth and predictions
        """
        if "predictions" not in dataset.get_field_schema():
            print("No predictions found in dataset. Skipping mistake analysis.")
            return
        
        print("Analyzing detection mistakes")
        
        try:
            # Compute mistakenness for predictions
            fob.compute_mistakenness(
                dataset,
                pred_field="predictions",
                label_field="ground_truth",
                brain_key="mistakenness"
            )
            print("Mistake analysis completed successfully")
            
        except Exception as e:
            print(f"Error analyzing mistakes: {e}")
    
    def get_analysis_views(self, dataset: fo.Dataset) -> Dict[str, fo.DatasetView]:
        """
        Create useful views for analysis.
        
        Args:
            dataset: FiftyOne dataset
            
        Returns:
            Dictionary of named dataset views
        """
        views = {}
        
        # Safety-critical objects
        views["safety_critical"] = dataset.filter_labels(
            "ground_truth",
            fo.ViewField("is_safety_critical") == True
        )
        
        # High confidence predictions
        if "predictions" in dataset.get_field_schema():
            views["high_confidence_predictions"] = dataset.filter_labels(
                "predictions",
                fo.ViewField("confidence") > 0.5
            )
            
            # Low confidence predictions
            views["low_confidence_predictions"] = dataset.filter_labels(
                "predictions",
                fo.ViewField("confidence") < 0.1
            )
        
        # Mistakes (if available)
        if "mistakenness" in dataset.get_brain_info():
            views["potential_mistakes"] = dataset.sort_by("mistakenness", reverse=True).limit(100)
        
        return views


def main():
    """Main function to run the FiftyOne integration."""
    
    # Configuration
    annotations_file = "data/analysis/processed_10class_corrected/val_annotations_10class.csv"
    images_root = "data/raw/bdd100k/images/10k/val"  # Update this path
    model_checkpoint = "checkpoints/complete_10class_demo/checkpoint_epoch_048.pth"
    
    # Create converter with small subset for testing
    converter = BDD100KFiftyOneConverter(
        annotations_file=annotations_file,
        images_root=images_root,
        model_checkpoint=model_checkpoint,
        dataset_name="bdd100k_test",
        subset_size=100  # Start with 100 images for testing
    )
    
    # Create FiftyOne dataset
    dataset = converter.create_fiftyone_dataset()
    
    # Add model predictions
    converter.add_model_predictions(dataset)
    
    # Compute brain features
    converter.compute_embeddings(dataset, "ground_truth")
    if "predictions" in dataset.get_field_schema():
        converter.compute_embeddings(dataset, "predictions")
        converter.compute_similarity(dataset, "predictions")
        converter.analyze_mistakes(dataset)
    
    # Create analysis views
    views = converter.get_analysis_views(dataset)
    
    print("\nDataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Available views: {list(views.keys())}")
    print("\nTo explore the dataset, run:")
    print(f"session = fo.launch_app(dataset=fo.load_dataset('{converter.dataset_name}'))")


if __name__ == "__main__":
    main()