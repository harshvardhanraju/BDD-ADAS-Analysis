#!/usr/bin/env python3
"""
Complete BDD100K Dataset Parser for All 10 Object Detection Classes

This module provides comprehensive parsing functionality for the BDD100K dataset,
extracting all 10 object detection classes with proper data validation and analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BDD100KCompleteParser:
    """
    Complete parser for BDD100K dataset with all 10 object detection classes.
    
    The BDD100K dataset contains 10 object detection classes:
    1. pedestrian - People walking on streets
    2. rider - People on bicycles/motorcycles  
    3. car - Standard passenger vehicles
    4. truck - Large commercial vehicles
    5. bus - Public transportation buses
    6. train - Railway trains
    7. motorcycle - Two-wheeled motor vehicles
    8. bicycle - Human-powered bicycles
    9. traffic light - Traffic control lights
    10. traffic sign - Road signage
    """
    
    # Complete set of 10 BDD100K object detection classes (using actual JSON category names)
    DETECTION_CLASSES = [
        "person",      # pedestrian in BDD100K JSON
        "rider", 
        "car",
        "truck",
        "bus",
        "train",
        "motor",       # motorcycle in BDD100K JSON
        "bike",        # bicycle in BDD100K JSON
        "traffic light",
        "traffic sign"
    ]
    
    # Mapping from JSON categories to standard names
    CATEGORY_MAPPING = {
        "person": "pedestrian",
        "rider": "rider",
        "car": "car", 
        "truck": "truck",
        "bus": "bus",
        "train": "train",
        "motor": "motorcycle",
        "bike": "bicycle",
        "traffic light": "traffic light",
        "traffic sign": "traffic sign"
    }
    
    # Class mapping for model training (0-indexed)
    CLASS_TO_ID = {cls: idx for idx, cls in enumerate(DETECTION_CLASSES)}
    ID_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_ID.items()}
    
    def __init__(self, data_root: str):
        """
        Initialize parser with dataset root directory.
        
        Args:
            data_root: Path to BDD100K dataset root directory
        """
        self.data_root = Path(data_root)
        self.labels_dir = self.data_root / "labels"
        self.images_dir = self.data_root / "images" / "100k"
        
        logger.info(f"Initialized BDD100K parser with root: {self.data_root}")
        logger.info(f"Number of detection classes: {len(self.DETECTION_CLASSES)}")
        
    def parse_split(self, split: str = "train") -> pd.DataFrame:
        """
        Parse a specific split of the dataset.
        
        Args:
            split: Dataset split ('train' or 'val')
            
        Returns:
            DataFrame with complete annotation data
        """
        logger.info(f"Parsing {split} split...")
        
        # Load JSON annotations
        labels_file = self.labels_dir / f"bdd100k_labels_images_{split}.json"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
            
        with open(labels_file, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data)} images from {split} split")
        
        # Parse annotations
        annotations = []
        
        for img_data in tqdm(data, desc=f"Processing {split} images"):
            image_name = img_data["name"]
            
            # Extract image-level attributes
            img_attrs = img_data.get("attributes", {})
            weather = img_attrs.get("weather", "unknown")
            scene = img_attrs.get("scene", "unknown") 
            timeofday = img_attrs.get("timeofday", "unknown")
            
            # Extract objects
            labels = img_data.get("labels", [])
            
            for obj_idx, label in enumerate(labels):
                category = label.get("category", "")
                
                # Only process detection classes
                if category not in self.DETECTION_CLASSES:
                    continue
                    
                # Map to standard category name
                standard_category = self.CATEGORY_MAPPING.get(category, category)
                    
                # Extract bounding box
                box2d = label.get("box2d", {})
                if not box2d:
                    continue
                    
                x1, y1 = box2d.get("x1", 0), box2d.get("y1", 0)
                x2, y2 = box2d.get("x2", 0), box2d.get("y2", 0)
                
                # Basic validation
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # Extract object attributes
                obj_attrs = label.get("attributes", {})
                occluded = obj_attrs.get("occluded", False)
                truncated = obj_attrs.get("truncated", False) 
                crowd = obj_attrs.get("crowd", False)
                traffic_light_color = obj_attrs.get("trafficLightColor", "none")
                
                # Calculate derived metrics
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                annotation = {
                    # Image information
                    'image_name': image_name,
                    'split': split,
                    
                    # Image attributes
                    'img_attr_weather': weather,
                    'img_attr_scene': scene, 
                    'img_attr_timeofday': timeofday,
                    
                    # Object information
                    'object_id': obj_idx,
                    'category': standard_category,
                    'class_id': self.CLASS_TO_ID[category],
                    
                    # Bounding box coordinates
                    'bbox_x1': x1,
                    'bbox_y1': y1, 
                    'bbox_x2': x2,
                    'bbox_y2': y2,
                    
                    # Derived metrics
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'bbox_area': bbox_area,
                    'bbox_aspect_ratio': bbox_aspect_ratio,
                    'center_x': center_x,
                    'center_y': center_y,
                    
                    # Object attributes
                    'occluded': occluded,
                    'truncated': truncated,
                    'crowd': crowd,
                    'obj_attr_occluded': occluded,
                    'obj_attr_truncated': truncated,
                    'obj_attr_trafficLightColor': traffic_light_color
                }
                
                annotations.append(annotation)
        
        df = pd.DataFrame(annotations)
        
        # Add image-level statistics
        if not df.empty:
            img_stats = df.groupby('image_name').size().reset_index(name='total_objects')
            df = df.merge(img_stats, on='image_name')
        
        logger.info(f"Parsed {len(df)} annotations from {len(df['image_name'].unique() if not df.empty else 0)} images")
        logger.info(f"Class distribution:")
        if not df.empty:
            class_counts = df['category'].value_counts()
            for cls, count in class_counts.items():
                logger.info(f"  {cls}: {count}")
        
        return df
        
    def parse_complete_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse both train and validation splits.
        
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info("Parsing complete BDD100K dataset...")
        
        train_df = self.parse_split("train")
        val_df = self.parse_split("val")
        
        logger.info(f"Complete dataset parsed:")
        logger.info(f"  Training: {len(train_df)} annotations")
        logger.info(f"  Validation: {len(val_df)} annotations") 
        logger.info(f"  Total: {len(train_df) + len(val_df)} annotations")
        
        return train_df, val_df
        
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           output_dir: str = "data/analysis/processed_10class"):
        """
        Save processed data to CSV files.
        
        Args:
            train_df: Training annotations DataFrame
            val_df: Validation annotations DataFrame  
            output_dir: Output directory for processed files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        train_file = output_path / "train_annotations_10class.csv"
        val_file = output_path / "val_annotations_10class.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        
        logger.info(f"Saved processed data:")
        logger.info(f"  Training: {train_file}")
        logger.info(f"  Validation: {val_file}")
        
        # Save class mapping
        class_mapping_file = output_path / "class_mapping.json"
        standard_classes = [self.CATEGORY_MAPPING[cls] for cls in self.DETECTION_CLASSES]
        class_info = {
            'json_categories': self.DETECTION_CLASSES,
            'standard_classes': standard_classes,
            'category_mapping': self.CATEGORY_MAPPING,
            'class_to_id': self.CLASS_TO_ID,
            'id_to_class': self.ID_TO_CLASS,
            'num_classes': len(self.DETECTION_CLASSES)
        }
        
        with open(class_mapping_file, 'w') as f:
            json.dump(class_info, f, indent=2)
            
        logger.info(f"  Class mapping: {class_mapping_file}")
        
    def get_dataset_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics.
        
        Args:
            train_df: Training annotations DataFrame
            val_df: Validation annotations DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        stats = {
            'dataset_info': {
                'total_annotations': len(combined_df),
                'total_images': combined_df['image_name'].nunique(),
                'num_classes': len(self.DETECTION_CLASSES),
                'classes': [self.CATEGORY_MAPPING[cls] for cls in self.DETECTION_CLASSES]
            },
            'split_info': {
                'train': {
                    'annotations': len(train_df),
                    'images': train_df['image_name'].nunique() if not train_df.empty else 0,
                    'avg_objects_per_image': train_df.groupby('image_name').size().mean() if not train_df.empty else 0
                },
                'val': {
                    'annotations': len(val_df), 
                    'images': val_df['image_name'].nunique() if not val_df.empty else 0,
                    'avg_objects_per_image': val_df.groupby('image_name').size().mean() if not val_df.empty else 0
                }
            },
            'class_distribution': {
                'overall': combined_df['category'].value_counts().to_dict() if not combined_df.empty else {},
                'train': train_df['category'].value_counts().to_dict() if not train_df.empty else {},
                'val': val_df['category'].value_counts().to_dict() if not val_df.empty else {}
            }
        }
        
        return stats


def main():
    """Main function to run complete dataset parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse BDD100K dataset with all 10 classes')
    parser.add_argument('--data-root', type=str, 
                       default='data/raw/bdd100k_labels_release/bdd100k',
                       help='Path to BDD100K dataset root')
    parser.add_argument('--output-dir', type=str,
                       default='data/analysis/processed_10class',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Initialize parser
    bdd_parser = BDD100KCompleteParser(args.data_root)
    
    # Parse complete dataset
    train_df, val_df = bdd_parser.parse_complete_dataset()
    
    # Save processed data
    bdd_parser.save_processed_data(train_df, val_df, args.output_dir)
    
    # Generate and save statistics
    stats = bdd_parser.get_dataset_statistics(train_df, val_df)
    
    stats_file = Path(args.output_dir) / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics saved to: {stats_file}")
    
    print("\n" + "="*60)
    print("BDD100K COMPLETE DATASET PARSING COMPLETE")
    print("="*60)
    print(f"Total annotations: {stats['dataset_info']['total_annotations']:,}")
    print(f"Total images: {stats['dataset_info']['total_images']:,}")
    print(f"Number of classes: {stats['dataset_info']['num_classes']}")
    print("\nClass distribution (combined):")
    for cls, count in sorted(stats['class_distribution']['overall'].items(), 
                           key=lambda x: x[1], reverse=True):
        percentage = (count / stats['dataset_info']['total_annotations']) * 100
        print(f"  {cls:12}: {count:7,} ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()