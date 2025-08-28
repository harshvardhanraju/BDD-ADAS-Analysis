"""
BDD100K Dataset Preparation for DETR Training

This module handles loading and preprocessing BDD100K data for DETR model training,
including proper format conversion and augmentation strategies.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BDD100KDETRDataset(Dataset):
    """
    BDD100K Dataset for DETR training.
    
    Converts BDD100K annotations to DETR-compatible format with proper
    data augmentation and class mapping.
    """
    
    def __init__(
        self,
        annotations_file: str,
        images_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        max_objects: int = 100,
        use_enhanced_augmentation: bool = True,
        augmentation_strength: str = 'medium'
    ):
        """
        Initialize BDD100K DETR dataset.
        
        Args:
            annotations_file: Path to processed annotations CSV
            images_root: Root directory containing images
            split: Dataset split ('train' or 'val')
            image_size: Target image size (H, W)
            augment: Whether to apply data augmentation
            max_objects: Maximum number of objects per image
            use_enhanced_augmentation: Whether to use enhanced augmentation strategies
            augmentation_strength: Augmentation strength ('light', 'medium', 'strong')
        """
        self.annotations_file = Path(annotations_file)
        self.images_root = Path(images_root)
        self.split = split
        self.use_enhanced_augmentation = use_enhanced_augmentation
        self.augmentation_strength = augmentation_strength
        self.image_size = image_size
        self.augment = augment
        self.max_objects = max_objects
        
        # Complete 10-class mapping for BDD100K
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
        self.num_classes = len(self.class_mapping)
        
        # Complete class names for enhanced augmentation (consistent naming)
        self.class_names = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign']
        
        # Load annotations
        self._load_annotations()
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Loaded {len(self.image_annotations)} images for {split} split")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_annotations(self):
        """Load and process annotations from CSV file."""
        try:
            # Load processed annotations
            df = pd.read_csv(self.annotations_file)
            
            # Filter by split
            df = df[df['split'] == self.split].copy()
            
            # Group by image
            self.image_annotations = []
            
            for image_name, group in df.groupby('image_name'):
                # Skip images without object annotations
                objects = group[group['category'].notna()].copy()
                if len(objects) == 0:
                    continue
                
                # Map categories to class IDs
                valid_objects = []
                for _, obj in objects.iterrows():
                    category = obj['category']
                    if category in self.class_mapping:
                        valid_objects.append({
                            'class_id': self.class_mapping[category],
                            'category': category,
                            'bbox': [
                                float(obj['bbox_x1']),
                                float(obj['bbox_y1']),
                                float(obj['bbox_x2']),
                                float(obj['bbox_y2'])
                            ],
                            'area': float(obj['bbox_area']),
                            'iscrowd': 0
                        })
                
                if valid_objects:
                    self.image_annotations.append({
                        'image_name': image_name,
                        'split': self.split,
                        'annotations': valid_objects
                    })
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image transforms and augmentations."""
        # TEMPORARY FIX: Disable enhanced augmentation due to bbox coordinate bug
        if False:  # self.use_enhanced_augmentation:
            # Use enhanced augmentation from enhanced_augmentation module
            try:
                from .enhanced_augmentation import create_enhanced_transforms
                enhanced_transforms = create_enhanced_transforms(
                    image_size=self.image_size,
                    split=self.split,
                    augmentation_strength=self.augmentation_strength,
                    class_names=self.class_names
                )
                self.transform = enhanced_transforms.transform
                print(f"✅ Using enhanced augmentation (strength: {self.augmentation_strength})")
                return
            except ImportError:
                print("⚠️  Enhanced augmentation not available, falling back to basic augmentation")
        
        if self.augment and self.split == 'train':
            # Basic training augmentations
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.2),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            # Validation transforms
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        class_counts = {}
        for img_ann in self.image_annotations:
            for ann in img_ann['annotations']:
                category = ann['category']
                class_counts[category] = class_counts.get(category, 0) + 1
        return class_counts
    
    def _load_image(self, image_name: str) -> np.ndarray:
        """
        Smart image loader that handles mixed directory structures.
        
        Handles BDD100K's inconsistent directory structure where:
        - Val images are directly in val/ folder
        - Most train images are in train/trainA/, train/trainB/ subdirectories
        - Some train images are directly in train/ folder
        
        Args:
            image_name: Name of the image file
            
        Returns:
            Image as numpy array in RGB format
        """
        # Try direct path first (works for val, some train images)
        image_path = self.images_root / self.split / image_name
        
        if image_path.exists():
            return self._read_image_file(image_path)
        
        # If not found, search in subdirectories (trainA, trainB, testA, testB)
        split_dir = self.images_root / self.split
        if split_dir.exists():
            for subdir in split_dir.iterdir():
                if subdir.is_dir():
                    candidate_path = subdir / image_name
                    if candidate_path.exists():
                        return self._read_image_file(candidate_path)
        
        # Fallback: return placeholder with warning
        print(f"Warning: Image not found: {image_name}, using placeholder")
        return np.zeros((720, 1280, 3), dtype=np.uint8)  # Standard BDD100K size
    
    def _read_image_file(self, image_path: Path) -> np.ndarray:
        """
        Read and convert image file to RGB numpy array.
        
        Args:
            image_path: Full path to the image file
            
        Returns:
            Image as numpy array in RGB format
        """
        # Load image using OpenCV and convert to RGB
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Cannot read image: {image_path}, using placeholder")
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _normalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Normalize bounding box coordinates to [0, 1] range.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            Normalized bounding box [x_center, y_center, width, height]
        """
        x1, y1, x2, y2 = bbox
        
        # Convert to center coordinates and normalize
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return [x_center, y_center, width, height]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item from dataset.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image, target) where target contains DETR-format annotations
        """
        img_annotation = self.image_annotations[idx]
        image_name = img_annotation['image_name']
        annotations = img_annotation['annotations']
        
        # Load image
        image = self._load_image(image_name)
        original_height, original_width = image.shape[:2]
        
        # Prepare bboxes and labels for augmentation
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            bboxes.append(ann['bbox'])
            class_labels.append(ann['class_id'])
        
        # Apply transforms
        if bboxes:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            # No objects, just transform image
            transformed = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])(image=image)
            image = transformed['image']
        
        # Convert to DETR format
        target = {}
        
        if bboxes and class_labels:
            # Convert bboxes to normalized center format for DETR
            # Note: bboxes are already in resized image coordinates after albumentations transform
            normalized_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                # Convert to normalized center format for DETR (already in resized image space)
                x_center = (x1 + x2) / 2.0 / self.image_size[1]
                y_center = (y1 + y2) / 2.0 / self.image_size[0] 
                width = (x2 - x1) / self.image_size[1]
                height = (y2 - y1) / self.image_size[0]
                normalized_bboxes.append([x_center, y_center, width, height])
            
            target['class_labels'] = torch.tensor(class_labels, dtype=torch.long)
            target['boxes'] = torch.tensor(normalized_bboxes, dtype=torch.float32)
        else:
            # No objects
            target['class_labels'] = torch.zeros((0,), dtype=torch.long)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        
        # Add additional metadata
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.tensor([ann.get('area', 0) for ann in annotations], dtype=torch.float32)
        target['iscrowd'] = torch.tensor([ann.get('iscrowd', 0) for ann in annotations], dtype=torch.long)
        target['orig_size'] = torch.tensor([original_height, original_width])
        target['size'] = torch.tensor(self.image_size)
        
        return image, target
    
    def collate_fn(self, batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Custom collate function for DETR training.
        
        Args:
            batch: List of (image, target) tuples
            
        Returns:
            Batched images and list of targets
        """
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets


def create_bdd_dataloaders(
    train_annotations: str,
    val_annotations: str,
    images_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for BDD100K.
    
    Args:
        train_annotations: Path to training annotations CSV
        val_annotations: Path to validation annotations CSV
        images_root: Root directory containing images
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Target image size (H, W)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = BDD100KDETRDataset(
        annotations_file=train_annotations,
        images_root=images_root,
        split='train',
        image_size=image_size,
        augment=True
    )
    
    val_dataset = BDD100KDETRDataset(
        annotations_file=val_annotations,
        images_root=images_root,
        split='val',
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test dataset creation
    print("Testing BDD100K DETR Dataset...")
    
    # Test paths (adjust as needed)
    train_ann = "data/analysis/processed/train_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    if Path(train_ann).exists() and Path(images_root).exists():
        dataset = BDD100KDETRDataset(
            annotations_file=train_ann,
            images_root=images_root,
            split='train',
            image_size=(512, 512)
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, target = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Number of objects: {len(target['class_labels'])}")
            print(f"Classes: {target['class_labels']}")
            print(f"Boxes shape: {target['boxes'].shape}")
            
        print("Dataset creation successful!")
    else:
        print("Data files not found. Please ensure annotations and images are available.")