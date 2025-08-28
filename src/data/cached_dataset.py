"""
Cached BDD100K Dataset for Fast DETR Training

This module implements a cached dataset that loads all images into memory
before training, eliminating file I/O during training and handling missing images.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CachedBDD100KDataset(Dataset):
    """
    Cached BDD100K dataset that loads all images into memory for fast training.
    """
    
    def __init__(
        self,
        annotations_file: str,
        images_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        max_objects: int = 100,
        cache_dir: str = "dataset_cache",
        max_images: int = None
    ):
        """
        Initialize cached dataset.
        
        Args:
            annotations_file: Path to processed annotations CSV
            images_root: Root directory containing images
            split: Dataset split ('train' or 'val')
            image_size: Target image size (H, W)
            augment: Whether to apply data augmentation
            max_objects: Maximum number of objects per image
            cache_dir: Directory to save/load cached data
            max_images: Maximum number of images to cache (None for all)
        """
        self.annotations_file = Path(annotations_file)
        self.images_root = Path(images_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.max_objects = max_objects
        self.max_images = max_images
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Class mapping for BDD100K
        self.class_mapping = {
            'car': 0, 'truck': 1, 'bus': 2, 'train': 3,
            'rider': 4, 'traffic sign': 5, 'traffic light': 6
        }
        
        self.class_names = ['car', 'truck', 'bus', 'train', 'rider', 'traffic_sign', 'traffic_light']
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Setup transforms
        self._setup_transforms()
        
        # Load or create cache
        self._load_or_create_cache()
        
        print(f"✅ Cached dataset ready: {len(self.cached_data)} images")
    
    def _setup_transforms(self):
        """Setup image transforms and augmentations."""
        if self.augment and self.split == 'train':
            # Enhanced training augmentations to reduce overfitting
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                # More aggressive brightness/contrast
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=0.6
                ),
                # Enhanced color augmentation
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=15,
                    p=0.5
                ),
                # Additional geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=10,
                    p=0.4
                ),
                # Weather/noise augmentations
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                # Cutout for regularization
                A.CoarseDropout(
                    max_holes=2,
                    max_height=32,
                    max_width=32,
                    p=0.2
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.1  # Reduced min visibility for more augmented samples
            ))
        else:
            # Validation transforms
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _get_cache_file(self) -> Path:
        """Get cache file path."""
        if self.max_images:
            cache_name = f"{self.split}_cache_{self.image_size[0]}x{self.image_size[1]}_{self.max_images}img.pkl"
        else:
            cache_name = f"{self.split}_cache_{self.image_size[0]}x{self.image_size[1]}.pkl"
        return self.cache_dir / cache_name
    
    def _load_or_create_cache(self):
        """Load existing cache or create new one."""
        cache_file = self._get_cache_file()
        
        if cache_file.exists():
            print(f"📁 Loading cached dataset from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.cached_data = cache_data['data']
                    self.class_distribution = cache_data['class_distribution']
                print(f"✅ Loaded {len(self.cached_data)} cached images")
                self._print_class_distribution()
                return
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}, creating new cache...")
        
        print(f"🔄 Creating cache for {self.split} split...")
        self._create_cache()
        
        # Save cache
        cache_data = {
            'data': self.cached_data,
            'class_distribution': self.class_distribution
        }
        
        print(f"💾 Saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Cache created and saved: {len(self.cached_data)} images")
    
    def _create_cache(self):
        """Create cache by loading all images and annotations."""
        # Load annotations
        print("📄 Loading annotations...")
        df = pd.read_csv(self.annotations_file, low_memory=False)
        df = df[df['split'] == self.split].copy()
        
        # Group by image
        self.cached_data = []
        self.class_distribution = {name: 0 for name in self.class_names}
        
        failed_images = 0
        total_images = 0
        
        print("🖼️  Loading and caching images...")
        
        # Limit images if max_images is specified
        image_groups = list(df.groupby('image_name'))
        if self.max_images:
            image_groups = image_groups[:self.max_images]
            print(f"   • Limiting to first {self.max_images} images")
        
        for image_name, group in tqdm(image_groups, desc="Caching images"):
            total_images += 1
            
            # Load image - check multiple possible subdirectories
            image = self._find_and_load_image(image_name)
            
            if image is None:
                failed_images += 1
                continue
            
            # Process annotations
            objects = group[group['category'].notna()].copy()
            valid_objects = []
            
            for _, obj in objects.iterrows():
                category = obj['category']
                if category in self.class_mapping:
                    try:
                        bbox = [
                            float(obj['bbox_x1']), float(obj['bbox_y1']),
                            float(obj['bbox_x2']), float(obj['bbox_y2'])
                        ]
                        
                        # Basic validation
                        if all(x >= 0 for x in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            class_id = self.class_mapping[category]
                            valid_objects.append({
                                'class_id': class_id,
                                'category': category,
                                'bbox': bbox
                            })
                            
                            # Update class distribution
                            class_name = self.class_names[class_id]
                            self.class_distribution[class_name] += 1
                            
                    except (ValueError, TypeError):
                        continue
            
            # Only cache images with valid objects
            if valid_objects:
                self.cached_data.append({
                    'image_name': image_name,
                    'image': image,  # Store the actual image array
                    'objects': valid_objects,
                    'original_size': (image.shape[0], image.shape[1])
                })
        
        print(f"📊 Caching complete:")
        print(f"   • Total images processed: {total_images}")
        print(f"   • Successfully cached: {len(self.cached_data)}")
        print(f"   • Failed to load: {failed_images}")
        
        self._print_class_distribution()
    
    def _find_and_load_image(self, image_name: str) -> Optional[np.ndarray]:
        """Find and load image from multiple possible subdirectories."""
        # BDD100K has nested directory structure
        split_dir = self.images_root / self.split
        
        # Possible subdirectories in BDD100K structure
        subdirs = ['trainA', 'trainB', 'trainVal', 'testA', 'testB', 'val', '']
        
        for subdir in subdirs:
            if subdir:
                image_path = split_dir / subdir / image_name
            else:
                image_path = split_dir / image_name
            
            try:
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return image
            except Exception:
                continue
        
        return None
    
    def _load_image_safe(self, image_path: Path) -> Optional[np.ndarray]:
        """Safely load image, return None if fails."""
        try:
            if not image_path.exists():
                return None
            
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception:
            return None
    
    def _print_class_distribution(self):
        """Print class distribution."""
        total_objects = sum(self.class_distribution.values())
        print("Class distribution:")
        for class_name, count in self.class_distribution.items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"   • {class_name}: {count:,} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.cached_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get item from cached dataset."""
        item = self.cached_data[idx]
        
        # Get cached image (already in RGB)
        image = item['image'].copy()
        objects = item['objects']
        original_size = item['original_size']
        
        # Prepare bboxes and labels for transform
        if objects:
            bboxes = [obj['bbox'] for obj in objects]
            class_labels = [obj['class_id'] for obj in objects]
            
            # Apply transforms
            try:
                if self.augment and self.split == 'train':
                    transformed = self.transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_labels = transformed['class_labels']
                    
                else:
                    # Validation - just resize and normalize
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    transformed_bboxes = bboxes
                    transformed_labels = class_labels
                
                # Convert to DETR format
                if transformed_bboxes and transformed_labels:
                    # Convert to normalized center format
                    height, width = self.image_size
                    normalized_boxes = []
                    
                    for bbox in transformed_bboxes:
                        x1, y1, x2, y2 = bbox
                        
                        # Normalize coordinates
                        center_x = (x1 + x2) / 2.0 / width
                        center_y = (y1 + y2) / 2.0 / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Clamp values
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        box_width = max(0.01, min(1, box_width))
                        box_height = max(0.01, min(1, box_height))
                        
                        normalized_boxes.append([center_x, center_y, box_width, box_height])
                    
                    # Pad or truncate to max_objects
                    if len(normalized_boxes) > self.max_objects:
                        normalized_boxes = normalized_boxes[:self.max_objects]
                        transformed_labels = transformed_labels[:self.max_objects]
                    
                    target = {
                        'class_labels': torch.tensor(transformed_labels, dtype=torch.long),
                        'boxes': torch.tensor(normalized_boxes, dtype=torch.float32),
                        'image_id': torch.tensor([idx]),
                        'area': torch.tensor([b[2]*b[3] for b in normalized_boxes], dtype=torch.float32),
                        'iscrowd': torch.zeros(len(transformed_labels), dtype=torch.long),
                        'orig_size': torch.tensor(original_size),
                        'size': torch.tensor(self.image_size)
                    }
                else:
                    # Empty target
                    target = {
                        'class_labels': torch.zeros((0,), dtype=torch.long),
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'image_id': torch.tensor([idx]),
                        'area': torch.zeros((0,), dtype=torch.float32),
                        'iscrowd': torch.zeros((0,), dtype=torch.long),
                        'orig_size': torch.tensor(original_size),
                        'size': torch.tensor(self.image_size)
                    }
                    
            except Exception as e:
                # Fallback to simple processing
                print(f"Warning: Transform failed for image {idx}, using simple processing")
                
                # Simple resize and normalize
                image = cv2.resize(image, self.image_size)
                image = image.astype(np.float32) / 255.0
                
                # Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std
                
                # Convert to tensor
                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
                
                # Empty target
                target = {
                    'class_labels': torch.zeros((0,), dtype=torch.long),
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'image_id': torch.tensor([idx]),
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.long),
                    'orig_size': torch.tensor(original_size),
                    'size': torch.tensor(self.image_size)
                }
        
        else:
            # No objects - just process image
            transformed = self.transform(image=image)
            image = transformed['image']
            
            target = {
                'class_labels': torch.zeros((0,), dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.long),
                'orig_size': torch.tensor(original_size),
                'size': torch.tensor(self.image_size)
            }
        
        return image, target
    
    def collate_fn(self, batch):
        """Custom collate function for batch processing."""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, dim=0)
        return images, targets
    
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        cache_file = self._get_cache_file()
        
        info = {
            'cached_images': len(self.cached_data),
            'class_distribution': self.class_distribution,
            'cache_file': str(cache_file),
            'cache_exists': cache_file.exists(),
            'split': self.split
        }
        
        if cache_file.exists():
            info['cache_size_mb'] = cache_file.stat().st_size / (1024 * 1024)
        
        return info


if __name__ == "__main__":
    # Test the cached dataset
    print("Testing Cached BDD100K Dataset")
    
    # Test dataset creation
    dataset = CachedBDD100KDataset(
        annotations_file="data/analysis/processed/train_annotations.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='train',
        cache_dir="test_cache"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting an item
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target keys: {target.keys()}")
        print(f"Number of objects: {len(target['class_labels'])}")
    
    # Print cache info
    cache_info = dataset.get_cache_info()
    print(f"Cache info: {cache_info}")