"""
Enhanced Data Augmentation Strategies for BDD100K DETR Training

This module implements advanced data augmentation techniques based on
qualitative analysis findings to improve model performance.
"""

import random
from typing import Dict, List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class EnhancedBDDTransforms:
    """
    Enhanced data augmentation pipeline for BDD100K based on analysis findings.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        split: str = 'train',
        augmentation_strength: str = 'medium'
    ):
        """
        Initialize enhanced transforms.
        
        Args:
            image_size: Target image size (height, width)
            split: Dataset split ('train', 'val')
            augmentation_strength: 'light', 'medium', 'strong'
        """
        self.image_size = image_size
        self.split = split
        self.augmentation_strength = augmentation_strength
        
        # Create transforms based on split
        if split == 'train':
            self.transform = self._create_training_transforms()
        else:
            self.transform = self._create_validation_transforms()
    
    def _create_training_transforms(self) -> A.Compose:
        """Create enhanced training transforms based on analysis findings."""
        
        # Base transforms that are always applied
        base_transforms = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ]
        
        # Augmentation strength configurations
        strength_configs = {
            'light': {
                'horizontal_flip': 0.3,
                'brightness_contrast': 0.2,
                'hue_saturation': 0.2,
                'gamma': 0.1,
                'noise': 0.1,
                'blur': 0.05,
                'cutout': 0.0,
                'mixup': 0.0
            },
            'medium': {
                'horizontal_flip': 0.5,
                'brightness_contrast': 0.4,
                'hue_saturation': 0.3,
                'gamma': 0.2,
                'noise': 0.2,
                'blur': 0.1,
                'cutout': 0.1,
                'mixup': 0.0
            },
            'strong': {
                'horizontal_flip': 0.7,
                'brightness_contrast': 0.6,
                'hue_saturation': 0.4,
                'gamma': 0.3,
                'noise': 0.3,
                'blur': 0.15,
                'cutout': 0.2,
                'mixup': 0.1
            }
        }
        
        config = strength_configs[self.augmentation_strength]
        
        # Geometric transforms
        geometric_transforms = []
        if config['horizontal_flip'] > 0:
            geometric_transforms.append(
                A.HorizontalFlip(p=config['horizontal_flip'])
            )
        
        # Enhanced photometric transforms for driving conditions
        photometric_transforms = []
        
        # Brightness/Contrast - crucial for different lighting conditions
        if config['brightness_contrast'] > 0:
            photometric_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=config['brightness_contrast']
                )
            )
        
        # HSV transforms for different camera characteristics
        if config['hue_saturation'] > 0:
            photometric_transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=config['hue_saturation']
                )
            )
        
        # Gamma correction for exposure variations
        if config['gamma'] > 0:
            photometric_transforms.append(
                A.RandomGamma(gamma_limit=(70, 130), p=config['gamma'])
            )
        
        # Weather and environmental conditions
        weather_transforms = []
        
        # Rain simulation
        weather_transforms.append(
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=20, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=7,
                brightness_coefficient=0.7,
                rain_type='drizzle',
                p=0.1
            )
        )
        
        # Fog simulation  
        weather_transforms.append(
            A.RandomFog(
                fog_coef_lower=0.1, fog_coef_upper=0.3,
                alpha_coef=0.08,
                p=0.1
            )
        )
        
        # Shadow simulation
        weather_transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.15
            )
        )
        
        # Quality degradation transforms
        quality_transforms = []
        
        # Gaussian noise for sensor variations
        if config['noise'] > 0:
            quality_transforms.append(
                A.GaussNoise(var_limit=(10.0, 50.0), p=config['noise'])
            )
        
        # Motion blur for movement
        if config['blur'] > 0:
            quality_transforms.append(
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.2)
                ], p=config['blur'])
            )
        
        # Advanced augmentation techniques
        advanced_transforms = []
        
        # Cutout for occlusion robustness
        if config['cutout'] > 0:
            advanced_transforms.append(
                A.CoarseDropout(
                    max_holes=2,
                    max_height=64,
                    max_width=64,
                    min_holes=1,
                    min_height=32,
                    min_width=32,
                    fill_value=0,
                    p=config['cutout']
                )
            )
        
        # Channel shuffle for color robustness
        advanced_transforms.append(
            A.ChannelShuffle(p=0.1)
        )
        
        # CLAHE for contrast enhancement
        advanced_transforms.append(
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2)
        )
        
        # Multi-scale training
        multiscale_transforms = []
        if random.random() < 0.3:  # 30% chance
            # Random scale between 0.8 and 1.2
            scale_factor = random.uniform(0.85, 1.15)
            new_size = int(self.image_size[0] * scale_factor)
            multiscale_transforms = [
                A.Resize(height=new_size, width=new_size),
                A.Resize(height=self.image_size[0], width=self.image_size[1])
            ]
        
        # Normalization
        normalization_transforms = [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
        
        # Combine all transforms
        all_transforms = (
            base_transforms +
            multiscale_transforms +
            geometric_transforms +
            photometric_transforms +
            weather_transforms +
            quality_transforms +
            advanced_transforms +
            normalization_transforms
        )
        
        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.2,  # Reduced from 0.3 for more training data
                min_area=64         # Minimum bounding box area
            )
        )
    
    def _create_validation_transforms(self) -> A.Compose:
        """Create validation transforms (no augmentation)."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray, bboxes: List, class_labels: List) -> Dict:
        """
        Apply transforms to image and annotations.
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes in pascal_voc format
            class_labels: List of class labels
            
        Returns:
            Dictionary with transformed image and annotations
        """
        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed
            
        except Exception as e:
            # Fallback to basic transform if augmentation fails
            basic_transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.1
            ))
            
            return basic_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )


class ClassSpecificAugmentation:
    """
    Class-specific augmentation strategies based on analysis findings.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize class-specific augmentation.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.class_strategies = self._define_class_strategies()
    
    def _define_class_strategies(self) -> Dict:
        """Define class-specific augmentation strategies."""
        return {
            'pedestrian': {
                'extra_horizontal_flip': 0.1,  # Pedestrians can move in both directions
                'scale_variation': 0.4,        # Pedestrians vary greatly in size/distance
                'motion_blur': 0.3,            # Walking pedestrians
                'occlusion_simulation': 0.4,   # Pedestrians often partially occluded
                'lighting_variation': 0.3      # Pedestrians in various lighting
            },
            'car': {
                'extra_horizontal_flip': 0.1,  # Cars can appear from both directions
                'scale_variation': 0.2,        # Cars vary in distance/size
                'lighting_variation': 0.3      # Cars in various lighting
            },
            'truck': {
                'extra_horizontal_flip': 0.2,  # Need more truck variations
                'scale_variation': 0.3,        # Trucks vary more in size
                'weather_augmentation': 0.2    # Trucks in bad weather
            },
            'bus': {
                'extra_horizontal_flip': 0.3,  # Buses from both directions
                'occlusion_simulation': 0.2,   # Buses often partially occluded
                'urban_augmentation': 0.3      # Buses in urban environments
            },
            'train': {
                'horizontal_flip': 0.5,        # Trains from both directions
                'motion_blur': 0.4,            # Fast-moving trains
                'side_view_emphasis': 0.6      # Trains mostly seen from side
            },
            'rider': {
                'scale_variation': 0.4,        # Riders vary greatly in size
                'motion_blur': 0.3,            # Moving riders
                'occlusion_simulation': 0.3    # Riders often partially hidden
            },
            'motorcycle': {
                'extra_horizontal_flip': 0.3,  # Motorcycles from both directions
                'scale_variation': 0.3,        # Motorcycles vary in size
                'motion_blur': 0.4,            # Fast-moving motorcycles
                'occlusion_simulation': 0.2    # Sometimes partially hidden
            },
            'bicycle': {
                'extra_horizontal_flip': 0.2,  # Bicycles from both directions
                'scale_variation': 0.3,        # Bicycles vary in distance
                'motion_blur': 0.2,            # Moving bicycles
                'occlusion_simulation': 0.3    # Often partially hidden
            },
            'traffic_sign': {
                'lighting_variation': 0.4,     # Signs in various lighting
                'perspective_variation': 0.3,  # Signs from different angles
                'weather_resistance': 0.2      # Signs must be visible in weather
            },
            'traffic_light': {
                'lighting_variation': 0.5,     # Lights in day/night
                'color_variation': 0.3,        # Different light colors
                'urban_context': 0.4           # Lights in urban settings
            }
        }
    
    def apply_class_specific_augmentation(
        self,
        image: np.ndarray,
        bboxes: List,
        class_labels: List,
        dominant_class: Optional[str] = None
    ) -> Tuple[np.ndarray, List, List]:
        """
        Apply class-specific augmentation based on dominant class.
        
        Args:
            image: Input image
            bboxes: Bounding boxes
            class_labels: Class labels
            dominant_class: Dominant class in the image
            
        Returns:
            Augmented image, bboxes, and labels
        """
        if not dominant_class or dominant_class not in self.class_strategies:
            return image, bboxes, class_labels
        
        strategy = self.class_strategies[dominant_class]
        
        # Apply class-specific transforms
        class_transforms = []
        
        # Extra horizontal flip for certain classes
        if 'extra_horizontal_flip' in strategy:
            if random.random() < strategy['extra_horizontal_flip']:
                class_transforms.append(A.HorizontalFlip(p=1.0))
        
        # Scale variation
        if 'scale_variation' in strategy:
            if random.random() < strategy['scale_variation']:
                scale = random.uniform(0.9, 1.1)
                class_transforms.append(A.RandomScale(scale_limit=scale, p=1.0))
        
        # Motion blur for moving objects
        if 'motion_blur' in strategy:
            if random.random() < strategy['motion_blur']:
                class_transforms.append(A.MotionBlur(blur_limit=5, p=1.0))
        
        # Apply transforms if any
        if class_transforms:
            transform = A.Compose(
                class_transforms,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    label_fields=['class_labels'],
                    min_visibility=0.2
                )
            )
            
            try:
                result = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                return result['image'], result['bboxes'], result['class_labels']
            except:
                pass
        
        return image, bboxes, class_labels


class BalancedSampling:
    """
    Balanced sampling strategy to ensure rare classes appear more frequently.
    """
    
    def __init__(self, class_distribution: Dict[str, int], target_balance: float = 0.3):
        """
        Initialize balanced sampling.
        
        Args:
            class_distribution: Current class distribution
            target_balance: Target balance ratio (0-1)
        """
        self.class_distribution = class_distribution
        self.target_balance = target_balance
        self.sampling_weights = self._calculate_sampling_weights()
    
    def _calculate_sampling_weights(self) -> Dict[str, float]:
        """Calculate sampling weights for balanced training."""
        total_samples = sum(self.class_distribution.values())
        num_classes = len(self.class_distribution)
        target_per_class = total_samples * self.target_balance / num_classes
        
        weights = {}
        for class_name, count in self.class_distribution.items():
            if count > 0:
                # Higher weight for underrepresented classes
                weights[class_name] = min(target_per_class / count, 10.0)
            else:
                weights[class_name] = 10.0  # Maximum weight for missing classes
        
        return weights
    
    def get_sample_weight(self, class_labels: List[str]) -> float:
        """
        Get sampling weight for an image based on its classes.
        
        Args:
            class_labels: List of class names in the image
            
        Returns:
            Sampling weight for the image
        """
        if not class_labels:
            return 1.0
        
        # Use maximum weight of classes in the image
        weights = [self.sampling_weights.get(label, 1.0) for label in class_labels]
        return max(weights)


def create_enhanced_transforms(
    image_size: Tuple[int, int] = (512, 512),
    split: str = 'train',
    augmentation_strength: str = 'medium',
    class_names: Optional[List[str]] = None
) -> EnhancedBDDTransforms:
    """
    Factory function to create enhanced transforms.
    
    Args:
        image_size: Target image size
        split: Dataset split
        augmentation_strength: Augmentation strength level
        class_names: List of class names for class-specific augmentation
        
    Returns:
        Enhanced transforms object
    """
    transforms = EnhancedBDDTransforms(
        image_size=image_size,
        split=split,
        augmentation_strength=augmentation_strength
    )
    
    return transforms


if __name__ == "__main__":
    # Test the enhanced augmentation
    print("Enhanced BDD100K Augmentation Module")
    print("Features:")
    print("  • Weather simulation (rain, fog, shadows)")
    print("  • Multi-scale training")
    print("  • Class-specific augmentation strategies")
    print("  • Advanced quality degradation")
    print("  • Balanced sampling for rare classes")
    
    # Example usage
    transforms = create_enhanced_transforms(
        image_size=(512, 512),
        split='train',
        augmentation_strength='medium'
    )
    
    print(f"\nCreated enhanced transforms for training:")
    print(f"Augmentation strength: medium")
    print(f"Image size: (512, 512)")
    print("Ready for integration with DETR training!")