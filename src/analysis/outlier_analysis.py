"""
Comprehensive Outlier and Noise Analysis for BDD100K Dataset

This module identifies outliers, noise points, missing annotations,
and problematic images for manual inspection and dataset quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from PIL import Image
import shutil
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set2")

class BDDOutlierAnalyzer:
    """Comprehensive outlier and noise analysis for BDD100K dataset."""
    
    def __init__(self, data_dir: str = "data", images_root: str = "data/raw/bdd100k/bdd100k/images/100k"):
        """Initialize outlier analyzer."""
        self.data_dir = Path(data_dir)
        self.images_root = Path(images_root)
        self.analysis_dir = self.data_dir / "analysis"
        self.outlier_dir = self.analysis_dir / "outliers"
        self.outlier_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outliers
        (self.outlier_dir / "size_outliers").mkdir(exist_ok=True)
        (self.outlier_dir / "aspect_ratio_outliers").mkdir(exist_ok=True)
        (self.outlier_dir / "position_outliers").mkdir(exist_ok=True)
        (self.outlier_dir / "annotation_outliers").mkdir(exist_ok=True)
        (self.outlier_dir / "missing_annotations").mkdir(exist_ok=True)
        (self.outlier_dir / "quality_outliers").mkdir(exist_ok=True)
        
        # Load processed data
        self.train_data = None
        self.val_data = None
        self.combined_data = None
        self.stats = None
        
        self._load_data()
        
    def _load_data(self):
        """Load processed data and statistics."""
        try:
            self.train_data = pd.read_csv(self.data_dir / "analysis" / "processed" / "train_annotations.csv")
            self.val_data = pd.read_csv(self.data_dir / "analysis" / "processed" / "val_annotations.csv")
            self.combined_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
            
            with open(self.data_dir / "analysis" / "processed" / "parsing_statistics.json", 'r') as f:
                self.stats = json.load(f)
            
            print(f"Loaded {len(self.combined_data):,} annotation records")
            print(f"Found {self.combined_data['image_name'].nunique():,} unique images")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def analyze_size_outliers(self, z_threshold: float = 3.0) -> Dict[str, Any]:
        """Identify size outliers using statistical methods."""
        print("🔍 Analyzing size outliers...")
        
        # Filter data with bounding boxes
        bbox_data = self.combined_data.dropna(subset=['bbox_area', 'bbox_width', 'bbox_height']).copy()
        
        size_outliers = {}
        
        # 1. Area outliers (Z-score method)
        area_mean = bbox_data['bbox_area'].mean()
        area_std = bbox_data['bbox_area'].std()
        area_z_scores = np.abs((bbox_data['bbox_area'] - area_mean) / area_std)
        
        area_outliers = bbox_data[area_z_scores > z_threshold].copy()
        area_outliers['z_score'] = area_z_scores[area_z_scores > z_threshold]
        area_outliers['outlier_reason'] = 'extreme_area'
        
        # 2. IQR method for robust outlier detection
        Q1_area = bbox_data['bbox_area'].quantile(0.25)
        Q3_area = bbox_data['bbox_area'].quantile(0.75)
        IQR_area = Q3_area - Q1_area
        
        area_lower = Q1_area - 1.5 * IQR_area
        area_upper = Q3_area + 1.5 * IQR_area
        
        iqr_outliers = bbox_data[
            (bbox_data['bbox_area'] < area_lower) | 
            (bbox_data['bbox_area'] > area_upper)
        ].copy()
        iqr_outliers['outlier_reason'] = 'area_iqr_outlier'
        
        # 3. Extreme dimension ratios
        bbox_data['dimension_ratio'] = bbox_data['bbox_width'] / bbox_data['bbox_height']
        extreme_ratios = bbox_data[
            (bbox_data['dimension_ratio'] < 0.1) | 
            (bbox_data['dimension_ratio'] > 10)
        ].copy()
        extreme_ratios['outlier_reason'] = 'extreme_aspect_ratio'
        
        # 4. Unusually small objects (potential annotation errors)
        tiny_objects = bbox_data[bbox_data['bbox_area'] < 10].copy()  # Less than 10 pixels
        tiny_objects['outlier_reason'] = 'tiny_object'
        
        # 5. Unusually large objects (potential annotation errors)  
        huge_objects = bbox_data[bbox_data['bbox_area'] > 100000].copy()  # More than 100k pixels
        huge_objects['outlier_reason'] = 'huge_object'
        
        # Combine all size outliers
        all_size_outliers = pd.concat([
            area_outliers, iqr_outliers, extreme_ratios, tiny_objects, huge_objects
        ], ignore_index=True).drop_duplicates(subset=['image_name', 'bbox_x1', 'bbox_y1'])
        
        size_outliers = {
            'area_z_score_outliers': len(area_outliers),
            'area_iqr_outliers': len(iqr_outliers),
            'extreme_aspect_ratio': len(extreme_ratios),
            'tiny_objects': len(tiny_objects),
            'huge_objects': len(huge_objects),
            'total_size_outliers': len(all_size_outliers),
            'outlier_data': all_size_outliers
        }
        
        # Save examples
        self._save_outlier_images(all_size_outliers, 'size_outliers', max_samples=50)
        
        return size_outliers
    
    def analyze_position_outliers(self) -> Dict[str, Any]:
        """Identify spatial/position outliers."""
        print("📍 Analyzing position outliers...")
        
        bbox_data = self.combined_data.dropna(subset=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']).copy()
        
        # Calculate normalized positions (assuming typical BDD100K dimensions)
        typical_width, typical_height = 1280, 720
        bbox_data['norm_center_x'] = ((bbox_data['bbox_x1'] + bbox_data['bbox_x2']) / 2) / typical_width
        bbox_data['norm_center_y'] = ((bbox_data['bbox_y1'] + bbox_data['bbox_y2']) / 2) / typical_height
        
        position_outliers = []
        
        # 1. Objects too close to edges (potential cropping issues)
        edge_threshold = 0.02  # 2% from edge
        edge_outliers = bbox_data[
            (bbox_data['norm_center_x'] < edge_threshold) |
            (bbox_data['norm_center_x'] > 1 - edge_threshold) |
            (bbox_data['norm_center_y'] < edge_threshold) |
            (bbox_data['norm_center_y'] > 1 - edge_threshold)
        ].copy()
        edge_outliers['outlier_reason'] = 'near_edge'
        position_outliers.append(edge_outliers)
        
        # 2. Class-specific position outliers
        class_position_outliers = []
        
        for class_name in bbox_data['category'].unique():
            if pd.isna(class_name):
                continue
                
            class_data = bbox_data[bbox_data['category'] == class_name]
            if len(class_data) < 100:  # Skip classes with few samples
                continue
            
            # Find objects in unusual positions for their class
            x_mean, x_std = class_data['norm_center_x'].mean(), class_data['norm_center_x'].std()
            y_mean, y_std = class_data['norm_center_y'].mean(), class_data['norm_center_y'].std()
            
            # Objects more than 3 std deviations from class mean position
            unusual_x = np.abs(class_data['norm_center_x'] - x_mean) > 3 * x_std
            unusual_y = np.abs(class_data['norm_center_y'] - y_mean) > 3 * y_std
            
            class_outliers = class_data[unusual_x | unusual_y].copy()
            class_outliers['outlier_reason'] = f'{class_name}_unusual_position'
            class_position_outliers.append(class_outliers)
        
        if class_position_outliers:
            position_outliers.extend(class_position_outliers)
        
        # 3. Objects outside typical image bounds (annotation errors)
        invalid_bounds = bbox_data[
            (bbox_data['bbox_x1'] < 0) |
            (bbox_data['bbox_y1'] < 0) |
            (bbox_data['bbox_x2'] > typical_width) |
            (bbox_data['bbox_y2'] > typical_height) |
            (bbox_data['bbox_x1'] >= bbox_data['bbox_x2']) |
            (bbox_data['bbox_y1'] >= bbox_data['bbox_y2'])
        ].copy()
        invalid_bounds['outlier_reason'] = 'invalid_coordinates'
        position_outliers.append(invalid_bounds)
        
        # Combine all position outliers
        all_position_outliers = pd.concat(position_outliers, ignore_index=True) if position_outliers else pd.DataFrame()
        if not all_position_outliers.empty:
            all_position_outliers = all_position_outliers.drop_duplicates(subset=['image_name', 'bbox_x1', 'bbox_y1'])
        
        position_analysis = {
            'edge_outliers': len(edge_outliers),
            'class_position_outliers': sum(len(df) for df in class_position_outliers),
            'invalid_bounds': len(invalid_bounds),
            'total_position_outliers': len(all_position_outliers),
            'outlier_data': all_position_outliers
        }
        
        # Save examples
        if not all_position_outliers.empty:
            self._save_outlier_images(all_position_outliers, 'position_outliers', max_samples=50)
        
        return position_analysis
    
    def analyze_annotation_outliers(self) -> Dict[str, Any]:
        """Identify annotation-related outliers."""
        print("📝 Analyzing annotation outliers...")
        
        annotation_outliers = {}
        
        # 1. Images with unusually high object counts
        objects_per_image = self.combined_data.groupby(['split', 'image_name']).agg({
            'category': 'count',
            'bbox_area': 'sum'
        }).rename(columns={'category': 'object_count', 'bbox_area': 'total_area'})
        
        # Statistical outliers in object count
        obj_count_mean = objects_per_image['object_count'].mean()
        obj_count_std = objects_per_image['object_count'].std()
        
        high_count_threshold = obj_count_mean + 3 * obj_count_std
        low_count_threshold = max(0, obj_count_mean - 3 * obj_count_std)
        
        high_count_images = objects_per_image[
            objects_per_image['object_count'] > high_count_threshold
        ].reset_index()
        high_count_images['outlier_reason'] = 'too_many_objects'
        
        # 2. Images with no annotations (missing annotations)
        all_image_names = set()
        for split in ['train', 'val']:
            split_dir = self.images_root / split
            if split_dir.exists():
                all_image_names.update(img.name for img in split_dir.glob('*.jpg'))
        
        annotated_images = set(self.combined_data['image_name'].unique())
        missing_annotation_images = all_image_names - annotated_images
        
        print(f"Found {len(missing_annotation_images):,} images without annotations")
        
        # 3. Images with only background/no-object annotations
        images_with_objects = self.combined_data[
            self.combined_data['category'].notna()
        ]['image_name'].unique()
        
        all_annotated_images = self.combined_data['image_name'].unique()
        background_only_images = set(all_annotated_images) - set(images_with_objects)
        
        # 4. Images with suspicious annotation patterns
        suspicious_patterns = []
        
        # Very small objects that might be annotation errors
        tiny_annotations = self.combined_data[
            (self.combined_data['bbox_area'] < 5) & 
            (self.combined_data['category'].notna())
        ].copy()
        tiny_annotations['outlier_reason'] = 'tiny_annotation'
        suspicious_patterns.append(tiny_annotations)
        
        # Objects with extreme aspect ratios
        extreme_aspect = self.combined_data[
            (self.combined_data['bbox_aspect_ratio'] < 0.05) |
            (self.combined_data['bbox_aspect_ratio'] > 20)
        ].copy()
        extreme_aspect['outlier_reason'] = 'extreme_aspect_annotation'
        suspicious_patterns.append(extreme_aspect)
        
        # Combine suspicious patterns
        all_suspicious = pd.concat(suspicious_patterns, ignore_index=True) if suspicious_patterns else pd.DataFrame()
        if not all_suspicious.empty:
            all_suspicious = all_suspicious.drop_duplicates(subset=['image_name', 'bbox_x1', 'bbox_y1'])
        
        annotation_outliers = {
            'high_object_count_images': len(high_count_images),
            'missing_annotation_images': len(missing_annotation_images),
            'background_only_images': len(background_only_images),
            'suspicious_annotations': len(all_suspicious),
            'high_count_data': high_count_images,
            'missing_images_list': list(missing_annotation_images),
            'background_only_list': list(background_only_images),
            'suspicious_data': all_suspicious
        }
        
        # Save examples of problematic images
        self._save_outlier_images(high_count_images, 'annotation_outliers', max_samples=20, 
                                image_list=True, reason_col='outlier_reason')
        
        # Save missing annotation images
        self._save_missing_annotation_images(list(missing_annotation_images)[:100])
        
        return annotation_outliers
    
    def analyze_quality_outliers(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Identify image quality outliers."""
        print("🖼️  Analyzing image quality outliers...")
        
        # Sample images for quality analysis
        unique_images = self.combined_data[['split', 'image_name']].drop_duplicates()
        if len(unique_images) > sample_size:
            sampled_images = unique_images.sample(n=sample_size, random_state=42)
        else:
            sampled_images = unique_images
        
        quality_data = []
        processing_errors = []
        
        for _, row in sampled_images.iterrows():
            split = row['split']
            image_name = row['image_name']
            image_path = self.images_root / split / image_name
            
            try:
                if image_path.exists():
                    # Load image
                    img = cv2.imread(str(image_path))
                    if img is None:
                        processing_errors.append({'image': image_name, 'error': 'Cannot read image'})
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Quality metrics
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Check for corruption indicators
                    height, width = gray.shape
                    is_corrupted = False
                    corruption_reasons = []
                    
                    # Extremely dark or bright images
                    if brightness < 10:
                        is_corrupted = True
                        corruption_reasons.append('too_dark')
                    elif brightness > 245:
                        is_corrupted = True
                        corruption_reasons.append('too_bright')
                    
                    # Very low contrast (flat images)
                    if contrast < 5:
                        is_corrupted = True
                        corruption_reasons.append('low_contrast')
                    
                    # Very blurry images
                    if sharpness < 10:
                        is_corrupted = True
                        corruption_reasons.append('blurry')
                    
                    # Unusual dimensions
                    if width != 1280 or height != 720:
                        is_corrupted = True
                        corruption_reasons.append('unusual_dimensions')
                    
                    quality_data.append({
                        'split': split,
                        'image_name': image_name,
                        'brightness': brightness,
                        'contrast': contrast,
                        'sharpness': sharpness,
                        'width': width,
                        'height': height,
                        'is_corrupted': is_corrupted,
                        'corruption_reasons': ','.join(corruption_reasons)
                    })
                else:
                    processing_errors.append({'image': image_name, 'error': 'File not found'})
                    
            except Exception as e:
                processing_errors.append({'image': image_name, 'error': str(e)})
        
        if not quality_data:
            return {'error': 'No images could be processed for quality analysis'}
        
        df_quality = pd.DataFrame(quality_data)
        
        # Identify quality outliers
        corrupted_images = df_quality[df_quality['is_corrupted']].copy()
        
        quality_outliers = {
            'total_analyzed': len(df_quality),
            'corrupted_images': len(corrupted_images),
            'processing_errors': len(processing_errors),
            'corruption_breakdown': corrupted_images['corruption_reasons'].value_counts().to_dict(),
            'corrupted_data': corrupted_images,
            'processing_errors_list': processing_errors
        }
        
        # Save quality outliers
        if not corrupted_images.empty:
            self._save_outlier_images(corrupted_images, 'quality_outliers', max_samples=50,
                                    image_list=True, reason_col='corruption_reasons')
        
        return quality_outliers
    
    def _save_outlier_images(self, outlier_data: pd.DataFrame, outlier_type: str, 
                           max_samples: int = 50, image_list: bool = False, 
                           reason_col: str = 'outlier_reason'):
        """Save outlier images to inspection folder."""
        if outlier_data.empty:
            return
        
        print(f"  💾 Saving {min(len(outlier_data), max_samples)} {outlier_type} examples...")
        
        outlier_folder = self.outlier_dir / outlier_type
        
        # Sample if too many
        if len(outlier_data) > max_samples:
            samples = outlier_data.sample(n=max_samples, random_state=42)
        else:
            samples = outlier_data
        
        # Save images and create report
        saved_images = []
        report_data = []
        
        for idx, row in samples.iterrows():
            try:
                if image_list:
                    # Row contains image info directly
                    split = row.get('split', 'unknown')
                    image_name = row.get('image_name', f'unknown_{idx}')
                else:
                    # Row contains annotation info
                    split = row['split']
                    image_name = row['image_name']
                
                source_path = self.images_root / split / image_name
                
                if source_path.exists():
                    # Create descriptive filename
                    reason = row.get(reason_col, 'unknown')
                    safe_reason = "".join(c for c in str(reason) if c.isalnum() or c in "._-")
                    dest_filename = f"{split}_{image_name.split('.')[0]}_{safe_reason}.jpg"
                    dest_path = outlier_folder / dest_filename
                    
                    # Copy image
                    shutil.copy2(source_path, dest_path)
                    saved_images.append(dest_filename)
                    
                    # Prepare report data
                    report_entry = {
                        'original_path': str(source_path),
                        'saved_as': dest_filename,
                        'split': split,
                        'image_name': image_name,
                        'outlier_reason': reason
                    }
                    
                    # Add relevant metrics
                    if 'bbox_area' in row:
                        report_entry['bbox_area'] = row['bbox_area']
                    if 'bbox_aspect_ratio' in row:
                        report_entry['aspect_ratio'] = row['bbox_aspect_ratio']
                    if 'brightness' in row:
                        report_entry['brightness'] = row['brightness']
                    if 'contrast' in row:
                        report_entry['contrast'] = row['contrast']
                    if 'object_count' in row:
                        report_entry['object_count'] = row['object_count']
                        
                    report_data.append(report_entry)
                    
            except Exception as e:
                print(f"    Error saving {image_name}: {e}")
        
        # Save report
        report_df = pd.DataFrame(report_data)
        report_path = outlier_folder / f"{outlier_type}_report.csv"
        report_df.to_csv(report_path, index=False)
        
        print(f"  ✅ Saved {len(saved_images)} images and report to {outlier_folder}")
    
    def _save_missing_annotation_images(self, missing_images: List[str]):
        """Save images that are missing annotations."""
        if not missing_images:
            return
        
        print(f"  💾 Saving {len(missing_images)} missing annotation examples...")
        
        missing_folder = self.outlier_dir / "missing_annotations"
        saved_count = 0
        report_data = []
        
        for image_name in missing_images:
            # Try to find the image in train or val
            for split in ['train', 'val']:
                source_path = self.images_root / split / image_name
                if source_path.exists():
                    dest_path = missing_folder / f"{split}_{image_name}"
                    try:
                        shutil.copy2(source_path, dest_path)
                        saved_count += 1
                        
                        report_data.append({
                            'image_name': image_name,
                            'split': split,
                            'original_path': str(source_path),
                            'saved_as': f"{split}_{image_name}",
                            'issue': 'missing_annotation'
                        })
                        break
                    except Exception as e:
                        print(f"    Error saving {image_name}: {e}")
        
        # Save report
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_path = missing_folder / "missing_annotations_report.csv"
            report_df.to_csv(report_path, index=False)
        
        print(f"  ✅ Saved {saved_count} missing annotation images")
    
    def generate_outlier_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive outlier analysis report."""
        print("📄 Generating outlier summary report...")
        
        report_lines = []
        
        report_lines.extend([
            "# BDD100K Dataset - Outlier & Noise Analysis Report",
            "=" * 60,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ])
        
        # Summary statistics
        total_images = self.combined_data['image_name'].nunique()
        total_objects = len(self.combined_data[self.combined_data['category'].notna()])
        
        report_lines.extend([
            f"- Total Images Analyzed: {total_images:,}",
            f"- Total Objects: {total_objects:,}",
            ""
        ])
        
        # Size outliers
        if 'size_outliers' in results:
            size_data = results['size_outliers']
            report_lines.extend([
                "## Size Outliers Analysis",
                f"- Total size outliers detected: {size_data.get('total_size_outliers', 0):,}",
                f"- Area Z-score outliers: {size_data.get('area_z_score_outliers', 0):,}",
                f"- Area IQR outliers: {size_data.get('area_iqr_outliers', 0):,}",
                f"- Extreme aspect ratios: {size_data.get('extreme_aspect_ratio', 0):,}",
                f"- Tiny objects (<10 px²): {size_data.get('tiny_objects', 0):,}",
                f"- Huge objects (>100k px²): {size_data.get('huge_objects', 0):,}",
                ""
            ])
        
        # Position outliers
        if 'position_outliers' in results:
            pos_data = results['position_outliers']
            report_lines.extend([
                "## Position Outliers Analysis", 
                f"- Total position outliers: {pos_data.get('total_position_outliers', 0):,}",
                f"- Edge outliers: {pos_data.get('edge_outliers', 0):,}",
                f"- Class position outliers: {pos_data.get('class_position_outliers', 0):,}",
                f"- Invalid coordinates: {pos_data.get('invalid_bounds', 0):,}",
                ""
            ])
        
        # Annotation outliers
        if 'annotation_outliers' in results:
            ann_data = results['annotation_outliers']
            report_lines.extend([
                "## Annotation Quality Issues",
                f"- Missing annotation images: {ann_data.get('missing_annotation_images', 0):,}",
                f"- Background-only images: {ann_data.get('background_only_images', 0):,}",
                f"- High object count images: {ann_data.get('high_object_count_images', 0):,}",
                f"- Suspicious annotations: {ann_data.get('suspicious_annotations', 0):,}",
                ""
            ])
            
            # Missing images breakdown
            missing_count = ann_data.get('missing_annotation_images', 0)
            if missing_count > 0:
                missing_pct = missing_count / (total_images + missing_count) * 100
                report_lines.extend([
                    f"### Missing Annotations Analysis",
                    f"- {missing_count:,} images found without annotations",
                    f"- Represents {missing_pct:.1f}% of total image files",
                    f"- These images exist in the dataset but have no annotation records",
                    f"- Possible causes: annotation errors, file naming issues, or incomplete labeling",
                    ""
                ])
        
        # Quality outliers  
        if 'quality_outliers' in results:
            qual_data = results['quality_outliers']
            report_lines.extend([
                "## Image Quality Issues",
                f"- Total images analyzed for quality: {qual_data.get('total_analyzed', 0):,}",
                f"- Corrupted/poor quality images: {qual_data.get('corrupted_images', 0):,}",
                f"- Processing errors: {qual_data.get('processing_errors', 0):,}",
                ""
            ])
            
            # Quality breakdown
            corruption_breakdown = qual_data.get('corruption_breakdown', {})
            if corruption_breakdown:
                report_lines.extend([
                    "### Quality Issue Breakdown:",
                    *[f"- {issue}: {count}" for issue, count in corruption_breakdown.items()],
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Immediate Actions:",
            "1. **Missing Annotations**: Review images without annotations",
            "   - Check if these are valid driving scenes requiring labeling",
            "   - Verify file naming consistency",
            "   - Consider excluding if not relevant to object detection",
            "",
            "2. **Size Outliers**: Manual review of extreme size annotations",
            "   - Verify tiny objects are correctly annotated",
            "   - Check huge objects for annotation errors",
            "   - Consider separate handling for extreme sizes",
            "",
            "3. **Position Outliers**: Review spatial anomalies",
            "   - Verify objects in unusual positions are correctly labeled",
            "   - Check for annotation coordinate errors",
            "   - Consider data augmentation implications",
            "",
            "4. **Quality Issues**: Address image quality problems",
            "   - Remove corrupted or unreadable images",
            "   - Consider separate preprocessing for poor quality images",
            "   - Verify image format consistency",
            "",
            "### Training Implications:",
            "- Filter out clear annotation errors before training",
            "- Consider robust loss functions for noisy data",
            "- Implement data validation checks in training pipeline",
            "- Monitor model performance on outlier cases",
            "",
            "### Quality Assurance:",
            "- Implement automated outlier detection in data pipeline",
            "- Create validation rules for new annotations",
            "- Regular quality audits of annotation process",
            "- Human review of flagged outliers",
            ""
        ])
        
        # File locations
        report_lines.extend([
            "## Generated Files",
            "",
            "Outlier images saved to folders:",
            f"- Size outliers: {self.outlier_dir / 'size_outliers'}",
            f"- Position outliers: {self.outlier_dir / 'position_outliers'}",
            f"- Annotation outliers: {self.outlier_dir / 'annotation_outliers'}",
            f"- Quality outliers: {self.outlier_dir / 'quality_outliers'}",
            f"- Missing annotations: {self.outlier_dir / 'missing_annotations'}",
            "",
            "Each folder contains:",
            "- Sample outlier images for visual inspection",
            "- Detailed CSV report with metrics and reasons",
            "- Ready for manual quality review",
            ""
        ])
        
        # Save report
        report_file = self.outlier_dir / "outlier_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Outlier analysis report saved: {report_file}")
        return str(report_file)
    
    def run_complete_outlier_analysis(self) -> Dict[str, Any]:
        """Run comprehensive outlier analysis."""
        print("🚀 Starting comprehensive outlier analysis...")
        print("=" * 60)
        
        results = {}
        
        # 1. Size outliers
        try:
            results['size_outliers'] = self.analyze_size_outliers()
        except Exception as e:
            print(f"Error in size analysis: {e}")
            results['size_outliers'] = {'error': str(e)}
        
        # 2. Position outliers  
        try:
            results['position_outliers'] = self.analyze_position_outliers()
        except Exception as e:
            print(f"Error in position analysis: {e}")
            results['position_outliers'] = {'error': str(e)}
        
        # 3. Annotation outliers
        try:
            results['annotation_outliers'] = self.analyze_annotation_outliers()
        except Exception as e:
            print(f"Error in annotation analysis: {e}")
            results['annotation_outliers'] = {'error': str(e)}
        
        # 4. Quality outliers
        try:
            results['quality_outliers'] = self.analyze_quality_outliers()
        except Exception as e:
            print(f"Error in quality analysis: {e}")
            results['quality_outliers'] = {'error': str(e)}
        
        # Generate summary report
        report_file = self.generate_outlier_summary_report(results)
        results['report_file'] = report_file
        
        print("=" * 60)
        print("🎉 Outlier analysis complete!")
        print(f"📄 Report: {report_file}")
        print(f"📁 Outlier images: {self.outlier_dir}")
        
        return results

if __name__ == "__main__":
    analyzer = BDDOutlierAnalyzer()
    results = analyzer.run_complete_outlier_analysis()
    
    # Print summary
    print("\n📋 ANALYSIS SUMMARY:")
    for analysis_type, data in results.items():
        if isinstance(data, dict) and 'error' not in data:
            if analysis_type == 'size_outliers':
                print(f"  Size Outliers: {data.get('total_size_outliers', 0):,}")
            elif analysis_type == 'position_outliers':
                print(f"  Position Outliers: {data.get('total_position_outliers', 0):,}")
            elif analysis_type == 'annotation_outliers':
                print(f"  Missing Annotations: {data.get('missing_annotation_images', 0):,}")
                print(f"  Annotation Issues: {data.get('suspicious_annotations', 0):,}")
            elif analysis_type == 'quality_outliers':
                print(f"  Quality Issues: {data.get('corrupted_images', 0):,}")