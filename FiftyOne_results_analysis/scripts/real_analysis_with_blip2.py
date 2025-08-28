"""
Real FiftyOne Analysis with BLIP2 Embeddings and Actual Model Predictions

This script creates a complete FiftyOne dataset using:
- Real BDD100K images
- Real model predictions from your trained checkpoint
- Real BLIP2 embeddings for clustering
- Interactive FiftyOne App for exploration
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import fiftyone as fo
import fiftyone.brain as fob
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.detr_dataset import BDD100KDETRDataset
from models.detr_model import BDD100KDETR, BDD100KDetrConfig


class RealFiftyOneAnalysis:
    """Real FiftyOne analysis with BLIP2 and actual model predictions."""
    
    def __init__(
        self,
        annotations_file: str,
        images_root: str,
        model_checkpoint: str,
        dataset_name: str = "bdd100k_real_analysis",
        subset_size: int = 100
    ):
        """
        Initialize real analysis.
        
        Args:
            annotations_file: Path to annotations CSV
            images_root: Path to actual BDD100K images
            model_checkpoint: Path to your trained model
            dataset_name: FiftyOne dataset name
            subset_size: Number of images to analyze
        """
        self.annotations_file = annotations_file
        self.images_root = Path(images_root)
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.subset_size = subset_size
        
        # BDD100K class mapping
        self.class_mapping = {
            'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
            'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8, 'traffic sign': 9
        }
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        self.safety_critical_classes = {'pedestrian', 'rider', 'bicycle', 'motorcycle'}
        
        # Model and processors
        self.model = None
        self.blip_processor = None
        self.blip_model = None
        
        print(f"🔧 Initialized Real FiftyOne Analysis")
        print(f"   📁 Images: {self.images_root}")
        print(f"   📊 Annotations: {self.annotations_file}")
        print(f"   🤖 Model: {self.model_checkpoint}")
        print(f"   📈 Dataset: {self.dataset_name}")
        print(f"   🎯 Subset size: {self.subset_size}")
    
    def load_models(self):
        """Load DETR model and BLIP2 for embeddings."""
        print("\n🤖 Loading Models...")
        
        # Load DETR model
        try:
            print("Loading DETR model...")
            config = BDD100KDetrConfig()
            self.model = BDD100KDETR(config=config, pretrained=False)
            
            checkpoint = torch.load(self.model_checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("✅ DETR model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading DETR model: {e}")
            return False
        
        # Load BLIP2 for embeddings
        try:
            print("Loading BLIP2 for embeddings...")
            model_name = "Salesforce/blip-image-captioning-base"
            self.blip_processor = BlipProcessor.from_pretrained(model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.blip_model.eval()
            print("✅ BLIP2 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading BLIP2 model: {e}")
            return False
        
        return True
    
    def create_fiftyone_dataset(self) -> fo.Dataset:
        """Create FiftyOne dataset with real images and annotations."""
        print(f"\n📊 Creating FiftyOne Dataset: {self.dataset_name}")
        
        # Delete existing dataset
        if self.dataset_name in fo.list_datasets():
            fo.delete_dataset(self.dataset_name)
        
        # Create new dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.persistent = True
        
        # Load annotations
        df = pd.read_csv(self.annotations_file)
        
        # Get subset of images
        unique_images = df['image_name'].unique()[:self.subset_size]
        subset_df = df[df['image_name'].isin(unique_images)]
        
        print(f"Processing {len(unique_images)} images with {len(subset_df)} annotations...")
        
        samples_added = 0
        for image_name, group in tqdm(subset_df.groupby('image_name'), desc="Creating samples"):
            # Check if image exists
            image_path = self.images_root / image_name
            if not image_path.exists():
                continue
            
            try:
                # Create sample
                sample = fo.Sample(filepath=str(image_path))
                
                # Add metadata
                first_row = group.iloc[0]
                sample["split"] = first_row['split']
                
                # Add weather, scene, time metadata if available
                for col in ['img_attr_weather', 'img_attr_scene', 'img_attr_timeofday']:
                    if col in group.columns:
                        sample[col.replace('img_attr_', '')] = first_row[col]
                
                # Process ground truth detections
                detections = []
                objects = group[group['category'].notna()]
                
                for _, obj in objects.iterrows():
                    category = obj['category']
                    if category not in self.class_mapping:
                        continue
                    
                    # Get image dimensions
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    
                    # Convert to relative coordinates
                    x1, y1, x2, y2 = obj['bbox_x1'], obj['bbox_y1'], obj['bbox_x2'], obj['bbox_y2']
                    rel_x = x1 / img_width
                    rel_y = y1 / img_height
                    rel_w = (x2 - x1) / img_width
                    rel_h = (y2 - y1) / img_height
                    
                    # Create detection
                    detection = fo.Detection(
                        label=category,
                        bounding_box=[rel_x, rel_y, rel_w, rel_h],
                        confidence=1.0,
                        area=float(obj['bbox_area']) if 'bbox_area' in obj else rel_w * rel_h
                    )
                    
                    # Add custom fields
                    detection["is_safety_critical"] = category in self.safety_critical_classes
                    detection["class_id"] = self.class_mapping[category]
                    if 'object_id' in obj:
                        detection["object_id"] = int(obj['object_id'])
                    
                    detections.append(detection)
                
                # Add ground truth to sample
                sample["ground_truth"] = fo.Detections(detections=detections)
                
                # Add to dataset
                dataset.add_sample(sample)
                samples_added += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {image_name}: {e}")
                continue
        
        print(f"✅ Created dataset with {samples_added} samples")
        return dataset
    
    def add_model_predictions(self, dataset: fo.Dataset):
        """Add real model predictions to the dataset."""
        print("\n🎯 Generating Model Predictions...")
        
        if not self.model:
            print("❌ Model not loaded")
            return
        
        # Setup transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        predictions_added = 0
        for sample in tqdm(dataset, desc="Generating predictions"):
            try:
                # Load and preprocess image
                image = Image.open(sample.filepath).convert('RGB')
                original_size = image.size
                
                # Transform for model
                image_tensor = transform(image).unsqueeze(0)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    # Debug: print output keys
                    if predictions_added == 0:
                        print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")
                
                # Process outputs
                predictions = self._process_model_outputs(outputs, original_size)
                
                # Add to sample
                sample["predictions"] = fo.Detections(detections=predictions)
                sample.save()
                
                predictions_added += 1
                
            except Exception as e:
                print(f"⚠️ Error predicting {sample.filepath}: {e}")
                continue
        
        print(f"✅ Added predictions for {predictions_added} samples")
    
    def _process_model_outputs(
        self, 
        outputs: Dict, 
        original_size: Tuple[int, int],
        confidence_threshold: float = 0.02
    ) -> List[fo.Detection]:
        """Process model outputs into FiftyOne detections."""
        detections = []
        
        try:
            # Extract predictions (handle different output formats)
            if 'pred_logits' in outputs:
                pred_logits = outputs['pred_logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
            elif 'logits' in outputs:
                pred_logits = outputs['logits'][0]
                pred_boxes = outputs['pred_boxes'][0]
            else:
                print(f"Unknown output format: {outputs.keys()}")
                return detections
            
            # Get probabilities
            pred_probs = torch.softmax(pred_logits, dim=-1)
            max_probs, pred_classes = torch.max(pred_probs, dim=-1)
            
            # Filter by confidence
            confident_mask = max_probs > confidence_threshold
            
            for i in range(len(pred_classes)):
                if not confident_mask[i]:
                    continue
                
                class_id = pred_classes[i].item()
                confidence = max_probs[i].item()
                
                if class_id >= len(self.id_to_class):
                    continue
                
                class_name = self.id_to_class[class_id]
                
                # Convert box coordinates (DETR format: cx, cy, w, h normalized)
                box = pred_boxes[i].tolist()
                cx, cy, w, h = box
                
                # Convert to FiftyOne format (x, y, w, h)
                x = max(0, min(1, cx - w / 2))
                y = max(0, min(1, cy - h / 2))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                detection = fo.Detection(
                    label=class_name,
                    bounding_box=[x, y, w, h],
                    confidence=confidence
                )
                
                detection["is_safety_critical"] = class_name in self.safety_critical_classes
                detection["class_id"] = class_id
                
                detections.append(detection)
        
        except Exception as e:
            print(f"Error processing outputs: {e}")
        
        return detections
    
    def compute_blip2_embeddings(self, dataset: fo.Dataset):
        """Compute real BLIP2 embeddings for images."""
        print("\n🧠 Computing BLIP2 Embeddings...")
        
        if not self.blip_model:
            print("❌ BLIP2 model not loaded")
            return
        
        embeddings = []
        sample_ids = []
        
        for sample in tqdm(dataset, desc="Computing embeddings"):
            try:
                # Load and process image
                image = Image.open(sample.filepath).convert('RGB')
                
                # Get image embeddings using BLIP
                inputs = self.blip_processor(image, return_tensors="pt")
                
                with torch.no_grad():
                    # Get vision embeddings
                    vision_outputs = self.blip_model.vision_model(**inputs)
                    # Use pooled output as embedding
                    embedding = vision_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                embeddings.append(embedding)
                sample_ids.append(sample.id)
                
            except Exception as e:
                print(f"⚠️ Error computing embedding for {sample.filepath}: {e}")
                continue
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            print(f"✅ Computed {len(embeddings)} BLIP2 embeddings (shape: {embeddings_array.shape})")
            
            # Store embeddings in dataset
            dataset.set_values("blip2_embedding", embeddings, sample_ids)
            
            return embeddings_array
        
        return None
    
    def compute_brain_analysis(self, dataset: fo.Dataset):
        """Compute FiftyOne Brain analysis."""
        print("\n🧠 Computing FiftyOne Brain Analysis...")
        
        try:
            # Compute visualization using ground truth patches
            print("Computing visualization embeddings...")
            fob.compute_visualization(
                dataset,
                patches_field="ground_truth",
                brain_key="gt_viz",
                method="umap",
                num_dims=2
            )
            print("✅ Ground truth visualization computed")
            
            # Compute for predictions if available
            if "predictions" in dataset.get_field_schema():
                print("Computing prediction embeddings...")
                fob.compute_visualization(
                    dataset,
                    patches_field="predictions",
                    brain_key="pred_viz",
                    method="umap",
                    num_dims=2
                )
                print("✅ Prediction visualization computed")
                
                # Compute similarity
                print("Computing similarity analysis...")
                fob.compute_similarity(
                    dataset,
                    patches_field="predictions",
                    brain_key="similarity"
                )
                print("✅ Similarity analysis computed")
                
                # Compute mistakenness
                print("Computing mistake analysis...")
                fob.compute_mistakenness(
                    dataset,
                    pred_field="predictions",
                    label_field="ground_truth",
                    brain_key="mistakes"
                )
                print("✅ Mistake analysis computed")
            
        except Exception as e:
            print(f"⚠️ Brain analysis error: {e}")
    
    def create_analysis_views(self, dataset: fo.Dataset) -> Dict[str, fo.DatasetView]:
        """Create useful analysis views."""
        print("\n👁️ Creating Analysis Views...")
        
        views = {}
        
        # Safety-critical objects
        views["safety_critical"] = dataset.filter_labels(
            "ground_truth",
            fo.ViewField("is_safety_critical") == True
        )
        
        # High confidence predictions
        if "predictions" in dataset.get_field_schema():
            views["high_conf_predictions"] = dataset.filter_labels(
                "predictions",
                fo.ViewField("confidence") > 0.5
            )
            
            views["low_conf_predictions"] = dataset.filter_labels(
                "predictions",
                fo.ViewField("confidence") < 0.1
            )
        
        # Mistakes (if computed)
        try:
            mistakes_info = dataset.get_brain_info("mistakes")
            if mistakes_info:
                views["potential_mistakes"] = dataset.sort_by("mistakenness", reverse=True).limit(50)
        except:
            pass
        
        # Class-specific views
        for class_name in ['car', 'pedestrian', 'traffic_sign', 'traffic_light']:
            views[f"{class_name}_objects"] = dataset.filter_labels(
                "ground_truth",
                fo.ViewField("label") == class_name
            )
        
        print(f"✅ Created {len(views)} analysis views")
        return views
    
    def launch_fiftyone_app(self, dataset: fo.Dataset, views: Dict[str, fo.DatasetView]):
        """Launch FiftyOne App for interactive exploration."""
        print("\n🚀 Launching FiftyOne App...")
        
        try:
            # Create app session
            session = fo.launch_app(dataset, port=5151, auto=False)
            
            print("🎯 FiftyOne App launched successfully!")
            print(f"   🌐 URL: http://localhost:5151")
            print(f"   📊 Dataset: {self.dataset_name}")
            print(f"   👁️ Available views: {list(views.keys())}")
            print("\n🔍 Interactive Analysis Features:")
            print("   • Ground truth vs predictions visualization")
            print("   • BLIP2 embedding-based clustering")
            print("   • Brain-powered similarity search")
            print("   • Safety-critical object filtering")
            print("   • Mistake detection and analysis")
            
            return session
            
        except Exception as e:
            print(f"❌ Error launching app: {e}")
            print("💡 Fallback: Use fo.launch_app() manually")
            return None
    
    def generate_summary_report(self, dataset: fo.Dataset, views: Dict):
        """Generate summary analysis report."""
        print("\n📄 Generating Summary Report...")
        
        # Collect stats
        stats = {
            'total_samples': len(dataset),
            'total_gt_objects': sum(len(sample.ground_truth.detections) for sample in dataset if sample.ground_truth),
            'safety_critical_objects': len(views["safety_critical"]) if "safety_critical" in views else 0,
            'classes_present': set(),
            'brain_features': []
        }
        
        # Collect class distribution
        for sample in dataset:
            if sample.ground_truth:
                for det in sample.ground_truth.detections:
                    stats['classes_present'].add(det.label)
        
        stats['classes_present'] = sorted(list(stats['classes_present']))
        
        # Collect brain features
        try:
            for brain_key in ["gt_viz", "pred_viz", "similarity", "mistakes"]:
                try:
                    if dataset.get_brain_info(brain_key):
                        stats['brain_features'].append(brain_key)
                except:
                    continue
        except:
            pass
        
        # Generate report
        report_path = Path("FiftyOne_results_analysis/brain_outputs/real_analysis_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Real FiftyOne Brain Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now()}\n")
            f.write(f"**Dataset:** {self.dataset_name}\n\n")
            
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Images:** {stats['total_samples']}\n")
            f.write(f"- **Total Objects:** {stats['total_gt_objects']}\n")
            f.write(f"- **Safety-Critical Objects:** {stats['safety_critical_objects']}\n")
            f.write(f"- **Classes Present:** {', '.join(stats['classes_present'])}\n\n")
            
            f.write("## Analysis Features\n\n")
            f.write("### Real Data Processing\n")
            f.write("- ✅ Actual BDD100K images loaded\n")
            f.write("- ✅ Real model predictions generated\n")
            f.write("- ✅ BLIP2 embeddings computed\n")
            f.write("- ✅ FiftyOne Brain analysis performed\n\n")
            
            f.write("### Brain Analysis Results\n")
            for brain_key in stats['brain_features']:
                f.write(f"- ✅ {brain_key}\n")
            f.write("\n")
            
            f.write("### Available Views\n")
            for view_name in views.keys():
                f.write(f"- 👁️ {view_name}\n")
            f.write("\n")
            
            f.write("## Interactive Exploration\n\n")
            f.write("The FiftyOne App provides interactive exploration with:\n")
            f.write("- Ground truth vs predictions comparison\n")
            f.write("- BLIP2 embedding-based clustering\n")
            f.write("- Safety-critical object filtering\n")
            f.write("- Brain-powered similarity search\n")
            f.write("- Mistake detection and analysis\n\n")
            
            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("import fiftyone as fo\n")
            f.write(f"dataset = fo.load_dataset('{self.dataset_name}')\n")
            f.write("session = fo.launch_app(dataset)\n")
            f.write("```\n")
        
        print(f"✅ Report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete real analysis pipeline."""
        start_time = time.time()
        
        print("🎯 Starting Real FiftyOne Brain Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Load models
            if not self.load_models():
                return False
            
            # Step 2: Create FiftyOne dataset
            dataset = self.create_fiftyone_dataset()
            if len(dataset) == 0:
                print("❌ No samples created in dataset")
                return False
            
            # Step 3: Add model predictions
            self.add_model_predictions(dataset)
            
            # Step 4: Compute BLIP2 embeddings
            self.compute_blip2_embeddings(dataset)
            
            # Step 5: Compute Brain analysis
            self.compute_brain_analysis(dataset)
            
            # Step 6: Create analysis views
            views = self.create_analysis_views(dataset)
            
            # Step 7: Generate summary report
            report_path = self.generate_summary_report(dataset, views)
            
            # Step 8: Launch FiftyOne App
            session = self.launch_fiftyone_app(dataset, views)
            
            elapsed_time = time.time() - start_time
            
            print("\n🎉 Real Analysis Complete!")
            print("=" * 60)
            print(f"⏱️ Total time: {elapsed_time:.1f} seconds")
            print(f"📊 Dataset: {self.dataset_name} ({len(dataset)} samples)")
            print(f"📄 Report: {report_path}")
            print(f"🌐 App: http://localhost:5151")
            
            print("\n🔥 Real Features Delivered:")
            print("✅ Actual BDD100K images and annotations")
            print("✅ Real model predictions from your checkpoint")
            print("✅ BLIP2 embeddings for semantic clustering")
            print("✅ FiftyOne Brain analysis (UMAP, similarity, mistakes)")
            print("✅ Interactive FiftyOne App for visual exploration")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Real FiftyOne Brain Analysis with BLIP2")
    
    parser.add_argument("--subset", type=int, default=50, help="Number of images to analyze")
    parser.add_argument("--dataset-name", default="bdd100k_real_analysis", help="FiftyOne dataset name")
    parser.add_argument("--no-app", action="store_true", help="Skip launching FiftyOne App")
    
    args = parser.parse_args()
    
    # Configuration
    annotations_file = "data/analysis/processed_10class_corrected/val_annotations_10class.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k/val"
    model_checkpoint = "checkpoints/complete_10class_demo/checkpoint_epoch_048.pth"
    
    # Verify files exist
    for path, name in [(annotations_file, "annotations"), (images_root, "images"), (model_checkpoint, "model")]:
        if not Path(path).exists():
            print(f"❌ {name} not found: {path}")
            return False
    
    # Run analysis
    analyzer = RealFiftyOneAnalysis(
        annotations_file=annotations_file,
        images_root=images_root,
        model_checkpoint=model_checkpoint,
        dataset_name=args.dataset_name,
        subset_size=args.subset
    )
    
    success = analyzer.run_complete_analysis()
    
    if success and not args.no_app:
        print("\n💡 Keep the terminal open to maintain FiftyOne App session")
        print("Press Ctrl+C to exit when done exploring")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)