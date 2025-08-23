"""
BDD100K Dataset Parser for Object Detection Analysis

This module provides comprehensive parsing functionality for BDD100K dataset
including JSON annotation parsing, image metadata extraction, and data validation.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with validation."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(
                f"Invalid bbox: ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


@dataclass
class DetectionObject:
    """Represents a detected object in BDD100K format."""

    category: str
    bbox: BoundingBox
    occluded: bool = False
    truncated: bool = False
    crowd: bool = False
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ImageAnnotation:
    """Represents complete annotation for a single image."""

    name: str
    url: str
    video_name: str
    index: int
    timestamp: int
    objects: List[DetectionObject]
    attributes: Dict[str, Any]

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class BDDParser:
    """Comprehensive parser for BDD100K dataset."""

    # Standard BDD100K object detection classes
    DETECTION_CLASSES = {
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    }

    def __init__(self, data_root: str):
        """Initialize parser with dataset root directory."""
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images" / "100k"
        self.labels_dir = self.data_root / "labels"

        # Statistics tracking
        self.stats = {
            "total_images": 0,
            "total_objects": 0,
            "class_distribution": defaultdict(int),
            "split_distribution": defaultdict(int),
            "parsing_errors": [],
            "missing_images": [],
            "invalid_annotations": [],
        }

    def validate_dataset_structure(self) -> Dict[str, bool]:
        """Validate expected dataset directory structure."""
        validation_results = {}

        # Check main directories
        validation_results["data_root_exists"] = self.data_root.exists()
        validation_results["images_dir_exists"] = self.images_dir.exists()
        validation_results["labels_dir_exists"] = self.labels_dir.exists()

        # Check for expected label files
        expected_splits = ["train", "val"]
        for split in expected_splits:
            label_file = self.labels_dir / f"bdd100k_labels_images_{split}.json"
            validation_results[f"{split}_labels_exist"] = label_file.exists()

        return validation_results

    def parse_bdd_annotation(self, annotation_data: Dict) -> ImageAnnotation:
        """Parse a single BDD annotation entry."""
        try:
            # Extract basic image information
            name = annotation_data["name"]
            url = annotation_data.get("url", "")
            video_name = annotation_data.get("videoName", "")
            index = annotation_data.get("index", 0)
            timestamp = annotation_data.get("timestamp", 0)
            attributes = annotation_data.get("attributes", {})

            # Parse objects/labels
            objects = []
            labels = annotation_data.get("labels", [])

            for label in labels:
                category = label.get("category", "")

                # Only process detection classes
                if category in self.DETECTION_CLASSES:
                    # Parse bounding box
                    box2d = label.get("box2d")
                    if box2d:
                        try:
                            bbox = BoundingBox(
                                x1=box2d["x1"],
                                y1=box2d["y1"],
                                x2=box2d["x2"],
                                y2=box2d["y2"],
                            )

                            obj = DetectionObject(
                                category=category,
                                bbox=bbox,
                                occluded=label.get("attributes", {}).get(
                                    "occluded", False
                                ),
                                truncated=label.get("attributes", {}).get(
                                    "truncated", False
                                ),
                                crowd=label.get("attributes", {}).get("crowd", False),
                                attributes=label.get("attributes", {}),
                            )
                            objects.append(obj)

                        except (KeyError, ValueError) as e:
                            self.stats["invalid_annotations"].append(
                                {"image": name, "error": str(e), "label": label}
                            )

            return ImageAnnotation(
                name=name,
                url=url,
                video_name=video_name,
                index=index,
                timestamp=timestamp,
                objects=objects,
                attributes=attributes,
            )

        except Exception as e:
            self.stats["parsing_errors"].append(
                {"annotation": annotation_data.get("name", "unknown"), "error": str(e)}
            )
            raise

    def load_split_annotations(self, split: str) -> List[ImageAnnotation]:
        """Load all annotations for a specific split (train/val)."""
        label_file = self.labels_dir / f"bdd100k_labels_images_{split}.json"

        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        logger.info(f"Loading {split} annotations from {label_file}")

        with open(label_file, "r") as f:
            raw_data = json.load(f)

        annotations = []
        for item in raw_data:
            try:
                annotation = self.parse_bdd_annotation(item)
                annotations.append(annotation)

                # Update statistics
                self.stats["total_images"] += 1
                self.stats["total_objects"] += len(annotation.objects)
                self.stats["split_distribution"][split] += 1

                for obj in annotation.objects:
                    self.stats["class_distribution"][obj.category] += 1

            except Exception as e:
                logger.warning(f"Failed to parse annotation: {e}")
                continue

        logger.info(f"Loaded {len(annotations)} annotations for {split} split")
        return annotations

    def load_all_annotations(self) -> Dict[str, List[ImageAnnotation]]:
        """Load annotations for all available splits."""
        all_annotations = {}

        for split in ["train", "val"]:
            try:
                all_annotations[split] = self.load_split_annotations(split)
            except FileNotFoundError:
                logger.warning(f"No annotations found for {split} split")
                all_annotations[split] = []

        return all_annotations

    def get_image_path(self, image_name: str, split: str) -> Path:
        """Get full path to image file."""
        return self.images_dir / split / image_name

    def validate_image_annotation_pairs(
        self, annotations: Dict[str, List[ImageAnnotation]]
    ) -> Dict[str, Any]:
        """Validate that all annotated images exist on disk."""
        validation_results = {
            "missing_images": defaultdict(list),
            "accessible_images": defaultdict(int),
            "image_statistics": defaultdict(dict),
        }

        for split, split_annotations in annotations.items():
            logger.info(f"Validating images for {split} split...")

            for annotation in split_annotations:
                image_path = self.get_image_path(annotation.name, split)

                if image_path.exists():
                    validation_results["accessible_images"][split] += 1

                    # Get image statistics if possible
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            mode = img.mode

                            validation_results["image_statistics"][split][
                                annotation.name
                            ] = {
                                "width": width,
                                "height": height,
                                "mode": mode,
                                "aspect_ratio": width / height,
                            }
                    except Exception as e:
                        logger.warning(f"Could not read image {image_path}: {e}")
                else:
                    validation_results["missing_images"][split].append(annotation.name)
                    self.stats["missing_images"].append(
                        {
                            "split": split,
                            "image": annotation.name,
                            "expected_path": str(image_path),
                        }
                    )

        return validation_results

    def export_parsed_data(
        self, annotations: Dict[str, List[ImageAnnotation]], output_dir: str
    ) -> Dict[str, str]:
        """Export parsed annotations to various formats for analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export as CSV for easy analysis
        for split, split_annotations in annotations.items():
            # Flatten annotations to rows
            rows = []
            for annotation in split_annotations:
                base_row = {
                    "image_name": annotation.name,
                    "video_name": annotation.video_name,
                    "split": split,
                    "timestamp": annotation.timestamp,
                    "total_objects": len(annotation.objects),
                }

                # Add image attributes
                for key, value in annotation.attributes.items():
                    base_row[f"img_attr_{key}"] = value

                if annotation.objects:
                    for i, obj in enumerate(annotation.objects):
                        row = base_row.copy()
                        row.update(
                            {
                                "object_id": i,
                                "category": obj.category,
                                "bbox_x1": obj.bbox.x1,
                                "bbox_y1": obj.bbox.y1,
                                "bbox_x2": obj.bbox.x2,
                                "bbox_y2": obj.bbox.y2,
                                "bbox_width": obj.bbox.width,
                                "bbox_height": obj.bbox.height,
                                "bbox_area": obj.bbox.area,
                                "bbox_aspect_ratio": obj.bbox.aspect_ratio,
                                "center_x": obj.bbox.center[0],
                                "center_y": obj.bbox.center[1],
                                "occluded": obj.occluded,
                                "truncated": obj.truncated,
                                "crowd": obj.crowd,
                            }
                        )

                        # Add object attributes
                        for key, value in obj.attributes.items():
                            row[f"obj_attr_{key}"] = value

                        rows.append(row)
                else:
                    # Image with no objects
                    rows.append(base_row)

            # Save to CSV
            df = pd.DataFrame(rows)
            csv_file = output_path / f"{split}_annotations.csv"
            df.to_csv(csv_file, index=False)
            exported_files[f"{split}_csv"] = str(csv_file)

            logger.info(f"Exported {len(df)} rows for {split} split to {csv_file}")

        # Export statistics
        stats_file = output_path / "parsing_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        exported_files["statistics"] = str(stats_file)

        return exported_files

    def get_parsing_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of parsing results."""
        return {
            "dataset_statistics": {
                "total_images": self.stats["total_images"],
                "total_objects": self.stats["total_objects"],
                "average_objects_per_image": self.stats["total_objects"]
                / max(self.stats["total_images"], 1),
            },
            "class_distribution": dict(self.stats["class_distribution"]),
            "split_distribution": dict(self.stats["split_distribution"]),
            "data_quality": {
                "parsing_errors": len(self.stats["parsing_errors"]),
                "missing_images": len(self.stats["missing_images"]),
                "invalid_annotations": len(self.stats["invalid_annotations"]),
            },
            "detection_classes": list(self.DETECTION_CLASSES),
        }


if __name__ == "__main__":
    # Example usage
    parser = BDDParser("/path/to/bdd100k/dataset")

    # Validate dataset structure
    validation = parser.validate_dataset_structure()
    print("Dataset Structure Validation:", validation)

    # Load all annotations
    annotations = parser.load_all_annotations()

    # Export processed data
    exported = parser.export_parsed_data(annotations, "data/processed")

    # Print summary
    summary = parser.get_parsing_summary()
    print("Parsing Summary:", json.dumps(summary, indent=2))
