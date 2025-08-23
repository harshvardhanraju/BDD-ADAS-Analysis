"""
Comprehensive Spatial and Bounding Box Analysis for BDD100K Dataset

This module provides detailed analysis of spatial patterns, bounding box characteristics,
and geometric properties of object annotations.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("Set2")


class SpatialAnalyzer:
    """Comprehensive spatial analysis for object detection annotations."""

    def __init__(
        self,
        data: pd.DataFrame,
        image_stats: Optional[Dict] = None,
        output_dir: str = "data/analysis/plots",
    ):
        """
        Initialize spatial analyzer.

        Args:
            data: DataFrame with bbox coordinates and class information
            image_stats: Dictionary with image dimensions if available
            output_dir: Directory to save analysis outputs
        """
        self.data = data
        self.image_stats = image_stats or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Filter out rows without bounding boxes
        bbox_columns = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        self.bbox_data = data.dropna(subset=bbox_columns).copy()

        # Add computed columns
        self.bbox_data["bbox_center_x"] = (
            self.bbox_data["bbox_x1"] + self.bbox_data["bbox_x2"]
        ) / 2
        self.bbox_data["bbox_center_y"] = (
            self.bbox_data["bbox_y1"] + self.bbox_data["bbox_y2"]
        ) / 2

        # Normalize coordinates if image dimensions are available
        if self.image_stats:
            self._normalize_coordinates()

        self.classes = sorted(self.bbox_data["category"].unique())
        self.analysis_results = {}

    def _normalize_coordinates(self):
        """
        Normalize coordinates using actual image dimensions.

        Converts absolute pixel coordinates to normalized coordinates (0-1)
        using image statistics when available.
        """
        normalized_data = []

        for split in self.bbox_data["split"].unique():
            split_data = self.bbox_data[self.bbox_data["split"] == split]

            if split in self.image_stats:
                split_stats = self.image_stats[split]

                for _, row in split_data.iterrows():
                    image_name = row["image_name"]
                    if image_name in split_stats:
                        img_width = split_stats[image_name]["width"]
                        img_height = split_stats[image_name]["height"]

                        # Normalize coordinates
                        norm_x1 = row["bbox_x1"] / img_width
                        norm_y1 = row["bbox_y1"] / img_height
                        norm_x2 = row["bbox_x2"] / img_width
                        norm_y2 = row["bbox_y2"] / img_height

                        norm_center_x = (norm_x1 + norm_x2) / 2
                        norm_center_y = (norm_y1 + norm_y2) / 2
                        norm_width = norm_x2 - norm_x1
                        norm_height = norm_y2 - norm_y1

                        normalized_data.append(
                            {
                                "image_name": image_name,
                                "category": row["category"],
                                "split": row["split"],
                                "norm_x1": norm_x1,
                                "norm_y1": norm_y1,
                                "norm_x2": norm_x2,
                                "norm_y2": norm_y2,
                                "norm_center_x": norm_center_x,
                                "norm_center_y": norm_center_y,
                                "norm_width": norm_width,
                                "norm_height": norm_height,
                                "norm_area": norm_width * norm_height,
                            }
                        )

        if normalized_data:
            self.normalized_data = pd.DataFrame(normalized_data)
        else:
            self.normalized_data = None

    def analyze_bbox_dimensions(self) -> Dict[str, Any]:
        """
        Analyze bounding box dimensions and aspect ratios.

        Computes statistics for width, height, area, and aspect ratios
        both overall and per-class. Categorizes objects by size.

        Returns:
            Dictionary containing dimension analysis and size categorization
        """
        bbox_analysis = {}

        # Overall dimension statistics
        dimension_stats = {
            "width": {
                "mean": self.bbox_data["bbox_width"].mean(),
                "std": self.bbox_data["bbox_width"].std(),
                "min": self.bbox_data["bbox_width"].min(),
                "max": self.bbox_data["bbox_width"].max(),
                "median": self.bbox_data["bbox_width"].median(),
                "q25": self.bbox_data["bbox_width"].quantile(0.25),
                "q75": self.bbox_data["bbox_width"].quantile(0.75),
            },
            "height": {
                "mean": self.bbox_data["bbox_height"].mean(),
                "std": self.bbox_data["bbox_height"].std(),
                "min": self.bbox_data["bbox_height"].min(),
                "max": self.bbox_data["bbox_height"].max(),
                "median": self.bbox_data["bbox_height"].median(),
                "q25": self.bbox_data["bbox_height"].quantile(0.25),
                "q75": self.bbox_data["bbox_height"].quantile(0.75),
            },
            "area": {
                "mean": self.bbox_data["bbox_area"].mean(),
                "std": self.bbox_data["bbox_area"].std(),
                "min": self.bbox_data["bbox_area"].min(),
                "max": self.bbox_data["bbox_area"].max(),
                "median": self.bbox_data["bbox_area"].median(),
                "q25": self.bbox_data["bbox_area"].quantile(0.25),
                "q75": self.bbox_data["bbox_area"].quantile(0.75),
            },
            "aspect_ratio": {
                "mean": self.bbox_data["bbox_aspect_ratio"].mean(),
                "std": self.bbox_data["bbox_aspect_ratio"].std(),
                "min": self.bbox_data["bbox_aspect_ratio"].min(),
                "max": self.bbox_data["bbox_aspect_ratio"].max(),
                "median": self.bbox_data["bbox_aspect_ratio"].median(),
                "q25": self.bbox_data["bbox_aspect_ratio"].quantile(0.25),
                "q75": self.bbox_data["bbox_aspect_ratio"].quantile(0.75),
            },
        }

        bbox_analysis["overall_statistics"] = dimension_stats

        # Per-class dimension analysis
        class_statistics = {}
        for class_name in self.classes:
            class_data = self.bbox_data[self.bbox_data["category"] == class_name]

            class_statistics[class_name] = {
                "count": len(class_data),
                "avg_width": class_data["bbox_width"].mean(),
                "avg_height": class_data["bbox_height"].mean(),
                "avg_area": class_data["bbox_area"].mean(),
                "avg_aspect_ratio": class_data["bbox_aspect_ratio"].mean(),
                "width_std": class_data["bbox_width"].std(),
                "height_std": class_data["bbox_height"].std(),
                "area_std": class_data["bbox_area"].std(),
                "aspect_ratio_std": class_data["bbox_aspect_ratio"].std(),
            }

        bbox_analysis["class_statistics"] = class_statistics

        # Size categories analysis
        area_percentiles = self.bbox_data["bbox_area"].quantile([0.33, 0.67]).values
        self.bbox_data["size_category"] = pd.cut(
            self.bbox_data["bbox_area"],
            bins=[0, area_percentiles[0], area_percentiles[1], float("inf")],
            labels=["Small", "Medium", "Large"],
        )

        size_distribution = (
            self.bbox_data.groupby(["category", "size_category"])
            .size()
            .unstack(fill_value=0)
        )
        bbox_analysis["size_category_distribution"] = size_distribution.to_dict()

        self.analysis_results["bbox_dimensions"] = bbox_analysis
        return bbox_analysis

    def analyze_spatial_distribution(self) -> Dict[str, Any]:
        """
        Analyze spatial distribution of object centers across images.

        Examines object positioning patterns using grid-based analysis
        and spatial clustering techniques.

        Returns:
            Dictionary containing center statistics and spatial patterns
        """
        spatial_analysis = {}

        # Overall center distribution
        center_stats = {
            "center_x": {
                "mean": self.bbox_data["bbox_center_x"].mean(),
                "std": self.bbox_data["bbox_center_x"].std(),
                "median": self.bbox_data["bbox_center_x"].median(),
            },
            "center_y": {
                "mean": self.bbox_data["bbox_center_y"].mean(),
                "std": self.bbox_data["bbox_center_y"].std(),
                "median": self.bbox_data["bbox_center_y"].median(),
            },
        }

        spatial_analysis["center_statistics"] = center_stats

        # Grid-based spatial analysis (divide image into 3x3 grid)
        if self.normalized_data is not None:
            # Use normalized data for grid analysis
            data_for_grid = self.normalized_data
            x_col, y_col = "norm_center_x", "norm_center_y"
        else:
            # Use absolute coordinates with assumption of common image size
            data_for_grid = self.bbox_data
            x_col, y_col = "bbox_center_x", "bbox_center_y"
            # Normalize by typical image dimensions if available
            if self.image_stats:
                typical_width = 1280  # Common BDD100K width
                typical_height = 720  # Common BDD100K height
                data_for_grid = data_for_grid.copy()
                data_for_grid[x_col] = data_for_grid[x_col] / typical_width
                data_for_grid[y_col] = data_for_grid[y_col] / typical_height

        # Create 3x3 grid
        x_bins = [0, 1 / 3, 2 / 3, 1.0]
        y_bins = [0, 1 / 3, 2 / 3, 1.0]

        data_for_grid["grid_x"] = pd.cut(
            data_for_grid[x_col], bins=x_bins, labels=["Left", "Center", "Right"]
        )
        data_for_grid["grid_y"] = pd.cut(
            data_for_grid[y_col], bins=y_bins, labels=["Top", "Middle", "Bottom"]
        )
        data_for_grid["grid_cell"] = (
            data_for_grid["grid_y"].astype(str)
            + "_"
            + data_for_grid["grid_x"].astype(str)
        )

        # Grid distribution overall
        grid_distribution = data_for_grid["grid_cell"].value_counts()
        spatial_analysis["grid_distribution"] = grid_distribution.to_dict()

        # Grid distribution per class
        class_grid_distribution = {}
        for class_name in self.classes:
            class_data = data_for_grid[data_for_grid["category"] == class_name]
            class_grid = class_data["grid_cell"].value_counts()
            class_grid_distribution[class_name] = class_grid.to_dict()

        spatial_analysis["class_grid_distribution"] = class_grid_distribution

        # Spatial clustering analysis
        if len(data_for_grid) > 100:  # Only if we have enough data points
            spatial_features = data_for_grid[[x_col, y_col]].values

            # Standardize features
            scaler = StandardScaler()
            spatial_features_scaled = scaler.fit_transform(spatial_features)

            # DBSCAN clustering
            clustering = DBSCAN(eps=0.1, min_samples=10)
            cluster_labels = clustering.fit_predict(spatial_features_scaled)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)

            spatial_analysis["clustering"] = {
                "n_clusters": n_clusters,
                "n_noise_points": n_noise,
                "clustering_labels": cluster_labels.tolist(),
            }

        self.analysis_results["spatial_distribution"] = spatial_analysis
        return spatial_analysis

    def analyze_class_spatial_patterns(self) -> Dict[str, Any]:
        """
        Analyze spatial patterns specific to each object class.

        Identifies class-specific positioning preferences and vertical
        distribution patterns for each object type.

        Returns:
            Dictionary containing per-class spatial pattern analysis
        """
        class_patterns = {}

        for class_name in self.classes:
            class_data = self.bbox_data[self.bbox_data["category"] == class_name]

            if len(class_data) < 10:  # Skip classes with too few samples
                continue

            # Positional preferences
            if self.normalized_data is not None:
                class_norm = self.normalized_data[
                    self.normalized_data["category"] == class_name
                ]
                center_x_mean = class_norm["norm_center_x"].mean()
                center_y_mean = class_norm["norm_center_y"].mean()
                center_x_std = class_norm["norm_center_x"].std()
                center_y_std = class_norm["norm_center_y"].std()
            else:
                center_x_mean = class_data["bbox_center_x"].mean()
                center_y_mean = class_data["bbox_center_y"].mean()
                center_x_std = class_data["bbox_center_x"].std()
                center_y_std = class_data["bbox_center_y"].std()

            # Vertical position preference analysis
            if self.normalized_data is not None:
                y_positions = class_norm["norm_center_y"]
            else:
                # Assume typical BDD100K image height for normalization
                y_positions = class_data["bbox_center_y"] / 720

            # Categorize vertical positions
            top_third = (y_positions <= 1 / 3).sum()
            middle_third = ((y_positions > 1 / 3) & (y_positions <= 2 / 3)).sum()
            bottom_third = (y_positions > 2 / 3).sum()
            total = len(y_positions)

            vertical_distribution = {
                "top_third_ratio": top_third / total,
                "middle_third_ratio": middle_third / total,
                "bottom_third_ratio": bottom_third / total,
            }

            # Determine dominant position
            dominant_position = max(vertical_distribution.items(), key=lambda x: x[1])[
                0
            ]

            class_patterns[class_name] = {
                "sample_count": len(class_data),
                "center_x_mean": center_x_mean,
                "center_y_mean": center_y_mean,
                "center_x_std": center_x_std,
                "center_y_std": center_y_std,
                "vertical_distribution": vertical_distribution,
                "dominant_vertical_position": dominant_position,
                "spatial_concentration": (
                    1 / (center_x_std * center_y_std)
                    if center_x_std > 0 and center_y_std > 0
                    else 0
                ),
            }

        self.analysis_results["class_spatial_patterns"] = class_patterns
        return class_patterns

    def detect_spatial_anomalies(self) -> Dict[str, Any]:
        """
        Detect spatial anomalies in object positions and sizes.

        Uses statistical methods to identify outliers in object dimensions,
        aspect ratios, and positioning.

        Returns:
            Dictionary containing anomaly detection results and examples
        """
        anomaly_analysis = {}

        # Size anomalies using IQR method
        Q1_area = self.bbox_data["bbox_area"].quantile(0.25)
        Q3_area = self.bbox_data["bbox_area"].quantile(0.75)
        IQR_area = Q3_area - Q1_area

        area_lower_bound = Q1_area - 1.5 * IQR_area
        area_upper_bound = Q3_area + 1.5 * IQR_area

        size_anomalies = self.bbox_data[
            (self.bbox_data["bbox_area"] < area_lower_bound)
            | (self.bbox_data["bbox_area"] > area_upper_bound)
        ]

        anomaly_analysis["size_anomalies"] = {
            "count": len(size_anomalies),
            "percentage": len(size_anomalies) / len(self.bbox_data) * 100,
            "examples": size_anomalies[["image_name", "category", "bbox_area"]]
            .head(10)
            .to_dict("records"),
        }

        # Aspect ratio anomalies
        Q1_ar = self.bbox_data["bbox_aspect_ratio"].quantile(0.25)
        Q3_ar = self.bbox_data["bbox_aspect_ratio"].quantile(0.75)
        IQR_ar = Q3_ar - Q1_ar

        ar_lower_bound = Q1_ar - 1.5 * IQR_ar
        ar_upper_bound = Q3_ar + 1.5 * IQR_ar

        ar_anomalies = self.bbox_data[
            (self.bbox_data["bbox_aspect_ratio"] < ar_lower_bound)
            | (self.bbox_data["bbox_aspect_ratio"] > ar_upper_bound)
        ]

        anomaly_analysis["aspect_ratio_anomalies"] = {
            "count": len(ar_anomalies),
            "percentage": len(ar_anomalies) / len(self.bbox_data) * 100,
            "examples": ar_anomalies[["image_name", "category", "bbox_aspect_ratio"]]
            .head(10)
            .to_dict("records"),
        }

        # Position anomalies (objects too close to edges)
        if self.normalized_data is not None:
            edge_threshold = 0.05  # 5% from edge
            edge_anomalies = self.normalized_data[
                (self.normalized_data["norm_x1"] < edge_threshold)
                | (self.normalized_data["norm_x2"] > 1 - edge_threshold)
                | (self.normalized_data["norm_y1"] < edge_threshold)
                | (self.normalized_data["norm_y2"] > 1 - edge_threshold)
            ]

            anomaly_analysis["edge_anomalies"] = {
                "count": len(edge_anomalies),
                "percentage": len(edge_anomalies) / len(self.normalized_data) * 100,
                "examples": edge_anomalies[["image_name", "category"]]
                .head(10)
                .to_dict("records"),
            }

        self.analysis_results["spatial_anomalies"] = anomaly_analysis
        return anomaly_analysis

    def create_spatial_plots(self) -> List[str]:
        """
        Create comprehensive spatial analysis visualization plots.

        Generates plots for dimensions, spatial distributions, heatmaps,
        and class-specific comparisons.

        Returns:
            List of paths to generated plot files
        """
        plot_files = []

        # 1. Bounding box dimension distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Width distribution
        self.bbox_data["bbox_width"].hist(
            bins=50, ax=axes[0, 0], alpha=0.7, color="skyblue"
        )
        axes[0, 0].set_title("Bounding Box Width Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Width (pixels)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            self.bbox_data["bbox_width"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        axes[0, 0].legend()

        # Height distribution
        self.bbox_data["bbox_height"].hist(
            bins=50, ax=axes[0, 1], alpha=0.7, color="lightcoral"
        )
        axes[0, 1].set_title("Bounding Box Height Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Height (pixels)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            self.bbox_data["bbox_height"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        axes[0, 1].legend()

        # Area distribution (log scale)
        axes[0, 2].hist(
            self.bbox_data["bbox_area"], bins=50, alpha=0.7, color="lightgreen"
        )
        axes[0, 2].set_title("Bounding Box Area Distribution", fontweight="bold")
        axes[0, 2].set_xlabel("Area (pixels²)")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_yscale("log")
        axes[0, 2].axvline(
            self.bbox_data["bbox_area"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        axes[0, 2].legend()

        # Aspect ratio distribution
        self.bbox_data["bbox_aspect_ratio"].hist(
            bins=50, ax=axes[1, 0], alpha=0.7, color="orange"
        )
        axes[1, 0].set_title("Aspect Ratio Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Aspect Ratio (Width/Height)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].axvline(
            1.0, color="black", linestyle="--", alpha=0.5, label="Square (1:1)"
        )
        axes[1, 0].axvline(
            self.bbox_data["bbox_aspect_ratio"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        axes[1, 0].legend()

        # Width vs Height scatter
        sample_size = min(5000, len(self.bbox_data))  # Sample for performance
        sample_data = self.bbox_data.sample(n=sample_size)

        scatter = axes[1, 1].scatter(
            sample_data["bbox_width"],
            sample_data["bbox_height"],
            c=sample_data.index,
            alpha=0.6,
            cmap="viridis",
            s=1,
        )
        axes[1, 1].set_title("Width vs Height Relationship", fontweight="bold")
        axes[1, 1].set_xlabel("Width (pixels)")
        axes[1, 1].set_ylabel("Height (pixels)")

        # Size category distribution
        if hasattr(self.bbox_data, "size_category"):
            size_counts = self.bbox_data["size_category"].value_counts()
            axes[1, 2].pie(
                size_counts.values, labels=size_counts.index, autopct="%1.1f%%"
            )
            axes[1, 2].set_title("Size Category Distribution", fontweight="bold")

        plt.tight_layout()
        plot_file = self.output_dir / "bbox_dimension_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plot_files.append(str(plot_file))
        plt.close()

        # 2. Spatial distribution heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Overall center heatmap
        if self.normalized_data is not None:
            x_data = self.normalized_data["norm_center_x"]
            y_data = self.normalized_data["norm_center_y"]
        else:
            # Use original coordinates with assumed normalization
            x_data = self.bbox_data["bbox_center_x"] / 1280
            y_data = self.bbox_data["bbox_center_y"] / 720

        # Sample for performance if too many points
        if len(x_data) > 10000:
            sample_idx = np.random.choice(len(x_data), 10000, replace=False)
            x_sample = x_data.iloc[sample_idx]
            y_sample = y_data.iloc[sample_idx]
        else:
            x_sample = x_data
            y_sample = y_data

        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x_sample, y_sample, bins=20)

        im1 = axes[0, 0].imshow(
            hist.T, origin="lower", extent=[0, 1, 0, 1], cmap="YlOrRd", aspect="equal"
        )
        axes[0, 0].set_title("Overall Object Center Heatmap", fontweight="bold")
        axes[0, 0].set_xlabel("Normalized X Position")
        axes[0, 0].set_ylabel("Normalized Y Position")
        plt.colorbar(im1, ax=axes[0, 0])

        # Grid-based distribution
        if "spatial_distribution" in self.analysis_results:
            grid_dist = self.analysis_results["spatial_distribution"][
                "grid_distribution"
            ]

            # Create 3x3 grid visualization
            grid_matrix = np.zeros((3, 3))
            position_map = {
                "Top_Left": (0, 0),
                "Top_Center": (0, 1),
                "Top_Right": (0, 2),
                "Middle_Left": (1, 0),
                "Middle_Center": (1, 1),
                "Middle_Right": (1, 2),
                "Bottom_Left": (2, 0),
                "Bottom_Center": (2, 1),
                "Bottom_Right": (2, 2),
            }

            for position, count in grid_dist.items():
                if position in position_map:
                    i, j = position_map[position]
                    grid_matrix[i, j] = count

            im2 = axes[0, 1].imshow(grid_matrix, cmap="Blues", aspect="equal")
            axes[0, 1].set_title("Grid-based Object Distribution", fontweight="bold")
            axes[0, 1].set_xticks([0, 1, 2])
            axes[0, 1].set_xticklabels(["Left", "Center", "Right"])
            axes[0, 1].set_yticks([0, 1, 2])
            axes[0, 1].set_yticklabels(["Top", "Middle", "Bottom"])

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    axes[0, 1].text(
                        j,
                        i,
                        f"{int(grid_matrix[i, j]):,}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

            plt.colorbar(im2, ax=axes[0, 1])

        # Class-wise spatial distribution (top 4 classes)
        top_classes = self.bbox_data["category"].value_counts().head(4).index

        for idx, class_name in enumerate(top_classes):
            if idx >= 2:  # Only show top 2 classes in remaining subplots
                break

            class_data = self.bbox_data[self.bbox_data["category"] == class_name]

            if self.normalized_data is not None:
                class_norm = self.normalized_data[
                    self.normalized_data["category"] == class_name
                ]
                class_x = class_norm["norm_center_x"]
                class_y = class_norm["norm_center_y"]
            else:
                class_x = class_data["bbox_center_x"] / 1280
                class_y = class_data["bbox_center_y"] / 720

            # Sample if too many points
            if len(class_x) > 2000:
                sample_idx = np.random.choice(len(class_x), 2000, replace=False)
                class_x = class_x.iloc[sample_idx]
                class_y = class_y.iloc[sample_idx]

            hist_class, _, _ = np.histogram2d(class_x, class_y, bins=15)

            ax_idx = (1, idx)
            im = axes[ax_idx].imshow(
                hist_class.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="YlOrRd",
                aspect="equal",
            )
            axes[ax_idx].set_title(
                f"{class_name.title()} Spatial Distribution", fontweight="bold"
            )
            axes[ax_idx].set_xlabel("Normalized X Position")
            axes[ax_idx].set_ylabel("Normalized Y Position")
            plt.colorbar(im, ax=axes[ax_idx])

        plt.tight_layout()
        plot_file = self.output_dir / "spatial_distribution_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plot_files.append(str(plot_file))
        plt.close()

        # 3. Class-specific dimension comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Box plots for dimensions by class
        class_sample = self.bbox_data.sample(
            n=min(5000, len(self.bbox_data))
        )  # Sample for performance

        # Width by class
        sns.boxplot(data=class_sample, x="category", y="bbox_width", ax=axes[0, 0])
        axes[0, 0].set_title("Width Distribution by Class", fontweight="bold")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Height by class
        sns.boxplot(data=class_sample, x="category", y="bbox_height", ax=axes[0, 1])
        axes[0, 1].set_title("Height Distribution by Class", fontweight="bold")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Area by class (log scale)
        sns.boxplot(data=class_sample, x="category", y="bbox_area", ax=axes[1, 0])
        axes[1, 0].set_title(
            "Area Distribution by Class (Log Scale)", fontweight="bold"
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Aspect ratio by class
        sns.boxplot(
            data=class_sample, x="category", y="bbox_aspect_ratio", ax=axes[1, 1]
        )
        axes[1, 1].set_title("Aspect Ratio Distribution by Class", fontweight="bold")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].axhline(
            y=1.0, color="red", linestyle="--", alpha=0.5, label="Square (1:1)"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plot_file = self.output_dir / "class_dimension_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plot_files.append(str(plot_file))
        plt.close()

        return plot_files

    def generate_spatial_report(self) -> str:
        """
        Generate comprehensive spatial analysis report in text format.

        Creates detailed report covering all spatial analysis aspects
        including dimensions, patterns, and anomalies.

        Returns:
            Path to the generated report file
        """
        report_lines = []

        report_lines.append("# BDD100K Spatial Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        if "bbox_dimensions" in self.analysis_results:
            bbox_stats = self.analysis_results["bbox_dimensions"]["overall_statistics"]

            report_lines.append("## Bounding Box Dimension Statistics")
            report_lines.append("")

            for dimension, stats in bbox_stats.items():
                report_lines.append(f"### {dimension.replace('_', ' ').title()}")
                report_lines.append(f"- Mean: {stats['mean']:.2f}")
                report_lines.append(f"- Std Dev: {stats['std']:.2f}")
                report_lines.append(f"- Median: {stats['median']:.2f}")
                report_lines.append(f"- Range: {stats['min']:.2f} - {stats['max']:.2f}")
                report_lines.append("")

            # Class-specific insights
            class_stats = self.analysis_results["bbox_dimensions"]["class_statistics"]
            report_lines.append("### Top 5 Classes by Average Area")

            # Sort classes by average area
            sorted_classes = sorted(
                class_stats.items(), key=lambda x: x[1]["avg_area"], reverse=True
            )

            for i, (class_name, stats) in enumerate(sorted_classes[:5]):
                report_lines.append(
                    f"{i+1}. {class_name}: {stats['avg_area']:.0f} pixels² "
                    f"({stats['avg_width']:.0f}×{stats['avg_height']:.0f})"
                )
            report_lines.append("")

        if "spatial_distribution" in self.analysis_results:
            spatial_stats = self.analysis_results["spatial_distribution"]

            report_lines.append("## Spatial Distribution Analysis")

            if "grid_distribution" in spatial_stats:
                grid_dist = spatial_stats["grid_distribution"]
                total_objects = sum(grid_dist.values())

                report_lines.append("### Grid Distribution (3×3)")
                for position, count in sorted(
                    grid_dist.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = count / total_objects * 100
                    report_lines.append(
                        f"- {position.replace('_', ' ')}: {count:,} ({percentage:.1f}%)"
                    )
                report_lines.append("")

            if "clustering" in spatial_stats:
                clustering = spatial_stats["clustering"]
                report_lines.append("### Spatial Clustering Analysis")
                report_lines.append(
                    f"- Number of spatial clusters: {clustering['n_clusters']}"
                )
                report_lines.append(f"- Noise points: {clustering['n_noise_points']}")
                report_lines.append("")

        if "class_spatial_patterns" in self.analysis_results:
            patterns = self.analysis_results["class_spatial_patterns"]

            report_lines.append("## Class Spatial Patterns")
            report_lines.append("")

            for class_name, pattern in patterns.items():
                vertical_dist = pattern["vertical_distribution"]
                dominant_pos = (
                    pattern["dominant_vertical_position"]
                    .replace("_", " ")
                    .replace(" ratio", "")
                )

                report_lines.append(f"### {class_name.title()}")
                report_lines.append(f"- Sample count: {pattern['sample_count']:,}")
                report_lines.append(f"- Dominant vertical position: {dominant_pos}")
                report_lines.append(
                    f"- Spatial concentration score: {pattern['spatial_concentration']:.3f}"
                )

                report_lines.append("- Vertical distribution:")
                report_lines.append(
                    f"  - Top third: {vertical_dist['top_third_ratio']:.1%}"
                )
                report_lines.append(
                    f"  - Middle third: {vertical_dist['middle_third_ratio']:.1%}"
                )
                report_lines.append(
                    f"  - Bottom third: {vertical_dist['bottom_third_ratio']:.1%}"
                )
                report_lines.append("")

        if "spatial_anomalies" in self.analysis_results:
            anomalies = self.analysis_results["spatial_anomalies"]

            report_lines.append("## Spatial Anomaly Detection")
            report_lines.append("")

            for anomaly_type, info in anomalies.items():
                anomaly_name = anomaly_type.replace("_", " ").title()
                report_lines.append(f"### {anomaly_name}")
                report_lines.append(f"- Count: {info['count']:,}")
                report_lines.append(f"- Percentage: {info['percentage']:.2f}%")

                if "examples" in info and info["examples"]:
                    report_lines.append("- Examples:")
                    for example in info["examples"][:3]:
                        report_lines.append(
                            f"  - {example['image_name']} ({example['category']})"
                        )
                report_lines.append("")

        # Save report
        report_file = self.output_dir / "spatial_analysis_report.txt"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        return str(report_file)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run all spatial analysis components in sequence.

        Executes dimension analysis, spatial distribution analysis,
        class patterns, anomaly detection, and generates plots and reports.

        Returns:
            Dictionary containing all analysis results and generated files
        """
        print("Analyzing bounding box dimensions...")
        self.analyze_bbox_dimensions()

        print("Analyzing spatial distribution...")
        self.analyze_spatial_distribution()

        print("Analyzing class spatial patterns...")
        self.analyze_class_spatial_patterns()

        print("Detecting spatial anomalies...")
        self.detect_spatial_anomalies()

        print("Creating spatial plots...")
        plot_files = self.create_spatial_plots()

        print("Generating spatial report...")
        report_file = self.generate_spatial_report()

        return {
            "analysis_results": self.analysis_results,
            "plot_files": plot_files,
            "report_file": report_file,
            "summary": {
                "total_bboxes": len(self.bbox_data),
                "classes_analyzed": len(self.classes),
                "avg_bbox_area": self.bbox_data["bbox_area"].mean(),
                "avg_aspect_ratio": self.bbox_data["bbox_aspect_ratio"].mean(),
            },
        }


if __name__ == "__main__":
    print("Spatial Analyzer for BDD100K Dataset")
    print("Load your annotation data and run spatial analysis")
    print("Example:")
    print("  df = pd.read_csv('data/processed/train_annotations.csv')")
    print("  analyzer = SpatialAnalyzer(df)")
    print("  results = analyzer.run_complete_analysis()")
