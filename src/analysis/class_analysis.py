"""
Comprehensive Class Distribution Analysis for BDD100K Dataset

This module provides in-depth statistical analysis of object classes,
including distribution analysis, imbalance detection, and co-occurrence patterns.
"""

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for consistent plotting
plt.style.use("default")
sns.set_palette("husl")


class ClassDistributionAnalyzer:
    """Comprehensive class distribution analysis for object detection dataset."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "data/analysis/plots"):
        """
        Initialize analyzer with annotation data.

        Args:
            data: DataFrame with columns ['split', 'category', 'image_name', etc.]
            output_dir: Directory to save analysis plots and reports
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Filter only detection objects (remove background/no-object rows)
        self.object_data = data[
            data["category"].notna() & (data["category"] != "")
        ].copy()

        self.classes = sorted(self.object_data["category"].unique())
        self.splits = sorted(self.data["split"].unique())

        # Analysis results storage
        self.analysis_results = {}

    def compute_basic_statistics(self) -> Dict[str, Any]:
        """
        Compute basic class distribution statistics.

        Calculates overall and per-split statistics including object counts,
        unique classes, and class imbalance metrics.

        Returns:
            Dictionary containing comprehensive class distribution statistics
        """
        stats = {}

        # Overall class distribution
        overall_counts = self.object_data["category"].value_counts()
        stats["overall_distribution"] = overall_counts.to_dict()
        stats["total_objects"] = len(self.object_data)
        stats["num_classes"] = len(self.classes)

        # Per-split statistics
        split_stats = {}
        for split in self.splits:
            split_data = self.object_data[self.object_data["split"] == split]
            split_counts = split_data["category"].value_counts()

            split_stats[split] = {
                "total_objects": len(split_data),
                "unique_classes": len(split_counts),
                "distribution": split_counts.to_dict(),
                "images_with_objects": split_data["image_name"].nunique(),
                "avg_objects_per_image": len(split_data)
                / split_data["image_name"].nunique(),
            }

        stats["split_statistics"] = split_stats

        # Class imbalance metrics
        stats["imbalance_metrics"] = self._compute_imbalance_metrics(overall_counts)

        self.analysis_results["basic_statistics"] = stats
        return stats

    def _compute_imbalance_metrics(self, class_counts: pd.Series) -> Dict[str, float]:
        """
        Compute various class imbalance metrics.

        Calculates imbalance ratio, Gini coefficient, entropy measures,
        and identifies most/least frequent classes.

        Args:
            class_counts: Pandas Series with class counts

        Returns:
            Dictionary containing imbalance metrics and class extremes
        """
        counts = class_counts.values

        # Imbalance ratio (max/min)
        imbalance_ratio = np.max(counts) / np.max([np.min(counts), 1])

        # Gini coefficient for class imbalance
        def gini_coefficient(x):
            x = np.array(x, dtype=float)
            n = len(x)
            mean_x = np.mean(x)
            return np.sum(np.abs(x[:, None] - x[None, :])) / (2 * n**2 * mean_x)

        gini = gini_coefficient(counts)

        # Entropy (higher = more balanced)
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        max_entropy = np.log2(len(counts))
        normalized_entropy = entropy / max_entropy

        return {
            "imbalance_ratio": imbalance_ratio,
            "gini_coefficient": gini,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "most_frequent_class": class_counts.index[0],
            "least_frequent_class": class_counts.index[-1],
            "most_frequent_count": int(class_counts.iloc[0]),
            "least_frequent_count": int(class_counts.iloc[-1]),
        }

    def analyze_split_consistency(self) -> Dict[str, Any]:
        """
        Analyze consistency of class distribution across train/validation splits.

        Uses chi-square test and coefficient of variation to assess whether
        class distributions are consistent between splits.

        Returns:
            Dictionary containing consistency analysis and statistical tests
        """
        consistency_analysis = {}

        # Create cross-tabulation
        crosstab = pd.crosstab(self.object_data["category"], self.object_data["split"])

        # Normalize to get proportions
        proportions = crosstab.div(crosstab.sum(axis=1), axis=0)

        # Compute chi-square test for independence
        from scipy.stats import chi2_contingency

        chi2, p_value, dof, expected = chi2_contingency(crosstab)

        consistency_analysis["chi_square_test"] = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "is_consistent": p_value > 0.05,  # Not significantly different
        }

        # Compute coefficient of variation for each class across splits
        cv_analysis = {}
        for class_name in self.classes:
            class_proportions = proportions.loc[class_name]
            mean_prop = class_proportions.mean()
            std_prop = class_proportions.std()
            cv = std_prop / mean_prop if mean_prop > 0 else np.inf

            cv_analysis[class_name] = {
                "coefficient_of_variation": cv,
                "mean_proportion": mean_prop,
                "std_proportion": std_prop,
                "proportions_by_split": class_proportions.to_dict(),
            }

        consistency_analysis["class_cv_analysis"] = cv_analysis

        # Find most and least consistent classes
        cv_values = {
            k: v["coefficient_of_variation"]
            for k, v in cv_analysis.items()
            if not np.isinf(v["coefficient_of_variation"])
        }

        if cv_values:
            consistency_analysis["most_consistent_class"] = min(
                cv_values, key=cv_values.get
            )
            consistency_analysis["least_consistent_class"] = max(
                cv_values, key=cv_values.get
            )

        self.analysis_results["split_consistency"] = consistency_analysis
        return consistency_analysis

    def analyze_co_occurrence(self) -> Dict[str, Any]:
        """
        Analyze co-occurrence patterns between object classes.

        Examines which classes frequently appear together in the same images
        and calculates isolation scores for each class.

        Returns:
            Dictionary containing co-occurrence matrices and isolation scores
        """
        # Group by image to find co-occurring classes
        image_classes = self.object_data.groupby(["split", "image_name"])[
            "category"
        ].apply(list)

        co_occurrence_matrix = pd.DataFrame(0, index=self.classes, columns=self.classes)
        co_occurrence_counts = defaultdict(int)

        # Count co-occurrences
        for classes_in_image in image_classes:
            unique_classes = list(set(classes_in_image))
            for i, class1 in enumerate(unique_classes):
                for j, class2 in enumerate(unique_classes):
                    if i != j:  # Don't count self-occurrence
                        co_occurrence_matrix.loc[class1, class2] += 1
                        pair = tuple(sorted([class1, class2]))
                        co_occurrence_counts[pair] += 1

        # Convert to conditional probabilities
        class_counts = self.object_data["category"].value_counts()
        conditional_prob_matrix = co_occurrence_matrix.div(class_counts, axis=0).fillna(
            0
        )

        # Find most frequent pairs
        top_pairs = sorted(
            co_occurrence_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        co_occurrence_analysis = {
            "co_occurrence_matrix": co_occurrence_matrix.to_dict(),
            "conditional_probability_matrix": conditional_prob_matrix.to_dict(),
            "top_co_occurring_pairs": [
                {
                    "classes": list(pair),
                    "count": count,
                    "percentage": count / len(image_classes) * 100,
                }
                for pair, count in top_pairs
            ],
        }

        # Analyze class isolation (classes that rarely co-occur)
        isolation_scores = {}
        for class_name in self.classes:
            total_occurrences = class_counts[class_name]
            co_occurrence_sum = co_occurrence_matrix.loc[class_name].sum()
            isolation_score = (
                1 - (co_occurrence_sum / total_occurrences)
                if total_occurrences > 0
                else 0
            )
            isolation_scores[class_name] = isolation_score

        co_occurrence_analysis["isolation_scores"] = isolation_scores
        co_occurrence_analysis["most_isolated_class"] = max(
            isolation_scores, key=isolation_scores.get
        )
        co_occurrence_analysis["least_isolated_class"] = min(
            isolation_scores, key=isolation_scores.get
        )

        self.analysis_results["co_occurrence"] = co_occurrence_analysis
        return co_occurrence_analysis

    def create_distribution_plots(self) -> List[str]:
        """
        Create comprehensive distribution visualization plots.

        Generates plots for overall distribution, split comparisons,
        and co-occurrence heatmaps.

        Returns:
            List of paths to generated plot files
        """
        plot_files = []

        # 1. Overall class distribution bar plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Linear scale
        overall_counts = self.object_data["category"].value_counts()
        bars1 = ax1.bar(
            range(len(overall_counts)),
            overall_counts.values,
            color=sns.color_palette("husl", len(overall_counts)),
        )
        ax1.set_xlabel("Object Classes")
        ax1.set_ylabel("Count (Linear Scale)")
        ax1.set_title(
            "Overall Class Distribution - Linear Scale", fontsize=14, fontweight="bold"
        )
        ax1.set_xticks(range(len(overall_counts)))
        ax1.set_xticklabels(overall_counts.index, rotation=45, ha="right")

        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Log scale
        bars2 = ax2.bar(
            range(len(overall_counts)),
            overall_counts.values,
            color=sns.color_palette("husl", len(overall_counts)),
        )
        ax2.set_xlabel("Object Classes")
        ax2.set_ylabel("Count (Log Scale)")
        ax2.set_title(
            "Overall Class Distribution - Log Scale", fontsize=14, fontweight="bold"
        )
        ax2.set_yscale("log")
        ax2.set_xticks(range(len(overall_counts)))
        ax2.set_xticklabels(overall_counts.index, rotation=45, ha="right")

        plt.tight_layout()
        plot_file = self.output_dir / "class_distribution_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plot_files.append(str(plot_file))
        plt.close()

        # 2. Split comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # Stacked bar chart
        split_crosstab = pd.crosstab(
            self.object_data["category"], self.object_data["split"]
        )
        split_crosstab.plot(
            kind="bar", stacked=True, ax=axes[0, 0], color=["#1f77b4", "#ff7f0e"]
        )
        axes[0, 0].set_title("Class Distribution by Split (Stacked)", fontweight="bold")
        axes[0, 0].set_xlabel("Object Classes")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend(title="Split")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Side-by-side comparison
        split_crosstab.plot(kind="bar", ax=axes[0, 1], color=["#1f77b4", "#ff7f0e"])
        axes[0, 1].set_title(
            "Class Distribution by Split (Side-by-side)", fontweight="bold"
        )
        axes[0, 1].set_xlabel("Object Classes")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend(title="Split")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Proportional view
        split_proportions = split_crosstab.div(split_crosstab.sum(axis=1), axis=0)
        split_proportions.plot(kind="bar", ax=axes[1, 0], color=["#1f77b4", "#ff7f0e"])
        axes[1, 0].set_title(
            "Class Distribution Proportions by Split", fontweight="bold"
        )
        axes[1, 0].set_xlabel("Object Classes")
        axes[1, 0].set_ylabel("Proportion")
        axes[1, 0].legend(title="Split")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Imbalance visualization
        imbalance_ratios = []
        class_names = []
        for class_name in split_crosstab.index:
            if len(self.splits) >= 2:
                ratio = (
                    split_crosstab.loc[class_name].max()
                    / split_crosstab.loc[class_name].min()
                )
                imbalance_ratios.append(ratio)
                class_names.append(class_name)

        axes[1, 1].bar(
            range(len(imbalance_ratios)), imbalance_ratios, color="red", alpha=0.7
        )
        axes[1, 1].set_title("Class Imbalance Ratio Between Splits", fontweight="bold")
        axes[1, 1].set_xlabel("Object Classes")
        axes[1, 1].set_ylabel("Max/Min Ratio")
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1, 1].axhline(y=1, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_file = self.output_dir / "split_comparison_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plot_files.append(str(plot_file))
        plt.close()

        # 3. Co-occurrence heatmap
        if (
            hasattr(self, "analysis_results")
            and "co_occurrence" in self.analysis_results
        ):
            co_occ_matrix = pd.DataFrame(
                self.analysis_results["co_occurrence"]["conditional_probability_matrix"]
            )

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                co_occ_matrix,
                annot=True,
                cmap="YlOrRd",
                fmt=".3f",
                square=True,
                cbar_kws={"label": "Conditional Probability"},
            )
            plt.title(
                "Class Co-occurrence Conditional Probability Matrix",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Given Class Present")
            plt.ylabel("Target Class")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

            plot_file = self.output_dir / "co_occurrence_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plot_files.append(str(plot_file))
            plt.close()

        return plot_files

    def create_statistical_summary_plot(self) -> str:
        """
        Create comprehensive statistical summary visualization.

        Generates multi-panel plot with imbalance metrics, distributions,
        cumulative coverage, and class comparisons.

        Returns:
            Path to the generated summary plot file
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Imbalance metrics
        if "basic_statistics" in self.analysis_results:
            metrics = self.analysis_results["basic_statistics"]["imbalance_metrics"]

            metric_names = [
                "Imbalance\nRatio",
                "Gini\nCoefficient",
                "Normalized\nEntropy",
            ]
            metric_values = [
                metrics["imbalance_ratio"],
                metrics["gini_coefficient"],
                metrics["normalized_entropy"],
            ]

            bars = axes[0, 0].bar(
                metric_names, metric_values, color=["red", "orange", "green"]
            )
            axes[0, 0].set_title("Class Imbalance Metrics", fontweight="bold")
            axes[0, 0].set_ylabel("Value")

            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        # 2. Objects per image distribution
        objects_per_image = self.data.groupby(["split", "image_name"]).size()

        for i, split in enumerate(self.splits):
            split_objects = objects_per_image[
                objects_per_image.index.get_level_values(0) == split
            ]
            axes[0, 1].hist(
                split_objects.values,
                bins=30,
                alpha=0.7,
                label=f"{split} (μ={split_objects.mean():.1f})",
                density=True,
            )

        axes[0, 1].set_title("Objects per Image Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Number of Objects")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()

        # 3. Class frequency distribution
        overall_counts = self.object_data["category"].value_counts()
        axes[0, 2].hist(
            overall_counts.values,
            bins=20,
            color="skyblue",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 2].set_title("Class Frequency Distribution", fontweight="bold")
        axes[0, 2].set_xlabel("Number of Objects")
        axes[0, 2].set_ylabel("Number of Classes")
        axes[0, 2].axvline(
            overall_counts.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {overall_counts.mean():.0f}",
        )
        axes[0, 2].axvline(
            overall_counts.median(),
            color="orange",
            linestyle="--",
            label=f"Median: {overall_counts.median():.0f}",
        )
        axes[0, 2].legend()

        # 4. Cumulative class coverage
        sorted_counts = overall_counts.sort_values(ascending=False)
        cumulative_percentage = sorted_counts.cumsum() / sorted_counts.sum() * 100

        axes[1, 0].plot(
            range(1, len(cumulative_percentage) + 1),
            cumulative_percentage,
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axes[1, 0].set_title("Cumulative Class Coverage", fontweight="bold")
        axes[1, 0].set_xlabel("Number of Top Classes")
        axes[1, 0].set_ylabel("Cumulative Percentage of Objects")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(
            y=80, color="red", linestyle="--", alpha=0.7, label="80% Coverage"
        )
        axes[1, 0].axhline(
            y=90, color="orange", linestyle="--", alpha=0.7, label="90% Coverage"
        )
        axes[1, 0].legend()

        # 5. Split distribution pie chart
        split_totals = self.object_data["split"].value_counts()
        axes[1, 1].pie(
            split_totals.values,
            labels=split_totals.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("husl", len(split_totals)),
        )
        axes[1, 1].set_title("Objects Distribution Across Splits", fontweight="bold")

        # 6. Top and bottom classes comparison
        top_5 = overall_counts.head(5)
        bottom_5 = overall_counts.tail(5)

        x_pos = np.arange(len(top_5))
        width = 0.35

        axes[1, 2].bar(
            x_pos - width / 2,
            top_5.values,
            width,
            label="Top 5 Classes",
            color="darkgreen",
            alpha=0.7,
        )
        axes[1, 2].bar(
            x_pos + width / 2,
            bottom_5.values,
            width,
            label="Bottom 5 Classes",
            color="darkred",
            alpha=0.7,
        )

        axes[1, 2].set_title("Top vs Bottom Classes", fontweight="bold")
        axes[1, 2].set_xlabel("Class Rank")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels([f"#{i+1}" for i in range(5)])
        axes[1, 2].legend()
        axes[1, 2].set_yscale("log")

        plt.tight_layout()
        plot_file = self.output_dir / "statistical_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(plot_file)

    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive text report of class analysis.

        Creates detailed report covering statistics, imbalance analysis,
        split consistency, and co-occurrence patterns.

        Returns:
            Path to the generated report file
        """
        report_lines = []

        report_lines.append("# BDD100K Class Distribution Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        if "basic_statistics" in self.analysis_results:
            stats = self.analysis_results["basic_statistics"]

            report_lines.append("## Basic Statistics")
            report_lines.append(f"- Total objects: {stats['total_objects']:,}")
            report_lines.append(f"- Number of classes: {stats['num_classes']}")
            report_lines.append("")

            report_lines.append("### Class Distribution (Top 10)")
            for i, (class_name, count) in enumerate(
                list(stats["overall_distribution"].items())[:10]
            ):
                percentage = count / stats["total_objects"] * 100
                report_lines.append(
                    f"{i+1:2d}. {class_name:<15}: {count:>8,} ({percentage:5.1f}%)"
                )
            report_lines.append("")

            # Imbalance metrics
            imb = stats["imbalance_metrics"]
            report_lines.append("### Imbalance Analysis")
            report_lines.append(
                f"- Imbalance Ratio (max/min): {imb['imbalance_ratio']:.2f}"
            )
            report_lines.append(
                f"- Gini Coefficient: {imb['gini_coefficient']:.3f} (0=perfect equality, 1=perfect inequality)"
            )
            report_lines.append(
                f"- Normalized Entropy: {imb['normalized_entropy']:.3f} (1=perfect balance, 0=perfect imbalance)"
            )
            report_lines.append(
                f"- Most frequent class: {imb['most_frequent_class']} ({imb['most_frequent_count']:,} objects)"
            )
            report_lines.append(
                f"- Least frequent class: {imb['least_frequent_class']} ({imb['least_frequent_count']:,} objects)"
            )
            report_lines.append("")

            # Split statistics
            report_lines.append("### Split Statistics")
            for split, split_stats in stats["split_statistics"].items():
                report_lines.append(f"#### {split.upper()} Split")
                report_lines.append(
                    f"- Total objects: {split_stats['total_objects']:,}"
                )
                report_lines.append(
                    f"- Images with objects: {split_stats['images_with_objects']:,}"
                )
                report_lines.append(
                    f"- Average objects per image: {split_stats['avg_objects_per_image']:.1f}"
                )
                report_lines.append("")

        if "split_consistency" in self.analysis_results:
            consistency = self.analysis_results["split_consistency"]
            report_lines.append("## Split Consistency Analysis")

            chi2_test = consistency["chi_square_test"]
            report_lines.append(
                f"- Chi-square test p-value: {chi2_test['p_value']:.6f}"
            )
            report_lines.append(
                f"- Splits are {'consistent' if chi2_test['is_consistent'] else 'inconsistent'} (α=0.05)"
            )

            if "most_consistent_class" in consistency:
                report_lines.append(
                    f"- Most consistent class across splits: {consistency['most_consistent_class']}"
                )
                report_lines.append(
                    f"- Least consistent class across splits: {consistency['least_consistent_class']}"
                )
            report_lines.append("")

        if "co_occurrence" in self.analysis_results:
            co_occ = self.analysis_results["co_occurrence"]
            report_lines.append("## Co-occurrence Analysis")
            report_lines.append("### Top Co-occurring Class Pairs")

            for i, pair_info in enumerate(co_occ["top_co_occurring_pairs"][:5]):
                classes = pair_info["classes"]
                count = pair_info["count"]
                percentage = pair_info["percentage"]
                report_lines.append(
                    f"{i+1}. {classes[0]} & {classes[1]}: {count:,} images ({percentage:.1f}%)"
                )

            report_lines.append("")
            report_lines.append(
                "### Class Isolation Scores (0=always co-occurs, 1=always alone)"
            )
            isolation = co_occ["isolation_scores"]
            sorted_isolation = sorted(
                isolation.items(), key=lambda x: x[1], reverse=True
            )

            for class_name, score in sorted_isolation[:5]:
                report_lines.append(f"- {class_name}: {score:.3f}")

            report_lines.append("")

        # Save report
        report_file = self.output_dir / "class_analysis_report.txt"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        return str(report_file)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run all analysis components and generate comprehensive outputs.

        Executes statistics, consistency analysis, co-occurrence analysis,
        creates plots, and generates report.

        Returns:
            Dictionary containing all analysis results and generated files
        """
        print("Running basic statistics analysis...")
        self.compute_basic_statistics()

        print("Analyzing split consistency...")
        self.analyze_split_consistency()

        print("Analyzing co-occurrence patterns...")
        self.analyze_co_occurrence()

        print("Creating distribution plots...")
        plot_files = self.create_distribution_plots()

        print("Creating statistical summary plot...")
        summary_plot = self.create_statistical_summary_plot()
        plot_files.append(summary_plot)

        print("Generating analysis report...")
        report_file = self.generate_analysis_report()

        return {
            "analysis_results": self.analysis_results,
            "plot_files": plot_files,
            "report_file": report_file,
            "summary": {
                "total_objects": self.analysis_results["basic_statistics"][
                    "total_objects"
                ],
                "num_classes": len(self.classes),
                "splits": self.splits,
                "most_frequent_class": self.analysis_results["basic_statistics"][
                    "imbalance_metrics"
                ]["most_frequent_class"],
                "imbalance_ratio": self.analysis_results["basic_statistics"][
                    "imbalance_metrics"
                ]["imbalance_ratio"],
            },
        }


if __name__ == "__main__":
    # Example usage
    print("Class Distribution Analyzer")
    print("Load your data using the BDD parser first, then run this analysis")
    print("Example:")
    print("  df = pd.read_csv('data/processed/train_annotations.csv')")
    print("  analyzer = ClassDistributionAnalyzer(df)")
    print("  results = analyzer.run_complete_analysis()")
