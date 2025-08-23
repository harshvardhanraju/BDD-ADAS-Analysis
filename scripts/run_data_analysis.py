"""
Comprehensive BDD100K Data Analysis Runner

This script coordinates all data analysis components and generates
a complete analysis report with visualizations and insights.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.parsers.bdd_parser import BDDParser
from src.analysis.class_analysis import ClassDistributionAnalyzer
from src.analysis.spatial_analysis import SpatialAnalyzer
from src.analysis.image_analysis import ImageCharacteristicsAnalyzer

def setup_logging():
    """Setup logging configuration."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/analysis/analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run comprehensive BDD100K data analysis')
    
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of BDD100K dataset')
    parser.add_argument('--output-dir', type=str, default='data/analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--skip-parsing', action='store_true',
                        help='Skip data parsing (use existing processed data)')
    parser.add_argument('--skip-image-analysis', action='store_true',
                        help='Skip image analysis (faster execution)')
    parser.add_argument('--sample-images', type=int, default=1000,
                        help='Number of images to sample for image analysis')
    parser.add_argument('--sample-quality', type=int, default=500,
                        help='Number of images to sample for quality analysis')
    
    return parser.parse_args()

def run_data_parsing(data_root: str, output_dir: str, logger) -> Dict[str, Any]:
    """Run BDD data parsing and export processed data."""
    logger.info("Starting BDD100K data parsing...")
    
    parser = BDDParser(data_root)
    
    # Validate dataset structure
    logger.info("Validating dataset structure...")
    validation = parser.validate_dataset_structure()
    
    for key, value in validation.items():
        status = "✓" if value else "✗"
        logger.info(f"{status} {key}: {value}")
    
    if not all(validation.values()):
        logger.warning("Dataset structure validation failed. Some files may be missing.")
    
    # Load annotations
    logger.info("Loading annotations...")
    annotations = parser.load_all_annotations()
    
    # Validate image-annotation pairs
    logger.info("Validating image-annotation pairs...")
    image_validation = parser.validate_image_annotation_pairs(annotations)
    
    # Export processed data
    logger.info("Exporting processed data...")
    processed_dir = Path(output_dir) / "processed"
    exported_files = parser.export_parsed_data(annotations, str(processed_dir))
    
    # Generate parsing summary
    summary = parser.get_parsing_summary()
    
    # Save detailed results
    results = {
        'validation': validation,
        'image_validation': image_validation,
        'exported_files': exported_files,
        'summary': summary,
        'annotations': annotations
    }
    
    # Save parsing results
    results_file = processed_dir / "parsing_results.json"
    with open(results_file, 'w') as f:
        # Convert annotations to serializable format for summary
        serializable_summary = summary.copy()
        json.dump(serializable_summary, f, indent=2, default=str)
    
    logger.info(f"Data parsing completed. Results saved to {results_file}")
    return results

def run_class_analysis(data: pd.DataFrame, output_dir: str, logger) -> Dict[str, Any]:
    """Run comprehensive class distribution analysis."""
    logger.info("Starting class distribution analysis...")
    
    analyzer = ClassDistributionAnalyzer(data, output_dir)
    results = analyzer.run_complete_analysis()
    
    logger.info(f"Class analysis completed. Generated {len(results['plot_files'])} plots")
    logger.info(f"Analysis report: {results['report_file']}")
    
    return results

def run_spatial_analysis(data: pd.DataFrame, image_stats: Dict, output_dir: str, logger) -> Dict[str, Any]:
    """Run comprehensive spatial analysis."""
    logger.info("Starting spatial distribution analysis...")
    
    analyzer = SpatialAnalyzer(data, image_stats, output_dir)
    results = analyzer.run_complete_analysis()
    
    logger.info(f"Spatial analysis completed. Generated {len(results['plot_files'])} plots")
    logger.info(f"Analysis report: {results['report_file']}")
    
    return results

def run_image_analysis(data: pd.DataFrame, images_root: str, output_dir: str, 
                      sample_images: int, sample_quality: int, logger) -> Dict[str, Any]:
    """Run comprehensive image characteristics analysis."""
    logger.info("Starting image characteristics analysis...")
    
    analyzer = ImageCharacteristicsAnalyzer(data, images_root, output_dir)
    results = analyzer.run_complete_analysis(
        dimension_sample=sample_images,
        quality_sample=sample_quality
    )
    
    logger.info(f"Image analysis completed. Generated {len(results['plot_files'])} plots")
    logger.info(f"Analysis report: {results['report_file']}")
    
    return results

def generate_executive_summary(all_results: Dict[str, Any], output_dir: str, logger):
    """Generate executive summary of all analyses."""
    logger.info("Generating executive summary...")
    
    summary_lines = []
    
    # Header
    summary_lines.extend([
        "# BDD100K Dataset - Executive Analysis Summary",
        "=" * 60,
        f"Analysis completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    # Dataset Overview
    if 'parsing' in all_results:
        parsing_summary = all_results['parsing']['summary']
        dataset_stats = parsing_summary.get('dataset_statistics', {})
        
        summary_lines.extend([
            "## Dataset Overview",
            f"- Total Images: {dataset_stats.get('total_images', 'N/A'):,}",
            f"- Total Objects: {dataset_stats.get('total_objects', 'N/A'):,}",
            f"- Average Objects per Image: {dataset_stats.get('average_objects_per_image', 'N/A'):.1f}",
            f"- Object Classes: {len(parsing_summary.get('detection_classes', []))}",
            ""
        ])
        
        # Class distribution
        class_dist = parsing_summary.get('class_distribution', {})
        if class_dist:
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
            summary_lines.extend([
                "### Top 5 Most Common Classes:",
                *[f"  {i+1}. {cls}: {count:,} objects" for i, (cls, count) in enumerate(sorted_classes[:5])],
                ""
            ])
    
    # Key Insights from Class Analysis
    if 'class_analysis' in all_results:
        class_results = all_results['class_analysis']
        if 'summary' in class_results:
            summary_lines.extend([
                "## Class Distribution Insights",
                f"- Most frequent class: {class_results['summary'].get('most_frequent_class', 'N/A')}",
                f"- Class imbalance ratio: {class_results['summary'].get('imbalance_ratio', 'N/A'):.2f}:1",
                ""
            ])
    
    # Spatial Analysis Insights
    if 'spatial_analysis' in all_results:
        spatial_results = all_results['spatial_analysis']
        if 'summary' in spatial_results:
            summary_lines.extend([
                "## Spatial Distribution Insights",
                f"- Total bounding boxes analyzed: {spatial_results['summary'].get('total_bboxes', 'N/A'):,}",
                f"- Average bounding box area: {spatial_results['summary'].get('avg_bbox_area', 'N/A'):.0f} pixels²",
                f"- Average aspect ratio: {spatial_results['summary'].get('avg_aspect_ratio', 'N/A'):.2f}",
                ""
            ])
    
    # Image Analysis Insights
    if 'image_analysis' in all_results:
        image_results = all_results['image_analysis']
        if 'summary' in image_results:
            summary_lines.extend([
                "## Image Characteristics Insights",
                f"- Total unique images: {image_results['summary'].get('total_unique_images', 'N/A'):,}",
                f"- Analysis components completed: {len(image_results['summary'].get('analysis_components', []))}",
                ""
            ])
    
    # Data Quality Assessment
    summary_lines.extend([
        "## Data Quality Assessment",
        ""
    ])
    
    if 'parsing' in all_results:
        data_quality = all_results['parsing']['summary'].get('data_quality', {})
        
        total_errors = sum([
            data_quality.get('parsing_errors', 0),
            data_quality.get('missing_images', 0),
            data_quality.get('invalid_annotations', 0)
        ])
        
        if total_errors == 0:
            summary_lines.append("✅ **Excellent Data Quality**: No parsing errors or missing images detected")
        elif total_errors < 100:
            summary_lines.append(f"⚠️ **Good Data Quality**: {total_errors} minor issues detected")
        else:
            summary_lines.append(f"❌ **Data Quality Issues**: {total_errors} problems detected - review required")
        
        summary_lines.extend([
            f"- Parsing errors: {data_quality.get('parsing_errors', 0)}",
            f"- Missing images: {data_quality.get('missing_images', 0)}",
            f"- Invalid annotations: {data_quality.get('invalid_annotations', 0)}",
            ""
        ])
    
    # Recommendations
    summary_lines.extend([
        "## Recommendations",
        ""
    ])
    
    # Add specific recommendations based on analysis results
    if 'class_analysis' in all_results:
        imbalance_ratio = all_results['class_analysis']['summary'].get('imbalance_ratio', 1)
        if imbalance_ratio > 10:
            summary_lines.append("⚠️ **Address Class Imbalance**: Consider data augmentation, weighted sampling, or focal loss")
        elif imbalance_ratio > 5:
            summary_lines.append("ℹ️ **Monitor Class Balance**: Ensure validation metrics account for class distribution")
    
    if 'spatial_analysis' in all_results:
        summary_lines.append("📍 **Spatial Patterns**: Review spatial distribution plots for class-specific positioning biases")
    
    summary_lines.extend([
        "🔍 **Model Selection**: Use analysis insights to inform architecture choice and training strategy",
        "📊 **Validation Strategy**: Ensure test set reflects training distribution patterns",
        "🎯 **Evaluation Metrics**: Consider class-weighted metrics given the imbalance patterns",
        ""
    ])
    
    # Generated Files Summary
    summary_lines.extend([
        "## Generated Analysis Files",
        ""
    ])
    
    for analysis_name, results in all_results.items():
        if isinstance(results, dict) and 'plot_files' in results:
            summary_lines.append(f"### {analysis_name.replace('_', ' ').title()}")
            summary_lines.append(f"- Report: {results.get('report_file', 'N/A')}")
            summary_lines.append(f"- Plots generated: {len(results['plot_files'])}")
            summary_lines.append("")
    
    # Save executive summary
    summary_file = Path(output_dir) / "reports" / "executive_summary.md"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Executive summary saved to: {summary_file}")
    return str(summary_file)

def main():
    """Main analysis execution function."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("Starting BDD100K comprehensive data analysis")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)
    (output_path / "processed").mkdir(exist_ok=True)
    
    all_results = {}
    
    try:
        # Step 1: Data Parsing
        if not args.skip_parsing:
            parsing_results = run_data_parsing(args.data_root, args.output_dir, logger)
            all_results['parsing'] = parsing_results
        else:
            logger.info("Skipping data parsing as requested")
        
        # Load processed data
        logger.info("Loading processed data for analysis...")
        
        try:
            train_data = pd.read_csv(output_path / "processed" / "train_annotations.csv")
            val_data = pd.read_csv(output_path / "processed" / "val_annotations.csv")
            combined_data = pd.concat([train_data, val_data], ignore_index=True)
            
            logger.info(f"Loaded {len(combined_data):,} annotation records")
        except FileNotFoundError:
            logger.error("Processed data files not found. Please run parsing first.")
            return 1
        
        # Step 2: Class Distribution Analysis
        logger.info("\n" + "="*50)
        class_results = run_class_analysis(combined_data, str(output_path / "plots"), logger)
        all_results['class_analysis'] = class_results
        
        # Step 3: Spatial Analysis
        logger.info("\n" + "="*50)
        
        # Get image statistics if available
        image_stats = None
        if 'parsing' in all_results and 'image_validation' in all_results['parsing']:
            image_stats = all_results['parsing']['image_validation'].get('image_statistics', {})
        
        spatial_results = run_spatial_analysis(combined_data, image_stats, str(output_path / "plots"), logger)
        all_results['spatial_analysis'] = spatial_results
        
        # Step 4: Image Analysis (optional)
        if not args.skip_image_analysis:
            logger.info("\n" + "="*50)
            images_root = Path(args.data_root) / "images" / "100k"
            
            image_results = run_image_analysis(
                combined_data, 
                str(images_root), 
                str(output_path / "plots"),
                args.sample_images,
                args.sample_quality,
                logger
            )
            all_results['image_analysis'] = image_results
        else:
            logger.info("Skipping image analysis as requested")
        
        # Step 5: Generate Executive Summary
        logger.info("\n" + "="*50)
        summary_file = generate_executive_summary(all_results, args.output_dir, logger)
        
        # Final Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 ANALYSIS COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 Executive Summary: {summary_file}")
        logger.info(f"📁 All outputs saved to: {args.output_dir}")
        
        # Count generated files
        total_plots = sum(len(r.get('plot_files', [])) for r in all_results.values() if isinstance(r, dict))
        total_reports = sum(1 for r in all_results.values() if isinstance(r, dict) and 'report_file' in r)
        
        logger.info(f"📈 Generated {total_plots} plots and {total_reports} detailed reports")
        logger.info(f"🚀 Start the dashboard: streamlit run src/visualization/dashboard.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)