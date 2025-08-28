#!/usr/bin/env python3
"""
Complete 6-Phase Evaluation Pipeline for BDD100K Model (1000 images subset)
Runs all evaluation phases sequentially with 1000 validation images for faster processing.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and handle errors."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"✅ Completed: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description}")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    """Run complete 6-phase evaluation pipeline with 1000 images."""
    
    # Configuration
    model_path = "checkpoints/complete_10class_demo/checkpoint_epoch_048.pth"
    data_dir = "data/analysis/processed_10class_corrected"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    base_output_dir = "evaluation_results_48epoch_1000images"
    confidence_threshold = 0.02
    max_images = 1000  # Use only 1000 images for faster processing
    
    # Create base output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("🚀 STARTING COMPLETE 6-PHASE EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Images: {max_images} validation images")
    logger.info(f"Output Directory: {base_output_dir}")
    logger.info("="*80)
    
    # Phase 1: Quantitative Analysis
    logger.info("📊 PHASE 1: QUANTITATIVE ANALYSIS")
    phase1_output = f"{base_output_dir}/phase1_quantitative"
    cmd = f"""python3 scripts/run_comprehensive_evaluation.py \
        --model-path {model_path} \
        --data-dir {data_dir} \
        --images-root {images_root} \
        --output-dir {phase1_output} \
        --confidence-threshold {confidence_threshold} \
        --max-images {max_images}"""
    
    if not run_command(cmd, "Phase 1: Quantitative Analysis"):
        logger.error("Phase 1 failed. Stopping pipeline.")
        return False
    
    # Phase 2: Visualization Generation
    logger.info("📈 PHASE 2: VISUALIZATION GENERATION")
    phase2_output = f"{base_output_dir}/phase2_visualizations"
    results_path = f"{phase1_output}/evaluation_results.json"
    cmd = f"""python3 scripts/generate_evaluation_visualizations.py \
        --results-path {results_path} \
        --output-dir {phase2_output}"""
    
    if not run_command(cmd, "Phase 2: Visualization Generation"):
        logger.warning("Phase 2 failed, but continuing with other phases.")
    
    # Phase 3: Failure Analysis
    logger.info("🔍 PHASE 3: FAILURE ANALYSIS")
    phase3_output = f"{base_output_dir}/phase3_failure_analysis"
    cmd = f"""python3 scripts/analyze_model_failures.py \
        --results-path {results_path} \
        --model-path {model_path} \
        --data-dir {data_dir} \
        --images-root {images_root} \
        --output-dir {phase3_output} \
        --max-samples {max_images}"""
    
    if not run_command(cmd, "Phase 3: Failure Analysis"):
        logger.warning("Phase 3 failed, but continuing with other phases.")
    
    # Phase 4: Performance Clustering
    logger.info("🎯 PHASE 4: PERFORMANCE CLUSTERING")
    phase4_output = f"{base_output_dir}/phase4_clustering"
    cmd = f"""python3 scripts/run_performance_clustering.py \
        --results-path {results_path} \
        --output-dir {phase4_output} \
        --max-samples {max_images}"""
    
    if not run_command(cmd, "Phase 4: Performance Clustering"):
        logger.warning("Phase 4 failed, but continuing with other phases.")
    
    # Phase 5: Comprehensive Reporting
    logger.info("📋 PHASE 5: COMPREHENSIVE REPORTING")
    phase5_output = f"{base_output_dir}/phase5_reports"
    cmd = f"""python3 scripts/generate_comprehensive_reports.py \
        --results-path {results_path} \
        --output-dir {phase5_output} \
        --model-info "BDD100K DETR Epoch 48" \
        --dataset-info "{max_images} validation images" \
        --evaluation-date "{datetime.now().strftime('%Y-%m-%d')}" """
    
    if not run_command(cmd, "Phase 5: Comprehensive Reporting"):
        logger.warning("Phase 5 failed, but continuing with final phase.")
    
    # Phase 6: Improvement Recommendations
    logger.info("💡 PHASE 6: IMPROVEMENT RECOMMENDATIONS")
    phase6_output = f"{base_output_dir}/phase6_recommendations"
    cmd = f"""python3 scripts/generate_improvement_recommendations.py \
        --results-path {results_path} \
        --output-dir {phase6_output} \
        --model-path {model_path}"""
    
    if not run_command(cmd, "Phase 6: Improvement Recommendations"):
        logger.warning("Phase 6 failed.")
    
    # Final Summary
    logger.info("="*80)
    logger.info("🎉 EVALUATION PIPELINE COMPLETED")
    logger.info("="*80)
    logger.info(f"Results saved to: {base_output_dir}")
    
    # List generated artifacts
    logger.info("Generated Artifacts:")
    for phase_dir in Path(base_output_dir).glob("phase*"):
        if phase_dir.is_dir():
            file_count = len(list(phase_dir.rglob("*")))
            logger.info(f"  {phase_dir.name}: {file_count} files")
    
    logger.info("="*80)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)