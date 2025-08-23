#!/bin/bash

# BDD100K Analysis Docker Runner Script
# Usage: ./docker/run_docker.sh [analysis|dashboard|complete] [path-to-bdd100k-dataset]

set -e

# Configuration
IMAGE_NAME="bdd-analysis"
CONTAINER_PREFIX="bdd"
RESULTS_DIR="./docker/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Show usage
show_usage() {
    echo "Usage: $0 [MODE] [DATASET_PATH]"
    echo ""
    echo "Modes:"
    echo "  analysis   - Run data analysis only"
    echo "  dashboard  - Run dashboard only (requires existing results)"
    echo "  complete   - Run analysis then dashboard"
    echo ""
    echo "Example:"
    echo "  $0 analysis /path/to/bdd100k/dataset"
    echo "  $0 dashboard"
    echo "  $0 complete /path/to/bdd100k/dataset"
}

# Validate inputs
if [ $# -lt 1 ]; then
    print_error "Missing mode argument"
    show_usage
    exit 1
fi

MODE=$1
DATASET_PATH=${2:-""}

# Validate mode
if [[ ! "$MODE" =~ ^(analysis|dashboard|complete)$ ]]; then
    print_error "Invalid mode: $MODE"
    show_usage
    exit 1
fi

# Check if dataset path is required
if [[ "$MODE" == "analysis" || "$MODE" == "complete" ]]; then
    if [ -z "$DATASET_PATH" ]; then
        print_error "Dataset path is required for mode: $MODE"
        show_usage
        exit 1
    fi
    
    if [ ! -d "$DATASET_PATH" ]; then
        print_error "Dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    
    # Convert to absolute path
    DATASET_PATH=$(realpath "$DATASET_PATH")
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
RESULTS_PATH=$(realpath "$RESULTS_DIR")

print_info "BDD100K Analysis Docker Runner"
print_info "Mode: $MODE"
print_info "Results will be saved to: $RESULTS_PATH"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Build Docker image
print_info "Building Docker image..."
docker build -t "$IMAGE_NAME" -f docker/Dockerfile . || {
    print_error "Failed to build Docker image"
    exit 1
}
print_success "Docker image built successfully"

# Clean up existing containers
print_info "Cleaning up existing containers..."
docker rm -f "${CONTAINER_PREFIX}-analysis" "${CONTAINER_PREFIX}-dashboard" 2>/dev/null || true

# Run based on mode
case $MODE in
    "analysis")
        print_info "Running data analysis..."
        print_info "Dataset path: $DATASET_PATH"
        
        docker run -it --rm \
            --name "${CONTAINER_PREFIX}-analysis" \
            -v "$DATASET_PATH:/data:ro" \
            -v "$RESULTS_PATH:/app/data/analysis" \
            "$IMAGE_NAME" \
            python scripts/run_data_analysis.py --data-root /data --output-dir /app/data/analysis
        
        if [ $? -eq 0 ]; then
            print_success "Analysis completed successfully!"
            print_info "Results saved to: $RESULTS_PATH"
            print_info "To view dashboard: $0 dashboard"
        else
            print_error "Analysis failed"
            exit 1
        fi
        ;;
        
    "dashboard")
        print_info "Starting dashboard..."
        
        # Check if results exist
        if [ ! -f "$RESULTS_PATH/processed/train_annotations.csv" ]; then
            print_warning "No processed data found. Running analysis first..."
            if [ -z "$DATASET_PATH" ]; then
                print_error "Dataset path required for analysis"
                exit 1
            fi
            $0 analysis "$DATASET_PATH"
        fi
        
        print_info "Dashboard will be available at: http://localhost:8501"
        print_info "Press Ctrl+C to stop the dashboard"
        
        docker run -it --rm \
            --name "${CONTAINER_PREFIX}-dashboard" \
            -p 8501:8501 \
            -v "$RESULTS_PATH:/app/data/analysis:ro" \
            "$IMAGE_NAME" \
            streamlit run src/visualization/dashboard.py --server.address 0.0.0.0
        ;;
        
    "complete")
        print_info "Running complete analysis and dashboard..."
        print_info "Dataset path: $DATASET_PATH"
        
        # Run analysis first
        docker run -it --rm \
            --name "${CONTAINER_PREFIX}-analysis" \
            -v "$DATASET_PATH:/data:ro" \
            -v "$RESULTS_PATH:/app/data/analysis" \
            "$IMAGE_NAME" \
            python scripts/run_data_analysis.py --data-root /data --output-dir /app/data/analysis
        
        if [ $? -eq 0 ]; then
            print_success "Analysis completed successfully!"
            print_info "Starting dashboard..."
            print_info "Dashboard will be available at: http://localhost:8501"
            print_info "Press Ctrl+C to stop the dashboard"
            
            docker run -it --rm \
                --name "${CONTAINER_PREFIX}-dashboard" \
                -p 8501:8501 \
                -v "$RESULTS_PATH:/app/data/analysis:ro" \
                "$IMAGE_NAME" \
                streamlit run src/visualization/dashboard.py --server.address 0.0.0.0
        else
            print_error "Analysis failed"
            exit 1
        fi
        ;;
esac

print_success "Docker operation completed successfully!"