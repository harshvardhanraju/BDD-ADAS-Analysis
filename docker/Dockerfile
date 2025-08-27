# BDD100K Comprehensive Object Detection Evaluation Framework
# Production-ready Docker image with full evaluation capabilities

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    # Additional system dependencies
    git \
    wget \
    curl \
    build-essential \
    pkg-config \
    # For matplotlib and plotting
    libfreetype6-dev \
    libpng-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY *.py ./
COPY *.md ./

# Copy evaluation framework components
COPY src/evaluation/ ./src/evaluation/

# Create necessary directories
RUN mkdir -p data/bdd100k/images/100k/train \
    && mkdir -p data/bdd100k/images/100k/val \
    && mkdir -p data/bdd100k/labels/detection_20 \
    && mkdir -p models/trained_models \
    && mkdir -p evaluation_results \
    && mkdir -p logs

# Set proper permissions
RUN chmod +x scripts/*.py

# Install additional ML dependencies that might not be in requirements.txt
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    mlflow \
    jupyter

# Expose ports
EXPOSE 8501  # Streamlit dashboard
EXPOSE 6006  # TensorBoard
EXPOSE 8888  # Jupyter

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD python -c "import torch, torchvision, transformers, src.evaluation.metrics.coco_metrics; print('Health check passed')" || exit 1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to run evaluation pipeline\n\
run_evaluation() {\n\
    echo "🔬 Starting BDD100K Comprehensive Evaluation Pipeline"\n\
    if [ -z "$MODEL_PATH" ]; then\n\
        echo "❌ MODEL_PATH environment variable not set"\n\
        echo "Usage: docker run -e MODEL_PATH=/app/models/model.pth bdd100k-evaluation"\n\
        exit 1\n\
    fi\n\
    \n\
    if [ ! -f "$MODEL_PATH" ]; then\n\
        echo "❌ Model file not found at $MODEL_PATH"\n\
        exit 1\n\
    fi\n\
    \n\
    echo "📊 Running Phase 1: Quantitative Metrics"\n\
    python scripts/run_comprehensive_evaluation.py --model-path "$MODEL_PATH"\n\
    \n\
    echo "📈 Running Phase 2: Visualization Generation"\n\
    python scripts/generate_evaluation_visualizations.py\n\
    \n\
    echo "🔍 Running Phase 3: Failure Analysis"\n\
    python scripts/test_failure_analysis.py\n\
    \n\
    echo "🎯 Running Phase 4: Advanced Clustering"\n\
    python scripts/run_phase4_analysis.py\n\
    \n\
    echo "📋 Running Phase 5: Comprehensive Reporting"\n\
    python scripts/generate_comprehensive_report.py\n\
    \n\
    echo "🎯 Running Phase 6: Improvement Recommendations"\n\
    python scripts/generate_improvement_recommendations.py\n\
    \n\
    echo "✅ Complete evaluation pipeline finished!"\n\
    echo "📁 Results available in /app/evaluation_results"\n\
}\n\
\n\
# Function to run dashboard\n\
run_dashboard() {\n\
    echo "🚀 Starting BDD100K Analysis Dashboard"\n\
    if [ -f "src/visualization/dashboard.py" ]; then\n\
        streamlit run src/visualization/dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
    else\n\
        echo "⚠️ Dashboard not found, running simple web server for results"\n\
        cd evaluation_results && python -m http.server 8501\n\
    fi\n\
}\n\
\n\
# Function to run training\n\
run_training() {\n\
    echo "🏋️ Starting BDD100K Model Training"\n\
    python scripts/train_complete_10class_detr.py ${TRAINING_ARGS:-}\n\
}\n\
\n\
# Function to run specific script\n\
run_script() {\n\
    if [ -f "scripts/$1.py" ]; then\n\
        echo "🔧 Running script: $1"\n\
        python "scripts/$1.py" "${@:2}"\n\
    else\n\
        echo "❌ Script scripts/$1.py not found"\n\
        echo "Available scripts:"\n\
        ls scripts/*.py | sed "s|scripts/||g" | sed "s|.py||g"\n\
        exit 1\n\
    fi\n\
}\n\
\n\
# Main execution logic\n\
case "${1:-evaluation}" in\n\
    "evaluation"|"eval")\n\
        run_evaluation\n\
        ;;\n\
    "dashboard"|"dash")\n\
        run_dashboard\n\
        ;;\n\
    "training"|"train")\n\
        run_training\n\
        ;;\n\
    "script")\n\
        run_script "${@:2}"\n\
        ;;\n\
    "bash"|"shell")\n\
        /bin/bash\n\
        ;;\n\
    "help"|"--help"|"-h")\n\
        echo "BDD100K Comprehensive Evaluation Framework Docker Container"\n\
        echo ""\n\
        echo "Usage modes:"\n\
        echo "  evaluation    Run complete 6-phase evaluation pipeline (default)"\n\
        echo "  dashboard     Start Streamlit dashboard on port 8501"\n\
        echo "  training      Run model training"\n\
        echo "  script <name> Run specific script from scripts/ directory"\n\
        echo "  bash          Open interactive bash shell"\n\
        echo "  help          Show this help message"\n\
        echo ""\n\
        echo "Environment Variables:"\n\
        echo "  MODEL_PATH      Path to model file (required for evaluation)"\n\
        echo "  DATASET_PATH    Path to BDD100K dataset (default: /app/data/bdd100k)"\n\
        echo "  OUTPUT_DIR      Output directory (default: /app/evaluation_results)"\n\
        echo "  TRAINING_ARGS   Additional training arguments"\n\
        echo ""\n\
        echo "Examples:"\n\
        echo "  docker run -e MODEL_PATH=/app/models/model.pth bdd100k-evaluation"\n\
        echo "  docker run -p 8501:8501 bdd100k-evaluation dashboard"\n\
        echo "  docker run bdd100k-evaluation script run_comprehensive_evaluation --help"\n\
        echo "  docker run -it bdd100k-evaluation bash"\n\
        ;;\n\
    *)\n\
        echo "❌ Unknown command: $1"\n\
        echo "Use '\''help'\'' for usage information"\n\
        exit 1\n\
        ;;\n\
esac' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["evaluation"]