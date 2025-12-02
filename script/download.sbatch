#!/bin/bash
#SBATCH -J download_data
#SBATCH -N 1
#SBATCH --mem=16G          # Increased memory slightly for unzipping/processing if needed
#SBATCH --cpus-per-task=4
#SBATCH -t 300             # Increased time (5 hours) as OpenNeuro downloads can be slow
#SBATCH -o ./logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL_HERE

# --- CLUSTER SPECIFICS ---
# #SBATCH --account=YOUR_ACCOUNT_NAME
# #SBATCH -q YOUR_QOS_NAME

echo "Starting data download at $(date)"

# Robust Project Root Detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create logs directory for this script
mkdir -p "$PROJECT_ROOT/script/logs/download_data"

# Create a clean data directory
DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR"

echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"

# Activate environment to ensure 'aws' and 'spacy' are available
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mindtransformer_env

# ---------------------------------------------------------
# 1. Download spaCy Models
# ---------------------------------------------------------
echo "Downloading spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download zh_core_web_sm

# ---------------------------------------------------------
# 2. Download OpenNeuro Dataset (ds003643)
# ---------------------------------------------------------
echo "Downloading OpenNeuro ds003643..."
# Syncing to a specific folder in data/
aws s3 sync --no-sign-request s3://openneuro.org/ds003643 "$DATA_DIR/ds003643"

# ---------------------------------------------------------
# 3. Download GloVe Embeddings
# ---------------------------------------------------------
echo "Downloading GloVe embeddings..."
mkdir -p "$DATA_DIR/glove"
cd "$DATA_DIR/glove"

wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip -o glove.6B.zip # -o overwrites without asking
rm glove.6B.zip       # Clean up zip file to save space

echo "Download complete at $(date)"
