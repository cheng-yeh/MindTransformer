#!/bin/bash
#SBATCH -J setup
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -t 120
#SBATCH -o ./logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL_HERE

# --- CLUSTER SPECIFICS ---
# #SBATCH --account=YOUR_ACCOUNT_NAME
# #SBATCH -q YOUR_QOS_NAME

echo "Starting environment setup at $(date)"

# Robust Project Root Detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p "$PROJECT_ROOT/script/logs/setup"

echo "Loading required modules..."
# NOTE: Users should check their cluster's module naming conventions
# module load anaconda3/2023.03
# module load cuda/11.8.0

# With Anaconda
conda create -y --name mindtransformer_env python=3.10
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mindtransformer_env

pip install -r requirements.txt
pip install --upgrade transformers

echo "Environment setup complete! Please run 'sbatch script/download.sbatch' next."
