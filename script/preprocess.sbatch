#!/bin/bash
#SBATCH -J preprocess
#SBATCH -N 1
#SBATCH --mem=64G          # Increased to 64G for LLM model loading
#SBATCH --cpus-per-task=8  # Increased CPUs for faster data loading
#SBATCH --gres=gpu:1       # <--- CRITICAL: Added GPU request for LLMs
#SBATCH -t 1440            # Increased to 24 hours (LLM inference takes time)
#SBATCH -o ./logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL_HERE

# --- CLUSTER SPECIFICS ---
# #SBATCH --account=YOUR_ACCOUNT_NAME
# #SBATCH -q YOUR_QOS_NAME

echo "Starting preprocessing and extraction at $(date)"

# Robust Project Root Detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p "$PROJECT_ROOT/script/logs/preprocess"

echo "Project Root: $PROJECT_ROOT"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mindtransformer_env

# =========================================================
# 1. Resample fMRI data to 4x4x4mm voxels
# =========================================================
echo "Step 1: Resampling fMRI data..."
python resample_fmri_data.py --config config_lpp.yaml

# =========================================================
# 2. Create a mask common to all subjects
# =========================================================
echo "Step 2: Computing masks..."
python compute_mask.py --config config_lpp.yaml

# =========================================================
# 3. Compute the average subject fMRI
# =========================================================
echo "Step 3: Computing average subject fMRI..."
python compute_average_subject_fmri.py --config config_lpp.yaml

echo "Step 3b: Computing per-subject fMRI..."
python compute_per_subject_fmri.py --config config_lpp.yaml

# =========================================================
# 4. Extract GloVe embeddings
# =========================================================
echo "Step 4: Extracting GloVe embeddings..."
python extract_glove_activations.py --config config_lpp.yaml

# =========================================================
# 5. Generate random baseline activations
# =========================================================
echo "Step 5: Generating random baselines..."
python generate_random_activations.py --config config_lpp.yaml

# =========================================================
# 6. Extract LLM Activations (New Section)
# =========================================================
echo "Step 6: Extracting LLM Activations..."

# Process Llama models
echo "Processing Llama..."
python extract_llm_activations.py --config config_lpp_llama.yaml

# Process Gemma models
echo "Processing Gemma..."
python extract_llm_activations.py --config config_lpp_gemma.yaml

# Process Qwen models
echo "Processing Qwen..."
python extract_llm_activations.py --config config_lpp_qwen.yaml

# Process Mistral models
echo "Processing Mistral..."
python extract_llm_activations.py --config config_lpp_mistral.yaml

# Process GPT-OSS models
echo "Processing GPT-OSS..."
python extract_llm_activations.py --config config_lpp_gptoss.yaml

echo "All preprocessing and extraction complete at $(date)"
