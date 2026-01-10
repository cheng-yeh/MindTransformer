#!/bin/bash
#SBATCH -J job_mindtransformer
#SBATCH -N 1
#SBATCH --mem=64G          # Adjusted to reasonable default (300G is huge, user can adjust)
#SBATCH --cpus-per-task=6
#SBATCH -t 2880            # 48 hours
#SBATCH -o ./logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL_HERE

# --- CLUSTER SPECIFICS ---
# #SBATCH --account=YOUR_ACCOUNT_NAME
# #SBATCH -q YOUR_QOS_NAME

echo "Starting preprocessing and extraction at $(date)"

# Project Root Detection
PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"
cd "$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"

echo "Loading required modules..."
# NOTE: Users should check their cluster's module naming conventions
# module load anaconda3/2023.03
# module load cuda/11.8.0

# Set Hugging Face cache directory
export HF_HOME=/storage/scratch1/3/cchen847/.cache/huggingface_cache

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mindtransformer_env


# --- Argument Handling ---
# Expecting 7 arguments (Parcel is the 7th, strictly positional)
if [ "$#" -ne 7 ]; then
    echo "Error: Illegal number of parameters. Expected 7, got $#."
    echo "Usage: sbatch job_mindtransformer.sbatch <config> <subj> <run_type> <layer> <inputs> <pivot> <parcel_arg>"
    exit 1
fi

CONFIG_FILE=$1
SUBJECT_ARG=$2
RUN_TYPE=$3
LAYER_NUM=$4
LLM_INPUTS_ARG=$5 
PIVOT_INPUT_ARG=$6 
PARCEL_ARG=$7      # Can be a specific parcel or "internal_parcel_list"

# Create log directory
mkdir -p ./logs

echo "------------------------------------------------"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: ${CONFIG_FILE}"
echo "Subject: ${SUBJECT_ARG}"
echo "Run Type: ${RUN_TYPE}"
echo "Layer: ${LAYER_NUM}"
echo "Inputs: ${LLM_INPUTS_ARG}"
echo "Pivot: ${PIVOT_INPUT_ARG}"
echo "Parcel Arg: ${PARCEL_ARG}"
echo "------------------------------------------------"

# --- Environment Setup ---
echo "Loading modules..."
# NOTE: Users should check their cluster's module naming conventions
# module load anaconda3/2023.03
# module load cuda/11.8.0

# Robust Project Root Detection
# Assumes this script is in ./ (project root) or ./script/
# If submitted via all_mindtransformer.bash, we are usually in project root.
if [ -d "script" ]; then
    # We are in project root
    PROJECT_ROOT=$(pwd)
else
    # We might be in script dir
    PROJECT_ROOT="$(cd ".." && pwd)"
fi
cd "$PROJECT_ROOT"
echo "Working Dir: $(pwd)"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mindtransformer_env

# Optional: Set HF Cache if needed
# export HF_HOME=/path/to/your/cache

# Base Python Command (Renamed to mindtransformer.py as requested)
PYTHON_CMD_BASE="python mindtransformer.py --output_suffix \"\" --subject \"${SUBJECT_ARG}\" --run ${RUN_TYPE}"

# --- EXECUTION LOGIC ---

if [[ "$RUN_TYPE" == "llm-mode1" ]]; then
    # === MODE 1: Loop over features (No Parcels) ===
    echo "Running LLM-MODE1 (Per-Feature Loop)..."

    PIVOT_CMD=""
    if [[ -n "$PIVOT_INPUT_ARG" && "$PIVOT_INPUT_ARG" != "None" ]]; then
        PIVOT_CMD="--pivot_input \"${PIVOT_INPUT_ARG}\""
    fi

    for single_input in ${LLM_INPUTS_ARG}; do
        echo ">> Feature: ${single_input} (Layer ${LAYER_NUM})"
        CMD="${PYTHON_CMD_BASE} --config ${CONFIG_FILE} --layers ${LAYER_NUM} --inputs \"${single_input}\" ${PIVOT_CMD}"
        echo "Exec: ${CMD}"
        eval ${CMD}
    done

elif [[ "$RUN_TYPE" == "llm-mode2" ]]; then
    # === MODE 2: Internal Loop over Parcels ===
    echo "Running LLM-MODE2 (Internal Parcel Loop)..."

    # 1. Define the Parcels List internally
    # You can uncomment "all" if you want full brain analysis
    # PARCELS=("all")
    
    PARCELS=(
        "Heschl's Gyrus (includes H1 and H2)"
        "Planum Temporale"
        "Superior Temporal Gyrus, posterior division"
        "Superior Temporal Gyrus, anterior division"
        "Middle Temporal Gyrus, temporooccipital part"
        "Middle Temporal Gyrus, posterior division"
        "Middle Temporal Gyrus, anterior division"
        "Inferior Frontal Gyrus, pars opercularis"
        "Inferior Frontal Gyrus, pars triangularis"
        "Angular Gyrus"
    )

    # 2. Prepare Pivot
    PIVOT_CMD=""
    if [[ -n "$PIVOT_INPUT_ARG" && "$PIVOT_INPUT_ARG" != "None" ]]; then
        PIVOT_CMD="--pivot_input \"${PIVOT_INPUT_ARG}\""
    fi
    
    # 3. Loop through Parcels
    for parcel in "${PARCELS[@]}"; do
        echo "=========================================================="
        echo ">> Processing Parcel: ${parcel}"
        echo "=========================================================="
        
        # Pass the specific parcel to the python script
        CMD="${PYTHON_CMD_BASE} --config ${CONFIG_FILE} --layers ${LAYER_NUM} --inputs ${LLM_INPUTS_ARG} ${PIVOT_CMD} --parcel \"${parcel}\""
        
        echo "Exec: ${CMD}"
        eval ${CMD}
        
        echo ">> Finished Parcel: ${parcel}"
        echo ""
    done

elif [[ "$RUN_TYPE" == "glove" || "$RUN_TYPE" == "random" ]]; then
    # === BASELINE ===
    echo "Running Baseline ${RUN_TYPE}..."
    CMD="${PYTHON_CMD_BASE} --config ${CONFIG_FILE}"
    echo "Exec: ${CMD}"
    eval ${CMD}

else
    echo "Error: Invalid RUN_TYPE '${RUN_TYPE}'."
    exit 1
fi

echo "------------------------------------------------"
echo "Job Finished: $(date)"
echo "------------------------------------------------"
