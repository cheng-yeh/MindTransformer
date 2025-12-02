# The Mind's Transformer: Computational Neuroanatomy of LLM-Brain Alignment

## 📝 Overview

**MindTransformer** systematically dissects 13 intermediate computational states within transformer blocks to achieve superior brain alignment. Our framework achieves a **31% improvement** in the primary auditory cortex, surpassing the gains typically achieved by 456× model scaling.

<img src="assets/mindtransformer_overview.png" alt="13 intermediate states extracted from each transformer block inside LLMs" width="100%" style="max-width: 900px;">

---

## 📂 Project Structure

```text
.
├── config_lpp.yaml             # Main configuration template
├── config_lpp_*.yaml           # Model-specific configs (Llama, Mistral, etc.)
├── mindtransformer.py          # Core encoding framework (Modes 1 & 2)
├── requirements.txt            # Dependencies
├── script/                     # HPC Automation Scripts
│   ├── setup.sbatch            # Environment setup
│   ├── download.sbatch         # Data download
│   ├── preprocess.sbatch       # Preprocessing pipeline
│   ├── all_mindtransformer.bash # Main Experiment Controller
│   └── job_mindtransformer.sbatch
└── data/                       # Data storage
````

-----

## 🛠️ Installation

You can set up the environment manually using the commands below, or use the automated script (`script/setup.sbatch`) if you are on a Slurm cluster.

```bash
# 1. Create environment
conda create -y --name mindtransformer_env python=3.10
conda activate mindtransformer_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy models for tokenization
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download zh_core_web_sm
```

-----

## 📊 Data Setup

We utilize the **Le Petit Prince (LPP)** fMRI corpus (OpenNeuro ds003643).

```bash
# Download fMRI Data
aws s3 sync --no-sign-request s3://openneuro.org/ds003643 data/ds003643/

# Download GloVe embeddings (Baseline)
mkdir -p data/glove
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip -P data/glove/
unzip data/glove/glove.6B.zip -d data/glove/
```

-----

## 🔧 Configuration

The pipeline is controlled by YAML configuration files. You must update the authentication section to access gated models (e.g., Llama-3) and the encrypted text data.

**File:** `config_lpp_*.yaml`

```yaml
auth:
  huggingface_token: "YOUR_HF_TOKEN"  # Required for meta-llama
text_processing:
  text_archive_password: "YOUR_TEXT_PASSWORD"
```

-----

## 🔄 Scientific Pipeline

This section details the step-by-step Python commands to reproduce our computational neuroanatomy analysis.

### Step 1: fMRI Preprocessing

We standardize the fMRI data to 4x4x4mm voxels, compute intersection masks, and generate group-level signals.

```bash
# 1. Resample fMRI data to standard voxel size
python resample_fmri_data.py --config config_lpp.yaml

# 2. Compute common brain mask across subjects
python compute_mask.py --config config_lpp.yaml

# 3. Compute Average Subject (for noise ceiling & group analysis)
python compute_average_subject_fmri.py --config config_lpp.yaml

# 4. Prepare Per-Subject data (for individual validation)
python compute_per_subject_fmri.py --config config_lpp.yaml
```

### Step 2: Extract LLM Intermediate States

We decompose each transformer block into 13 distinct states. Extraction is typically run per model family to manage memory resources.

```bash
# Extract Llama (Example)
python extract_llm_activations.py --config config_lpp_llama.yaml

# Extract Mistral
python extract_llm_activations.py --config config_lpp_mistral.yaml
```

### Step 3: MindTransformer Analysis

We propose two modes to align these extracted states with brain activity. The core script is `mindtransformer.py`.

#### A. Baselines

We benchmark against static embeddings (GloVe) and random networks.

```bash
# 1. Extract baseline features
python extract_glove_activations.py --config config_lpp.yaml
python generate_random_activations.py --config config_lpp.yaml

# 2. Run encoding
python mindtransformer.py --config config_lpp.yaml --run "glove"
python mindtransformer.py --config config_lpp.yaml --run "random"
```

#### B. Mode 1: Optimal Single-State Selection

Systematically evaluates all 13 states to identify the single best predictor for each voxel. This reveals the **intra-block hierarchy**.

```bash
# Example: Fit the 'per_head_q_rope' state
python mindtransformer.py \
    --config config_lpp_llama.yaml \
    --subject "average" \
    --run "llm-mode1" \
    --layers 80 \
    --inputs "per_head_q_rope"
```

#### C. Mode 2: Multi-State Feature Integration

Learns a combined representation from multiple states using a two-stage feature selection process (using a "Pivot" layer). This yields **SOTA alignment**.

```bash
# Example: Combine attention and FFN states, using input_hidden_state as pivot
python mindtransformer.py \
    --config config_lpp_llama.yaml \
    --subject "average" \
    --run "llm-mode2" \
    --layers 80 \
    --inputs "per_head_q_rope" "ffn_activated_state" "attn_output" \
    --pivot_input "input_hidden_state" \
    --parcel "Heschl's Gyrus"
```

-----

## ⚡ HPC Automation (Slurm)

To facilitate large-scale analysis (e.g., sweeping all layers of a 70B model), we provide an automated orchestration suite in the `script/` directory.

**1. Setup & Preprocessing**

```bash
sbatch script/setup.sbatch       # Builds env
sbatch script/download.sbatch    # Downloads data
sbatch script/preprocess.sbatch  # Runs ALL Step 1 & Step 2 commands
```

**2. Experiment Runner**
The `all_mindtransformer.bash` script is the master controller. It automatically submits separate jobs for every layer and configuration to your cluster.

**Important:** You must run this script from inside the `script/` directory.

```bash
cd script/

# Configure the run mode inside the script:
# RUN="llm-mode1" (or "llm-mode2")
# SUBJECT="average"
nano all_mindtransformer.bash

# Launch the full experiment suite
./all_mindtransformer.bash
```

-----

## 🧠 Intermediate States Analyzed

We extract 13 states per block, categorized into three computational stages.

| Stage | State Key | Description |
| :--- | :--- | :--- |
| **Block Input** | `input_hidden_state` | Standard baseline input |
| | `pre_attn_norm` | Layer-normalized input |
| **Attention** | `per_head_q` / `_k` / `_v` | Query/Key/Value projections |
| | `per_head_q_rope` | Query with **Rotary Positional Embeddings** (Critical for auditory alignment) |
| | `per_head_k_rope` | Key with Rotary Positional Embeddings |
| | `per_head_context_vector` | Attention output per head |
| | `attn_output` | Multi-head attention output (projection) |
| **FFN & Residuals** | `post_attn_hidden_state` | Residual stream after Attention |
| | `pre_ffn_norm` | Before FFN layer norm |
| | `ffn_activated_state` | FFN intermediate activation (e.g., SwiGLU) |
| | `ffn_output` | Final FFN block output |

-----

## 📈 Key Results

Our computational neuroanatomy analysis reveals:

  * **Auditory Alignment:** MindTransformer Mode 2 achieves a correlation of **0.467** in Heschl's Gyrus, a **31.0% improvement** over standard baselines.
  * **RoPE Importance:** Per-Head Query with RoPE explains **73.88%** of voxels in the auditory cortex, compared to just 7.82% without RoPE.
  * **Intra-Block Hierarchy:** Early attention states map to sensory regions (HG, PT), while later FFN states map to association regions (IFG, AG).

-----

## 📚 Citation

```bibtex
@inproceedings{mindtransformer2026,
  title={The Mind's Transformer: Computational Neuroanatomy of LLM-Brain Alignment},
  author={Anonymous Authors},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
