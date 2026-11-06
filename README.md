# Dynamic Multi-Adapter Routing: Improving Generalization Loss in LLMs without Resource Overhead

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Routers-blue)](https://huggingface.co/mutli-adapter-routing/multi-adapter)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets%20%26%20Logs-orange)](https://huggingface.co/datasets/mutli-adapter-routing/multi-adapter-datasets)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C.svg)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-0.19.1-yellow.svg)](https://github.com/huggingface/peft)



## Architecture Overview


1. **Base Feature Extractors (6D Probability Vector):**
   - **Syntax Router (`SyntaxRouter`):** A TF-IDF vectorizer (word and punctuation n-grams ranging from 1 to 3, `min_df=100`, `max_features=10000`) paired with a Logistic Regression classifier to detect structural patterns (`def`, `import math`, equations). Outputs `[syn_math, syn_code, syn_general]`.
   - **Semantic Router (`SemanticRouter`):** Encodes prompts with `sentence-transformers/all-mpnet-base-v2`, computes Cosine Similarity against pre-calculated domain centroids, and applies a Softmax transformation. Outputs `[sem_math, sem_code, sem_general]`.
2. **Random Forest Meta-Classifier (`MetaRouter`):**
   - A Random Forest classifier (`n_estimators=100`, `max_depth=4`, `class_weight="balanced"`) that maps the 6D feature vector to one of four routing actions:
     - `math`  Activates the **Math LoRA Adapter**.
     - `code`  Activates the **Code LoRA Adapter**.
     - `hybrid`  Activates the **TIES-Merged Adapter**.
     - `general`  Disables all adapters and uses the **Frozen Base Model**.


## Repository Structure

```text
confidence-driven-lora/
├── config/
│   ├── domain_registry.py               # Domain-specific decoding hyperparameters
│   └── router_dataset_configs.py        # Dataset paths and split rules for router training
├── src/
│   ├── data_ops/
│   │   ├── base_preprocessor.py         # Abstract Hugging Face dataset processor
│   │   ├── gsm8k_preprocessor.py        # GSM8K math formatting pipeline
│   │   ├── codefeedback_preprocessor.py # Python-filtered CodeFeedback pipeline
│   │   └── routing_dataset_manager.py   # Unified loader for SQuAD, UltraChat, and domain data
│   ├── model_ops/
│   │   ├── lora_manager.py              # 4-bit NF4 QLoRA initialization and PEFT injection
│   │   └── inference_manager.py         # Multi-adapter VRAM manager and batched generator
│   ├── router/
│   │   ├── syntax_router.py             # TF-IDF + Logistic Regression router
│   │   ├── semantic_router.py           # MPNet centroid cosine-similarity router
│   │   └── meta_router.py               # 6D Random Forest Meta-Classifier & guardrails
│   ├── training/
│       └── trainer.py                   # Custom Hugging Face Trainer with -100 prompt masking
├── scripts/
│   ├── train_adapter.py                 # CLI to train single-domain or mixed QLoRA adapters
│   ├── ties_merge.py                    # Offline TIES weight fusion for Math + Code adapters
│   ├── build_syntax_router.py           # Fit and evaluate the TF-IDF Syntax Router
│   ├── build_semantic_router.py         # Fit and evaluate the MPNet Semantic Router
│   ├── train_meta_router.py             # Fit and evaluate the Random Forest Meta-Router
│   ├── eval_end_to_end_grid.py          # End-to-end batched evaluation across 5 configs x 5 datasets
├── models/                              # Local directory for downloaded/trained routers & adapters
```

## Environment Setup

The experiments were executed on a Google Colab NVIDIA T4 GPU (16 GB VRAM) using the following core dependencies:

- `torch==2.11.0`
- `transformers==5.13.1`
- `peft==0.19.1`
- `bitsandbytes`
- `datasets`
- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `python-dotenv`
- `huggingface-hub`

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/ozgur-coban/confidence-driven-lora.git
cd confidence-driven-lora
pip install torch==2.11.0 transformers==5.13.1 peft==0.19.1 bitsandbytes datasets sentence-transformers scikit-learn pandas matplotlib seaborn python-dotenv huggingface-hub tqdm joblib
```

### 2. Configure Hugging Face Token

Accessing `meta-llama/Llama-3.2-3B` requires a Hugging Face access token. Create a `.env` file in the root directory:

```dotenv
HF_TOKEN=your_huggingface_token_here
```

## Quickstart: Using Pre-Trained Artifacts

You can download all trained LoRA adapters and router artifacts directly from [mutli-adapter-routing/multi-adapter](https://huggingface.co/mutli-adapter-routing/multi-adapter):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mutli-adapter-routing/multi-adapter",
    local_dir="./models"
)
```

## Reproducing the Training Pipeline

### 1. Train Domain Expert LoRA Adapters

Adapters are trained on the unaligned `meta-llama/Llama-3.2-3B` base model for 1 epoch using 4-bit NF4 QLoRA (bitsandbytes), `adamw_torch`, $r=32$, $\\alpha=64$, and -100 label masking on prompt tokens.

**Train the Math Adapter (GSM8K - 7,500 samples):**

```bash
python scripts/train_adapter.py \
  --model_id "meta-llama/Llama-3.2-3B" \
  --adapter_name "math_gsm8k_main_3b_base" \
  --dataset "openai/gsm8k" \
  --data_size 7500 \
  --max_length 512 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 1 \
  --learning_rate 3e-4 \
  --warmup_steps 50 \
  --r 32 \
  --lora_alpha 64 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --seed 42 \
  --output_dir "./models/adapters"
```

**Train the Code Adapter (CodeFeedback Python - 15,000 samples):**

```bash
python scripts/train_adapter.py \
  --model_id "meta-llama/Llama-3.2-3B" \
  --adapter_name "code_codefeedback_main_3b_base" \
  --dataset "m-a-p/CodeFeedback-Filtered-Instruction" \
  --data_size 15000 \
  --max_length 512 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 1 \
  --learning_rate 3e-4 \
  --warmup_steps 50 \
  --r 32 \
  --lora_alpha 64 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --seed 42 \
  --output_dir "./models/adapters"
```

### 2. Fuse Adapters with Offline TIES-Merging

Combine the trained Math and Code adapters into a static hybrid baseline (density=0.8, equal weights 0.5 / 0.5):

```bash
python scripts/ties_merge.py \
  --base_model_id "meta-llama/Llama-3.2-3B" \
  --math_adapter "./models/adapters/math_gsm8k_main_3b_base/final" \
  --code_adapter "./models/adapters/code_codefeedback_main_3b_base/final" \
  --output_dir "./models/adapters/hybrid_ties_code_math_3b" \
  --density 0.8 \
  --weight 0.5
```

### 3. Train the Routers

Train the base Syntax and Semantic routers first, then fit the Random Forest Meta-Classifier on their 6D probability outputs:

```bash
# 1. Fit TF-IDF + Logistic Regression Syntax Router
python scripts/build_syntax_router.py --enable_training --specialist_samples 6000

# 2. Fit MPNet Centroid Semantic Router
python scripts/build_semantic_router.py --enable_training --specialist_samples 6000 --threshold 0.25

# 3. Fit 6D Random Forest Meta-Router
python scripts/train_meta_router.py --enable_training --specialist_samples 240 --test_samples 60
```

## Evaluation & Visualization

To run the full evaluation grid across all 5 model configurations (base, pure_math, pure_code, hybrid_adapter, routed) and all 5 evaluation datasets (gsm8k, codefeedback, mixed, hybrid, wildchat):

```bash
python scripts/eval_end_to_end_grid.py \
  --datasets all \
  --configs all \
  --batch_size 8 \
  --math_adapter "./models/math_adapter_3b" \
  --code_adapter "./models/code_adapter_3b" \
  --hybrid_adapter "./models/hybrid_ties_code_math_3b" \
  --out_dir "logs/eval_grid"
```
