  

#  Confidence Driven LoRA (v0.1)

**v0.1** of the confidence-driven-lora. This project implements a lightweight, high-speed routing architecture built on top of **Meta-Llama-3.1-8B-Instruct**.

Instead of relying on a massive, slow model to do everything, this system uses a classic Machine Learning router (Semantic + Syntax analysis) to intercept user prompts and dynamically swap specialized **LoRA Adapters** (Math, Code, Medical) into VRAM on the fly.

## Architecture

1. **The Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (Quantized to 4-bit NF4).
2. **The Router:** A hybrid `scikit-learn` pipeline utilizing TF-IDF semantic centroids and regex-based syntax heuristics to classify queries in <0.05 seconds.
3. **The Experts (LoRAs):**

   * **Math Expert:** Trained on `openai/gsm8k` (Socratic).

   * **Code Expert:** Trained on `m-a-p/CodeFeedback`.

   * **Medical Expert:** Trained on a custom 10k-sample clinical diagnostic dataset (`ozgur-coban/medical-lora-10k`).

---

##  Quick Start: Live Colab Demo

The easiest way to test the routing and generation pipeline is through Google Colab.

>  **IMPORTANT: Hugging Face Token Required**

> Because Llama 3.1 8B is a gated model, you must have a Hugging Face account and have accepted the Meta license agreement.

1. Open a new Google Colab Notebook with a **T4 GPU** runtime.
2. Open the **Secrets** tab (key icon on the left sidebar), create a new secret named `HF_TOKEN`, and paste your Hugging Face Write/Read token. Grant notebook access.
3. Run the following cells:

**Cell 1: Setup & Authentication**

```python

!pip install -q transformers accelerate bitsandbytes peft scikit-learn joblib python-dotenv

!git clone [https://github.com/ozgur-coban/confidence-driven-lora.git](https://github.com/ozgur-coban/confidence-driven-lora.git)

%cd /content/capstone-project

  

import os

from google.colab import userdata

from huggingface_hub import login

  

try:

    hf_token = userdata.get('HF_TOKEN')

    login(hf_token)

    print("Successfully logged into Hugging Face!")

except userdata.SecretNotFoundError:

    print("CRITICAL: Please add your HF_TOKEN to the Colab secrets tab!")

```

**Cell 2: Run the Interactive Demo**

Python

```

!python scripts/interactive_demo.py \

  --code_adapter "ozgur-coban/code-expert-lora" \

  --math_adapter "ozgur-coban/math-expert-lora" \

  --medical_adapter "ozgur-coban/medical-expert-lora"

```

_Note: The script will load the 4-bit base model and all three adapters into VRAM. An interactive `[USER]:` prompt will appear below the cell._

---

##  Local Developer Setup

If you want to run the pipeline locally or train your own adapters:

**1. Clone and Install**

Bash

```

!git clone [https://github.com/ozgur-coban/confidence-driven-lora.git](https://github.com/ozgur-coban/confidence-driven-lora.git)

pip install -r requirements.txt

```

**2. Set Environment Variables** Create a `.env` file in the root directory:

Code snippet

```

HF_TOKEN=your_huggingface_token_here

```

**3. Run the Demo**

Bash

```

python scripts/run_full_pipeline.py

```

---

##  Repository Structure

Plaintext

```

capstone-project/

├── artifacts/             # Generated JSON results and evaluations

├── config/                # Domain registry, few-shot prompts, hyperparams

├── models/                # Router artifacts (.npy, .joblib)

├── scripts/               # CLI entry points (train, evaluate, demo)

├── src/                   # Core pipeline logic

│   ├── data_ops/          # Preprocessors for HuggingFace datasets

│   ├── model_ops/         # Inference Manager and PEFT loading

│   ├── router/            # Hybrid Semantic/Syntax Router logic

│   └── training/          # Custom Trainer classes

├── COPYING                # GNU GPLv3 License text

├── COPYING.LESSER         # GNU LGPLv3 Exceptions

├── .env.example          

├── environment.yml                

└── requirements.txt

```

---

##  License

This project is licensed under the **GNU Lesser General Public License v3.0 (LGPLv3)**.

Because the LGPL is a set of additional permissions added to the base GPL, this repository includes both licenses:

- `COPYING`: Contains the standard GNU General Public License (GPLv3).
- `COPYING.LESSER`: Contains the LGPLv3 exceptions and additional permissions.

Copyright (c) 2026 Özgür ÇOBAN.
