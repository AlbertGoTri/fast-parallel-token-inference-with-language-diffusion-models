# LLaDA Evaluation Methods

This directory contains different evaluation approaches for the LLaDA models.

## Available Evaluation Methods

### 1. Promptfoo Evaluation (`promptfoo/`)

**Purpose:** Qualitative evaluation using LLM-as-judge with Yes/No questions.

**Use Case:**
- Evaluating fine-tuned student models
- Checking for hallucinations
- Verifying instruction following
- Quick qualitative assessment

**How to Run:**
```powershell
evaluation\promptfoo\run_evaluation.ps1 -ApiKey "your-google-api-key"
```

**Output:** HTML report with pass/fail percentages

See: [promptfoo/README.md](promptfoo/README.md)

---

### 2. lm-eval (via `../eval_llada_lm_eval.sh`)

**Purpose:** Standard benchmarks (MMLU, BBH, GSM8K, etc.) using EleutherAI's lm-evaluation-harness.

**Use Case:**
- Standard academic benchmarks
- Perplexity-based tasks (for Base model)
- Generation tasks (for Base/Instruct)

**How to Run:**
```bash
bash eval_llada_lm_eval.sh
```

See: [EVAL.md](../EVAL.md)

---

### 3. OpenCompass (via `../opencompass/`)

**Purpose:** Comprehensive benchmark suite with batch generation support.

**Use Case:**
- Reproducing LLaDA paper results
- Batch evaluation for efficiency
- Detailed metric breakdowns

**How to Run:**
```bash
bash eval_llada_opencompass.sh
```

See: [opencompass/](../opencompass/)

---

### 4. Reversal Curse Evaluation (`../eval_reverse.py`)

**Purpose:** Test bidirectional knowledge in classical Chinese poetry.

**Use Case:**
- Specific capability testing
- Directional generation evaluation

**How to Run:**
```bash
python eval_reverse.py --type ftb --eos_inf  # Forward to backward
python eval_reverse.py --type btf --eos_inf  # Backward to forward
```

---

### 5. Internal Evaluation (`../eval_llada.py`)

**Purpose:** Custom evaluation script for internal benchmarking.

**Use Case:**
- Development testing
- Model comparison

---

## Which Evaluation Should You Use?

| Goal | Method | Location |
|------|--------|----------|
| Quick qualitative check | Promptfoo | `promptfoo/` |
| Standard benchmarks | lm-eval | `eval_llada_lm_eval.sh` |
| Paper reproduction | OpenCompass | `opencompass/` |
| Hallucination detection | Promptfoo | `promptfoo/` |
| Directional reasoning | Reversal Curse | `eval_reverse.py` |

## Common Setup

All methods require:
- LLaDA model files (base or fine-tuned)
- Python environment with dependencies
- GPU (recommended)

See individual method documentation for specific requirements.