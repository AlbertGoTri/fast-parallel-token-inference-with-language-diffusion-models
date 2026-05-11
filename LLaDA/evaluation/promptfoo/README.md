# Promptfoo Evaluation for LLaDA

This directory contains the promptfoo-based evaluation system for LLaDA models.

## Evaluation Method

Evaluation uses:
- **LLaDA Student Model** - Generates responses via local Flask server
- **Mistral-7B-Instruct** - Local LLM judge (from Hugging Face) evaluates responses
- **No external services** - Everything runs locally, no API keys needed

The Mistral model is automatically downloaded on first run (~15GB) and cached locally.

## Files

| File | Purpose |
|------|---------|
| `promptfooconfig_mistral.yaml` | Main evaluation config (12 prompts, ~5 assertions each) |
| `mistral_judge_provider.py` | Mistral-7B-Instruct judge provider |
| `llada_api_provider.py` | Python provider that calls the LLaDA Flask server |
| `run_evaluation.ps1` | Automated PowerShell runner script |
| `generate_report.py` | HTML report generator |

## Quick Start

From the **LLaDA root directory**:

```powershell
evaluation\promptfoo\run_evaluation.ps1
```

The script will:
1. Start the LLaDA server automatically (if not running)
2. Download Mistral-7B-Instruct on first run (one-time, ~15GB download)
3. Run evaluation (LLaDA generates, Mistral judges sequentially)
4. Generate HTML report
5. Optionally open the report in your browser

## Manual Setup

If you prefer to run steps manually:

### 1. Start the LLaDA Server

```bash
python serve_llada.py
```

Wait until you see: **"Model loaded and ready to serve."**

### 2. Run Evaluation

```bash
cd evaluation/promptfoo
npx promptfoo eval -c promptfooconfig_mistral.yaml
```

### 3. Generate Report

```bash
python generate_report.py
```

## System Requirements

- **GPU** with CUDA support (VRAM usage: LLaDA ~16GB + Mistral ~15GB sequentially)
- **Python 3.10+** with transformers, torch
- **Node.js 18+** for promptfoo
- **~30GB free disk** for model caches
- **~30 mins** for full evaluation (15-30 depending on hardware)