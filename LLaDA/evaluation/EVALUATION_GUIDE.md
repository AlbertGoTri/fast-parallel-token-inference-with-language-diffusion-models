# LLaDA Evaluation Guide

This guide explains how to evaluate your LLaDA student model using **promptfoo** with a **local LLM judge** that answers Yes/No questions about model outputs.

## Overview

The evaluation system supports **two evaluation backends** (choose one):

### Option A: Mistral-7B-Instruct (Recommended - No External Service)
✅ **Default option** - Uses free Hugging Face model, runs locally
- **LLaDA Model Server** (`serve_llada.py`) - Generates responses
- **Mistral-7B-Instruct** (`mistral_judge_provider.py`) - Evaluates locally
- **No Ollama required** - Everything is local Python
- **No external services** - No API keys, no rate limits
- **One-time download** (~15GB) - Then cached locally

See `./promptfoo/MISTRAL_SETUP.md` for detailed Mistral setup.

### Option B: Ollama + llama3.1:8b (Legacy)
⚠️ **Alternative option** - Requires running Ollama as external service
- See instructions below for manual Ollama setup
- Config: `promptfooconfig.yaml` (uses Ollama HTTP API)

### Plus: Perplexity Evaluation (Text Fluency)
Measures how natural/fluent the generated text is using GPT-2 perplexity:
- **Lower perplexity** = More natural, fluent text
- **Higher perplexity** = More surprising, potentially lower quality text
- See `./perplexity/README.md` for details

## Prerequisites

**For Mistral-7B-Instruct (Recommended):**
1. **Node.js** (v18 or later) - Download from [nodejs.org](https://nodejs.org/)
2. **Python** (3.10 or later) with `transformers`, `torch` installed
3. **GPU with CUDA** (recommended) or CPU (slower)

**For Ollama (Legacy):**
1. All of the above, plus:
2. **Ollama** - Install from [ollama.com](https://ollama.com)
3. Pull the model: `ollama pull llama3.1:8b`

## Quick Start (Automated)

The easiest way to run evaluation with **Mistral** (default):

```powershell
# Run everything (starts LLaDA, downloads Mistral on first run, generates report)
.\evaluation\promptfoo\run_evaluation.ps1

# If LLaDA server is already running
.\evaluation\promptfoo\run_evaluation.ps1 -SkipServer

# Just generate report from existing results
.\evaluation\promptfoo\run_evaluation.ps1 -JustReport
```

The script will:
1. Download Mistral-7B-Instruct if it is not present locally (one-time, ~15GB)
2. Check that all dependencies are installed
3. Run the evaluation (LLaDA generates, Mistral judges sequentially to avoid VRAM collision)
4. Generate an HTML report
5. Optionally open the report in your browser

## Manual Setup

If you prefer to run steps manually:

### 1. Prepare the local judge (Mistral)

No separate server is required for the judge. The `mistral_judge_provider.py` will download and load `mistralai/Mistral-7B-Instruct-v0.2` automatically on first use. Ensure you have an internet connection for the initial download and `transformers` + `torch` installed.

To test the Mistral download and tokenizer quickly:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2'); print('Mistral tokenizer OK')"
```

### 2. Start the LLaDA Server

```bash
python serve_llada.py
```

Wait until you see: **"Model loaded and ready to serve."**

This loads your trained LoRA weights and starts a Flask API on port 5000.

### 3. Run Evaluation

```bash
cd evaluation/promptfoo
npx promptfoo eval
```

Or with explicit output:
```bash
npx promptfoo eval -o promptfoo_results.json
```

### 4. View Results

Interactive viewer:
```bash
npx promptfoo view
```

Generate HTML report:
```bash
python generate_report.py
```

## Understanding the Results

### Output Files

- `evaluation/promptfoo/promptfoo_results.json` - Raw evaluation data (JSON)
- `evaluation/promptfoo/evaluation_report.html` - Visual HTML report

### Running Perplexity Evaluation

After running promptfoo evaluation, you can also compute perplexity:

```powershell
cd evaluation/perplexity
.\run_perplexity_eval.ps1 -GenerateHtml
```

Or manually:

```bash
python calculate_perplexity.py \
  --input ../promptfoo/promptfoo_results.json \
  --output perplexity_results.json \
  --html perplexity_report.html
```

Perplexity interpretation:
- **< 20**: Excellent - Very natural, fluent text
- **20-50**: Good - Natural text with minor awkwardness
- **> 50**: High - Significant awkwardness or errors

### Scoring System

Each prompt is evaluated with 5 Yes/No questions (assertions). The final score is:

```
Accuracy = (Number of "Yes" answers / Total assertions) × 100%
```

Example:
- 60 assertions total (12 prompts × 5 questions each)
- 45 "Yes" answers
- **Accuracy: 75%**

### Report Sections

The HTML report shows:
- **Global Accuracy** - Overall percentage across all tests
- **Per-Prompt Results** - Individual scores for each prompt
- **Detailed Assertions** - Each Yes/No question with PASS/FAIL status

## Customizing the Evaluation

### Adding New Prompts

Edit `promptfooconfig.yaml` and add entries under `tests:`

```yaml
  - vars:
      prompt: "Your new prompt here"
    assert:
      - type: python
        value: |
          import json, urllib.request
          def judge(out, q):
            p = json.dumps({"model":"llama3.1:8b","stream":False,"options":{"temperature":0,"num_predict":256},"messages":[{"role":"system","content":"Respond ONLY with JSON: {\\"answer\\":\\"Yes\\",\\"reason\\":\\"...\\"} or {\\"answer\\":\\"No\\",\\"reason\\":\\"...\\"}. No other text."},{"role":"user","content":f"<o>{out}</o>\n<Question>{q}</Question>"}]}).encode()
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/chat",data=p,headers={"Content-Type":"application/json"}),timeout=300)
            return json.loads(json.loads(r.read()).get("message",{}).get("content","{}")).get("answer","").lower().strip()=="yes"
          return judge(output, "Your Yes/No question here?")
```

### Tips for Writing Good Evaluation Questions

1. **Be Specific** - "Does it mention X?" rather than "Is it good?"
2. **Make it Binary** - Answer should be clearly Yes or No
3. **Check for Hallucinations** - Include questions about what the model should NOT say
4. **Cover Requirements** - Check that all parts of the prompt are addressed

## Troubleshooting

### "Judge failed to load"

If the Mistral judge fails to load or respond, ensure that `transformers` and `torch` are installed and that the initial model download completed. Test with:
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2'); print('Mistral OK')"
```
If the download failed due to connectivity, re-run the evaluation with internet access or download manually.

### "Cannot connect to LLaDA server"

Start the LLaDA server in another terminal:
```bash
python serve_llada.py
```

### Evaluation Times Out

Increase the timeout:
```powershell
$env:PROMPTFOO_REQUEST_TIMEOUT_MS="7200000"  # 2 hours
```

### Server Takes Too Long to Start

Model loading can take 5-10 minutes depending on your hardware. Check that:
- Your GPU drivers are installed (if using CUDA)
- The LoRA weights path in `serve_llada.py` is correct
- You have sufficient RAM/VRAM

### VRAM Issues

The evaluation runs **sequentially** (concurrency=1) to avoid VRAM collision between LLaDA and the local judge. If you still have issues:
- Close other GPU-intensive applications
- Reduce batch size or generation length in `serve_llada.py`
-- Reduce model sizes or run on CPU where possible

## File Reference

| File | Purpose |
|------|---------|
| `llada_api_provider.py` | Promptfoo provider for LLaDA model |
| `promptfooconfig.yaml` | Main evaluation configuration (12 prompts, 5 rubrics each) |
| `serve_llada.py` | Flask server for model inference |
| `generate_report.py` | HTML report generator |
| `run_evaluation.ps1` | Automated evaluation script |

## Architecture Notes

Unlike the previous Gemini-based approach that required API keys and had rate limits, this evaluation:
- Runs entirely **locally** - no API keys needed
- Uses **Mistral-7B-Instruct** as the local judge (downloaded once from Hugging Face)
- Runs **sequentially** to avoid VRAM collision on consumer GPUs
- Completes in ~15-30 minutes on capable hardware (may be slower on CPU)

## Getting Help

1. Check server logs in the terminal running `serve_llada.py`
2. If the judge is failing, run the quick Mistral test shown above to verify the model/tokenizer
3. Run promptfoo with verbose output: `npx promptfoo eval --verbose`
4. Test the server directly:
   ```bash
   curl -X POST http://127.0.0.1:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello"}'
   ```
5. Test the server directly:
  ```bash
  curl -X POST http://127.0.0.1:5000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello"}'
  ```
