# LLaDA Evaluation Guide

This guide explains how to evaluate your LLaDA student model using promptfoo with a Gemini-based judge that answers Yes/No questions about model outputs.

## Overview

The evaluation system works as follows:
1. **LLaDA Model Server** (`serve_llada.py`) - Loads your trained model and serves it via a local API
2. **Promptfoo Configuration** (`promptfooconfig.yaml`) - Defines test prompts and evaluation criteria
3. **Gemini Judge** (`gemini_judge_provider.js`) - Evaluates each response with Yes/No questions
4. **Report Generator** (`generate_report.py`) - Creates a visual HTML report with results

## Prerequisites

1. **Node.js** (v18 or later) - Download from [nodejs.org](https://nodejs.org/)
2. **Python** (3.10 or later) with your LLaDA environment activated
3. **Google API Key** - Get from [Google AI Studio](https://aistudio.google.com/) (free tier available)

## Quick Start (Automated)

The easiest way to run the evaluation is using the provided PowerShell script:

```powershell
# Run everything (starts server, runs evaluation, generates report)
.\run_evaluation.ps1 -ApiKey "your-google-api-key"

# If server is already running (in another terminal)
.\run_evaluation.ps1 -ApiKey "your-google-api-key" -SkipServer

# Just generate report from existing results
.\run_evaluation.ps1 -ApiKey "your-google-api-key" -JustReport
```

The script will:
1. Check that all dependencies are installed
2. Start the LLaDA server (if not already running)
3. Run the evaluation with proper timeouts
4. Generate an HTML report
5. Optionally open the report in your browser

## Manual Setup

If you prefer to run steps manually:

### 1. Start the LLaDA Server

```bash
python serve_llada.py
```

Wait until you see: **"Model loaded and ready to serve."**

This loads your trained LoRA weights and starts a Flask API on port 5000.

### 2. Set Environment Variables

PowerShell:
```powershell
$env:GOOGLE_API_KEY="your-api-key"
$env:PROMPTFOO_REQUEST_TIMEOUT_MS="3600000"  # 1 hour
```

CMD:
```cmd
set GOOGLE_API_KEY=your-api-key
set PROMPTFOO_REQUEST_TIMEOUT_MS=3600000
```

Bash/WSL:
```bash
export GOOGLE_API_KEY="your-api-key"
export PROMPTFOO_REQUEST_TIMEOUT_MS="3600000"
```

### 3. Run Evaluation

```bash
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

- `promptfoo_results.json` - Raw evaluation data (JSON)
- `evaluation_report.html` - Visual HTML report

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
- **Detailed Assertions** - Each Yes/No question with the judge's reasoning

## Customizing the Evaluation

### Adding New Prompts

Edit `promptfooconfig.yaml` and add entries under `tests:`

```yaml
  - vars:
      prompt: "Your new prompt here"
    assert:
      - type: llm-rubric
        value: |
          You are grading output according to a user-specified rubric.

          <Output>{{output}}</Output>

          <Question>Your Yes/No question here?</Question>

          Answer ONLY with a JSON object in this exact format:
          {"answer": "Yes"|"No", "reason": "brief explanation"}
```

### Tips for Writing Good Evaluation Questions

1. **Be Specific** - "Does it mention X?" rather than "Is it good?"
2. **Make it Binary** - Answer should be clearly Yes or No
3. **Check for Hallucinations** - Include questions about what the model should NOT say
4. **Cover Requirements** - Check that all parts of the prompt are addressed

## Troubleshooting

### "No connection could be made because the target machine actively refused it"

The LLaDA server is not running. Start it with:
```bash
python serve_llada.py
```

### "No API key found"

Set your Google API key:
```powershell
$env:GOOGLE_API_KEY="your-api-key"
```

### Rate Limiting Errors (429)

The Gemini free tier is limited to ~15 RPM. The judge provider already implements:
- Conservative 3 RPM pacing
- Exponential backoff on errors
- Serial execution of assertions

**Solutions:**
- Wait a few minutes and retry
- Use a paid Gemini API tier for higher limits
- Reduce the number of test prompts

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

## File Reference

- `llada_api_provider.py` - Promptfoo provider for LLaDA model (Python)
- `gemini_judge_provider.js` - Promptfoo provider for Gemini judge (JavaScript)
- `promptfooconfig.yaml` - Main evaluation configuration
- `serve_llada.py` - Flask server for model inference
- `generate_report.py` - HTML report generator
- `run_evaluation.ps1` - Automated evaluation script (PowerShell)

## Rate Limiting Details

The Gemini judge provider implements these safeguards:

| Setting | Value | Purpose |
|---------|-------|---------|
| RPM | 3 | Stay well under free tier limits |
| Min Gap | 20 seconds | Avoid burst detection |
| Retry Delay | 90s + 30s per consecutive 429 | Back off when rate limited |
| Max Attempts | 20 | Prevent infinite loops |

**Estimated Runtime:**
- 12 prompts × 5 assertions = 60 total calls
- 60 calls × 20 seconds = ~20 minutes minimum
- With retries: 25-40 minutes typical

## Getting Help

1. Check server logs in the terminal running `serve_llada.py`
2. Run promptfoo with verbose output: `npx promptfoo eval --verbose`
3. Test the server directly:
   ```bash
   curl -X POST http://127.0.0.1:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello"}'
   ```
