# Promptfoo Evaluation for LLaDA

This directory contains the promptfoo-based evaluation system for LLaDA models.

## Files

| File | Purpose |
|------|---------|
| `promptfooconfig.yaml` | Main evaluation configuration (12 prompts, 5 assertions each) |
| `llada_api_provider.py` | Python provider that calls the LLaDA Flask server |
| `gemini_judge_provider.js` | JavaScript provider for Gemini-based evaluation |
| `run_evaluation.ps1` | Automated PowerShell runner script |
| `generate_report.py` | HTML report generator |

## Quick Start

From the **LLaDA root directory**:

```powershell
evaluation\promptfoo\run_evaluation.ps1 -ApiKey "your-google-api-key"
```

Or manually:

1. Start the LLaDA server (from LLaDA root):
   ```powershell
   python serve_llada.py
   ```

2. Set environment variables:
   ```powershell
   $env:GOOGLE_API_KEY="your-key"
   $env:PROMPTFOO_REQUEST_TIMEOUT_MS="7200000"
   ```

3. Run evaluation (from this directory):
   ```powershell
   cd evaluation\promptfoo
   npx promptfoo eval
   ```

4. Generate report:
   ```powershell
   python generate_report.py
   ```

## Configuration

The `promptfooconfig.yaml` defines:
- 12 diverse test prompts
- 5 Yes/No evaluation assertions per prompt
- LLaDA model provider (local API)
- Gemini judge provider (for evaluation)

See `..EVALUATION_GUIDE.md` for full documentation.