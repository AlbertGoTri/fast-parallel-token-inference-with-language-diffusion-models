# Perplexity Evaluation

This directory contains tools for evaluating LLaDA model outputs using **perplexity** calculated with GPT-2.

## What is Perplexity?

Perplexity measures how well a language model (GPT-2) predicts a given text. It's essentially the exponential of the average negative log-likelihood:

- **Lower perplexity** = Text is more natural, fluent, and "expected" by the model
- **Higher perplexity** = Text is more surprising, unusual, or potentially lower quality

## Interpretation Guide

| Perplexity | Quality |
|------------|---------|
| < 20 | Excellent - Very natural, fluent text |
| 20-50 | Good - Natural text with minor awkwardness |
| 50-100 | Moderate - Some unusual phrasing |
| > 100 | High - Significant awkwardness or errors |

## Quick Start

### From Promptfoo Results (Recommended)

After running a promptfoo evaluation:

```powershell
# Change to the perplexity directory
cd evaluation/perplexity

# Run using the promptfoo results
python calculate_perplexity.py --input ../promptfoo/promptfoo_results.json --output results.json --html report.html

# Or use the PowerShell script (generates HTML automatically)
.\run_perplexity_eval.ps1 -GenerateHtml
```

### Evaluate Single Text

```bash
python calculate_perplexity.py --text "Hello, this is a sample text to evaluate."
```

### Evaluate From File

Create a file with one text per line:
```bash
python calculate_perplexity.py --file sample_texts.txt --output results.json
```

## Files

| File | Purpose |
|------|---------|
| `calculate_perplexity.py` | Main perplexity calculation script |
| `run_perplexity_eval.ps1` | PowerShell wrapper for easy execution |
| `README.md` | This documentation |

## How It Works

1. **Load GPT-2**: The script downloads GPT-2 (~500MB) on first run
2. **Tokenize Text**: Convert text to token IDs
3. **Calculate NLL**: For each token, compute the negative log-likelihood given previous tokens
4. **Sliding Window**: For long texts (>512 tokens), uses a sliding window approach
5. **Compute Perplexity**: `exp(sum(NLL) / num_tokens)`

## Output Format

### JSON Output

```json
{
  "average_perplexity": 35.42,
  "num_samples": 12,
  "samples": [
    {
      "prompt_id": 1,
      "prompt": "Explain the theory of relativity...",
      "output_preview": "The theory of relativity...",
      "perplexity": 28.15
    },
    ...
  ]
}
```

### HTML Report

The HTML report provides:
- Average perplexity with color-coded quality indicator
- Per-sample perplexity scores
- Prompt and output previews for each sample

## First-Time Setup

GPT-2 will be automatically downloaded on first run (~500MB). Ensure you have:
- Internet connection for the first run
- At least 1GB free disk space
- PyTorch and transformers installed:
  ```bash
  pip install torch transformers tqdm
  ```

## Integration with Promptfoo

The recommended workflow:

1. **Run promptfoo evaluation**:
   ```powershell
   cd evaluation/promptfoo
   .\run_evaluation.ps1
   ```

2. **Run perplexity evaluation**:
   ```powershell
   cd evaluation/perplexity
   .\run_perplexity_eval.ps1
   ```

3. **Compare both metrics**:
   - **Promptfoo score**: Task-specific accuracy (Yes/No rubrics)
   - **Perplexity**: General text fluency/naturalness

## Command-Line Options

```
python calculate_perplexity.py [options]

Options:
  --input, -i PATH     Path to promptfoo_results.json
  --text, -t TEXT      Single text to evaluate
  --file, -f PATH      Text file with one sample per line
  --output, -o PATH    Output JSON path (default: perplexity_results.json)
  --html PATH          Generate HTML report
  --device DEVICE      Device to use: cuda/cpu (default: auto)
```

## Notes

- GPU is recommended for faster processing but not required
- GPT-2 uses ~1GB VRAM - small enough to run alongside LLaDA on most GPUs
- If you encounter OOM errors, stop the LLaDA server first or use `--device cpu`
- The sliding window approach ensures consistent results for texts of any length
- Perplexity is most useful for comparing different models or generation settings on the same prompts
