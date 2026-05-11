# Mistral-7B-Instruct Evaluation Setup

## Changes Made

### Files Changed:
- **run_evaluation.ps1** - Removed all Ollama checks and references
- **promptfooconfig.yaml** - Kept for backwards compatibility (original Ollama version)
- **README.md** - Updated documentation

### Files Added:
- **mistral_judge_provider.py** - New local LLM judge using Mistral-7B-Instruct
- **promptfooconfig_mistral.yaml** - New config file using Mistral judge

---

## How It Works

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│ 1. run_evaluation.ps1                                    │
│    - Starts LLaDA server                                 │
│    - Calls: npx promptfoo eval -c promptfooconfig_mistral.yaml
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼────────┐         ┌───────▼──────┐
    │ LLaDA       │         │ Promptfoo    │
    │ (Generator) │         │ (Orchestrator)
    │ Port 5000   │         │              │
    └────┬────────┘         └───────┬──────┘
         │                          │
         │  Generate response       │  For each test:
         │◄──────────────────────────┤  1. Get LLaDA output
         │                          │  2. Run assertions
         │  Response text           │  3. Each assertion
         └──────────────────────────►   calls Mistral
                                       judge locally
                            ┌──────────────────┐
                            │ Mistral-7B       │
                            │ (Judge/Evaluator)│
                            │ - Downloaded     │
                            │   once: ~15GB    │
                            │ - Cached locally │
                            │ - No API calls   │
                            └──────────────────┘
```

### Key Points

1. **First Run**
   - Mistral-7B-Instruct is downloaded (~15GB)
   - Cached to: `~/.cache/huggingface/hub/`
   - Requires internet for download only

2. **Subsequent Runs**
   - No download needed
   - Everything local and offline

3. **Sequential Execution**
   - LLaDA runs first: generates response
   - Mistral runs after: evaluates response
   - Avoids VRAM collision (maxConcurrency: 1)

4. **No External Dependencies**
   - ✅ No Ollama server needed
   - ✅ No API keys required
   - ✅ No rate limits
   - ✅ No legal/licensing issues per server

---

## Usage

### Option 1: Automated (Recommended)

```powershell
# From LLaDA root directory
cd c:\Users\Gotri\Documents\tfg\LLaDA
.\evaluation\promptfoo\run_evaluation.ps1
```

### Option 2: Manual Steps

```powershell
# 1. Start LLaDA server (terminal 1)
python serve_llada.py

# 2. Run evaluation (terminal 2)
cd evaluation/promptfoo
npx promptfoo eval -c promptfooconfig_mistral.yaml -o promptfoo_results.json

# 3. Generate report
python generate_report.py
```

### Option 3: With Server Already Running

```powershell
.\evaluation\promptfoo\run_evaluation.ps1 -SkipServer
```

---

## Troubleshooting

### "CUDA out of memory"
Mistral (7B) + LLaDA (8B) together ≈ 30GB VRAM. Solutions:
- Run on GPU with ≥24GB VRAM
- Use CPU (slower, but works)
- Reduce batch sizes in configs

### "Model not downloading"
Mistral needs internet on first run. Check:
```bash
# Verify internet connectivity
ping huggingface.co

# Manually test download:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### "Provider import error"
Make sure `mistral_judge_provider.py` is in the same directory as config:
```bash
ls -la evaluation/promptfoo/mistral_judge_provider.py
```

### Evaluation Very Slow
Mistral on CPU is slow (~1 min per assertion). Use GPU if available or reduce number of assertions.

---

## Comparison: Old vs New

| Aspect | Old (Ollama) | New (Mistral) |
|--------|-------------|---------------|
| Judge Model | llama3.1:8b | Mistral-7B-Instruct |
| Deployment | External service | Local Python |
| Dependencies | `ollama serve` | `transformers` library |
| Internet Required | Yes (to run) | No (after download) |
| Legal/Licensing | Potential issues | Clean (Apache 2.0) |
| Accuracy | High | High |
| Speed | Medium | Medium |
| VRAM | 8GB | 15GB |

---

## Reverting to Ollama

If you want to revert to the old Ollama-based setup:

```powershell
# Use original config
npx promptfoo eval -c promptfooconfig.yaml

# Make sure Ollama is running
ollama serve  # (in another terminal)
```

---

## Next Steps

1. Run evaluation:
   ```powershell
   .\evaluation\promptfoo\run_evaluation.ps1
   ```

2. Check results in: `evaluation/promptfoo/evaluation_report.html`

3. View detailed JSON: `evaluation/promptfoo/promptfoo_results.json`

4. Fine-tune assertions in `promptfooconfig_mistral.yaml` as needed
