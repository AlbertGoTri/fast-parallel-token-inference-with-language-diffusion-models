# promptfoo evaluation

Two eval configs, both scored the same way (Yes/No judge assertions → `promptfoo_%` in the
leaderboard, same HTML report):

| Config | Prompts | Assertions | Use |
| --- | --- | --- | --- |
| `promptfooconfig.yaml` | 10 (hand-written) | ~3 each | Fast smoke / 8 GB default config |
| `promptfooconfig_alpaca500.yaml` | 500 (Alpaca) | 5 each = **2500** | Full capability grade (powerful config) |

## The 500-prompt Alpaca config

Generated from the [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
instruction dataset. Each prompt gets **5 reference-grounded rubric assertions**:

1. Relevant to the instruction.
2. Coherent / fluent (not degenerate or repeated).
3. Complete (finished, not cut off).
4. **Accomplishes the task, covering the key content of the Alpaca reference answer.** *(capability)*
5. **Satisfies the instruction's specific constraints.** *(capability)*

Assertions 4–5 embed *that prompt's* reference answer / instruction into the judge question, so
the grade measures task capability — not just fluency — without hand-writing 2500 bespoke checks.

**Grade** = total `Yes` / (500 × 5 = 2500), reported as `promptfoo_%` exactly like the 10-prompt
config. The judge (`llama3.1:8b` via Ollama) is called once per prompt with all 5 questions
batched (the pipeline's `create_promptfoo_config_for_round` does this at render time).

### Which config uses it

`powerful_config.yaml` already points at this file — that profile runs the Ollama judge on GPU
(`judge_num_gpu: 99`, `max_concurrency: 8`), so 2500 assertions are feasible. The 8 GB default
(`nested_distillation_config.yaml`) keeps the 10-prompt config, because 500 prompts × a
CPU/low-VRAM judge at `max_concurrency: 1` would take days.

To point any other profile at it, it's one line — no code change:

```yaml
evaluation:
  promptfoo:
    config_path: "evaluation/promptfoo/promptfooconfig_alpaca500.yaml"
```

### Regenerate

```bash
python LLaDA/evaluation/promptfoo/generate_alpaca_eval.py \
    --n 500 --out LLaDA/evaluation/promptfoo/promptfooconfig_alpaca500.yaml
```

`--n` sets the prompt count. The generator filters Alpaca to self-contained prompts (no extra
`input` context, reference length 20–1500 chars) and emits assertions in the same inline
`return judge(output, '<question>')` shape the pipeline's renderer expects.
