"""Generate a large promptfoo eval config from the Alpaca instruction dataset.

Each prompt gets 5 reference-grounded rubric assertions, judged Yes/No by the Ollama judge:
  1. relevant to the instruction
  2. coherent / fluent (not degenerate)
  3. complete (finished, no cut-off or repetition)
  4. accomplishes the task -- covers the key content of the Alpaca *reference answer*  (capability)
  5. satisfies the instruction's specific constraints                                   (capability)

Assertions 4-5 are the task-capability checks: they embed *this prompt's* reference answer /
instruction into the judge question, so we measure "did it do the task", not just "is it fluent"
-- without hand-writing anything per prompt.

Output format is the same inline `return judge(output, '<question>')` shape the pipeline's eval
renderer (nested_distillation_eval.create_promptfoo_config_for_round) expects, so its num_gpu
injection + per-prompt batching still apply. Run once, commit the result:

    python LLaDA/evaluation/promptfoo/generate_alpaca_eval.py \
        --n 500 --out LLaDA/evaluation/promptfoo/promptfooconfig_alpaca500.yaml
"""

import argparse
import os
import re

import yaml

# Exact judge boilerplate the existing config uses (so the renderer's extraction/batching match).
# NOTE: {out}/{q} are the generated code's f-string placeholders and the {...} are literal JSON
# braces -- this is a plain string, not an f-string, so nothing here is interpolated by us.
JUDGE_BOILERPLATE = (
    'import json, urllib.request\n'
    'def judge(out, q):\n'
    '  p = json.dumps({"model":"llama3.1:8b","stream":False,'
    '"options":{"temperature":0,"num_gpu":0,"num_predict":32},'
    '"messages":[{"role":"system","content":"Reply ONLY with JSON: '
    '{\\"answer\\":\\"Yes\\"} or {\\"answer\\":\\"No\\"}. No other text."},'
    '{"role":"user","content":f"Text: {out}\\nQuestion: {q}"}]}).encode()\n'
    '  try:\n'
    '    r = urllib.request.urlopen(urllib.request.Request('
    '"http://127.0.0.1:11434/api/chat",data=p,'
    'headers={"Content-Type":"application/json"}),timeout=300)\n'
    '  except Exception:\n'
    '    return False\n'
    '  return json.loads(json.loads(r.read()).get("message",{}).get("content","{}"))'
    '.get("answer","").lower().strip()=="yes"\n'
)


def _sanitize(text: str, maxlen: int) -> str:
    """Make text safe to embed inside a single-quoted, single-line judge question."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("\\", "").replace("'", "").replace('"', "")
    if len(text) > maxlen:
        text = text[:maxlen].rstrip() + " ..."
    return text


def _assertion(question: str) -> dict:
    return {"type": "python", "value": JUDGE_BOILERPLATE + f"return judge(output, '{question}')\n"}


def build_tests(rows, n: int) -> list:
    tests = []
    for r in rows[:n]:
        instr = _sanitize(r["instruction"], 300)
        ref = _sanitize(r["output"], 400)
        # The prompt actually sent to the student (keep the original text intact).
        prompt = r["instruction"].strip()
        if str(r.get("input", "")).strip():
            prompt = prompt + "\n" + r["input"].strip()
        questions = [
            f"Is the response relevant to this instruction? Instruction: {instr}",
            "Is the response coherent, fluent and grammatical (a real answer, not repeated or "
            "degenerate/gibberish text)?",
            "Is the response a complete, finished answer with no cut-off and no repeated tokens?",
            f"Does the response accomplish the task, covering the key content of this reference "
            f"answer? Reference: {ref}",
            f"Does the response satisfy the specific requirements or constraints stated in this "
            f"instruction? Instruction: {instr}",
        ]
        tests.append({"vars": {"prompt": prompt}, "assert": [_assertion(q) for q in questions]})
    return tests


def main():
    parser = argparse.ArgumentParser(description="Generate an Alpaca-based promptfoo eval config")
    parser.add_argument("--n", type=int, default=500, help="number of prompts")
    parser.add_argument("--out", default="LLaDA/evaluation/promptfoo/promptfooconfig_alpaca500.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Clean, self-contained prompts: no extra 'input' context, reasonable reference length.
    rows = [r for r in ds
            if not str(r["input"]).strip() and 20 < len(str(r["output"]).strip()) < 1500]
    if len(rows) < args.n:
        raise SystemExit(f"only {len(rows)} usable Alpaca rows, need {args.n}")

    tests = build_tests(rows, args.n)
    config = {
        "prompts": ["{{prompt}}"],
        "providers": [{"id": "file://llada_api_provider.py", "label": "LLaDA Student"}],
        "tests": tests,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True, width=1000000)
    print(f"Wrote {len(tests)} prompts x 5 assertions -> {args.out}")


if __name__ == "__main__":
    main()
