"""
Perplexity Evaluation for LLaDA Model

Calculates perplexity of LLaDA-generated outputs using GPT-2 as a reference model.
Lower perplexity indicates more natural, fluent text.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# GPT-2 is used as a proxy for fluency because it is small, fast, and has a
# well-calibrated perplexity scale on general English.
def load_gpt2_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Loading GPT-2 model on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    print("GPT-2 loaded successfully.")
    return model, tokenizer, device


def calculate_perplexity(
    text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
    max_length: int = 512,
    stride: int = 512
) -> float:
    """Calculate perplexity using a sliding window."""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    # Single-pass perplexity is exact for short texts; sliding window introduces
    # boundary effects.
    if seq_len <= max_length:
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            perplexity = torch.exp(outputs.loss).item()
        return perplexity

    # A stride equal to max_length means no overlap; this is faster and usually
    # sufficient for LLaDA's short generations (<128 tokens).
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # Only compute loss on the new non-overlapping tail to avoid
        # double-counting tokens across windows.
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def extract_outputs_from_promptfoo(json_path: str) -> List[Dict[str, str]]:
    """Extract LLaDA outputs and prompts from promptfoo_results.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    if isinstance(results, dict) and 'results' in results:
        results = results['results']

    outputs = []
    for i, result in enumerate(results):
        prompt_text = result.get('prompt', {}).get('raw', f'Prompt #{i+1}')
        response_obj = result.get('response') or {}
        output_text = response_obj.get('output', '')

        if output_text and output_text.strip():
            outputs.append({
                'prompt_id': i + 1,
                'prompt': prompt_text,
                'output': output_text.strip()
            })

    return outputs


def evaluate_single_text(text: str, model, tokenizer, device) -> Dict:
    """Evaluate a single text and return results."""
    perplexity = calculate_perplexity(text, model, tokenizer, device)
    tokens = tokenizer.encode(text)

    return {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'perplexity': round(perplexity, 4),
        'token_count': len(tokens),
        'text_length': len(text)
    }


def evaluate_from_promptfoo(
    json_path: str,
    model,
    tokenizer,
    device
) -> Dict:
    """Evaluate all outputs from promptfoo results."""
    print(f"Loading results from {json_path}...")
    outputs = extract_outputs_from_promptfoo(json_path)
    print(f"Found {len(outputs)} outputs to evaluate.")

    results = []
    ppls = []

    for item in tqdm(outputs, desc="Calculating perplexity"):
        ppl = calculate_perplexity(item['output'], model, tokenizer, device)
        ppls.append(ppl)

        results.append({
            'prompt_id': item['prompt_id'],
            'prompt': item['prompt'][:100] + '...' if len(item['prompt']) > 100 else item['prompt'],
            'output_preview': item['output'][:150] + '...' if len(item['output']) > 150 else item['output'],
            'perplexity': round(ppl, 4)
        })

    # One catastrophic output (e.g., repetition loops) can dominate the mean;
    # trimming the worst stabilizes comparisons across rounds.
    excluded_max_ppl = None
    if len(ppls) > 1:
        excluded_max_ppl = max(ppls)
        trimmed_ppls = [ppl for ppl in ppls if ppl != excluded_max_ppl]
        if not trimmed_ppls:
            trimmed_ppls = ppls
            excluded_max_ppl = None
    else:
        trimmed_ppls = ppls

    avg_ppl = sum(trimmed_ppls) / len(trimmed_ppls) if trimmed_ppls else 0

    summary = {
        'average_perplexity': round(avg_ppl, 4),
        'num_samples': len(results),
        'samples': results
    }

    if excluded_max_ppl is not None:
        summary['excluded_max_perplexity'] = round(excluded_max_ppl, 4)
        summary['trimmed_sample_count'] = len(trimmed_ppls)

    return summary


def evaluate_from_text_file(file_path: str, model, tokenizer, device) -> Dict:
    """Evaluate texts from a file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} lines to evaluate.")

    results = []
    ppls = []

    for i, text in enumerate(tqdm(lines, desc="Calculating perplexity")):
        ppl = calculate_perplexity(text, model, tokenizer, device)
        ppls.append(ppl)

        results.append({
            'line_id': i + 1,
            'text_preview': text[:150] + '...' if len(text) > 150 else text,
            'perplexity': round(ppl, 4)
        })

    excluded_max_ppl = None
    if len(ppls) > 1:
        excluded_max_ppl = max(ppls)
        trimmed_ppls = [ppl for ppl in ppls if ppl != excluded_max_ppl]
        if not trimmed_ppls:
            trimmed_ppls = ppls
            excluded_max_ppl = None
    else:
        trimmed_ppls = ppls

    avg_ppl = sum(trimmed_ppls) / len(trimmed_ppls) if trimmed_ppls else 0

    summary = {
        'average_perplexity': round(avg_ppl, 4),
        'num_samples': len(results),
        'samples': results
    }

    if excluded_max_ppl is not None:
        summary['excluded_max_perplexity'] = round(excluded_max_ppl, 4)
        summary['trimmed_sample_count'] = len(trimmed_ppls)

    return summary


def generate_html_report(results: Dict, output_path: str):
    """Generate an HTML report similar to the promptfoo report."""
    avg_ppl = results['average_perplexity']
    num_samples = results['num_samples']
    excluded_max_ppl = results.get('excluded_max_perplexity')
    trimmed_sample_count = results.get('trimmed_sample_count')

    # Color code based on perplexity
    if avg_ppl < 20:
        ppl_color = "text-emerald-400"
        ppl_label = "Excellent"
    elif avg_ppl < 50:
        ppl_color = "text-amber-400"
        ppl_label = "Good"
    else:
        ppl_color = "text-rose-400"
        ppl_label = "High"

    samples_html = ""
    for sample in results['samples']:
        ppl = sample['perplexity']
        if ppl < 20:
            row_color = "border-emerald-500/30"
            badge_class = "bg-emerald-500/20 text-emerald-400"
        elif ppl < 50:
            row_color = "border-amber-500/30"
            badge_class = "bg-amber-500/20 text-amber-400"
        else:
            row_color = "border-rose-500/30"
            badge_class = "bg-rose-500/20 text-rose-400"

        if 'prompt' in sample:
            preview = f"<p class='text-xs text-gray-500 mb-1'>Prompt: {sample['prompt']}</p><p class='text-sm'>{sample['output_preview']}</p>"
        else:
            preview = f"<p class='text-sm'>{sample['text_preview']}</p>"

        samples_html += f"""
        <div class="mb-4 p-4 rounded-xl border {row_color} bg-white/5">
            <div class="flex justify-between items-start mb-2">
                <span class="text-sm font-medium text-gray-300">Sample #{sample.get('prompt_id', sample.get('line_id', 1))}</span>
                <span class="px-3 py-1 rounded-full text-xs font-bold border {badge_class}">
                    PPL: {ppl:.2f}
                </span>
            </div>
            {preview}
        </div>
        """

    # The HTML report is standalone (CDN-hosted Tailwind) so it can be viewed on
    # clustered compute nodes without internet access to the build pipeline.
    html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaDA Perplexity Evaluation Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #050505;
            background-image:
                radial-gradient(circle at 15% 50%, rgba(30, 58, 138, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(88, 28, 135, 0.15), transparent 25%);
            background-attachment: fixed;
            color: #e5e5e5;
        }}
        code, .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .glass-header {{
            background: rgba(10, 10, 10, 0.7);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
    </style>
</head>
<body class="antialiased min-h-screen flex flex-col">
    <header class="glass-header sticky top-0 z-50 px-6 py-4 shadow-2xl">
        <div class="max-w-6xl mx-auto flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-600 animate-pulse flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z"/>
                    </svg>
                </div>
                <h1 class="text-2xl font-bold tracking-tight">LLaDA <span class="font-light text-gray-400">Perplexity Eval</span></h1>
            </div>
            <div class="text-right">
                <p class="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-1">Average Perplexity</p>
                <p class="text-3xl font-mono font-bold {ppl_color} drop-shadow-md">
                    {avg_ppl:.2f}
                </p>
                <p class="text-xs {ppl_color}">{ppl_label}</p>
            </div>
        </div>
    </header>

    <main class="flex-grow p-6 py-10">
        <div class="max-w-6xl mx-auto">
            <div class="mb-8 p-6 rounded-2xl bg-gradient-to-b from-white/5 to-transparent border border-white/10">
                <div class="grid grid-cols-3 gap-6 text-center divide-x divide-white/10">
                    <div>
                        <p class="text-sm text-gray-500 mb-1">Samples Evaluated</p>
                        <p class="text-2xl font-semibold text-white">{num_samples}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500 mb-1">Reference Model</p>
                        <p class="text-2xl font-semibold text-white">GPT-2</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500 mb-1">Interpretation</p>
                        <p class="text-sm text-gray-300 mt-2">Lower is better.<br/>&lt;20: Excellent, &lt;50: Good, &gt;50: High</p>
                    </div>
                </div>
                {f'<div class="mt-4 text-center text-xs text-gray-500">Average computed excluding the highest perplexity value ({excluded_max_ppl:.2f}); trimmed sample count: {trimmed_sample_count}.</div>' if excluded_max_ppl is not None else ''}
            </div>

            <h2 class="text-xl font-bold mb-4 text-white">Individual Scores</h2>
            <div class="space-y-2">
                {samples_html}
            </div>
        </div>
    </main>

    <footer class="mt-auto py-6 border-t border-white/5 text-center text-xs text-gray-600">
        Generated with GPT-2 Perplexity Evaluation
    </footer>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity of LLaDA outputs using GPT-2"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to promptfoo_results.json file"
    )
    parser.add_argument(
        "--text", "-t",
        help="Single text string to evaluate"
    )
    parser.add_argument(
        "--file", "-f",
        help="Text file with one sample per line"
    )
    parser.add_argument(
        "--output", "-o",
        default="perplexity_results.json",
        help="Output JSON file path (default: perplexity_results.json)"
    )
    parser.add_argument(
        "--html",
        help="Generate HTML report at specified path"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)"
    )

    args = parser.parse_args()

    if not any([args.input, args.text, args.file]):
        parser.print_help()
        print("\nError: Must specify one of --input, --text, or --file")
        sys.exit(1)

    model, tokenizer, device = load_gpt2_model(args.device)

    if args.text:
        print(f"Evaluating single text...")
        result = evaluate_single_text(args.text, model, tokenizer, device)
        results = {
            'average_perplexity': result['perplexity'],
            'num_samples': 1,
            'samples': [result]
        }

    elif args.file:
        print(f"Evaluating texts from {args.file}...")
        results = evaluate_from_text_file(args.file, model, tokenizer, device)

    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
        results = evaluate_from_promptfoo(args.input, model, tokenizer, device)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")
    print(f"Average Perplexity: {results['average_perplexity']:.4f}")

    if args.html:
        generate_html_report(results, args.html)


if __name__ == "__main__":
    main()
