"""
Evaluation module for nested distillation pipeline.
Integrates promptfoo and perplexity evaluation with result parsing.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import subprocess
import torch

try:
    from evaluation.promptfoo.generate_report import generate_report as generate_promptfoo_html_report
except Exception:
    generate_promptfoo_html_report = None

try:
    from evaluation.perplexity.calculate_perplexity import generate_html_report as generate_perplexity_html_report
except Exception:
    generate_perplexity_html_report = None


def run_promptfoo_eval(
    config_path: str,
    output_path: str,
    working_dir: Optional[str] = None,
    timeout: int = 3600,
    max_retries: int = 3,
    retry_delay: int = 10
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run promptfoo evaluation with retry logic for transient HTTP errors.

    Args:
        config_path: Path to promptfooconfig.yaml
        output_path: Path to save results JSON
        working_dir: Working directory (promptfoo runs from here)
        timeout: Timeout in seconds per attempt
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries

    Returns:
        Tuple of (success, results_dict)
    """
    print(f"Running promptfoo evaluation...")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_path}")
    print(f"  Max retries: {max_retries}")

    if working_dir is None:
        working_dir = os.path.dirname(config_path)
    if working_dir:
        working_dir = os.path.abspath(working_dir)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = f'npx promptfoo eval -c "{config_path}" -o "{output_path}"'

    print(f"  Working dir: {working_dir}")
    print(f"  Command: {cmd}")

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\n[Retry {attempt}/{max_retries-1}] Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
                print(f"Retrying promptfoo evaluation (attempt {attempt+1}/{max_retries})...")

            result = subprocess.run(
                cmd,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                # Check if error is transient (HTTP 500, connection error, etc.)
                stderr_text = (result.stderr or "").lower()
                stdout_text = (result.stdout or "").lower()
                combined_text = f"{stdout_text}\n{stderr_text}"
                if ('http error 500' in combined_text or
                        'internal server error' in combined_text or
                        'connection' in combined_text or
                        'timeout' in combined_text):
                    last_error = f"Transient HTTP error (code {result.returncode})"
                    print(f"WARNING: Transient error detected: {last_error}")
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        print(f"Max retries exhausted")
                        # Try to parse partial results if available
                        if os.path.exists(output_path):
                            parsed_ok, parsed_data = parse_promptfoo_results(output_path)
                            if parsed_ok:
                                parsed_data["warning"] = f"Completed with transient errors after {max_retries} attempts"
                                return True, parsed_data
                else:
                    print(f"WARNING: promptfoo exited with code {result.returncode}")
                    if os.path.exists(output_path):
                        parsed_ok, parsed_data = parse_promptfoo_results(output_path)
                        if parsed_ok:
                            parsed_data["warning"] = f"promptfoo exited with code {result.returncode}"
                            return True, parsed_data
                    return False, {"error": f"Exit code {result.returncode}", "stderr": result.stderr}
            else:
                # Success
                break

        except subprocess.TimeoutExpired:
            last_error = f"Timeout after {timeout}s"
            print(f"ERROR: {last_error}")
            if attempt < max_retries - 1:
                print(f"Will retry (attempt {attempt+1}/{max_retries})")
                continue
            return False, {"error": last_error}
        except Exception as e:
            last_error = str(e)
            print(f"ERROR: Failed to run promptfoo: {e}")
            if attempt < max_retries - 1:
                print(f"Will retry (attempt {attempt+1}/{max_retries})")
                continue
            return False, {"error": last_error}

    # Parse results
    return parse_promptfoo_results(output_path)


def parse_promptfoo_results(json_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse promptfoo results JSON and extract statistics.

    Args:
        json_path: Path to promptfoo_results.json

    Returns:
        Tuple of (success, results_dict with percent and details)
    """
    if not os.path.exists(json_path):
        return False, {"error": f"Results file not found: {json_path}"}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to parse JSON: {e}"}

    raw_results = data.get('results', [])
    if isinstance(raw_results, dict) and 'results' in raw_results:
        raw_results = raw_results['results']

    if isinstance(data, list):
        raw_results = data

    if not raw_results:
        return False, {"error": "No results found in file"}

    # Calculate statistics using the same logic as generate_report.py
    total_assertions = 0
    passed_assertions = 0

    for res in raw_results:
        grading = res.get('gradingResult') or {}
        components = grading.get('componentResults', [])

        # Handle nested componentResults
        flat_components = _flatten_components(components)

        for comp in flat_components:
            total_assertions += 1
            if comp.get('pass', False):
                passed_assertions += 1

    if total_assertions == 0:
        return False, {"error": "No assertions found in results"}

    percent = (passed_assertions / total_assertions) * 100

    return True, {
        "assertion_percent": round(percent, 2),
        "assertion_passed": passed_assertions,
        "assertion_total": total_assertions,
        # Backward-compatible aliases
        "percent": round(percent, 2),
        "passed": passed_assertions,
        "total": total_assertions,
        "num_prompts": len(raw_results)
    }


def _flatten_components(components: list) -> list:
    """Recursively flatten nested componentResults from promptfoo."""
    flat = []
    for comp in components:
        inner = comp.get('componentResults')
        if inner:
            flat.extend(_flatten_components(inner))
        else:
            flat.append(comp)
    return flat


def run_perplexity_eval(
    promptfoo_results_path: str,
    output_path: str,
    device: str = "cuda",
    html_report_path: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run perplexity evaluation on promptfoo outputs.

    Args:
        promptfoo_results_path: Path to promptfoo_results.json
        output_path: Path to save perplexity results
        device: Device to run on

    Returns:
        Tuple of (success, results_dict)
    """
    print(f"Running perplexity evaluation...")
    print(f"  Input: {promptfoo_results_path}")
    print(f"  Output: {output_path}")
    print(f"  Device: {device}")

    # Import perplexity module
    try:
        from evaluation.perplexity.calculate_perplexity import (
            load_gpt2_model, evaluate_from_promptfoo
        )
    except ImportError:
        print("ERROR: Could not import perplexity evaluation module")
        print("Make sure evaluation/perplexity/calculate_perplexity.py exists")
        return False, {"error": "Import failed"}

    # If CUDA is requested but unavailable, force CPU up front.
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested for perplexity but not available; falling back to CPU")
        device = "cpu"

    try:
        model, tokenizer, actual_device = load_gpt2_model(device)
        results = evaluate_from_promptfoo(promptfoo_results_path, model, tokenizer, actual_device)

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if html_report_path and generate_perplexity_html_report is not None:
            generate_perplexity_html_report(results, html_report_path)

        # Cleanup model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True, {
            "average_perplexity": results.get('average_perplexity', 0),
            "num_samples": results.get('num_samples', 0),
            "device": actual_device
        }

    except Exception as e:
        error_text = str(e)
        # Common case in this pipeline: evaluation server model still occupies VRAM.
        if device == "cuda" and ("out of memory" in error_text.lower() or "cuda" in error_text.lower()):
            print(f"WARNING: CUDA perplexity failed ({e}); retrying on CPU...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model, tokenizer, actual_device = load_gpt2_model("cpu")
                results = evaluate_from_promptfoo(promptfoo_results_path, model, tokenizer, actual_device)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                if html_report_path and generate_perplexity_html_report is not None:
                    generate_perplexity_html_report(results, html_report_path)

                del model, tokenizer
                return True, {
                    "average_perplexity": results.get('average_perplexity', 0),
                    "num_samples": results.get('num_samples', 0),
                    "device": actual_device,
                    "warning": f"CUDA failed, used CPU fallback: {e}"
                }
            except Exception as cpu_e:
                print(f"ERROR: Perplexity CPU fallback failed: {cpu_e}")
                import traceback
                traceback.print_exc()
                return False, {"error": f"CUDA error: {e}; CPU fallback error: {cpu_e}"}

        print(f"ERROR: Perplexity evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}


def create_promptfoo_config_for_round(
    template_path: str,
    output_path: str,
    provider_path: str,
    max_concurrency: int = 1,
    timeout_ms: int = 3600000,
    judge_num_predict: int = 256,
    judge_num_gpu: int = 1,
) -> str:
    """
    Create a promptfoo config file for a specific round.

    Args:
        template_path: Path to base promptfooconfig.yaml
        output_path: Path to write the round-specific config
        provider_path: Path to the LLaDA provider
        max_concurrency: Maximum concurrent evaluations
        timeout_ms: Request timeout in milliseconds

    Returns:
        Path to the created config file
    """
    import yaml

    # Load template
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Update settings
    config['evaluateOptions'] = {'maxConcurrency': max_concurrency}
    config['env'] = {'PROMPTFOO_REQUEST_TIMEOUT_MS': str(timeout_ms)}

    # Keep judge outputs short (Yes/No JSON), which reduces Ollama latency/VRAM
    # pressure during assertion execution.
    if judge_num_predict <= 0:
        judge_num_predict = 256

    # Force judge to CPU when requested (num_gpu=0) to avoid contention with
    # the evaluation generation server on small VRAM GPUs.
    if judge_num_gpu < 0:
        judge_num_gpu = 1

    # Make inline python judges resilient: if Ollama returns HTTP 500 or similar,
    # assertion should fail gracefully instead of aborting the whole promptfoo run.
    request_line_pattern = re.compile(
        r'(?m)^(\s*)r = urllib\.request\.urlopen\(urllib\.request\.Request\('
        r'"http://127\.0\.0\.1:11434/api/chat",data=p,headers=\{"Content-Type":"application/json"\}\),timeout=300\)\s*$'
    )

    def _harden_assertion_value(value: str) -> str:
        updated = value.replace('"num_predict":256', f'"num_predict":{judge_num_predict}')
        updated = updated.replace(
            '"options":{"temperature":0,',
            f'"options":{{"temperature":0,"num_gpu":{judge_num_gpu},'
        )
        return request_line_pattern.sub(
            r'\1try:\n'
            r'\1  r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/chat",data=p,headers={"Content-Type":"application/json"}),timeout=300)\n'
            r'\1except Exception:\n'
            r'\1  return False',
            updated
        )

    for test in config.get('tests', []):
        assertions = test.get('assert', [])
        if not isinstance(assertions, list):
            continue
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            if assertion.get('type') != 'python':
                continue
            value = assertion.get('value')
            if isinstance(value, str) and 'urllib.request.urlopen' in value:
                assertion['value'] = _harden_assertion_value(value)

    # Update provider path to be relative to output directory
    output_dir = os.path.dirname(output_path)

    # Make provider path relative
    if os.path.isabs(provider_path):
        # Calculate relative path
        rel_provider = os.path.relpath(provider_path, output_dir)
        config['providers'][0]['id'] = f'file://{rel_provider}'
    else:
        config['providers'][0]['id'] = f'file://{provider_path}'

    # Write config
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return output_path


def evaluate_round(
    round_num: int,
    student_steps: int,
    promptfoo_config_path: str,
    provider_path: str,
    eval_output_dir: str,
    reports_dir: Optional[str] = None,
    perplexity_device: str = "cuda",
    max_concurrency: int = 1,
    timeout_ms: int = 3600000,
    judge_num_predict: int = 256,
    judge_num_gpu: int = 1,
) -> Dict[str, Any]:
    """
    Run complete evaluation for a round.

    Args:
        round_num: Round number
        student_steps: Number of steps the student was trained with
        promptfoo_config_path: Path to base promptfoo config
        provider_path: Path to LLaDA provider
        eval_output_dir: Directory to save evaluation results
        perplexity_device: Device for perplexity
        max_concurrency: Max concurrent promptfoo evals
        timeout_ms: Timeout for promptfoo

    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(eval_output_dir, exist_ok=True)
    if reports_dir is None:
        reports_dir = os.path.join(eval_output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Create round-specific promptfoo config
    round_config_path = os.path.join(eval_output_dir, "promptfooconfig.yaml")
    create_promptfoo_config_for_round(
        promptfoo_config_path,
        round_config_path,
        provider_path,
        max_concurrency,
        timeout_ms,
        judge_num_predict,
        judge_num_gpu,
    )

    # Run promptfoo
    promptfoo_output = os.path.join(eval_output_dir, "promptfoo_results.json")
    promptfoo_success, promptfoo_data = run_promptfoo_eval(
        round_config_path,
        promptfoo_output,
        working_dir=eval_output_dir
    )

    promptfoo_html_path = os.path.join(reports_dir, "promptfoo_report.html")
    if os.path.exists(promptfoo_output):
        if generate_promptfoo_html_report is not None:
            try:
                generate_promptfoo_html_report(promptfoo_output, promptfoo_html_path)
            except Exception as e:
                print(f"WARNING: Failed to generate promptfoo HTML report: {e}")
        else:
            print("WARNING: promptfoo HTML report generator unavailable")

    if not promptfoo_success:
        print(f"WARNING: Promptfoo evaluation failed: {promptfoo_data.get('error', 'Unknown')}")
        promptfoo_assertion_percent = 0.0
    else:
        promptfoo_assertion_percent = promptfoo_data.get('assertion_percent', promptfoo_data.get('percent', 0.0))
        print(f"Promptfoo assertion-level evaluation complete: {promptfoo_assertion_percent:.2f}%")

    # Run perplexity on promptfoo outputs
    perplexity_output = os.path.join(eval_output_dir, "perplexity_results.json")
    perplexity_html_path = os.path.join(reports_dir, "perplexity_report.html")
    perplexity_success, perplexity_data = run_perplexity_eval(
        promptfoo_output,
        perplexity_output,
        device=perplexity_device,
        html_report_path=perplexity_html_path
    )

    if not perplexity_success:
        print(f"WARNING: Perplexity evaluation failed: {perplexity_data.get('error', 'Unknown')}")
        perplexity = 999.0
    else:
        perplexity = perplexity_data.get('average_perplexity', 999.0)
        print(f"Perplexity evaluation complete: {perplexity:.2f}")

    return {
        "promptfoo_assertion_percent": promptfoo_assertion_percent,
        # Backward-compatible key used by older call sites
        "promptfoo_percent": promptfoo_assertion_percent,
        "perplexity": perplexity,
        "promptfoo_success": promptfoo_success,
        "perplexity_success": perplexity_success,
        "promptfoo_details": promptfoo_data,
        "perplexity_details": perplexity_data,
        "output_dir": eval_output_dir,
        "reports_dir": reports_dir,
        "promptfoo_html_report": promptfoo_html_path,
        "perplexity_html_report": perplexity_html_path
    }


@dataclass
class EvaluationThresholds:
    """Thresholds for pass/fail decisions using assertion-level Promptfoo score."""
    promptfoo_min: float = 30.0

    def check_pass(self, promptfoo_assertion_percent: float) -> bool:
        """Check if assertion-level Promptfoo score passes the threshold."""
        return promptfoo_assertion_percent >= self.promptfoo_min


def check_continue(eval_results: Dict[str, Any], thresholds: EvaluationThresholds) -> bool:
    """
    Check if the experiment should continue based on evaluation results.

    Args:
        eval_results: Results from evaluate_round
        thresholds: Threshold configuration

    Returns:
        True if should continue, False to stop
    """
    promptfoo_assertion_percent = eval_results.get(
        'promptfoo_assertion_percent',
        eval_results.get('promptfoo_percent', 0)
    )
    should_continue = thresholds.check_pass(promptfoo_assertion_percent)

    if should_continue:
        print(f"Promptfoo assertion score {promptfoo_assertion_percent:.1f}% >= {thresholds.promptfoo_min}%: CONTINUE")
    else:
        print(f"Promptfoo assertion score {promptfoo_assertion_percent:.1f}% < {thresholds.promptfoo_min}%: STOP")

    return should_continue
