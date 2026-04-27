"""
Nested Distillation Pipeline - End-to-End Orchestrator

Pure recursive teacher-student distillation where each new student uses exactly
half denoising steps than its teacher.

Usage:
    python nested_distillation.py
    python nested_distillation.py --config custom_config.yaml
    python nested_distillation.py --resume
    python nested_distillation.py --dry-run
    python nested_distillation.py --status
    python nested_distillation.py --force
"""

import os
import sys
import shutil
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import gc

from nested_distillation_utils import (
    log_memory_usage, cleanup_gpu_memory, verify_gpu_empty, unload_model,
    set_seed, ensure_dir, load_yaml_config, save_json, load_json,
    save_csv, format_timestamp, RoundResult, ExperimentState,
    StateManager, ProgressLogger, print_leaderboard, validate_step_reduction,
    check_ollama_running
)
from nested_distillation_eval import (
    evaluate_round, EvaluationThresholds, check_continue
)
from nested_distillation_server import (
    managed_server, check_server_running, wait_for_server
)


def _resolve_run_dir(base_output_dir: str, args: argparse.Namespace) -> str:
    """Resolve the active run directory for this invocation.

    Behavior:
    - New runs create outputs under base_output_dir/runs/run_YYYYmmdd_HHMMSS.
    - --resume/--status use --run-dir when provided, else latest_run.txt.
    - Falls back to legacy base_output_dir for backward compatibility.
    """
    base_output_dir = os.path.abspath(base_output_dir)
    latest_run_file = os.path.join(base_output_dir, "latest_run.txt")

    if getattr(args, 'run_dir', None):
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.abspath(run_dir)
        return run_dir

    if args.resume or args.status:
        if os.path.exists(latest_run_file):
            try:
                with open(latest_run_file, 'r', encoding='utf-8') as f:
                    candidate = f.read().strip()
                if candidate and os.path.isdir(candidate):
                    return candidate
            except Exception:
                pass
        # Backward compatibility with older layout that stored everything at base_output_dir.
        return base_output_dir

    # New non-resume run: create an isolated timestamped directory.
    runs_root = os.path.join(base_output_dir, "runs")
    ensure_dir(runs_root)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_root, run_name)
    ensure_dir(run_dir)

    try:
        with open(latest_run_file, 'w', encoding='utf-8') as f:
            f.write(run_dir)
    except Exception as e:
        print(f"Warning: Could not update latest run pointer: {e}")

    return run_dir


def _build_run_scoped_paths(config_paths: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    """Build per-run output paths rooted at run_dir."""
    run_paths = deepcopy(config_paths)
    run_paths['base_output_dir'] = run_dir
    run_paths['state_file'] = os.path.join(run_dir, 'state.json')
    run_paths['leaderboard'] = {
        'markdown': os.path.join(run_dir, 'leaderboard.md'),
        'csv': os.path.join(run_dir, 'leaderboard.csv'),
        'json': os.path.join(run_dir, 'leaderboard.json'),
    }
    return run_paths


def resolve_teacher_model_paths(teacher_path: str, fallback_model_path: str) -> tuple[str, Optional[str]]:
    """Resolve the base model and optional LoRA adapter for a teacher checkpoint."""
    adapter_config_path = os.path.join(teacher_path, "adapter_config.json")
    if os.path.isdir(teacher_path) and os.path.exists(adapter_config_path):
        try:
            adapter_config = load_json(adapter_config_path)
            base_model_path = adapter_config.get("base_model_name_or_path") or fallback_model_path
            return base_model_path, teacher_path
        except Exception:
            return fallback_model_path, teacher_path
    return teacher_path, None


def cache_stage(
    config: Dict[str, Any],
    round_num: int,
    teacher_path: str,
    teacher_steps: int,
    target_step: int,
    cache_dir: str,
    logger: ProgressLogger
) -> bool:
    """
    Stage 1: Generate teacher trajectories for current round.

    The teacher generates trajectories at the target_step (intermediate state),
    which the student will learn to predict.
    """
    logger.start_stage("cache")

    # Verify GPU is empty before starting
    if config['execution']['verify_gpu_empty']:
        verify_gpu_empty()
    log_memory_usage("cache_start")

    # Import required modules
    try:
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
        from datasets import load_dataset
        from generate_cache import generate_and_cache_trajectory
    except ImportError as e:
        logger.log(f"ERROR: Failed to import required modules: {e}")
        return False

    # Setup quantization
    quant_config = config['system']['quantization']
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, quant_config['compute_dtype']),
        bnb_4bit_quant_type=quant_config['quant_type'],
        bnb_4bit_use_double_quant=quant_config['use_double_quant'],
    )

    # Setup environment
    os.environ["HF_HOME"] = config['system']['hf_home']
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SAFETENSORS_FAST_GPU"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.cuda.set_per_process_memory_fraction(config['system']['cuda_memory_fraction'])

    # Clear cache dir for this round
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.log(f"Cache directory prepared: {cache_dir}")

    # Calculate available RAM
    ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3

    # Load teacher model
    base_model_path, adapter_path = resolve_teacher_model_paths(
        teacher_path,
        config['teacher']['model_path']
    )
    logger.log(f"Loading teacher model: {teacher_path}")
    logger.log(f"Resolved base model: {base_model_path}")
    if adapter_path:
        logger.log(f"Resolved LoRA adapter: {adapter_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        model.tie_weights()
        model.eval()
        log_memory_usage("teacher_loaded")
    except Exception as e:
        logger.log(f"ERROR: Failed to load teacher model: {e}")
        return False

    # Load dataset
    logger.log("Loading dataset...")
    try:
        dataset = load_dataset(
            config['student']['dataset_name'],
            config['student']['dataset_config'],
            split=config['student']['dataset_split']
        )
        textos = [t for t in dataset['text']
                  if len(t.strip()) > config['student']['min_text_length']]
        textos = textos[:config['student']['num_train_examples']]
        logger.log(f"Loaded {len(textos)} training examples")
    except Exception as e:
        logger.log(f"ERROR: Failed to load dataset: {e}")
        unload_model(model, "teacher")
        return False

    # Generate trajectories
    logger.log(f"Generating trajectories with steps={teacher_steps}, target_step={target_step}")

    student_config = config['student']

    with torch.no_grad():
        for i, texto in enumerate(textos):
            prompt_content = f"Resume o continua este texto de forma breve:\n{texto[:200]}"
            conversation = [{"role": "user", "content": prompt_content}]

            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_len = input_ids.shape[1]
            logger.log(f"Processing example {i+1}/{len(textos)} (prompt_len={prompt_len})")

            try:
                estado_x, teacher_logits, attn_mask = generate_and_cache_trajectory(
                    model,
                    input_ids,
                    steps=teacher_steps,
                    gen_length=student_config['gen_length'],
                    block_length=student_config['block_length'],
                    target_step=target_step,
                )

                file_path = os.path.join(cache_dir, f"batch_{i}.pt")
                torch.save({
                    'input_x': estado_x.cpu(),
                    'target_logits': teacher_logits.cpu(),
                    'attn_mask': attn_mask.cpu() if attn_mask is not None else None,
                    'prompt_len': prompt_len,
                    'target_step': target_step,
                }, file_path)

                logger.log(f"Saved trajectory to {file_path}")

            except Exception as e:
                logger.log(f"ERROR: Failed to generate trajectory {i+1}: {e}")
                continue

            # Clear cache after each example
            del input_ids, estado_x, teacher_logits
            if attn_mask is not None:
                del attn_mask
            torch.cuda.empty_cache()

    # Cleanup
    num_cached = len([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
    logger.log(f"Cached {num_cached} trajectories")

    unload_model(model, "teacher")
    unload_model(tokenizer, "tokenizer")
    log_memory_usage("cache_end")

    logger.end_stage("cache", success=num_cached > 0)
    return num_cached > 0


def train_stage(
    config: Dict[str, Any],
    round_num: int,
    cache_dir: str,
    checkpoint_dir: str,
    student_steps: int,
    logger: ProgressLogger
) -> bool:
    """
    Stage 2: Train student model on cached trajectories.
    """
    logger.start_stage("train")

    # Verify GPU is empty before starting
    if config['execution']['verify_gpu_empty']:
        verify_gpu_empty()
    log_memory_usage("train_start")

    # Import required modules
    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig
        import glob
        import psutil
    except ImportError as e:
        logger.log(f"ERROR: Failed to import required modules: {e}")
        return False

    # Setup quantization
    quant_config = config['system']['quantization']
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, quant_config['compute_dtype']),
        bnb_4bit_quant_type=quant_config['quant_type'],
        bnb_4bit_use_double_quant=quant_config['use_double_quant'],
    )

    # Setup environment
    os.environ["HF_HOME"] = config['system']['hf_home']
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SAFETENSORS_FAST_GPU"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.cuda.set_per_process_memory_fraction(config['system']['cuda_memory_fraction'])

    # Load cache files
    archivos_cache = glob.glob(os.path.join(cache_dir, "*.pt"))
    if not archivos_cache:
        logger.log(f"ERROR: No cache files found in {cache_dir}")
        return False

    logger.log(f"Found {len(archivos_cache)} cached trajectories")

    # Load base model
    model_id = config['teacher']['model_path']
    ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3

    logger.log(f"Loading student base model: {model_id}")
    try:
        student_base = AutoModel.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        student_base.gradient_checkpointing_enable()

        # Freeze base parameters
        for param in student_base.parameters():
            param.requires_grad = False

        log_memory_usage("base_loaded")
    except Exception as e:
        logger.log(f"ERROR: Failed to load base model: {e}")
        return False

    # Add LoRA adapters
    lora_config = config['student']['lora']
    logger.log(f"Injecting LoRA adapters (r={lora_config['r']}, alpha={lora_config['alpha']})")

    try:
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        student_model = get_peft_model(student_base, peft_config)

        # Cast LoRA parameters to float16
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float16)

        student_model.train()
        log_memory_usage("lora_added")
    except Exception as e:
        logger.log(f"ERROR: Failed to add LoRA: {e}")
        unload_model(student_base, "base_model")
        return False

    # Setup optimizer
    learning_rate = float(config['student']['learning_rate'])
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in student_model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        logger.log("Using AdamW 8-bit optimizer")
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in student_model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        logger.log("Using standard AdamW optimizer")

    # Training loop
    temperature = config['student']['temperature']
    max_grad_norm = config['student']['max_grad_norm']

    logger.log(f"Starting training with temperature={temperature}")

    try:
        for epoch in range(1):  # Single epoch for now
            for i, archivo in enumerate(archivos_cache):
                torch.cuda.empty_cache()

                trayectoria = torch.load(archivo, weights_only=True)

                input_x = trayectoria['input_x'].to("cuda")
                target_logits = trayectoria['target_logits'].to("cuda").to(torch.float16)
                attn_mask = trayectoria['attn_mask'].to("cuda") if trayectoria['attn_mask'] is not None else None
                prompt_len = trayectoria.get('prompt_len', 0)

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_salida = student_model(input_x, attention_mask=attn_mask)
                    student_logits = student_salida.logits

                    student_logits_gen = student_logits[:, prompt_len:, :]
                    target_logits_gen = target_logits[:, prompt_len:, :]

                    loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(student_logits_gen / temperature, dim=-1),
                        torch.nn.functional.softmax(target_logits_gen / temperature, dim=-1),
                        reduction="batchmean"
                    ) * (temperature ** 2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_model.parameters() if p.requires_grad],
                    max_norm=max_grad_norm
                )
                optimizer.step()

                vram = torch.cuda.memory_allocated(0) / 1024**3
                logger.log(f"Epoch {epoch+1} | Batch {i+1}/{len(archivos_cache)} | Loss: {loss.item():.4f} | VRAM: {vram:.2f}GB")

                del input_x, target_logits, student_logits, loss
                if attn_mask is not None:
                    del attn_mask
                torch.cuda.empty_cache()

    except Exception as e:
        logger.log(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        unload_model(student_model, "student")
        return False

    # Save checkpoint
    logger.log(f"Saving checkpoint to {checkpoint_dir}")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        student_model.save_pretrained(checkpoint_dir)
    except Exception as e:
        logger.log(f"ERROR: Failed to save checkpoint: {e}")
        unload_model(student_model, "student")
        return False

    # Cleanup
    unload_model(student_model, "student")
    log_memory_usage("train_end")

    logger.end_stage("train", success=True)
    return True


def run_single_round(
    config: Dict[str, Any],
    state: ExperimentState,
    round_num: int,
    teacher_steps: int,
    student_steps: int,
    teacher_path: str,
    output_dirs: Dict[str, str],
    logger: ProgressLogger,
    thresholds: EvaluationThresholds
) -> Optional[RoundResult]:
    """
    Run a complete distillation round.

    Returns:
        RoundResult if successful, None otherwise
    """
    logger.start_round(round_num, student_steps)

    # Validate halving schedule
    try:
        expected_student_steps = max(config['schedule']['min_steps'], teacher_steps // 2)
        if student_steps != expected_student_steps:
            raise ValueError(
                f"Step halving validation failed: Teacher={teacher_steps}, Student={student_steps}, Expected={expected_student_steps}"
            )
    except ValueError as e:
        logger.log(f"ERROR: Step validation failed: {e}")
        return None

    logger.log(f"Teacher: {teacher_steps} steps | Student: {student_steps} steps")
    logger.log(f"Teacher path: {teacher_path}")

    # Calculate target_step as midpoint (as in original code)
    target_step = teacher_steps // 2

    # Stage 1: Cache teacher trajectories
    cache_dir = output_dirs['cache']
    if not cache_stage(config, round_num, teacher_path, teacher_steps, target_step, cache_dir, logger):
        logger.log("ERROR: Cache stage failed")
        return None

    # Stage 2: Train student
    checkpoint_dir = output_dirs['checkpoint']
    if not train_stage(config, round_num, cache_dir, checkpoint_dir, student_steps, logger):
        logger.log("ERROR: Train stage failed")
        return None

    # Stage 3 & 4: Evaluation
    logger.start_stage("evaluation")

    # Ensure clean GPU before evaluation
    if config['execution']['verify_gpu_empty']:
        verify_gpu_empty()
    cleanup_gpu_memory()
    log_memory_usage("eval_start")

    # Run evaluation with managed server
    eval_result = None
    try:
        eval_config = config['evaluation']
        eval_server_config = eval_config.get('server', {})
        eval_server_device = str(eval_server_config.get('device', 'cuda')).lower()
        if eval_server_device not in {'cuda', 'cpu'}:
            logger.log(f"WARNING: Invalid evaluation.server.device='{eval_server_device}', using 'cuda'")
            eval_server_device = 'cuda'
        eval_server_cuda_fraction = float(eval_server_config.get('cuda_memory_fraction', 0.85))
        if not (0 < eval_server_cuda_fraction <= 1.0):
            logger.log(
                f"WARNING: Invalid evaluation.server.cuda_memory_fraction='{eval_server_cuda_fraction}', using 0.85"
            )
            eval_server_cuda_fraction = 0.85
        eval_server_timeout = float(
            eval_server_config.get(
                'startup_timeout_seconds',
                1800.0 if eval_server_device == 'cpu' else 600.0,
            )
        )
        if eval_server_timeout <= 0:
            logger.log(
                f"WARNING: Invalid evaluation.server.startup_timeout_seconds='{eval_server_timeout}', "
                f"using {'1800.0' if eval_server_device == 'cpu' else '600.0'}"
            )
            eval_server_timeout = 1800.0 if eval_server_device == 'cpu' else 600.0

        # Use eval_steps if configured, otherwise use student_steps
        eval_steps = eval_config.get('eval_steps', 0)
        if eval_steps == 0:
            eval_steps = student_steps
        logger.log(f"Starting evaluation server for {student_steps} steps (eval using {eval_steps} steps)...")

        with managed_server(
            checkpoint_dir,
            eval_steps,
            port=5000,
            timeout=eval_server_timeout,
            device=eval_server_device,
            cuda_memory_fraction=eval_server_cuda_fraction,
        ) as server:
            logger.log("Server is ready, running evaluations...")

            eval_dirs = output_dirs['eval']
            reports_dir = output_dirs['reports']
            os.makedirs(eval_dirs, exist_ok=True)
            eval_dirs = os.path.abspath(eval_dirs)
            os.makedirs(reports_dir, exist_ok=True)
            reports_dir = os.path.abspath(reports_dir)

            # Use absolute paths for promptfoo config to avoid path issues
            promptfoo_template = os.path.abspath(eval_config['promptfoo']['config_path'])
            provider_abs = os.path.abspath(eval_config['promptfoo']['provider_path'])

            eval_result = evaluate_round(
                round_num=round_num,
                student_steps=student_steps,
                promptfoo_config_path=promptfoo_template,
                provider_path=provider_abs,
                eval_output_dir=eval_dirs,
                reports_dir=reports_dir,
                perplexity_device=eval_config['perplexity']['device'],
                max_concurrency=eval_config['promptfoo']['max_concurrency'],
                timeout_ms=eval_config['promptfoo']['timeout_ms'],
                judge_num_predict=int(eval_config['promptfoo'].get('judge_num_predict', 256)),
                judge_num_gpu=int(eval_config['promptfoo'].get('judge_num_gpu', 1)),
            )

        logger.log("Evaluation server stopped")

    except Exception as e:
        logger.log(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        # Hard evaluation failures should fail the round; do not promote this
        # checkpoint as next teacher.
        logger.end_stage("evaluation", success=False)
        logger.end_stage("round", success=False)
        return None

    log_memory_usage("eval_end")
    logger.end_stage("evaluation", success=eval_result.get('promptfoo_success', False))

    # Determine if passed threshold using assertion-level Promptfoo score
    promptfoo_percent = eval_result.get('promptfoo_assertion_percent', eval_result.get('promptfoo_percent', 0.0))
    perplexity = eval_result.get('perplexity', 999.9)
    passed = promptfoo_percent >= thresholds.promptfoo_min

    # Create result
    result = RoundResult(
        round_number=round_num,
        student_name=f"student_steps_{student_steps}",
        teacher_name=f"teacher_steps_{teacher_steps}",
        student_steps=student_steps,
        teacher_steps=teacher_steps,
        promptfoo_percent=promptfoo_percent,
        perplexity=perplexity,
        passed=passed,
        timestamp=format_timestamp(),
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        eval_dir=output_dirs['eval']
    )

    logger.end_stage("round", success=True)
    return result


def update_leaderboard(results: list, paths: Dict[str, str]) -> None:
    """Update all leaderboard files (MD, CSV, JSON)."""
    # JSON
    save_json([r.to_dict() for r in results], paths['json'])

    # CSV
    if results:
        rows = [r.to_dict() for r in results]
        save_csv(rows, paths['csv'])

    # Markdown
    md_lines = ["# Nested Distillation Leaderboard\n"]
    md_lines.append("\n| Round | Student Steps | Promptfoo Assertion % | Perplexity | Pass | Timestamp |")
    md_lines.append("|-------|---------------|-------------|------------|------|-----------|")

    for r in results:
        status = "✓" if r.passed else "✗"
        md_lines.append(
            f"| {r.round_number} | {r.student_steps} | {r.promptfoo_percent:.2f}% | "
            f"{r.perplexity:.2f} | {status} | {r.timestamp[:19]} |"
        )

    md_lines.append("\n## Summary\n")
    for r in results:
        md_lines.append(f"- {r.format_summary()}")

    with open(paths['markdown'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))


def calculate_max_rounds(initial_steps: int, min_steps: int) -> int:
    """Calculate the maximum number of halving rounds possible."""
    rounds = 0
    steps = initial_steps
    while steps > min_steps:
        steps = max(min_steps, steps // 2)
        rounds += 1
    return rounds


def print_dry_run(config: Dict[str, Any]) -> None:
    """Print planned rounds without training."""
    initial_steps = config['teacher']['initial_steps']
    min_steps = config['schedule']['min_steps']
    max_rounds = calculate_max_rounds(initial_steps, min_steps)

    print("\n" + "=" * 70)
    print("DRY RUN - Planned Rounds")
    print("=" * 70)

    teacher_steps = initial_steps
    for round_num in range(1, max_rounds + 1):
        student_steps = max(min_steps, teacher_steps // 2)
        if student_steps < min_steps:
            break

        print(f"Round {round_num}:")
        print(f"  Teacher: {teacher_steps} steps")
        print(f"  Student: {student_steps} steps (target)")
        print(f"  Target step for caching: {teacher_steps // 2}")
        print()

        teacher_steps = student_steps

    print("=" * 70)
    print(f"Maximum rounds: {max_rounds}")
    print(f"Promptfoo assertion threshold (reporting only): {config['evaluation']['promptfoo_threshold']}%")
    print("=" * 70 + "\n")


def get_output_dirs(base_dir: str, round_num: int, steps: int) -> Dict[str, str]:
    """Get output directories for a specific round."""
    base_dir = os.path.abspath(base_dir)
    round_dir = os.path.join(base_dir, f"round_{round_num:03d}_steps_{steps}")
    return {
        'cache': os.path.join(round_dir, "cache"),
        'checkpoint': os.path.join(round_dir, "checkpoint"),
        'eval': os.path.join(round_dir, "evaluation"),
        'reports': os.path.join(round_dir, "reports")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Nested Distillation Pipeline - Recursive teacher-student distillation"
    )
    parser.add_argument(
        "--config",
        default="nested_distillation_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed round"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all rounds even if completed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rounds without training"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current leaderboard status and exit"
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory to use for --resume/--status"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_yaml_config(args.config)

    # Setup run-scoped paths
    config_paths = config['paths']
    legacy_base_output_dir = config_paths['base_output_dir']
    ensure_dir(legacy_base_output_dir)
    run_dir = _resolve_run_dir(legacy_base_output_dir, args)
    paths = _build_run_scoped_paths(config_paths, run_dir)
    ensure_dir(paths['base_output_dir'])
    print(f"Run directory: {paths['base_output_dir']}")

    # Route all round artifacts under the active run directory.
    base_output_dir = paths['base_output_dir']

    # Initialize state manager
    state_manager = StateManager(paths['state_file'])

    # Status command
    if args.status:
        state = state_manager.load()
        if state:
            print(f"\nExperiment: {state.experiment_name}")
            print(f"Run dir: {paths['base_output_dir']}")
            print(f"Current round: {state.current_round}")
            print(f"Current teacher: {state.current_teacher_path}")
            print(f"Current teacher steps: {state.current_teacher_steps}")

            # Load and print leaderboard
            if os.path.exists(paths['leaderboard']['json']):
                data = load_json(paths['leaderboard']['json'])
                results = [RoundResult.from_dict(d) for d in data]
                print_leaderboard(results)
        else:
            print("No experiment state found. Pipeline has not been run yet.")
        return

    # Dry run
    if args.dry_run:
        print_dry_run(config)
        return

    # Setup experiment
    set_seed(config['execution']['seed'])

    thresholds = EvaluationThresholds(
        promptfoo_min=config['evaluation']['promptfoo_threshold']
    )

    # Initialize or resume experiment
    initial_steps = config['teacher']['initial_steps']
    min_steps = config['schedule']['min_steps']
    teacher_path = config['teacher']['model_path']

    if args.resume:
        state = state_manager.load()
        if state is None:
            print("No previous state found. Starting new experiment.")
            state = state_manager.initialize(
                config['experiment']['name'],
                teacher_path,
                initial_steps
            )
        else:
            print(f"Resuming from round {state.current_round}")
            teacher_path = state.current_teacher_path
            initial_steps = state.current_teacher_steps
    elif args.force:
        print("Force flag set - starting fresh experiment")
        state = state_manager.initialize(
            config['experiment']['name'],
            teacher_path,
            initial_steps
        )
    else:
        state = state_manager.load()
        if state and state.is_running:
            print("Found existing experiment. Use --resume to continue or --force to restart.")
            print(f"Current state: Round {state.current_round}, Teacher steps: {state.current_teacher_steps}")
            return
        state = state_manager.initialize(
            config['experiment']['name'],
            teacher_path,
            initial_steps
        )

    # Check Ollama for evaluation
    if not check_ollama_running():
        print("WARNING: Ollama server not detected at 127.0.0.1:11434")
        print("Evaluation will not work properly without Ollama running.")
        print("Start Ollama with: ollama serve")
        if not input("Continue anyway? (y/N): ").lower().startswith('y'):
            return

    # Calculate max rounds
    max_rounds = calculate_max_rounds(initial_steps, min_steps)
    logger = ProgressLogger(max_rounds)

    print("\n" + "=" * 70)
    print(f"Starting Nested Distillation Pipeline")
    print(f"Run directory: {paths['base_output_dir']}")
    print(f"Initial teacher: {teacher_path}")
    print(f"Initial steps: {initial_steps}")
    print(f"Min steps: {min_steps}")
    print(f"Max rounds: {max_rounds}")
    print(f"Promptfoo assertion threshold: {thresholds.promptfoo_min}%")
    print("=" * 70 + "\n")

    # Load existing results if resuming
    results = []
    if args.resume and os.path.exists(paths['leaderboard']['json']):
        data = load_json(paths['leaderboard']['json'])
        results = [RoundResult.from_dict(d) for d in data]
        print(f"Loaded {len(results)} previous results")

    # Main distillation loop
    teacher_steps = state.current_teacher_steps
    current_teacher_path = state.current_teacher_path
    start_round = state.current_round + 1 if args.resume else 1

    for round_num in range(start_round, max_rounds + 1):
        student_steps = max(min_steps, teacher_steps // 2)

        if student_steps < min_steps:
            print(f"\nReached minimum steps ({min_steps}). Stopping.")
            break

        # Get output directories
        output_dirs = get_output_dirs(base_output_dir, round_num, student_steps)

        # Check if already completed (unless force)
        if args.resume and os.path.exists(output_dirs['checkpoint']):
            existing = [r for r in results if r.round_number == round_num]
            if existing:
                logger.log(f"Round {round_num} already completed, skipping...")
                teacher_steps = student_steps
                current_teacher_path = existing[0].checkpoint_dir
                continue

        # Run the round
        result = run_single_round(
            config,
            state,
            round_num,
            teacher_steps,
            student_steps,
            current_teacher_path,
            output_dirs,
            logger,
            thresholds
        )

        if result is None:
            print(f"\nRound {round_num} failed. Stopping pipeline.")
            state.is_running = False
            state_manager.save(state)
            break

        # Update results
        results.append(result)
        update_leaderboard(results, paths['leaderboard'])

        # Update state
        state.current_round = round_num
        state.current_teacher_path = result.checkpoint_dir
        state.current_teacher_steps = student_steps
        state.completed_rounds.append(round_num)
        state_manager.save(state)

        # Print summary
        print("\n" + "-" * 70)
        print(f"Round {round_num} completed!")
        print(result.format_summary())
        print("-" * 70)

        # Keep going through the full halving ladder; the threshold is reported only.
        if not result.passed:
            print(f"\nPromptfoo assertion score {result.promptfoo_percent:.1f}% < {thresholds.promptfoo_min}% threshold")
            print("Continuing to the next halving round as requested.")

        # Prepare for next round
        teacher_steps = student_steps
        current_teacher_path = result.checkpoint_dir

        # Memory cleanup between rounds
        if config['execution']['memory_cleanup']:
            cleanup_gpu_memory()

    # Finalize
    state.is_running = False
    state_manager.save(state)

    print("\n" + "=" * 70)
    print("NESTED DISTILLATION PIPELINE COMPLETED")
    print("=" * 70)
    print_leaderboard(results)

    print(f"\nResults saved to:")
    print(f"  Markdown: {paths['leaderboard']['markdown']}")
    print(f"  CSV:      {paths['leaderboard']['csv']}")
    print(f"  JSON:     {paths['leaderboard']['json']}")


if __name__ == "__main__":
    import psutil
    main()
