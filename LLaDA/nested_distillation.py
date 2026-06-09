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
import time
import json
import pickle
import gzip
import urllib.request
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

import torch
import gc

from nested_distillation_utils import (
    log_memory_usage, cleanup_gpu_memory, verify_gpu_empty, unload_model,
    set_seed, ensure_dir, load_yaml_config, save_json, load_json,
    save_csv, format_timestamp, RoundResult, ExperimentState,
    StateManager, ProgressLogger, print_leaderboard, validate_step_reduction,
    check_ollama_running, compute_latency_aggregates
)
from nested_distillation_eval import (
    evaluate_round, EvaluationThresholds, check_continue
)
from nested_distillation_server import (
    managed_server, check_server_running, wait_for_server
)


def _save_cache_object(obj: Dict[str, Any], path: str) -> None:
    """Save cache dict using pickle+gzip to avoid PyTorch ZIP corruption on Windows."""
    temp_path = path + ".tmp"
    with gzip.open(temp_path, "wb", compresslevel=3) as f:
        # Convert tensors to numpy arrays for compatibility
        payload = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                payload[k] = ("tensor", v.detach().cpu().numpy())
            else:
                payload[k] = ("raw", v)
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, path)


def _load_cache_object(path: str) -> Dict[str, Any]:
    """Load cache dict saved with _save_cache_object and restore tensors."""
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    result = {}
    for k, (kind, v) in payload.items():
        if kind == "tensor":
            result[k] = torch.from_numpy(v)
        else:
            result[k] = v
    return result


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
    run_paths['timing'] = {
        'per_prompt': os.path.join(run_dir, 'per_prompt_latencies.jsonl'),
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


def _reload_external_server(checkpoint_dir: str, steps: int, port: int = 5000, timeout: float = 60.0) -> bool:
    """Hot-swap an externally-managed server via its /reload endpoint."""
    url = f"http://127.0.0.1:{port}/reload"
    data = json.dumps({"lora_dir": checkpoint_dir, "steps": steps}).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("status") == "ok":
                print(f"[ExternalServer] Reloaded LoRA from {checkpoint_dir} (steps={steps})")
                return True
            else:
                print(f"[ExternalServer] Reload failed: {result}")
                return False
    except Exception as e:
        print(f"[ExternalServer] Could not reach /reload: {e}")
        return False


@contextmanager
def _managed_or_external_server(
    checkpoint_dir: str,
    steps: int,
    use_external: bool = False,
    port: int = 5000,
    timeout: float = 600.0,
    device: str = "cuda",
    cuda_memory_fraction: float = 0.85,
    eval_dir: Optional[str] = None,
    hf_home: Optional[str] = None,
    timing_log_path: Optional[str] = None,
    gen_length: int = 128,
    block_length: int = 32,
):
    """Context manager that either reloads an external server or starts a managed one."""
    if use_external:
        if not check_server_running(port):
            print(f"ERROR: External server not detected on port {port}")
            print("Start it first with: python serve_llada.py")
            raise RuntimeError(f"No external server on port {port}")
        if not _reload_external_server(checkpoint_dir, steps, port):
            raise RuntimeError("External server /reload failed")
        yield None  # No server object needed
    else:
        with managed_server(
            checkpoint_dir, steps, port=port, timeout=timeout,
            device=device, cuda_memory_fraction=cuda_memory_fraction,
            eval_dir=eval_dir, hf_home=hf_home,
            timing_log_path=timing_log_path,
            gen_length=gen_length,
            block_length=block_length,
        ) as mgr:
            yield mgr


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
    os.environ["HF_HOME"] = os.path.expanduser(config['system']['hf_home'])
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
        texts = [t for t in dataset['text']
             if len(t.strip()) > config['student']['min_text_length']]
        texts = texts[:config['student']['num_train_examples']]
        logger.log(f"Loaded {len(texts)} training examples")
    except Exception as e:
        logger.log(f"ERROR: Failed to load dataset: {e}")
        unload_model(model, "teacher")
        return False

    # Generate trajectories
    logger.log(f"Generating trajectories with steps={teacher_steps}, target_step={target_step}")

    student_config = config['student']

    with torch.no_grad():
        for i, text in enumerate(texts):
            prompt_content = f"Briefly summarize or continue this text:\n{text[:200]}"
            conversation = [{"role": "user", "content": prompt_content}]

            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_len = input_ids.shape[1]
            logger.log(f"Processing example {i+1}/{len(texts)} (prompt_len={prompt_len})")

            try:
                state_x, teacher_logits, attn_mask = generate_and_cache_trajectory(
                    model,
                    input_ids,
                    steps=teacher_steps,
                    gen_length=student_config['gen_length'],
                    block_length=student_config['block_length'],
                    target_step=target_step,
                )

                file_path = os.path.join(cache_dir, f"batch_{i}.pkl.gz")
                try:
                    # Ensure GPU writes are complete before saving to disk
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    payload = {
                        'input_x': state_x.cpu(),
                        'target_logits': teacher_logits.cpu(),
                        'attn_mask': attn_mask.cpu() if attn_mask is not None else None,
                        'prompt_len': prompt_len,
                        'target_step': target_step,
                    }
                    _save_cache_object(payload, file_path)
                    logger.log(f"Saved trajectory to {file_path}")
                except Exception as e:
                    logger.log(f"ERROR: Failed to save trajectory {i+1}: {e}")
                    continue

            except Exception as e:
                logger.log(f"ERROR: Failed to generate trajectory {i+1}: {e}")
                continue

            # Clear cache after each example
            del input_ids, state_x, teacher_logits
            if attn_mask is not None:
                del attn_mask
            torch.cuda.empty_cache()

    # Cleanup and validate count
    all_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl.gz')]
    logger.log(f"Validating {len(all_files)} cache files...")
    valid_count = 0
    for f in all_files:
        fpath = os.path.join(cache_dir, f)
        try:
            _ = _load_cache_object(fpath)
            valid_count += 1
        except Exception:
            logger.log(f"WARNING: Removing corrupted cache file {f}")
            try:
                os.remove(fpath)
            except Exception:
                pass
    logger.log(f"Cached {valid_count} valid trajectories")

    unload_model(model, "teacher")
    unload_model(tokenizer, "tokenizer")
    log_memory_usage("cache_end")

    logger.end_stage("cache", success=valid_count > 0)
    return valid_count > 0


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
    os.environ["HF_HOME"] = os.path.expanduser(config['system']['hf_home'])
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SAFETENSORS_FAST_GPU"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.cuda.set_per_process_memory_fraction(config['system']['cuda_memory_fraction'])

    # Load cache files
    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl.gz"))
    if not cache_files:
        logger.log(f"ERROR: No cache files found in {cache_dir}")
        return False

    logger.log(f"Found {len(cache_files)} cached trajectories")

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
        if hasattr(student_base, "gradient_checkpointing_enable"):
            try:
                student_base.gradient_checkpointing_enable()
            except Exception as e:
                logger.log(f"WARNING: Gradient checkpointing not supported: {e}")

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
            for i, cache_file in enumerate(cache_files):
                torch.cuda.empty_cache()

                try:
                    trajectory = _load_cache_object(cache_file)
                except Exception as e:
                    logger.log(f"WARNING: Skipping corrupted cache file {cache_file}: {e}")
                    continue

                input_x = trajectory['input_x'].to("cuda")
                target_logits = trajectory['target_logits'].to("cuda").to(torch.float16)
                attn_mask = trajectory['attn_mask'].to("cuda") if trajectory['attn_mask'] is not None else None
                prompt_len = trajectory.get('prompt_len', 0)

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_output = student_model(input_x, attention_mask=attn_mask)
                    student_logits = student_output.logits

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
                logger.log(f"Epoch {epoch+1} | Batch {i+1}/{len(cache_files)} | Loss: {loss.item():.4f} | VRAM: {vram:.2f}GB")

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
    thresholds: EvaluationThresholds,
    use_external_server: bool = False,
    previous_result: Optional[RoundResult] = None,
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
    t0_cache = time.time()
    if not cache_stage(config, round_num, teacher_path, teacher_steps, target_step, cache_dir, logger):
        logger.log("ERROR: Cache stage failed")
        return None
    cache_duration = time.time() - t0_cache

    # Stage 2: Train student
    checkpoint_dir = output_dirs['checkpoint']
    t0_train = time.time()
    if not train_stage(config, round_num, cache_dir, checkpoint_dir, student_steps, logger):
        logger.log("ERROR: Train stage failed")
        return None
    train_duration = time.time() - t0_train

    # Clean up cache files after training to save disk space
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl.gz')]
        if cache_files:
            logger.log(f"Cleaning up {len(cache_files)} cache files to free disk space...")
            for f in cache_files:
                try:
                    os.remove(os.path.join(cache_dir, f))
                except Exception as e:
                    logger.log(f"WARNING: Could not remove cache file {f}: {e}")
            logger.log(f"Freed ~{sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if f.endswith('.pkl.gz')) / 1024**2:.0f} MB from cache dir")

    # Stage 3 & 4: Evaluation
    logger.start_stage("evaluation")

    # Ensure clean GPU before evaluation
    if config['execution']['verify_gpu_empty']:
        verify_gpu_empty()
    cleanup_gpu_memory()
    log_memory_usage("eval_start")

    # Per-round timing log path (used by managed server)
    timing_log_path = os.path.join(output_dirs['eval'], "generation_timing.jsonl")

    # Run evaluation with managed server
    eval_result = None
    t0_eval = time.time()
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
        if use_external_server:
            logger.log("Using external server on port 5000 (must be running separately)")
        logger.log(f"Starting evaluation server for {student_steps} steps (eval using {eval_steps} steps)...")

        with _managed_or_external_server(
            checkpoint_dir,
            eval_steps,
            use_external=use_external_server,
            port=5000,
            timeout=eval_server_timeout,
            device=eval_server_device,
            cuda_memory_fraction=eval_server_cuda_fraction,
            hf_home=config['system']['hf_home'],
            timing_log_path=None if use_external_server else timing_log_path,
            gen_length=config['student']['gen_length'],
            block_length=config['student']['block_length'],
        ) as server:
            if not use_external_server:
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

        if use_external_server:
            logger.log("External server left running")
        else:
            logger.log("Evaluation server stopped")

    except Exception as e:
        logger.log(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.end_stage("evaluation", success=False)
        logger.end_stage("round", success=False)
        return None

    eval_duration = time.time() - t0_eval
    log_memory_usage("eval_end")
    logger.end_stage("evaluation", success=eval_result.get('promptfoo_success', False))

    # Promptfoo score is informational only; continuation is no longer gated on it.
    promptfoo_percent = eval_result.get('promptfoo_assertion_percent', eval_result.get('promptfoo_percent', 0.0))
    perplexity = eval_result.get('perplexity', 999.9)
    passed = True

    # Compute per-prompt latency aggregates from timing log
    per_prompt_records = _read_timing_log(timing_log_path)
    if not per_prompt_records and use_external_server:
        # External server: timing log unavailable, try to read from promptfoo metadata if present
        promptfoo_details = eval_result.get('promptfoo_details', {})
        raw_results = promptfoo_details.get('results', [])
        if isinstance(raw_results, dict) and 'results' in raw_results:
            raw_results = raw_results['results']
        per_prompt_records = []
        for res in raw_results:
            meta = res.get('metadata', {})
            timing = meta.get('timing', {})
            if 'generation_ms' in timing:
                per_prompt_records.append(timing)

    latency_aggs = compute_latency_aggregates(per_prompt_records)
    latency_aggs.cache_s = cache_duration
    latency_aggs.train_s = train_duration
    latency_aggs.eval_s = eval_duration
    latency_aggs.total_pipeline_s = cache_duration + train_duration + eval_duration

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
        eval_dir=output_dirs['eval'],
        stage_timings=latency_aggs,
        per_prompt_latencies=per_prompt_records,
    )

    if previous_result is not None:
        result.previous_avg_generation_ms = previous_result.stage_timings.avg_generation_ms

    logger.end_stage("round", success=True)
    return result


def evaluate_teacher_baseline(
    config: Dict[str, Any],
    teacher_path: str,
    teacher_steps: int,
    base_output_dir: str,
    logger: ProgressLogger,
    thresholds: EvaluationThresholds,
    use_external_server: bool = False,
) -> RoundResult:
    """Run a baseline evaluation for the initial teacher model."""
    logger.start_stage("teacher_eval")

    if config['execution']['verify_gpu_empty']:
        verify_gpu_empty()
    cleanup_gpu_memory()
    log_memory_usage("teacher_eval_start")

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

    eval_steps = eval_config.get('eval_steps', 0)
    if eval_steps == 0:
        eval_steps = teacher_steps

    logger.log(
        f"Starting teacher evaluation server for {teacher_steps} steps (eval using {eval_steps} steps)..."
    )

    eval_result = None
    eval_dir = os.path.join(base_output_dir, "teacher_eval")
    reports_dir = os.path.join(eval_dir, "reports")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    timing_log_path = os.path.join(eval_dir, "generation_timing.jsonl")
    promptfoo_template = os.path.abspath(eval_config['promptfoo']['config_path'])
    provider_abs = os.path.abspath(eval_config['promptfoo']['provider_path'])

    t0_eval = time.time()
    with _managed_or_external_server(
        teacher_path,
        eval_steps,
        use_external=use_external_server,
        port=5000,
        timeout=eval_server_timeout,
        device=eval_server_device,
        cuda_memory_fraction=eval_server_cuda_fraction,
        eval_dir=eval_dir,
        hf_home=config['system']['hf_home'],
        timing_log_path=None if use_external_server else timing_log_path,
        gen_length=config['student']['gen_length'],
        block_length=config['student']['block_length'],
    ) as server:
        if use_external_server:
            logger.log("External server ready for teacher baseline...")
        else:
            logger.log("Server is ready, running teacher baseline evaluation...")
        eval_result = evaluate_round(
            round_num=0,
            student_steps=teacher_steps,
            promptfoo_config_path=promptfoo_template,
            provider_path=provider_abs,
            eval_output_dir=eval_dir,
            reports_dir=reports_dir,
            perplexity_device=eval_config['perplexity']['device'],
            max_concurrency=eval_config['promptfoo']['max_concurrency'],
            timeout_ms=eval_config['promptfoo']['timeout_ms'],
            judge_num_predict=int(eval_config['promptfoo'].get('judge_num_predict', 256)),
            judge_num_gpu=int(eval_config['promptfoo'].get('judge_num_gpu', 1)),
        )
    eval_duration = time.time() - t0_eval

    logger.log("Teacher evaluation server stopped")
    log_memory_usage("teacher_eval_end")
    logger.end_stage("teacher_eval", success=eval_result.get('promptfoo_success', False))

    promptfoo_percent = eval_result.get('promptfoo_assertion_percent', eval_result.get('promptfoo_percent', 0.0))
    perplexity = eval_result.get('perplexity', 999.9)
    passed = True

    # Compute per-prompt latency aggregates
    per_prompt_records = _read_timing_log(timing_log_path)
    latency_aggs = compute_latency_aggregates(per_prompt_records)
    latency_aggs.eval_s = eval_duration
    latency_aggs.total_pipeline_s = eval_duration

    return RoundResult(
        round_number=0,
        student_name="teacher_baseline",
        teacher_name=teacher_path,
        student_steps=teacher_steps,
        teacher_steps=teacher_steps,
        promptfoo_percent=promptfoo_percent,
        perplexity=perplexity,
        passed=passed,
        timestamp=format_timestamp(),
        cache_dir="",
        checkpoint_dir=teacher_path,
        eval_dir=eval_dir,
        stage_timings=latency_aggs,
        per_prompt_latencies=per_prompt_records,
    )


def update_leaderboard(results: list, paths: Dict[str, str]) -> None:
    """Update all leaderboard files (MD, CSV, JSON)."""
    # JSON
    save_json([r.to_dict() for r in results], paths['json'])

    # CSV — flatten stage_timings so columns are readable
    if results:
        flat_rows = []
        for r in results:
            st = r.stage_timings
            avg_speedup = ""
            if r.previous_avg_generation_ms > 0 and st.avg_generation_ms > 0:
                avg_speedup = round(r.previous_avg_generation_ms / st.avg_generation_ms, 2)
            flat_rows.append({
                "round_number": r.round_number,
                "student_steps": r.student_steps,
                "promptfoo_percent": round(r.promptfoo_percent, 2),
                "perplexity": round(r.perplexity, 2),
                "cache_s": round(st.cache_s, 1),
                "train_s": round(st.train_s, 1),
                "eval_s": round(st.eval_s, 1),
                "avg_gen_ms": round(st.avg_generation_ms, 1),
                "median_gen_ms": round(st.median_generation_ms, 1),
                "p95_gen_ms": round(st.p95_generation_ms, 1),
                "num_prompts": st.num_prompts,
                "avg_speedup": avg_speedup,
                "passed": "Yes" if r.passed else "No",
                "timestamp": r.timestamp[:19],
            })
        save_csv(flat_rows, paths['csv'])

    # Markdown
    md_lines = ["# Nested Distillation Leaderboard\n"]
    md_lines.append(
        "\n| Round | Steps | Promptfoo % | Perplexity | Cache(s) | Train(s) | Eval(s) | "
        "AvgGen(ms) | MedGen(ms) | P95Gen(ms) | Speedup | Pass | Timestamp |"
    )
    md_lines.append(
        "|-------|-------|-------------|------------|----------|----------|---------|"
        "------------|------------|------------|---------|------|-----------|"
    )

    for r in results:
        st = r.stage_timings
        avg_speedup = ""
        if r.previous_avg_generation_ms > 0 and st.avg_generation_ms > 0:
            avg_speedup = f"{r.previous_avg_generation_ms / st.avg_generation_ms:.2f}x"
        status = "✓" if r.passed else "✗"
        md_lines.append(
            f"| {r.round_number} | {r.student_steps} | {r.promptfoo_percent:.1f}% | "
            f"{r.perplexity:.2f} | {st.cache_s:.1f} | {st.train_s:.1f} | {st.eval_s:.1f} | "
            f"{st.avg_generation_ms:.0f} | {st.median_generation_ms:.0f} | {st.p95_generation_ms:.0f} | "
            f"{avg_speedup} | {status} | {r.timestamp[:19]} |"
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


def _read_timing_log(timing_log_path: str) -> list:
    """Read a JSONL timing log into a list of dicts."""
    if not os.path.exists(timing_log_path):
        return []
    records = []
    with open(timing_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


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

    parser.add_argument(
        "--use-external-server",
        action="store_true",
        help="Use an externally-managed server (serve_llada.py) instead of starting a new one per round"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_yaml_config(args.config)

    # Resolve paths relative to the config file's directory so relative
    # entries such as "workspace/outputs/..." are stable regardless of CWD.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.dirname(os.path.abspath(args.config))

    def _resolve_config_path(value: str) -> str:
        value = os.path.expanduser(value)
        if os.path.isabs(value):
            return value
        # Prefer resolution relative to the config file; fallback to script dir
        candidate = os.path.join(config_dir, value)
        if os.path.exists(candidate) or os.path.dirname(candidate):
            return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(script_dir, value))

    for key in ("base_output_dir", "cache_dir", "checkpoint_dir", "eval_dir", "state_file"):
        if key in config.get("paths", {}):
            config["paths"][key] = _resolve_config_path(config["paths"][key])
    if "leaderboard" in config.get("paths", {}):
        for key, value in config["paths"]["leaderboard"].items():
            config["paths"]["leaderboard"][key] = _resolve_config_path(value)
    if "hf_home" in config.get("system", {}):
        config["system"]["hf_home"] = _resolve_config_path(config["system"]["hf_home"])
    # Similarly resolve promptfoo config/template and provider paths
    for pkey in ("config_path", "provider_path"):
        if pkey in config.get("evaluation", {}).get("promptfoo", {}):
            config["evaluation"]["promptfoo"][pkey] = _resolve_config_path(
                config["evaluation"]["promptfoo"][pkey]
            )

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
        promptfoo_min=float(config['evaluation'].get('promptfoo_threshold', 0.0))
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
    print("Promptfoo assertion threshold: disabled (informational only)")
    print("=" * 70 + "\n")

    # Load existing results if resuming
    results = []
    if args.resume and os.path.exists(paths['leaderboard']['json']):
        data = load_json(paths['leaderboard']['json'])
        results = [RoundResult.from_dict(d) for d in data]
        print(f"Loaded {len(results)} previous results")

    # Baseline evaluation for initial teacher (round 0)
    baseline_done = any(r.round_number == 0 for r in results)
    if state.current_round == 0 and not baseline_done:
        try:
            print("\n" + "=" * 70)
            print(f"ROUND 0/BASELINE | Teacher Steps: {initial_steps}")
            print("=" * 70)
            print(f"[Round 0] Teacher path: {teacher_path}")
            baseline_result = evaluate_teacher_baseline(
                config,
                teacher_path,
                initial_steps,
                base_output_dir,
                logger,
                thresholds,
                use_external_server=args.use_external_server,
            )
            results.append(baseline_result)
            update_leaderboard(results, paths['leaderboard'])
            print("\n" + "-" * 70)
            print("Teacher baseline evaluation completed!")
            print(baseline_result.format_summary())
            print("-" * 70)
        except Exception as e:
            print(f"WARNING: Teacher baseline evaluation failed: {e}")

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

        # Determine previous result for speedup calculation
        previous_result = None
        if results:
            prev = [r for r in results if r.round_number == round_num - 1]
            if prev:
                previous_result = prev[0]

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
            thresholds,
            use_external_server=args.use_external_server,
            previous_result=previous_result,
        )

        if result is None:
            print(f"\nRound {round_num} failed. Stopping pipeline.")
            state.is_running = False
            state_manager.save(state)
            break

        # Update results
        results.append(result)
        update_leaderboard(results, paths['leaderboard'])

        # Write per-prompt latencies to master JSONL in run dir
        if 'timing' in paths and result.per_prompt_latencies:
            with open(paths['timing']['per_prompt'], 'a', encoding='utf-8') as f:
                for entry in result.per_prompt_latencies:
                    enriched = dict(entry)
                    enriched['round'] = result.round_number
                    enriched['student_steps'] = result.student_steps
                    f.write(json.dumps(enriched) + '\n')

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

        # Keep going through the full halving ladder; Promptfoo is informational only.
        if not result.passed:
            print(f"\nPromptfoo assertion score {result.promptfoo_percent:.1f}% recorded for reporting only.")

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
    if 'timing' in paths:
        print(f"  Per-prompt latencies: {paths['timing']['per_prompt']}")


if __name__ == "__main__":
    import psutil
    main()
