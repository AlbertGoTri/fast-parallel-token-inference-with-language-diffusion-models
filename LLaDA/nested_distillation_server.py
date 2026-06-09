"""
Server management for nested distillation evaluation.
Handles starting/stopping the LLaDA server for promptfoo evaluation.
"""

import os
import sys
import time
import signal
import subprocess
import socket
from contextlib import contextmanager
from typing import Optional, Tuple


def find_free_port(start_port: int = 5000, max_port: int = 6000) -> int:
    """Find a free port to use."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError("No free ports found")


def check_server_running(port: int = 5000, timeout: float = 1.0) -> bool:
    """Check if the LLaDA server is running."""
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            method='GET'
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, Exception):
        return False


def _find_pids_on_port(port: int) -> list[int]:
    """Best-effort lookup of PIDs listening on a TCP port."""
    pids: list[int] = []
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        try:
                            pids.append(int(parts[-1]))
                        except ValueError:
                            continue
        else:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True,
                text=True,
                check=False
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
    except Exception:
        return []

    # preserve order while deduplicating
    seen = set()
    deduped = []
    for pid in pids:
        if pid not in seen:
            deduped.append(pid)
            seen.add(pid)
    return deduped


def stop_server_on_port(port: int = 5000) -> None:
    """Force-stop any process listening on the given port."""
    pids = _find_pids_on_port(port)
    if not pids:
        return

    print(f"Found existing process(es) on port {port}: {pids}; stopping them...")
    for pid in pids:
        try:
            if sys.platform == 'win32':
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            continue

    # Give sockets a brief moment to release.
    time.sleep(1.0)


def wait_for_server(port: int = 5000, timeout: float = 120.0, interval: float = 5.0, verbose: bool = False) -> bool:
    """Wait for the server to be ready."""
    print(f"Waiting for server on port {port} (timeout: {timeout}s)...")
    start_time = time.time()
    last_print = start_time

    while time.time() - start_time < timeout:
        if check_server_running(port):
            print(f"Server ready in {time.time() - start_time:.1f}s")
            return True

        # Print progress every 10 seconds
        if verbose and time.time() - last_print > 10:
            elapsed = time.time() - start_time
            print(f"  Still waiting... ({elapsed:.0f}s elapsed, model loading)")
            last_print = time.time()

        time.sleep(interval)

    print(f"Server did not become ready within {timeout}s")
    return False


def create_server_script(
    checkpoint_dir: str,
    output_path: str,
    port: int = 5000,
    steps: int = 128,
    device: str = "cuda",
    cuda_memory_fraction: float = 0.85,
    hf_home: Optional[str] = None,
    gen_length: int = 128,
    block_length: int = 32,
) -> str:
    """
    Create a temporary server script for the specific checkpoint.

    Args:
        checkpoint_dir: Path to the LoRA checkpoint
        output_path: Path to write the server script
        port: Port to serve on
        steps: Number of generation steps
        gen_length: Number of tokens to generate
        block_length: Block size for semi-autoregressive generation

    Returns:
        Path to the created script
    """
    # Get absolute path to LLaDA directory (where generate.py lives)
    llada_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_doc_path = checkpoint_dir.replace('\\', '/')
    checkpoint_dir_posix = checkpoint_doc_path

    hf_home = os.path.expanduser(hf_home or "")

    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated LLaDA server for nested distillation evaluation.
Serves checkpoint: {checkpoint_doc_path}
"""

import os
import sys
import json
import torch
import psutil
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

# Add LLaDA directory to path for imports
sys.path.insert(0, r"{llada_dir}")
from generate import generate

# --- CONFIGURATION ---
os.environ["HF_HOME"] = r"{hf_home}"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

DEVICE = "{device}"
CUDA_MEMORY_FRACTION = {cuda_memory_fraction}
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

CHECKPOINT_DIR = r"{checkpoint_dir}"
CHECKPOINT_DIR_POSIX = r"{checkpoint_dir_posix}"
PORT = {port}
STEPS = {steps}
GEN_LENGTH = {gen_length}
BLOCK_LENGTH = {block_length}

BASE_MODEL_FALLBACK = "GSAI-ML/LLaDA-8B-Instruct"


def resolve_base_model_path(checkpoint_path: str, fallback: str) -> str:
    if not checkpoint_path:
        return fallback
    if os.path.isdir(checkpoint_path):
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                return adapter_config.get("base_model_name_or_path") or fallback
            except Exception:
                return fallback
        # Directory without adapter config; assume it is a base model dir.
        return checkpoint_path
    # Non-directory string; assume it is a model id or path.
    return checkpoint_path

# --- MODEL LOADING ---
model_id = "GSAI-ML/LLaDA-8B-Instruct"
ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
base_model_path = resolve_base_model_path(CHECKPOINT_DIR, BASE_MODEL_FALLBACK)

print(f"Loading tokenizer from {{base_model_path}}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print("Loading base model...")
if DEVICE == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModel.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={{0: "6GiB", "cpu": f"{{ram_gb}}GiB"}},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
else:
    print("CUDA disabled for eval server; running generation on CPU")
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
model.tie_weights()

use_lora = False
if os.path.isdir(CHECKPOINT_DIR):
    adapter_config_path = os.path.join(CHECKPOINT_DIR, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        use_lora = True

if use_lora:
    print(f"Loading LoRA checkpoint from {{CHECKPOINT_DIR}}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.tie_weights()
else:
    print("No LoRA adapter found; using base model only.")

model.eval()
print("Model loaded and ready to serve.")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{"status": "ok"}})

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    import time, hashlib

    t0_total = time.time()
    data = request.json
    prompt = data.get('prompt', '')

    conversation = [{{"role": "user", "content": prompt}}]
    t0_tok = time.time()
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)
    tok_ms = (time.time() - t0_tok) * 1000

    try:
        with torch.no_grad():
            t0_gen = time.time()
            output = generate(
                model,
                input_ids,
                steps=STEPS,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            gen_ms = (time.time() - t0_gen) * 1000

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        total_ms = (time.time() - t0_total) * 1000

        timing = {{
            "tokenization_ms": round(tok_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round(total_ms, 2),
            "steps": STEPS,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_preview": prompt[:120],
        }}

        timing_log = os.environ.get("LLADA_TIMING_LOG")
        if timing_log:
            try:
                with open(timing_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(timing) + "\\n")
            except Exception as e:
                print(f"[timing_log] ERROR writing to {{timing_log}}: {{e}}")

        return jsonify({{"response": response, "timing": timing}})
    except Exception as e:
        print(f"[generate_endpoint] ERROR: {{e}}")
        return jsonify({{"error": str(e)}}), 500

if __name__ == '__main__':
    # Suppress Flask startup messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print(f"Starting server on port {{PORT}}...")
    app.run(host='127.0.0.1', port=PORT, threaded=False)
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return output_path


class ServerManager:
    """Manages the LLaDA server lifecycle for evaluation."""

    def __init__(
        self,
        checkpoint_dir: str,
        steps: int,
        port: int = 5000,
        device: str = "cuda",
        cuda_memory_fraction: float = 0.85,
        eval_dir: Optional[str] = None,
        hf_home: Optional[str] = None,
        timing_log_path: Optional[str] = None,
        gen_length: int = 128,
        block_length: int = 32,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.steps = steps
        self.port = port
        self.device = device
        self.cuda_memory_fraction = cuda_memory_fraction
        self.eval_dir = eval_dir
        self.hf_home = hf_home
        self.timing_log_path = timing_log_path
        self.gen_length = gen_length
        self.block_length = block_length
        self.process: Optional[subprocess.Popen] = None
        self.script_path: Optional[str] = None
        self.log_file: Optional[str] = None

    def start(self, timeout: float = 600.0) -> bool:
        """Start the server."""
        if check_server_running(self.port):
            print(f"Server already running on port {self.port}; replacing stale server")
            stop_server_on_port(self.port)

            if check_server_running(self.port):
                print(f"ERROR: Could not free port {self.port}")
                return False

        # Create server script
        eval_dir = self.eval_dir or os.path.join(os.path.dirname(self.checkpoint_dir), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        self.script_path = os.path.join(eval_dir, "serve_eval.py")
        self.log_file = os.path.join(eval_dir, "server.log")

        create_server_script(
            self.checkpoint_dir,
            self.script_path,
            self.port,
            self.steps,
            self.device,
            self.cuda_memory_fraction,
            hf_home=self.hf_home,
            gen_length=self.gen_length,
            block_length=self.block_length,
        )

        print(f"Starting LLaDA server on port {self.port}...")
        print(f"  Checkpoint: {self.checkpoint_dir}")
        print(f"  Steps: {self.steps}")
        print(f"  Device: {self.device}")
        if self.device == "cuda":
            print(f"  CUDA memory fraction: {self.cuda_memory_fraction}")
        print(f"  Log file: {self.log_file}")

        # Start server process with output directed to file (avoids pipe deadlock)
        try:
            # Use CREATE_NEW_PROCESS_GROUP on Windows for proper termination
            creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)

            env = os.environ.copy()
            if self.timing_log_path:
                env["LLADA_TIMING_LOG"] = self.timing_log_path
                print(f"  Timing log: {self.timing_log_path}")

            with open(self.log_file, 'w') as log:
                self.process = subprocess.Popen(
                    [sys.executable, self.script_path],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                    env=env,
                )

            # Wait for server to be ready
            if not wait_for_server(self.port, timeout=timeout, verbose=True):
                # Print log contents on failure
                try:
                    with open(self.log_file, 'r') as f:
                        log_content = f.read()
                        if log_content:
                            print("Server log output:")
                            print(log_content[-2000:])  # Last 2000 chars
                except Exception as e:
                    print(f"Could not read log file: {e}")
                self.stop()
                return False

            return True

        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop(self) -> None:
        """Stop the server."""
        if self.process is None:
            return

        print(f"Stopping LLaDA server (PID {self.process.pid})...")

        try:
            # Try graceful termination first
            if hasattr(self.process, 'send_signal'):
                # On Windows, use CTRL_BREAK_EVENT
                if sys.platform == 'win32':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGTERM)

            # Wait for termination
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()

        except Exception as e:
            print(f"Error stopping server: {e}")
            try:
                self.process.kill()
            except:
                pass

        self.process = None
        print("Server stopped")

    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


@contextmanager
def managed_server(
    checkpoint_dir: str,
    steps: int,
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
    """
    Context manager for running the LLaDA server.

    Usage:
        with managed_server(checkpoint_dir, steps) as server:
            # Server is running, run evaluations
            run_promptfoo(...)
    """
    manager = ServerManager(
        checkpoint_dir,
        steps,
        port,
        device=device,
        cuda_memory_fraction=cuda_memory_fraction,
        eval_dir=eval_dir,
        hf_home=hf_home,
        timing_log_path=timing_log_path,
        gen_length=gen_length,
        block_length=block_length,
    )
    try:
        if not manager.start(timeout=timeout):
            raise RuntimeError("Failed to start server")
        yield manager
    finally:
        manager.stop()
