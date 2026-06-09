#!/usr/bin/env python3
"""
Auto-generated LLaDA server for nested distillation evaluation.
Serves checkpoint: C:/Users/Gotri/Documents/tfg/LLaDA/workspace/LLaDA_outputs/nested_distillation/runs/run_20260609_105305/round_005_steps_4/checkpoint
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
sys.path.insert(0, r"C:\Users\Gotri\Documents\tfg\LLaDA")
from generate import generate

# --- CONFIGURATION ---
os.environ["HF_HOME"] = r"D:/tfg/.cache"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

DEVICE = "cuda"
CUDA_MEMORY_FRACTION = 0.85
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

CHECKPOINT_DIR = r"C:\Users\Gotri\Documents\tfg\LLaDA\workspace\LLaDA_outputs\nested_distillation\runs\run_20260609_105305\round_005_steps_4\checkpoint"
CHECKPOINT_DIR_POSIX = r"C:/Users/Gotri/Documents/tfg/LLaDA/workspace/LLaDA_outputs/nested_distillation/runs/run_20260609_105305/round_005_steps_4/checkpoint"
PORT = 5000
STEPS = 4
GEN_LENGTH = 64
BLOCK_LENGTH = 64

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

print(f"Loading tokenizer from {base_model_path}...")
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
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
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
    print(f"Loading LoRA checkpoint from {CHECKPOINT_DIR}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.tie_weights()
else:
    print("No LoRA adapter found; using base model only.")

model.eval()
print("Model loaded and ready to serve.")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    import time, hashlib

    t0_total = time.time()
    data = request.json
    prompt = data.get('prompt', '')

    conversation = [{"role": "user", "content": prompt}]
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

        timing = {
            "tokenization_ms": round(tok_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round(total_ms, 2),
            "steps": STEPS,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_preview": prompt[:120],
        }

        timing_log = os.environ.get("LLADA_TIMING_LOG")
        if timing_log:
            try:
                with open(timing_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(timing) + "\n")
            except Exception as e:
                print(f"[timing_log] ERROR writing to {timing_log}: {e}")

        return jsonify({"response": response, "timing": timing})
    except Exception as e:
        print(f"[generate_endpoint] ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Suppress Flask startup messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print(f"Starting server on port {PORT}...")
    app.run(host='127.0.0.1', port=PORT, threaded=False)
