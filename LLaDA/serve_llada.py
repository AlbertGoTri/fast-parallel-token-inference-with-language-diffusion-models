import os
import sys
import torch
import psutil
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from generate import generate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Prefer explicit overrides, then local Windows workspace, then legacy cluster path.
BASE_DIR = os.environ.get("LLADA_BASE_DIR", PROJECT_ROOT)
workspace_candidates = [
    os.environ.get("LLADA_WORKSPACE_DIR"),
    os.path.join(PROJECT_ROOT, "workspace"),
    os.path.join(BASE_DIR, "workspace"),
    os.path.expanduser("~/groups/hpai-collaborators/albert-gomez-triunfante/fast-parallel-token-inference-with-language-diffusion-models/workspace"),
]
WORKSPACE_DIR = None
for candidate in workspace_candidates:
    if candidate and os.path.isdir(candidate):
        WORKSPACE_DIR = os.path.abspath(candidate)
        break
if WORKSPACE_DIR is None:
    WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "workspace"))

os.makedirs(WORKSPACE_DIR, exist_ok=True)
HF_HOME = os.environ.get("HF_HOME") or os.path.join(WORKSPACE_DIR, ".cache")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

if torch.cuda.is_available():
    # Leave headroom for concurrent Ollama judge or Windows compositor VRAM usage.
    torch.cuda.set_per_process_memory_fraction(0.85)

USE_LORA = os.environ.get("LLADA_USE_LORA", "1").lower() not in {"0", "false", "no"}
LORA_DIR = os.path.abspath(os.path.expanduser(os.environ.get("LORA_DIR", os.path.join(WORKSPACE_DIR, "llada_student_lora"))))

# NF4 quant is chosen over INT8 because LLaDA's masking dynamics are sensitive to
# outlier logits, and NF4 preserves tail behavior better.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "GSAI-ML/LLaDA-8B-Instruct"
ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading base model...")
model = AutoModel.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# LLaDA shares input/output embeddings; without tying, generation produces garbage
# tokens at the vocabulary boundaries.
model.tie_weights()

if USE_LORA:
    adapter_config = os.path.join(LORA_DIR, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"Loading LoRA adapters from {LORA_DIR}...")
        model = PeftModel.from_pretrained(model, LORA_DIR)
        model.tie_weights()
    else:
        print(f"WARNING: LoRA not found at {LORA_DIR} (missing adapter_config.json).")
        print("WARNING: Continuing with base model. Set LORA_DIR or LLADA_USE_LORA=0 to control this explicitly.")

model.eval()
print("Model loaded and ready to serve.")


def resolve_block_length(steps, gen_length):
    """
    generate() requires gen_length % block_length == 0 and steps % num_blocks == 0;
    this brute-forces the largest valid block from common divisors.
    """
    for bl in [32, 64, 128]:
        if gen_length % bl == 0:
            num_blocks = gen_length // bl
            if steps % num_blocks == 0:
                return bl
    return gen_length


STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = resolve_block_length(STEPS, GEN_LENGTH)
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/config', methods=['GET'])
def config_endpoint():
    """Report current generation and model configuration."""
    return jsonify({
        "steps": STEPS,
        "gen_length": GEN_LENGTH,
        "block_length": BLOCK_LENGTH,
        "temperature": TEMPERATURE,
        "cfg_scale": CFG_SCALE,
        "remasking": REMASKING,
        "lora_loaded": hasattr(model, 'peft_config'),
    })

@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """Hot-swap LoRA adapters and/or generation parameters without reloading the base model."""
    global model, STEPS, GEN_LENGTH, BLOCK_LENGTH, TEMPERATURE, CFG_SCALE, REMASKING
    data = request.json or {}

    STEPS = data.get('steps', STEPS)
    GEN_LENGTH = data.get('gen_length', GEN_LENGTH)
    # Auto-resolving prevents callers from accidentally violating divisibility
    # after changing steps or gen_length.
    explicit_block = data.get('block_length')
    if explicit_block is not None:
        BLOCK_LENGTH = explicit_block
    else:
        BLOCK_LENGTH = resolve_block_length(STEPS, GEN_LENGTH)
    TEMPERATURE = data.get('temperature', TEMPERATURE)
    CFG_SCALE = data.get('cfg_scale', CFG_SCALE)
    REMASKING = data.get('remasking', REMASKING)

    new_lora_dir = data.get('lora_dir')
    if new_lora_dir and os.path.isdir(new_lora_dir):
        adapter_config = os.path.join(new_lora_dir, "adapter_config.json")
        if os.path.exists(adapter_config):
            print(f"[/reload] Unloading old adapters and loading from {new_lora_dir}...")
            # unload() drops PEFT wrappers in-place; merge_and_unload() fuses weights permanently.
            # We prefer unload to keep the base model reusable.
            if hasattr(model, 'unload'):
                model = model.unload()
            elif hasattr(model, 'merge_and_unload'):
                model = model.merge_and_unload()
            else:
                # Full reload is a last resort when the PEFT version does not support
                # runtime adapter swapping; it costs ~30s on a fast NVMe.
                print("[/reload] Full model reload required (fallback)...")
                del model
                torch.cuda.empty_cache()
                model = AutoModel.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.tie_weights()

            model = PeftModel.from_pretrained(model, new_lora_dir)
            model.tie_weights()
            model.eval()
            print("[/reload] LoRA swapped successfully.")
            return jsonify({"status": "ok", "lora_loaded": new_lora_dir, "steps": STEPS})
        else:
            return jsonify({"status": "error", "reason": f"No adapter_config.json in {new_lora_dir}"}), 400
    else:
        if 'lora_dir' in data and not new_lora_dir:
            if hasattr(model, 'unload'):
                model = model.unload()
                model.tie_weights()
                model.eval()
                print("[/reload] Adapters unloaded; using base model.")
                return jsonify({"status": "ok", "lora_loaded": None, "steps": STEPS})
        return jsonify({"status": "ok", "steps": STEPS, "note": "params updated, no LoRA change"})

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
    ).to("cuda")
    tok_ms = (time.time() - t0_tok) * 1000

    with torch.no_grad():
        t0_gen = time.time()
        output = generate(
            model,
            input_ids,
            steps=STEPS,
            gen_length=GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            temperature=TEMPERATURE,
            cfg_scale=CFG_SCALE,
            remasking=REMASKING,
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

if __name__ == '__main__':
    app.run(port=5000)
