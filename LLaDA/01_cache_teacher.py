import os
import sys
import torch
import psutil
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
from generate_cache import generate_and_cache_trajectory

# --- ENVIRONMENT CONFIGURATION ---
# Use the project root relative to this script, falling back to sensible defaults.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.environ.get("LLADA_BASE_DIR", PROJECT_ROOT)

WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
# Hugging Face cache root: model weights, tokenizer files, and dataset caches
# will be stored here to avoid re-downloading each run.
# Prefer explicit HF_HOME env var or config override, fallback to workspace/.cache
HF_HOME = os.environ.get("HF_HOME") or os.path.join(WORKSPACE_DIR, ".cache")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
# Silence advisory warnings to reduce noisy logs.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Avoid symlink warnings on shared filesystems.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Disable a fast GPU path that can increase memory pressure on some setups.
os.environ["SAFETENSORS_FAST_GPU"] = "0"

# Folder where cached trajectories are stored.
# Each cached file is a single training trajectory for the student.
CACHE_DIR = os.path.join(WORKSPACE_DIR, "llada_trajectories")

# Clear any previous cache to ensure a clean run with current settings
# (steps, target_step, dataset slice) without mixing old artifacts.
import shutil
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory cleared and recreated: {CACHE_DIR}")

# --- TEACHER MODEL LOAD (4-BIT) ---
# Configure 4-bit quantization to fit the 8B model into limited VRAM.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Hugging Face model id for the teacher.
model_id = "GSAI-ML/LLaDA-8B-Instruct"

# Compute available system RAM and leave a safety margin so the OS remains stable.
ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"Available RAM for loading: {ram_gb} GB")

# Tokenizer is required to format prompts into model input ids.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the teacher in 4-bit quantized form with automatic device mapping
# and explicit memory limits for GPU/CPU offload.
print("Loading Teacher Model (LLaDA-8B) in 4-bit...")
try:
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Switch to eval mode because we only generate, no gradients needed here.
    model.eval()
    print(f"Model loaded successfully. VRAM used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
except Exception as e:
    print(f"Fatal error while loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- DATASET PREPARATION ---
# We sample a small number of longer texts to create short training trajectories.
print("Preparing dataset (Wikitext) for caching...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Keep only examples with sufficient length and take a small subset for speed.
    sample_texts = [t for t in dataset['text'] if len(t.strip()) > 80][:10]
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# --- TRAJECTORY GENERATION ---
# These parameters define how we capture the intermediate state from the teacher.
# target_step=64 captures the state halfway through the diffusion process.
# At that point, the teacher has revealed ~50% of the tokens, so the logits are
# informative and differ from an untrained student. The student learns to predict
# this intermediate state in a single step.
TARGET_STEP = 64
# Total steps in the teacher's diffusion process.
STEPS = 128
# Number of tokens to generate for each sample.
GEN_LENGTH = 64
# Chunk size used by the generator for efficiency.
BLOCK_LENGTH = 64

# Profiler captures CPU/GPU timing so we can analyze performance later.
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# High-level run banner for logs.
print(f"\nStarting cache generation in: {CACHE_DIR}")
print(f"target_step={TARGET_STEP} (halfway through diffusion)")
print("-" * 50)

# Folder for profiler traces and summaries.
profiling_dir = os.path.join(WORKSPACE_DIR, "profiling")
os.makedirs(profiling_dir, exist_ok=True)

# Disable gradients to save memory and speed up generation.
with torch.no_grad():
    # Profile a short warmup/active window for GPU timing analysis.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=False
    ) as prof:
        for i, text in enumerate(sample_texts):
            # Build a short prompt from the dataset sample.
            prompt_content = f"Briefly summarize or continue this text:\n{text[:200]}"
            conversation = [{"role": "user", "content": prompt_content}]

            # Convert the chat-style prompt into model input ids on GPU.
            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_len = input_ids.shape[1]
            print(f"[*] Processing example {i+1}/10 (prompt_len={prompt_len})...")

            try:
                # Capture the intermediate state and logits at TARGET_STEP.
                with record_function(f"generate_trajectory_{i}"):
                    state_x, teacher_logits, attn_mask = generate_and_cache_trajectory(
                        model,
                        input_ids,
                        steps=STEPS,
                        gen_length=GEN_LENGTH,
                        block_length=BLOCK_LENGTH,
                        target_step=TARGET_STEP,
                    )

                # Save tensors to disk for the student training stage.
                file_path = os.path.join(CACHE_DIR, f"batch_{i}.pt")
                torch.save({
                    'input_x': state_x.cpu(),
                    'target_logits': teacher_logits.cpu(),
                    'attn_mask': attn_mask.cpu() if attn_mask is not None else None,
                    'prompt_len': prompt_len,
                    'target_step': TARGET_STEP,
                }, file_path)

                print(f"    [OK] Saved to {file_path}")

            except Exception as e:
                # Per-sample failure should not kill the whole run.
                print(f"    [ERROR] Failed on example {i+1}: {e}")
                import traceback
                traceback.print_exc()

            # Advance the profiler step to record timing for this iteration.
            prof.step()

# Export profiling results
# Chrome trace lets you inspect a detailed timeline of CPU/GPU activity.
prof.export_chrome_trace(os.path.join(profiling_dir, "01_cache_teacher_trace.json"))
with open(os.path.join(profiling_dir, "01_cache_teacher_summary.txt"), "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(f"Profiling results saved under '{profiling_dir}'")

# Final summary and cleanup.
print("-" * 50)
print(f"PROCESS COMPLETE. Generated {len(os.listdir(CACHE_DIR))} trajectory files.")

# Explicitly free GPU memory after finishing the caching stage.
del model
torch.cuda.empty_cache()