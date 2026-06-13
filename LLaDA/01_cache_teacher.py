import os
import sys
import torch
import psutil
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
from generate_cache import generate_and_cache_trajectory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.environ.get("LLADA_BASE_DIR", PROJECT_ROOT)

WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
HF_HOME = os.environ.get("HF_HOME") or os.path.join(WORKSPACE_DIR, ".cache")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

CACHE_DIR = os.path.join(WORKSPACE_DIR, "llada_trajectories")

import shutil
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory cleared and recreated: {CACHE_DIR}")

# 4-bit NF4 with nested quantization minimizes VRAM for the 8B teacher while
# preserving enough precision for logit distillation.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "GSAI-ML/LLaDA-8B-Instruct"

# Leave 3 GB headroom for OS and dataset paging on machines with tight RAM.
ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"Available RAM for loading: {ram_gb} GB")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading Teacher Model (LLaDA-8B) in 4-bit...")
try:
    # Cap GPU memory at 6GiB to leave room for activation caches; offload the rest to CPU RAM.
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded successfully. VRAM used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
except Exception as e:
    print(f"Fatal error while loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Preparing dataset (Wikitext) for caching...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # 10 examples and >80 chars is a pragmatic trade-off: enough to fit a single
    # training batch, short enough to keep caching fast.
    sample_texts = [t for t in dataset['text'] if len(t.strip()) > 80][:10]
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# target_step = steps // 2 approximates the midpoint where the teacher's masked
# distribution is most informative for the student.
TARGET_STEP = 64
STEPS = 128
GEN_LENGTH = 64
# block_length must divide gen_length, and steps must divide num_blocks
# (enforced later in generate_cache.py).
BLOCK_LENGTH = 64

from torch.profiler import profile, record_function, ProfilerActivity, schedule

print(f"\nStarting cache generation in: {CACHE_DIR}")
print(f"target_step={TARGET_STEP} (halfway through diffusion)")
print("-" * 50)

profiling_dir = os.path.join(WORKSPACE_DIR, "profiling")
os.makedirs(profiling_dir, exist_ok=True)

with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=False
    ) as prof:
        for i, text in enumerate(sample_texts):
            prompt_content = f"Briefly summarize or continue this text:\n{text[:200]}"
            conversation = [{"role": "user", "content": prompt_content}]

            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_len = input_ids.shape[1]
            print(f"[*] Processing example {i+1}/10 (prompt_len={prompt_len})...")

            try:
                with record_function(f"generate_trajectory_{i}"):
                    state_x, teacher_logits, attn_mask = generate_and_cache_trajectory(
                        model,
                        input_ids,
                        steps=STEPS,
                        gen_length=GEN_LENGTH,
                        block_length=BLOCK_LENGTH,
                        target_step=TARGET_STEP,
                    )

                file_path = os.path.join(CACHE_DIR, f"batch_{i}.pt")
                # Move tensors to CPU before saving to avoid device-side memory
                # pinning issues on Windows.
                torch.save({
                    'input_x': state_x.cpu(),
                    'target_logits': teacher_logits.cpu(),
                    'attn_mask': attn_mask.cpu() if attn_mask is not None else None,
                    'prompt_len': prompt_len,
                    'target_step': TARGET_STEP,
                }, file_path)

                print(f"    [OK] Saved to {file_path}")

            except Exception as e:
                print(f"    [ERROR] Failed on example {i+1}: {e}")
                import traceback
                traceback.print_exc()

            prof.step()

prof.export_chrome_trace(os.path.join(profiling_dir, "01_cache_teacher_trace.json"))
with open(os.path.join(profiling_dir, "01_cache_teacher_summary.txt"), "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(f"Profiling results saved under '{profiling_dir}'")

print("-" * 50)
print(f"PROCESS COMPLETE. Generated {len(os.listdir(CACHE_DIR))} trajectory files.")

del model
torch.cuda.empty_cache()
