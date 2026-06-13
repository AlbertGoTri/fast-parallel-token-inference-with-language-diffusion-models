import os
import glob
import torch
import torch.nn.functional as F
import psutil
from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.environ.get("LLADA_BASE_DIR", PROJECT_ROOT)

WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
HF_HOME = os.environ.get("HF_HOME") or os.path.join(WORKSPACE_DIR, ".cache")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Reserve 10% of VRAM for CUDA overhead and temporary allocations during backward.
torch.cuda.set_per_process_memory_fraction(0.90)

CACHE_DIR = os.path.join(WORKSPACE_DIR, "llada_trajectories")
SAVE_DIR = os.path.join(WORKSPACE_DIR, "llada_student_lora")
# T=2 softens the teacher distribution so the student learns richer relative
# probabilities, not just argmax labels.
TEMPERATURE = 2.0

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model_id = "GSAI-ML/LLaDA-8B-Instruct"

ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"Available RAM for loading: {ram_gb} GB")

print("Loading Student Model (Base)...")
student_base = AutoModel.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Base model loaded. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

if hasattr(student_base, "gradient_checkpointing_enable"):
    try:
        student_base.gradient_checkpointing_enable()
    except Exception as e:
        print(f"WARNING: Gradient checkpointing not supported: {e}")

# Base parameters stay frozen; only LoRA deltas are updated, so the optimizer
# state is ~0.1% of full-model AdamW.
for param in student_base.parameters():
    param.requires_grad = False

print("Injecting LoRA adapters into the student...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
student_model = get_peft_model(student_base, lora_config)
student_model.print_trainable_parameters()

# PEFT initializes adapters in f32; downcasting saves ~50% optimizer state VRAM
# with negligible impact on LoRA convergence.
for name, param in student_model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float16)

student_model.train()
print(f"VRAM after LoRA: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

try:
    import bitsandbytes as bnb
    # AdamW8bit compresses optimizer states; on 8B models this often avoids OOM
    # during the backward pass.
    optimizer = bnb.optim.AdamW8bit(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=1e-4
    )
    print("Using AdamW 8-bit")
except Exception:
    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=1e-4
    )
    print("Using standard AdamW")

from torch.profiler import profile, record_function, ProfilerActivity, schedule
cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
print(f"Found {len(cache_files)} trajectories. Starting distillation...\n")

profiling_dir = os.path.join(WORKSPACE_DIR, "profiling")
os.makedirs(profiling_dir, exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
    record_shapes=False
) as prof:
    for epoch in range(1):
        for i, cache_file in enumerate(cache_files):
            torch.cuda.empty_cache()

            trajectory = torch.load(cache_file, weights_only=True)

            input_x = trajectory['input_x'].to("cuda")
            target_logits = trajectory['target_logits'].to("cuda").to(torch.float16)
            attn_mask = trajectory['attn_mask'].to("cuda") if trajectory['attn_mask'] is not None else None
            prompt_len = trajectory.get('prompt_len', 0)

            optimizer.zero_grad()

            with record_function(f"train_step_epoch_{epoch}_batch_{i}"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_output = student_model(input_x, attention_mask=attn_mask)
                    student_logits = student_output.logits

                    student_logits_gen = student_logits[:, prompt_len:, :]
                    target_logits_gen = target_logits[:, prompt_len:, :]

                    # Scale by T^2 so the gradients have comparable magnitude to a
                    # standard cross-entropy loss (Hinton et al.).
                    loss = F.kl_div(
                        F.log_softmax(student_logits_gen / TEMPERATURE, dim=-1),
                        F.softmax(target_logits_gen / TEMPERATURE, dim=-1),
                        reduction="batchmean"
                    ) * (TEMPERATURE ** 2)

                loss.backward()

                # Clip at 1.0 because 4-bit LoRA training is sensitive to gradient
                # spikes from outlier teacher logits.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                optimizer.step()

            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Epoch {epoch+1} | Batch {i+1}/{len(cache_files)} | KL Loss: {loss.item():.4f} | VRAM: {vram:.2f} GB")

            del input_x, target_logits, student_logits, loss
            if attn_mask is not None:
                del attn_mask
            torch.cuda.empty_cache()
            prof.step()

prof.export_chrome_trace(os.path.join(profiling_dir, "02_train_student_trace.json"))
with open(os.path.join(profiling_dir, "02_train_student_summary.txt"), "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(f"Profiling results saved under '{profiling_dir}'")

os.makedirs(SAVE_DIR, exist_ok=True)
student_model.save_pretrained(SAVE_DIR)
print(f"\nTraining complete! Student saved to {SAVE_DIR}")
