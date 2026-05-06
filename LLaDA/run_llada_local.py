import os
import sys
import torch
import psutil
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = os.path.expanduser("~/groups/hpai-collaborators/albert-gomez-triunfante/tfg")
HF_HOME = os.path.join(BASE_DIR, ".cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

torch.cuda.set_per_process_memory_fraction(0.85)

# --- CONFIGURACIÓN ---
# Cambiar a False para usar el modelo base sin LoRA (útil para comparar)
USE_LORA = True
LORA_DIR = os.path.join(HF_HOME, "llada_student_lora")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "GSAI-ML/LLaDA-8B-Instruct"
ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"RAM disponible: {ram_gb} GB")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Tokenizer loaded.")

print("Loading base model...")
try:
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Base model loaded. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
except Exception as e:
    print(f"FATAL ERROR during model load: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- CARGA DE ADAPTADORES LORA ---
if USE_LORA:
    if not os.path.exists(LORA_DIR):
        print(f"ERROR: LoRA weights not found at {LORA_DIR}")
        print("Run 02_train_student.py first to train the student.")
        sys.exit(1)
    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    print(f"LoRA loaded. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    model_label = "Student (base + LoRA)"
else:
    model_label = "Teacher (base, no LoRA)"

model.eval()
print(f"Running as: {model_label}")

# --- PROMPT ---
prompt = "Act as a football coach. My team is losing 2-0 at halftime. What should I say to my players to motivate them for the second half?"
conversation = [{"role": "user", "content": prompt}]

input_ids = tokenizer.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

from torch.profiler import profile, record_function, ProfilerActivity, schedule
print(f"\nPrompt: {prompt}")
print("Running generation...")

from generate import generate

os.makedirs("profiling", exist_ok=True)

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("generate_inference"):
            output = generate(
                model,
                input_ids,
                steps=128,
                gen_length=128,
                block_length=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n--- Output ({model_label}) ---\n{response}")

# Exportar resultados del profiling
prof.export_chrome_trace("profiling/run_llada_local_trace.json")
with open("profiling/run_llada_local_summary.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print("\nResultados de profiling guardados en la carpeta 'profiling/'")