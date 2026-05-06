import os
import glob
import torch
import torch.nn.functional as F
import psutil
from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

os.environ["HF_HOME"] = r"C:\Users\Gotri\.cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.cuda.set_per_process_memory_fraction(0.90)

CACHE_DIR = r"C:\Users\Gotri\.cache\trayectorias_llada"
SAVE_DIR = r"C:\Users\Gotri\.cache\llada_student_lora"
TEMPERATURE = 2.0

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model_id = "GSAI-ML/LLaDA-8B-Instruct"

ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"RAM disponible para carga: {ram_gb} GB")

print("Cargando Student Model (Base)...")
student_base = AutoModel.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Modelo base cargado. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# Activar gradient checkpointing si el modelo lo soporta
if hasattr(student_base, "gradient_checkpointing_enable"):
    try:
        student_base.gradient_checkpointing_enable()
    except Exception as e:
        print(f"WARNING: Gradient checkpointing no soportado: {e}")

# Asegurarse de que solo los parametros LoRA (que se añadiran) requieren gradiente
# Los parametros 4-bit del base model no necesitan grad
for param in student_base.parameters():
    param.requires_grad = False

print("Inyectando adaptadores LoRA al Student...")
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

# Los adaptadores LoRA se inicializan en float32 por defecto
# Los casteamos a float16 para ahorrar VRAM
for name, param in student_model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float16)

student_model.train()
print(f"VRAM tras LoRA: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# AdamW 8-bit para reducir memoria de estados del optimizador
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=1e-4
    )
    print("Usando AdamW 8-bit")
except Exception:
    optimizer = torch.optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=1e-4
    )
    print("Usando AdamW estandar")

from torch.profiler import profile, record_function, ProfilerActivity, schedule
archivos_cache = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
print(f"Encontradas {len(archivos_cache)} trayectorias. Iniciando destilacion...\n")

os.makedirs("profiling", exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
    record_shapes=False
) as prof:
    for epoch in range(1):
        for i, archivo in enumerate(archivos_cache):
            torch.cuda.empty_cache()

            trayectoria = torch.load(archivo, weights_only=True)

            input_x = trayectoria['input_x'].to("cuda")
            target_logits = trayectoria['target_logits'].to("cuda").to(torch.float16)
            attn_mask = trayectoria['attn_mask'].to("cuda") if trayectoria['attn_mask'] is not None else None
            prompt_len = trayectoria.get('prompt_len', 0)

            optimizer.zero_grad()

            with record_function(f"train_step_epoch_{epoch}_batch_{i}"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_salida = student_model(input_x, attention_mask=attn_mask)
                    student_logits = student_salida.logits

                    student_logits_gen = student_logits[:, prompt_len:, :]
                    target_logits_gen = target_logits[:, prompt_len:, :]

                    loss = F.kl_div(
                        F.log_softmax(student_logits_gen / TEMPERATURE, dim=-1),
                        F.softmax(target_logits_gen / TEMPERATURE, dim=-1),
                        reduction="batchmean"
                    ) * (TEMPERATURE ** 2)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                optimizer.step()

            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Epoca {epoch+1} | Lote {i+1}/{len(archivos_cache)} | Loss KL: {loss.item():.4f} | VRAM: {vram:.2f} GB")

            del input_x, target_logits, student_logits, loss
            if attn_mask is not None:
                del attn_mask
            torch.cuda.empty_cache()
            prof.step()

# Exportar resultados del profiling
prof.export_chrome_trace("profiling/02_train_student_trace.json")
with open("profiling/02_train_student_summary.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print("Resultados de profiling guardados en la carpeta 'profiling/'")

os.makedirs(SAVE_DIR, exist_ok=True)
student_model.save_pretrained(SAVE_DIR)
print(f"\nEntrenamiento completado! Student guardado en {SAVE_DIR}")