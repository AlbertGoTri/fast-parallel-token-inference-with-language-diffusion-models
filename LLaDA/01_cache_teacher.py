import os
import sys
import torch
import psutil
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
from generate_cache import generate_and_cache_trajectory

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["HF_HOME"] = r"C:\Users\Gotri\.cache"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

# Carpeta donde se guardarán los datos
CACHE_DIR = r"C:\Users\Gotri\.cache\trayectorias_llada"

# Limpiar cache anterior para regenerar con target_step correcto
import shutil
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache dir limpiada y recreada: {CACHE_DIR}")

# --- CARGA DEL MODELO TEACHER (4-BIT) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "GSAI-ML/LLaDA-8B-Instruct"

ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3
print(f"RAM disponible para carga: {ram_gb} GB")

print("Cargando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Cargando Teacher Model (LLaDA-8B) en 4-bits...")
try:
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Modelo cargado correctamente. VRAM usada: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- PREPARACIÓN DEL DATASET ---
print("Preparando dataset de prueba (Wikitext)...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    textos_prueba = [t for t in dataset['text'] if len(t.strip()) > 80][:10]
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    sys.exit(1)

# --- GENERACIÓN DE TRAYECTORIAS ---
# target_step=64 captura el estado a mitad del proceso de difusión.
# En ese punto el Teacher ya ha revelado ~50% de los tokens, por lo que
# sus logits son informativos y distintos a los del Student sin entrenar.
# El Student aprenderá a predecir ese estado intermedio en un solo paso.
TARGET_STEP = 64
STEPS = 128
GEN_LENGTH = 64
BLOCK_LENGTH = 64

from torch.profiler import profile, record_function, ProfilerActivity, schedule

print(f"\nIniciando generación de caché en: {CACHE_DIR}")
print(f"target_step={TARGET_STEP} (mitad del proceso de difusion)")
print("-" * 50)

os.makedirs("profiling", exist_ok=True)

with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=False
    ) as prof:
        for i, texto in enumerate(textos_prueba):
            prompt_content = f"Resume o continúa este texto de forma breve:\n{texto[:200]}"
            conversation = [{"role": "user", "content": prompt_content}]

            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_len = input_ids.shape[1]
            print(f"[*] Procesando ejemplo {i+1}/10 (prompt_len={prompt_len})...")

            try:
                with record_function(f"generate_trajectory_{i}"):
                    estado_x, teacher_logits, attn_mask = generate_and_cache_trajectory(
                        model,
                        input_ids,
                        steps=STEPS,
                        gen_length=GEN_LENGTH,
                        block_length=BLOCK_LENGTH,
                        target_step=TARGET_STEP,
                    )

                file_path = os.path.join(CACHE_DIR, f"batch_{i}.pt")
                torch.save({
                    'input_x': estado_x.cpu(),
                    'target_logits': teacher_logits.cpu(),
                    'attn_mask': attn_mask.cpu() if attn_mask is not None else None,
                    'prompt_len': prompt_len,
                    'target_step': TARGET_STEP,
                }, file_path)

                print(f"    [OK] Guardado en {file_path}")

            except Exception as e:
                print(f"    [ERROR] Fallo en el ejemplo {i+1}: {e}")
                import traceback
                traceback.print_exc()
            
            prof.step()

# Exportar resultados del profiling
prof.export_chrome_trace("profiling/01_cache_teacher_trace.json")
with open("profiling/01_cache_teacher_summary.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print("Resultados de profiling guardados en la carpeta 'profiling/'")

print("-" * 50)
print(f"PROCESO FINALIZADO. Se han generado {len(os.listdir(CACHE_DIR))} archivos de trayectoria.")

del model
torch.cuda.empty_cache()