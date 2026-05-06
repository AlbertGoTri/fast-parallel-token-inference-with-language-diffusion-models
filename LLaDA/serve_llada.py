import os
import sys
import torch
import psutil
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from generate import generate

# --- CONFIGURACIÓN DE ENTORNO ---
BASE_DIR = os.path.expanduser("~/groups/hpai-collaborators/albert-gomez-triunfante/tfg")
HF_HOME = os.path.join(BASE_DIR, ".cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

torch.cuda.set_per_process_memory_fraction(0.85)

USE_LORA = True
LORA_DIR = os.path.join(HF_HOME, "llada_student_lora")

# --- INICIALIZACIÓN DEL MODELO ---
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

# Fix for "The model weights are not tied" error
model.tie_weights()

if USE_LORA:
    print(f"Loading LoRA adapters from {LORA_DIR}...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    # Also need to tie weights after PEFT wrapping
    model.tie_weights()

model.eval()
print("Model loaded and ready to serve.")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    
    conversation = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
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
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000)
