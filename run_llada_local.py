import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# 1. Configure 8-bit Quantization to fit the 2080 Ti's 11GB VRAM
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True # Acts as a safety net if VRAM spikes
)

print("Loading LLaDA-8B tokenizer...")
model_id = "GSAI-ML/LLaDA-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading LLaDA-8B model in 8-bit precision (This will take a moment)...")
model = AutoModel.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    device_map="auto", # Automatically maps to your RTX 2080 Ti
    trust_remote_code=True
)

# 2. Setup your prompt
prompt = "Explain the concept of Progressive Distillation in AI."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating response...")
# Note: LLaDA generates via a parallel diffusion masking process. 
# The exact generation wrapper might depend on their specific repo architecture.
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        # Diffusion specific hyperparameters (like steps) might be added here 
        # based on LLaDA's specific generation API.
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Output ---\n", response)