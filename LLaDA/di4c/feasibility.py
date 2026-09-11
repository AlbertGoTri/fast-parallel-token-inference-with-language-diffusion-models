"""Di4C Phase 0 feasibility probe (GPU).

Runs ONE distillation-loss training step with the lambda-conditioned student and reports peak
VRAM + wall time, to decide whether faithful Di4C is practical on this hardware before building
the full loss set and training loop.

Memory trick: teacher and student SHARE the frozen 4-bit base. The teacher forward runs with
LoRA adapters disabled and lambda off; the student forward runs with adapters + lambda on. So
only one 8B model is resident (not two).

Run:
    python LLaDA/di4c/feasibility.py --config LLaDA/smoke_config.yaml
"""

import os
import sys
import time
import argparse

import torch

# Make `import LLaDA...` work when run as a script from the repo root or elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LLaDA.nested_distillation_utils import load_yaml_config
from LLaDA.di4c.model import wrap_lambda_conditioned, set_lambda
from LLaDA.di4c.losses import distillation_loss

MASK_ID = 126336


def main():
    parser = argparse.ArgumentParser(description="Di4C Phase 0 feasibility probe")
    parser.add_argument("--config", default="LLaDA/smoke_config.yaml")
    parser.add_argument("--batch", type=int, default=1, help="sequences per step")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig
    import psutil

    config = load_yaml_config(args.config)
    os.environ["HF_HOME"] = os.path.expanduser(config["system"]["hf_home"])
    torch.cuda.set_per_process_memory_fraction(config["system"]["cuda_memory_fraction"])

    q = config["system"]["quantization"]
    quant = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["compute_dtype"]),
        bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["use_double_quant"],
    )
    model_id = config["teacher"]["model_path"]
    ram_gb = int(psutil.virtual_memory().available / 1024**3) - 3

    print(f"[di4c-feasibility] Loading {model_id} (4-bit)...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        model_id, quantization_config=quant, device_map="auto",
        max_memory={0: "6GiB", "cpu": f"{ram_gb}GiB"},
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.tie_weights()
    for p in base.parameters():
        p.requires_grad = False

    lora = config["student"]["lora"]
    model = get_peft_model(base, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"], bias="none", task_type="FEATURE_EXTRACTION",
    ))
    model, state = wrap_lambda_conditioned(model)
    model.train()

    # Trainable = LoRA + lambda-conditioner.
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[di4c-feasibility] Trainable params: {n_train/1e6:.2f}M")

    # Build one partially-masked batch (prompt + gen_length with masked generated positions).
    gen_len = config["student"]["gen_length"]
    prompt_ids = tok("Briefly continue this text: The", return_tensors="pt").input_ids
    plen = prompt_ids.shape[1]
    x = torch.full((args.batch, plen + gen_len), MASK_ID, dtype=torch.long)
    x[:, :plen] = prompt_ids
    x = x.to("cuda")
    score_mask = (x == MASK_ID)

    opt = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    lam = torch.rand(args.batch, device="cuda")
    # Teacher: shared base, adapters + lambda OFF.
    with torch.no_grad(), model.disable_adapter():
        set_lambda(state, None)
        teacher_logits = model(x).logits.float()
    # Student: adapters + lambda ON (this pass carries gradients).
    set_lambda(state, lam)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits = model(x).logits
        loss = distillation_loss(student_logits.float(), teacher_logits, mask=score_mask)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print("\n[di4c-feasibility] === RESULT ===")
    print(f"  loss:            {loss.item():.4f}")
    print(f"  step wall time:  {dt:.1f}s")
    print(f"  peak VRAM alloc: {peak:.2f} GB")
    print("  -> if this fits and is reasonably fast, Di4C Phase 1 is viable on this hardware.")


if __name__ == "__main__":
    main()
