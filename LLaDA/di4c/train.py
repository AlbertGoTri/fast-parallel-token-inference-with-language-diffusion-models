"""Di4C training loop (GPU).

One optimization step combines:
  * distillation loss  -- student single-step matches the teacher single-step at x_t;
  * consistency loss   -- the direct student's mixture (over N sampled lambdas) is trained to
                          reproduce a target x_s sampled from the COMPOSED path
                          (teacher one step x_t->x_u, then student x_u->x_s), stop-gradiented.

Teacher and student share the frozen 4-bit base (teacher = adapters+lambda off via
`disable_adapter`; student = LoRA + lambda). Only the LoRA deltas and the lambda-conditioner
are trained.

This is a faithful-in-structure first implementation; the exact estimator, the x_t noise-level
schedule, and the loss weights are tuning knobs that need GPU iteration. Run:

    python LLaDA/di4c/train.py --config LLaDA/smoke_config.yaml --checkpoint <out_dir>
"""

import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LLaDA.nested_distillation_utils import load_yaml_config, ensure_dir
from LLaDA.generate_cache import _select_transfer
from LLaDA.di4c.model import wrap_lambda_conditioned, set_lambda
from LLaDA.di4c.losses import distillation_loss, consistency_loss

MASK_ID = 126336


def _load_model(config):
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig
    import psutil

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
    return tok, model, state


def di4c_train_step(model, state, x, block_end, n_lambda, consistency_weight):
    """Compute the combined Di4C loss for one fully-masked batch ``x`` (prompt + masked gen)."""
    mask_t = (x == MASK_ID)                              # positions to predict
    device = x.device
    lam = torch.rand(n_lambda, x.shape[0], device=device)

    # --- teacher single step at x_t (shared base, adapters+lambda off) ---
    with torch.no_grad(), model.disable_adapter():
        set_lambda(state, None)
        teacher_logits = model(x).logits.float()

    # --- composed target x_s: teacher one step (x_t->x_u), then student finishes (x_u->x_s) ---
    with torch.no_grad():
        # teacher unmasks ~half the generated tokens
        half = torch.clamp(mask_t.sum(dim=1) // 2, min=1)
        transfer_u, x0_u = _select_transfer(x, teacher_logits, half, block_end, MASK_ID)
        x_u = x.clone()
        x_u[transfer_u] = x0_u[transfer_u]
        # student (one lambda) finishes the rest, greedily
        set_lambda(state, lam[0])
        s_logits = model(x_u).logits
        x_s = x_u.clone()
        still = (x_u == MASK_ID)
        x_s[still] = s_logits.argmax(dim=-1)[still]
        x_s = x_s.detach()

    # --- direct student mixture at x_t over N lambdas (gradients here) ---
    direct = []
    for i in range(n_lambda):
        set_lambda(state, lam[i])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            direct.append(model(x).logits.float())
    direct = torch.stack(direct)                        # [N, B, L, V]

    loss_distil = distillation_loss(direct[0], teacher_logits, mask=mask_t)
    loss_consis = consistency_loss(direct, x_s, mask=mask_t)
    loss = loss_distil + consistency_weight * loss_consis
    return loss, loss_distil.detach(), loss_consis.detach()


def main():
    parser = argparse.ArgumentParser(description="Di4C training loop")
    parser.add_argument("--config", default="LLaDA/smoke_config.yaml")
    parser.add_argument("--checkpoint", default="workspace/di4c_checkpoint")
    parser.add_argument("--n-lambda", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    from datasets import load_dataset

    config = load_yaml_config(args.config)
    tok, model, state = _load_model(config)
    model.train()

    sc = config["student"]
    gen_len, block_len = sc["gen_length"], sc["block_length"]
    dataset = load_dataset(sc["dataset_name"], sc["dataset_config"], split=sc["dataset_split"])
    texts = [t for t in dataset["text"] if len(t.strip()) > sc["min_text_length"]][:sc["num_train_examples"]]
    print(f"[di4c-train] {len(texts)} examples | n_lambda={args.n_lambda}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    except Exception:
        opt = torch.optim.AdamW(trainable, lr=args.lr)
    max_grad_norm = config["student"]["max_grad_norm"]

    for i, text in enumerate(texts):
        conv = [{"role": "user", "content": f"Briefly summarize or continue this text:\n{text[:200]}"}]
        ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to("cuda")
        plen = ids.shape[1]
        x = torch.full((1, plen + gen_len), MASK_ID, dtype=torch.long, device="cuda")
        x[:, :plen] = ids
        block_end = plen + block_len

        opt.zero_grad()
        loss, ld, lc = di4c_train_step(model, state, x, block_end, args.n_lambda, args.consistency_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
        opt.step()
        vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[di4c-train] {i+1}/{len(texts)} | loss {loss.item():.4f} "
              f"(distil {ld.item():.4f}, consis {lc.item():.4f}) | peakVRAM {vram:.2f}GB")

    ensure_dir(args.checkpoint)
    model.save_pretrained(args.checkpoint)
    torch.save(model._lambda_conditioner.state_dict(), os.path.join(args.checkpoint, "lambda_conditioner.pt"))
    print(f"[di4c-train] saved LoRA + lambda-conditioner to {args.checkpoint}")


if __name__ == "__main__":
    main()
