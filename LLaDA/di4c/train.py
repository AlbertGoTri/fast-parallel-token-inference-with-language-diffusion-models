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
import math
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LLaDA.nested_distillation_utils import load_yaml_config, ensure_dir
from LLaDA.generate_cache import _select_transfer
from LLaDA.di4c.model import wrap_lambda_conditioned, set_lambda
from LLaDA.di4c.losses import distillation_loss, mixture_log_prob

MASK_ID = 126336


def _load_model(config, mem_fraction):
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig
    import psutil

    os.environ["HF_HOME"] = os.path.expanduser(config["system"]["hf_home"])
    # Di4C training runs standalone (no concurrent Ollama/eval), so use more of the GPU than the
    # eval-time default (config's cuda_memory_fraction, ~0.85). Overridable via --mem-fraction.
    torch.cuda.set_per_process_memory_fraction(mem_fraction)
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


def _seq_logprob(logits, tokens, mask):
    """Sequence log-prob of `tokens` under one factorized student (over masked positions), [B]."""
    return mixture_log_prob(logits.unsqueeze(0), tokens, mask)   # N=1 -> plain product


def di4c_train_step(model, state, x, block_end, n_lambda, consistency_weight):
    """One Di4C step with a MEMORY-EFFICIENT mixture gradient.

    Only one student forward graph is alive at a time, so peak VRAM ~= a single forward
    (~6 GB, which the Phase 0 gate proved fits) regardless of ``n_lambda``. Backward is done
    inside this function (gradients accumulate across lambdas); the caller runs the optimizer.

    Uses d/dtheta[-logsumexp_i seq_logp_i] = -sum_i softmax_i(seq_logp) * d seq_logp_i, so we
    compute the softmax weights once with no grad, then accumulate each lambda's weighted term.
    """
    mask_t = (x == MASK_ID)
    B = x.shape[0]
    lam = torch.rand(n_lambda, B, device=x.device)

    # teacher single-step (no grad; shared base with adapters+lambda off)
    with torch.no_grad(), model.disable_adapter():
        set_lambda(state, None)
        teacher_logits = model(x).logits

    # composed target x_s (no grad): teacher one step x_t->x_u, then student finishes x_u->x_s
    with torch.no_grad():
        half = torch.clamp(mask_t.sum(dim=1) // 2, min=1)
        transfer_u, x0_u = _select_transfer(x, teacher_logits.float(), half, block_end, MASK_ID)
        x_u = x.clone()
        x_u[transfer_u] = x0_u[transfer_u]
        set_lambda(state, lam[0])
        x_s = x_u.clone()
        still = (x_u == MASK_ID)
        x_s[still] = model(x_u).logits.argmax(dim=-1)[still]
        del x_u

    # mixture weights for the consistency gradient (no grad -> no graphs retained)
    with torch.no_grad():
        seq_logp = []
        for i in range(n_lambda):
            set_lambda(state, lam[i])
            seq_logp.append(_seq_logprob(model(x).logits, x_s, mask_t))
        seq_logp = torch.stack(seq_logp)                              # [N, B]
        weights = torch.softmax(seq_logp, dim=0)                      # [N, B]
        consis_val = (-(torch.logsumexp(seq_logp, dim=0) - math.log(n_lambda))).mean().item()

    # Reclaim the caching allocator's leftovers from the no-grad weight forwards before the
    # gradient loop, to reduce fragmentation on the tight 8GB budget.
    torch.cuda.empty_cache()

    # accumulate gradients one lambda at a time (single graph alive); distillation on i=0
    distil_val = 0.0
    for i in range(n_lambda):
        set_lambda(state, lam[i])
        logits = model(x).logits
        slp = _seq_logprob(logits, x_s, mask_t)                       # [B], carries grad
        loss_i = consistency_weight * (-(weights[i] * slp).sum() / B)
        if i == 0:
            d = distillation_loss(logits.float(), teacher_logits.float(), mask=mask_t)
            loss_i = loss_i + d
            distil_val = d.item()
        loss_i.backward()
        del logits, slp
    return distil_val, consis_val


def run_di4c_training(config, checkpoint, n_lambda=2, consistency_weight=1.0, lr=1e-4,
                      mem_fraction=0.92, logger=None):
    """Train a Di4C student and save LoRA + lambda-conditioner to ``checkpoint``.

    Callable from the pipeline (Di4CStrategy.train_round) or the CLI (main). Returns True on
    success. Frees the GPU on exit so the eval stage that follows can start its own server.
    """
    import gc
    from datasets import load_dataset

    def _log(msg):
        (logger.log if logger is not None else print)(msg)

    tok, model, state = _load_model(config, mem_fraction)
    model.train()

    sc = config["student"]
    gen_len, block_len = sc["gen_length"], sc["block_length"]
    dataset = load_dataset(sc["dataset_name"], sc["dataset_config"], split=sc["dataset_split"])
    texts = [t for t in dataset["text"] if len(t.strip()) > sc["min_text_length"]][:sc["num_train_examples"]]
    _log(f"[di4c-train] {len(texts)} examples | n_lambda={n_lambda}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=lr)
    except Exception:
        opt = torch.optim.AdamW(trainable, lr=lr)
    max_grad_norm = config["student"]["max_grad_norm"]

    for i, text in enumerate(texts):
        conv = [{"role": "user", "content": f"Briefly summarize or continue this text:\n{text[:200]}"}]
        ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to("cuda")
        plen = ids.shape[1]
        x = torch.full((1, plen + gen_len), MASK_ID, dtype=torch.long, device="cuda")
        x[:, :plen] = ids
        block_end = plen + block_len

        opt.zero_grad()
        ld, lc = di4c_train_step(model, state, x, block_end, n_lambda, consistency_weight)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
        opt.step()
        vram = torch.cuda.max_memory_allocated() / 1024**3
        _log(f"[di4c-train] {i+1}/{len(texts)} | distil {ld:.4f} | consis {lc:.4f} | peakVRAM {vram:.2f}GB")

    ensure_dir(checkpoint)
    model.save_pretrained(checkpoint)
    torch.save(model._lambda_conditioner.state_dict(), os.path.join(checkpoint, "lambda_conditioner.pt"))
    _log(f"[di4c-train] saved LoRA + lambda-conditioner to {checkpoint}")

    del model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser(description="Di4C training loop")
    parser.add_argument("--config", default="LLaDA/smoke_config.yaml")
    parser.add_argument("--checkpoint", default="workspace/di4c_checkpoint")
    parser.add_argument("--n-lambda", type=int, default=2)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mem-fraction", type=float, default=0.92,
                        help="GPU memory fraction (training runs without Ollama, so > eval's 0.85)")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_di4c_training(config, args.checkpoint, n_lambda=args.n_lambda,
                      consistency_weight=args.consistency_weight, lr=args.lr,
                      mem_fraction=args.mem_fraction)


if __name__ == "__main__":
    main()
