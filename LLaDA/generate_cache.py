import torch
import numpy as np
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_and_cache_trajectory(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, target_step=0, mask_id=126336):
    """
    Versión modificada que interrumpe la generación en el 'target_step' 
    y devuelve el estado de entrada (x) y los logits del Teacher.
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits

            # --- LA MODIFICACIÓN DE CACHÉ ---
            if i == target_step:
                # Devolvemos:
                # 1. 'x': El input con las máscaras actuales
                # 2. 'logits': El target (Soft-labels) del Teacher. Lo pasamos a float16 para ahorrar disco.
                return x.clone(), logits.clone().to(torch.float16), attention_mask.clone() if attention_mask is not None else None
            # --------------------------------

            # Lógica normal de desenmascarado de LLaDA si target_step > i
            logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, None, None