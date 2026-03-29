# main.py
# Script para generar texto usando el checkpoint y la función generate_samples

import torch
from logic import generate, flow
from model import Transformer
from transformers import CLIPTokenizer
from omegaconf import OmegaConf

# Cargar configuración desde el archivo YAML
cfg = OmegaConf.load("config_text.yaml")

def main():
    # Inicializar el tokenizer (ajusta la ruta si es necesario)
    tokenizer = CLIPTokenizer.from_pretrained("/gpfs/projects/bsc70/bsc193242/Models/clip-vit-large-patch14")
    vocab_size = tokenizer.vocab_size
    tokenizer.add_tokens(["<MASK>"])
    masked = True

    # Inicializar la distribución de origen y el modelo
    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size
    )
    model = Transformer(
        config=cfg.model, vocab_size=vocab_size, masked=masked
    )
    # Cargar pesos del checkpoint
    checkpoint = torch.load("checkpoint.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Inicializar el path para flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )

    # Generar muestras
    with torch.no_grad():
        samples = generate.generate_samples(
            model=model,
            step=0,
            sample_dir=None,  # Cambia esto si quieres guardar los samples
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            rank=0,
            device=torch.device("cpu"),
            path=path,
            source_distribution=source_distribution,
            sample_batch_size=cfg.eval.sample_batch_size,
            sequence_length=cfg.model.length,
            sampling_steps=cfg.flow.sampling_steps,
            time_epsilon=1e-3,
        )
        print("Muestras generadas:", samples)

if __name__ == "__main__":
    main()
