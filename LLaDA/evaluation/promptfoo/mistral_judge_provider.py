"""
Mistral-7B-Instruct Judge Provider for Promptfoo

Local LLM-based evaluation using Mistral-7B-Instruct from Hugging Face.
No external services required - everything runs locally.

Model will be automatically downloaded (~15GB) on first run and cached.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


# Global model cache (loaded once per process)
_mistral_model = None
_mistral_tokenizer = None
_device = None


def _load_mistral():
    """Load Mistral model once and cache it."""
    global _mistral_model, _mistral_tokenizer, _device
    
    if _mistral_model is not None:
        return _mistral_model, _mistral_tokenizer, _device
    
    print("Loading Mistral-7B-Instruct from Hugging Face (first time only, ~15GB)...")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {_device}")
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    _mistral_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _mistral_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    _mistral_model.eval()
    
    print("✓ Mistral-7B-Instruct loaded successfully")
    return _mistral_model, _mistral_tokenizer, _device


def _evaluate_with_mistral(output_text: str, question: str) -> dict:
    """
    Use Mistral to evaluate if output meets a criterion (Yes/No).
    
    Returns:
        {
            "pass": bool,
            "reason": str
        }
    """
    model, tokenizer, device = _load_mistral()
    
    # Prompt format: make it very explicit for Yes/No
    prompt = f"""[INST] You are an evaluator. Answer ONLY "Yes" or "No" with a brief reason.

Output to evaluate:
<output>
{output_text}
</output>

Question: {question}

Answer ONLY with:
Yes. [reason]
or
No. [reason]
[/INST]"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,  # Low temperature for deterministic Yes/No
            top_p=0.9,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer from response
    answer_text = response.split("[/INST]")[-1].strip().lower()
    
    # Parse Yes/No
    is_yes = answer_text.startswith("yes")
    
    # Extract reason (everything after Yes/No)
    reason = ""
    match = re.search(r'(?:yes|no)\.\s*(.+?)(?:\n|$)', answer_text, re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
    
    return {
        "pass": is_yes,
        "reason": reason,
        "raw_response": answer_text
    }


def call_api(prompt, options, context):
    """
    Promptfoo provider that evaluates LLaDA outputs with Mistral.
    
    This is called by Promptfoo for each test case.
    """
    # Get LLaDA output from context (it's the output we're evaluating)
    llada_output = context.get("output", "")
    
    if not llada_output:
        return {
            "error": "No output to evaluate",
            "output": "[Error: output missing]",
            "pass": False
        }
    
    # For this provider, we return the LLaDA output
    # The actual evaluation happens in the assertions below
    return {
        "output": llada_output
    }


# Helper functions for inline assertions in promptfooconfig.yaml

def evaluate_yes_no(output_text: str, question: str) -> bool:
    """
    Simple boolean evaluation: returns True if Mistral says "Yes".
    Used in assertion inline Python code.
    """
    try:
        result = _evaluate_with_mistral(output_text, question)
        return result["pass"]
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False


def evaluate_with_reason(output_text: str, question: str) -> dict:
    """
    Evaluation with detailed reason.
    Used when you need explanation.
    """
    return _evaluate_with_mistral(output_text, question)
