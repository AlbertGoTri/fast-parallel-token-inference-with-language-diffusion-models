from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "Translate the following English sentence to Spanish: 'I would have gone to the party, but it was raining.'"

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)

print(tok.decode(out[0], skip_special_tokens=True))
