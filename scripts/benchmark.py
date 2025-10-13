from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."}, # TEST MESSAGE
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
)

new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))