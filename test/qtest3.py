import time
import os
from sllm_store.transformers import save_model, load_model

# Load a model from HuggingFace model hub.
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
model_name = "facebook/opt-1.3b"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b', quantization_config=quantization_config, torch_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer('hello, my dog is cute', return_tensors='pt').to('cuda')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"memory footprint: {model.get_memory_footprint()}")
print(f"memory allocated: {torch.cuda.memory_allocated()}")
print(f"max memory allocated: {torch.cuda.max_memory_allocated()}")
