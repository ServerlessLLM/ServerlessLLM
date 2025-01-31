import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sllm_store.transformers import load_model, save_model

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_name = "facebook/opt-1.3b"
model_folder = os.getenv("MODEL_FOLDER")
model_path = os.path.join(model_folder, model_name)
# =======================================================================================================================
torch.cuda.empty_cache()
# warm up the GPU
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

before_mem = torch.cuda.memory_allocated()
print(f"getting model from {model_folder}")
model = load_model(
    model_name,
    device_map="auto",
    storage_path=model_folder,
    fully_parallel=True,
    quantization="fp4",
)
after_mem = torch.cuda.memory_allocated()
print(f"Memory difference: {after_mem - before_mem}")
print(f"memory footprint: {model.get_memory_footprint()}")
print(f"memory allocated: {torch.cuda.memory_allocated()}")
print(f"max memory allocated: {torch.cuda.max_memory_allocated()}")
print(f"getting model from {model_folder}")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}, {param.device}")
# =======================================================================================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")
