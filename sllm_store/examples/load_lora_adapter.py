import argparse
import time

import torch

from transformers import AutoTokenizer

from sllm_store.transformers import load_model, load_lora

parser = argparse.ArgumentParser(description="Load a model from ServerlessLLM")
parser.add_argument(
    "--model-name", type=str, required=True, help="Model name stored"
)
parser.add_argument(
    "--adapter-name", type=str, required=True, help="Lora adapter name"
)
parser.add_argument(
    "--adapter-path", type=str, required=True, help="Lora adapter path stored"
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path stored the model.",
)

args = parser.parse_args()

model_name = args.model_name
lora_adapter_name = args.adapter_name
lora_adapter_path = args.adapter_path
storage_path = args.storage_path


# warm up the GPU
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

start = time.time()
model = load_model(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    storage_path=storage_path,
    fully_parallel=True,
)

model = load_lora(
    model,
    lora_adapter_name,
    lora_adapter_path,
    device_map="auto",
    storage_path=storage_path,
    torch_dtype=torch.float16,
)
# Please note the loading time depends on model size and hardware bandwidth.
print(f"Model with lora adapter loading time: {time.time() - start:.2f}s")

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
generate_kwargs = {"adapter_names": [lora_adapter_name]}
outputs = model.generate(**inputs, **generate_kwargs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
