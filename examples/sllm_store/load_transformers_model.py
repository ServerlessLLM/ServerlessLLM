import argparse
import time

import torch

from sllm_store.transformers import load_model

parser = argparse.ArgumentParser(description="Load a model from ServerlessLLM")
parser.add_argument(
    "--model_name", type=str, required=True, help="Model name stored"
)
parser.add_argument(
    "--storage_path",
    type=str,
    default="./models",
    help="Local path stored the model.",
)

args = parser.parse_args()

model_name = args.model_name
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
# Please note the loading time depends on the model size and the hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
