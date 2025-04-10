import argparse
import time

import torch

from transformers import AutoTokenizer, BitsAndBytesConfig

from sllm_store.transformers import load_model

parser = argparse.ArgumentParser(description="Load a model from ServerlessLLM")
parser.add_argument(
    "--model-name", type=str, required=True, help="Model name stored"
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path stored the model.",
)

parser.add_argument(
    "--precision",
    type=str,
    default="int8",
    help="Precision of quantized model. Supports int8, fp4, and nf4",
)

args = parser.parse_args()

model_name = args.model_name
storage_path = args.storage_path

# Define quantization configuration with BitsAndBytesConfig
if args.precision == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.precision == "fp4":
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
elif args.precision == "nf4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4"
    )
else:
    quantization_config = None

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
    quantization_config=quantization_config,
)
# Please note the loading time depends on model size and hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")


tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
