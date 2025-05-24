import argparse
import importlib
import os

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel

from sllm_store.transformers import save_lora

parser = argparse.ArgumentParser(
    description="Save a lora adapter from HuggingFace model hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub.",
)
parser.add_argument(
    "--adapter-name",
    type=str,
    required=True,
    help="Lora adapter name from HuggingFace model hub.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path to save the model.",
)

args = parser.parse_args()

base_model_name = args.model_name
lora_adapter_name = args.adapter_name
storage_path = args.storage_path

config = AutoConfig.from_pretrained(
    os.path.join(storage_path, "transformers", base_model_name),
    trust_remote_code=True,
)
config.torch_dtype = torch.float16
module = importlib.import_module("transformers")
hf_model_cls = getattr(module, AutoModelForCausalLM)
base_model = hf_model_cls.from_config(
    config,
    trust_remote_code=True,
).to(config.torch_dtype)

# Load a lora adapter from HuggingFace model hub.
model = PeftModel.from_pretrained(base_model, lora_adapter_name)

# Save the model to the local path.
model_path = os.path.join(storage_path, lora_adapter_name)
save_lora(model, model_path)
