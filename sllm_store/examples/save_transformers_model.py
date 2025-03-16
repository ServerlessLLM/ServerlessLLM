import argparse
import os

import torch
from transformers import AutoModelForCausalLM

from sllm_store.transformers import save_model

parser = argparse.ArgumentParser(
    description="Save a model from HuggingFace model hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path to save the model.",
)

args = parser.parse_args()

model_name = args.model_name
storage_path = args.storage_path

# Load a model from HuggingFace model hub.
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16
)

# Save the model to the local path.
model_path = os.path.join(storage_path, model_name)
save_model(model, model_path)
