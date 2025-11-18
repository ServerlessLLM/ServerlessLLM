import argparse
import os

import torch
from transformers import AutoModelForVision2Seq

from sllm_store.transformers import save_model


parser = argparse.ArgumentParser(
    description="Save a VLM model from the HuggingFace Hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    default="Qwen/Qwen2-VL-2B-Instruct",
    help="Model name from the HuggingFace model hub.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path to save the converted model.",
)

args = parser.parse_args()

model_name = args.model_name
storage_path = args.storage_path

# Load a model from Hugging Face.
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Save the model to local storage using ServerlessLLM.
model_path = os.path.join(storage_path, model_name)
save_model(model, model_path)
