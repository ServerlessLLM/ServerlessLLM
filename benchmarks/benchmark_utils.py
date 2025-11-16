# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import gc
import os
import time

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sllm_store.transformers import load_model


def print_gpu_memory(prefix=""):
    """Print GPU memory usage via PyTorch."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[GPU Memory {prefix}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )
    else:
        print(f"[GPU Memory {prefix}] CUDA not available")


def _warmup_cuda():
    num_gpus = torch.cuda.device_count()
    print(f"Warming up {num_gpus} GPUs")
    for i in tqdm(range(num_gpus)):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()


def _warmup_inference():
    print("Warming up inference")
    model_name = "facebook/opt-6.7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    del outputs, tokenizer, inputs, model
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_inference(model: nn.Module, model_path: str):
    # Inference
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_to_end_time = end_time - start_time
    throughput = outputs.shape[1] / end_to_end_time

    del outputs, tokenizer, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return end_to_end_time, throughput, output_text


def measure(
    model_name: str, model_format: str, model_dir: str, loading_order: list
):
    results = []
    print(
        f"Measuring loading time for {model_format} model={model_name}, repeating {len(loading_order)} times"
    )
    print_gpu_memory("BEFORE benchmark")
    # loading_order = torch.randperm(num_replicas)
    for model_idx in loading_order:
        print(f"Loading {model_name}_{model_idx}")
        print_gpu_memory(f"before load #{model_idx}")
        model_record = {"model_name": f"{model_name}_{model_idx}"}

        # Model Loading
        if model_format == "sllm":
            model_path = os.path.join(model_dir, f"{model_name}_{model_idx}")
            start_time = time.time()
            model = load_model(
                f"{model_name}_{model_idx}",
                storage_path=model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            end_time = time.time()
        elif model_format == "safetensors":
            model_path = os.path.join(
                model_dir, f"{model_name}_safetensors_{model_idx}"
            )
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            end_time = time.time()
        elif model_format == "torch":
            model_path = os.path.join(
                model_dir, f"{model_name}_torch_{model_idx}"
            )
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                use_safetensors=False,
            )
            end_time = time.time()
        model_record["loading_time"] = end_time - start_time

        # Inference
        end_to_end_time, throughput, output_text = benchmark_inference(
            model, model_path
        )

        model_record["end_to_end_time"] = end_to_end_time
        model_record["throughput"] = throughput
        model_record["output_text"] = output_text

        results.append(model_record)

        print_gpu_memory(f"after inference #{model_idx}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print_gpu_memory(f"after cleanup #{model_idx}")

    print_gpu_memory("AFTER all benchmarks")
    return results
