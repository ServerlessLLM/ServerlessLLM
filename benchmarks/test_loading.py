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
import argparse
import json
import os

import torch
from benchmark_utils import (
    _warmup_cuda,
    _warmup_inference,
    measure,
    print_gpu_memory,
)


def get_args():
    parser = argparse.ArgumentParser(description="Load test")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to serve",
    )
    parser.add_argument(
        "--model-format",
        type=str,
        required=True,
        choices=["sllm", "safetensors", "torch"],
        help="Format to save the model in",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory to load models",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of replicas to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        required=True,
        choices=["random", "cached"],
        help="Name of the test.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print("=" * 80)
    print(f"Starting benchmark for {args.model_format} format")
    print_gpu_memory("START of script")
    print("=" * 80)

    _warmup_cuda()
    _warmup_inference()
    print_gpu_memory("after warmup")

    model_format = args.model_format
    model_name = args.model_name
    model_dir = args.model_dir
    num_replicas = args.num_replicas
    output_dir = args.output_dir
    benchmark_type = args.benchmark_type

    # Check if model_dir exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory {model_dir} does not exist")

    if benchmark_type == "random":
        loading_order = torch.randperm(num_replicas)
    elif benchmark_type == "cached":
        # For cached: do warmup load first, then measure num_replicas loads
        loading_order = [0] * num_replicas
        # Perform warmup load (not measured)
        print(f"Performing warmup load for cached benchmark...")
        _ = measure(model_name, model_format, model_dir, [0])
        print(f"Warmup complete. Now measuring {num_replicas} cached loads...")
    else:
        raise ValueError(f"Unknown benchmark type {benchmark_type}")

    results = measure(model_name, model_format, model_dir, loading_order)

    output_filename = (
        f"{model_name}_{model_format}_{num_replicas}_{benchmark_type}.json"
    )
    output_filename = output_filename.replace("/", "_")
    output_filename = os.path.join(output_dir, output_filename)

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")

    print("=" * 80)
    print_gpu_memory("END of script")
    print("=" * 80)


if __name__ == "__main__":
    main()
