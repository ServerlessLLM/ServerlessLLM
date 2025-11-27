#!/usr/bin/env python3
"""
MoE-CAP Evaluation Script for ServerlessLLM

This script evaluates MoE models using the MoE-CAP methodology by:
1. Loading benchmark datasets using MoE-CAP data loaders
2. Starting batch recording and expert distribution recording on ServerlessLLM
3. Sending inference requests to ServerlessLLM API
4. Collecting batch statistics and expert distribution data
5. Computing MoE-CAP metrics

Usage:
    python evaluate_moecap.py --model Qwen/Qwen3-30B-A3B --dataset gsm8k \
        --server-url http://127.0.0.1:8500 --output-dir ./results
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from moe_cap.configs import CAPConfig
from moe_cap.data_loader import GSM8KLoader, LongBenchV2Loader, NuminaMathLoader
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.acc_metrics import (
    compute_accuracy_metrics,
    format_accuracy_summary,
)
from moe_cap.utils.continuous_batching_utils import (
    _calculate_continuous_metrics,
)
from moe_cap.utils.cost_utils import calculate_cost
from tqdm.asyncio import tqdm_asyncio

# Import recording client APIs
from sllm.client.recording import (
    clear_batch_recording,
    clear_expert_distribution,
    dump_batch_recording,
    dump_expert_distribution,
    get_batch_recording_status,
    get_expert_distribution_status,
    start_batch_recording,
    start_expert_distribution_recording,
    stop_batch_recording,
    stop_expert_distribution_recording,
)


def load_dataset(
    dataset_name: str, model_id: str, num_samples: int = None
) -> Tuple[List[str], List[str]]:
    """Load benchmark dataset using MoE-CAP data loaders.

    Args:
        dataset_name: Name of dataset (gsm8k, numinamath, longbench_v2, etc.)
        model_id: Model identifier for config
        num_samples: Number of samples to load (None = all)

    Returns:
        Tuple of (prompts, targets) lists
    """
    # Create config for the dataset
    config = CAPConfig(
        dataset_names=[dataset_name],
        metrics=["em"],  # Can be extended based on needs
        model_id=model_id,
        dataset_split="test",
    )

    # Load the appropriate dataset
    if dataset_name.lower() == "gsm8k":
        loader = GSM8KLoader(config)
    elif dataset_name.lower() in ["numinamath", "numina"]:
        loader = NuminaMathLoader(config)
    elif dataset_name.lower() == "longbench_v2":
        loader = LongBenchV2Loader(config)
    else:
        print(f"Error: Unknown dataset {dataset_name}")
        print("Supported datasets: gsm8k, numinamath, longbench_v2")
        sys.exit(1)

    # Get inputs and targets
    prompts = loader.get_input()
    targets = loader.get_target()

    # Limit to num_samples if specified
    if num_samples is not None:
        prompts = prompts[:num_samples]
        targets = targets[:num_samples]

    return prompts, targets


def send_inference_request(
    server_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    """Send an inference request to ServerlessLLM.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model to use
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: System prompt for the assistant

    Returns:
        Response dictionary with generated text
    """
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Inference request failed: {e}")
        return {}


async def send_inference_request_async(
    session: aiohttp.ClientSession,
    server_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful assistant.",
) -> Dict[str, Any]:
    """Send an inference request asynchronously."""
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        print(f"✗ Inference request failed: {e}")
        return {}


async def run_inference_batch(
    server_url: str,
    model_name: str,
    prompts: List[str],
    targets: List[str],
    max_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Run a batch of inference requests concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            task = send_inference_request_async(
                session, server_url, model_name, prompt, max_tokens, temperature
            )
            tasks.append(task)

        # Use tqdm for progress tracking
        responses = await tqdm_asyncio.gather(*tasks, desc="Inference requests")

        outputs = []
        for prompt, target, response in zip(prompts, targets, responses):
            outputs.append(
                {
                    "prompt": prompt,
                    "expected": target,
                    "response": response,
                }
            )
        return outputs


def extract_generated_text(response: Dict[str, Any]) -> str:
    """Extract generated text from ServerlessLLM response.

    Args:
        response: Response dictionary from inference request

    Returns:
        Generated text string
    """
    try:
        # Standard OpenAI chat completion format
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "text" in choice:
                return choice["text"]
        return ""
    except Exception as e:
        print(f"Warning: Failed to extract text from response: {e}")
        return ""


def get_model_simple_name(model_name: str) -> str:
    """Get simplified model name for output directory."""
    norm_path = os.path.normpath(model_name)
    parts = norm_path.split(os.sep)
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return model_name


def save_results(
    output_dir: str,
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    detailed_results: List[Dict[str, Any]] = None,
    backend: str = "sllm",
):
    """Save evaluation results to files matching openai_api_profile format.

    Saves:
    - cap_metrics_{backend}_{dataset}.json: All metrics in flat format
    - detailed_results_{dataset}.jsonl: Per-request detailed results (optional)
    """
    from datetime import datetime

    # Create model-specific directory
    model_dir = os.path.join(output_dir, get_model_simple_name(model_name))
    os.makedirs(model_dir, exist_ok=True)

    # Save metrics with timestamp (matching openai_api_profile format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(
        model_dir, f"cap_metrics_{backend}_{dataset_name}_{timestamp}.json"
    )

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {metrics_file}")

    # Save detailed results if provided
    if detailed_results:
        detailed_file = os.path.join(
            model_dir, f"detailed_results_{dataset_name}.jsonl"
        )
        with open(detailed_file, "w", encoding="utf-8") as f:
            for record in detailed_results:
                f.write(json.dumps(record) + "\n")
        print(f"✓ Detailed results saved to {detailed_file}")


def warmup_model(server_url: str, model_name: str) -> bool:
    """Send a warmup request to ensure model is loaded."""
    print("Warming up model...")
    try:
        send_inference_request(server_url, model_name, "Hello", max_tokens=10)
        print("✓ Model warmed up and ready")
        return True
    except Exception as e:
        print(f"✗ Warmup failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MoE models with MoE-CAP methodology"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Qwen/Qwen3-30B-A3B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name (gsm8k, numinamath, longbench_v2)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8500",
        help="ServerlessLLM server URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./moecap_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs used for inference (for MoE-CAP metrics)",
    )
    parser.add_argument(
        "--clear-before",
        action="store_true",
        help="Clear previous recordings before starting",
    )
    # Expert distribution recording options
    parser.add_argument(
        "--enable-expert-recording",
        action="store_true",
        help="Enable expert distribution recording",
    )
    parser.add_argument(
        "--expert-recording-mode",
        type=str,
        default="per_pass",
        choices=["per_token", "per_pass", "stat", "stat_approx"],
        help="Expert distribution recording mode (default: per_pass)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MoE-CAP Evaluation for ServerlessLLM")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Server: {args.server_url}")
    print(f"Output: {args.output_dir}")
    if args.enable_expert_recording:
        print(f"Expert Recording: ENABLED (mode={args.expert_recording_mode})")
    else:
        print("Expert Recording: DISABLED")
    print("=" * 80)

    # Load dataset using MoE-CAP data loaders
    print("\n[1/7] Loading dataset...")
    try:
        prompts, targets = load_dataset(
            args.dataset, args.model, args.num_samples
        )
        print(f"✓ Loaded {len(prompts)} samples from {args.dataset}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Clear previous recordings if requested
    if args.clear_before:
        print("\n[2/7] Clearing previous recordings...")
        clear_batch_recording(args.server_url, args.model)
        if args.enable_expert_recording:
            clear_expert_distribution(args.server_url, args.model)
    else:
        print("\n[2/7] Skipping clear (use --clear-before to clear)")

    # Warmup
    warmup_model(args.server_url, args.model)

    # Start batch recording
    print("\n[3/7] Starting batch recording...")
    if not start_batch_recording(args.server_url, args.model):
        print("Failed to start batch recording. Exiting.")
        return

    # Start expert distribution recording if enabled
    if args.enable_expert_recording:
        print("\n[3.5/7] Starting expert distribution recording...")
        if not start_expert_distribution_recording(
            args.server_url, args.model, args.expert_recording_mode
        ):
            print(
                "Warning: Failed to start expert distribution recording. Continuing without it."
            )

    # Check status
    status = get_batch_recording_status(args.server_url, args.model)
    print(f"Batch recording status: {status}")
    if args.enable_expert_recording:
        expert_status = get_expert_distribution_status(
            args.server_url, args.model
        )
        print(f"Expert distribution status: {expert_status}")

    # Send inference requests and measure total time
    print(f"\n[4/7] Sending {len(prompts)} inference requests...")
    start_time = time.perf_counter()
    outputs = asyncio.run(
        run_inference_batch(
            args.server_url,
            args.model,
            prompts,
            targets,
            args.max_tokens,
            args.temperature,
        )
    )
    total_time = time.perf_counter() - start_time
    print(f"✓ Completed {len(outputs)} inference requests in {total_time:.2f}s")

    # Stop batch recording
    print("\n[5/7] Stopping batch recording...")
    if not stop_batch_recording(args.server_url, args.model):
        print("Failed to stop batch recording.")

    # Stop expert distribution recording if enabled
    expert_distribution_data = {}
    if args.enable_expert_recording:
        print("\n[5.5/7] Stopping expert distribution recording...")
        if not stop_expert_distribution_recording(args.server_url, args.model):
            print("Warning: Failed to stop expert distribution recording.")

        # Dump expert distribution data
        print("Dumping expert distribution data...")
        expert_distribution_data = dump_expert_distribution(
            args.server_url, args.model
        )

    # Dump and analyze batch results
    print("\n[6/7] Dumping and analyzing results...")
    batch_data = dump_batch_recording(args.server_url, args.model)

    if not batch_data.get("records"):
        print("✗ No batch data collected. Check if backend is vllm_moecap.")
        return

    server_records = batch_data.get("records", [])

    # Calculate batch performance metrics using _calculate_continuous_metrics
    print("\n[7/7] Computing metrics...")

    # Load model info for metrics calculation
    config = CAPConfig(
        dataset_names=[args.dataset],
        metrics=["em"],
        model_id=args.model,
    )
    model_info = HFModelInfoRetriever(config)

    # Get model architecture details
    arch_info = model_info.get_architecture_info()
    attn_info = model_info.get_attention_info()
    moe_info = model_info.get_moe_info()

    n_layers = arch_info.get("num_hidden_layers", 0)
    d_model = arch_info.get("hidden_size", 0)
    d_ff = moe_info.get("ffn_dim") or arch_info.get("intermediate_size", 0)
    n_attn_heads = attn_info.get("num_attention_heads", 0)
    n_kv_heads = attn_info.get("num_key_value_heads") or n_attn_heads
    d_head = attn_info.get("head_dim") or (
        d_model // n_attn_heads if n_attn_heads > 0 else 0
    )

    # Get precision
    precision = model_info.get_model_precision_bytes()
    used_dtype = f"float{int(precision * 8)}" if precision else "bfloat16"

    # Auto-detect GPU type and number from server records or locally
    num_gpus = args.num_gpus
    gpu_type = "Unknown"
    gpu_raw_type = None

    # Try to get GPU info from server records first
    if server_records:
        first_record = server_records[0]
        num_gpus = first_record.get("gpu_num", args.num_gpus)
        gpu_raw_type = first_record.get("gpu_raw_type", None)

    # If gpu_raw_type is not in server records, detect it locally
    if not gpu_raw_type:
        try:
            from moe_cap.utils.hardware_utils import get_gpu_details

            gpu_raw_type = get_gpu_details()
            print(f"✓ Detected GPU type locally: {gpu_raw_type}")
        except Exception as e:
            print(f"Warning: Could not detect GPU type: {e}")
            gpu_raw_type = None

    # Extract simplified GPU name for display
    if gpu_raw_type:
        import re

        gpu_name_pattern = re.compile(r"NVIDIA[\s-]+(RTX[\s-]+)?([A-Z0-9]+)")
        match = gpu_name_pattern.search(gpu_raw_type)
        if match:
            gpu_type = "".join(filter(None, match.groups())).strip()

    # Calculate metrics using MoE-CAP's function
    try:
        res_dict = _calculate_continuous_metrics(
            n_layers=n_layers,
            d_model=d_model,
            gpu_raw_type=gpu_raw_type,
            n_attn_heads=n_attn_heads,
            d_head=d_head,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            hf_config=model_info.hf_config,
            num_gpus=num_gpus,
            model_name=args.model,
            used_dtype=used_dtype,
            precision=precision,
            output_data=server_records,
        )
    except Exception as e:
        print(f"Warning: Could not calculate continuous batching metrics: {e}")
        import traceback

        traceback.print_exc()
        res_dict = {}

    # Compute accuracy metrics
    predictions = [
        extract_generated_text(o.get("response", {})) for o in outputs
    ]
    successful_count = sum(1 for p in predictions if p)

    # Import extract_answer for detailed results
    from moe_cap.utils.acc_metrics import extract_answer

    accuracy_metrics = compute_accuracy_metrics(
        predictions=predictions,
        targets=targets[: len(predictions)],
        dataset_name=args.dataset,
        extract_answers=True,
    )

    # Merge accuracy metrics into res_dict
    res_dict.update(accuracy_metrics)

    # Calculate cost
    cost = calculate_cost(round(total_time, 2), gpu_raw_type, num_gpus)
    if cost is not None:
        res_dict["cost"] = cost

    # Remove gpu_raw_type from output if present
    if "gpu_raw_type" in res_dict:
        del res_dict["gpu_raw_type"]

    # Add metadata fields matching openai_api_profile format
    res_dict["model_name"] = args.model
    res_dict["method"] = "sllm"
    res_dict["precision"] = used_dtype
    res_dict["e2e_s"] = round(total_time, 2)
    res_dict["batch_size"] = None  # Continuous batching
    res_dict["gpu_type"] = f"{num_gpus}x{gpu_type}"
    res_dict["dataset"] = args.dataset
    res_dict["model_type"] = (
        "instruct"
        if any(x in args.model.lower() for x in ["instruct", "chat"])
        else "thinking"
    )
    res_dict["total_requests"] = len(outputs)
    res_dict["successful_requests"] = successful_count
    res_dict["failed_requests"] = len(outputs) - successful_count

    # Add expert distribution summary if available
    if expert_distribution_data and "local_records" in expert_distribution_data:
        local_records = expert_distribution_data.get("local_records", [])
        if local_records:
            num_records = len(local_records)
            res_dict["expert_total_records"] = num_records
            res_dict["expert_avg_activation"] = (
                sum(r.get("expert_activation", 0) for r in local_records)
                / num_records
                if num_records > 0
                else 0
            )
            res_dict["expert_avg_utilization"] = (
                sum(r.get("expert_utilization", 0) for r in local_records)
                / num_records
                if num_records > 0
                else 0
            )

    # Print metrics summary
    print("\n" + "=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    for key, value in res_dict.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    print("=" * 80)

    # Print accuracy summary
    summary = format_accuracy_summary(accuracy_metrics)
    print(f"\nAccuracy: {summary}")

    # Build detailed results for JSONL output (with extracted answer)
    detailed_results = []
    for i, output in enumerate(outputs):
        response = output.get("response", {})
        generated_text = extract_generated_text(response)
        extracted = (
            extract_answer(generated_text, args.dataset)
            if generated_text
            else ""
        )
        expected = str(output.get("expected", ""))
        is_correct = (
            extracted.lower() == expected.lower() if extracted else False
        )
        detailed_results.append(
            {
                "index": i,
                "success": bool(generated_text),
                "is_correct": is_correct,
                "prompt": output.get("prompt", "")[:200] + "..."
                if len(output.get("prompt", "")) > 200
                else output.get("prompt", ""),
                "expected": expected,
                "extracted_answer": extracted,
                "generated": generated_text[:500] if generated_text else "",
                "error": "" if generated_text else "No response",
            }
        )

    # Save results
    save_results(
        output_dir=args.output_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        metrics=res_dict,
        detailed_results=detailed_results,
        backend="sllm",
    )

    print(
        f"\n✓ Evaluation complete! Results saved to {args.output_dir}/{get_model_simple_name(args.model)}/"
    )


if __name__ == "__main__":
    main()
