#!/usr/bin/env python3
"""
MoE-CAP Evaluation Script for ServerlessLLM

This script evaluates MoE models using the MoE-CAP methodology by:
1. Loading benchmark datasets using MoE-CAP data loaders
2. Starting batch recording on ServerlessLLM
3. Sending inference requests to ServerlessLLM API
4. Collecting batch statistics
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
from typing import Any, Dict, List, Tuple

import aiohttp
import requests
from tqdm.asyncio import tqdm_asyncio

# Add MoE-CAP to path
moecap_path = Path(__file__).parent.parent.parent.parent / "MoE-CAP-Real"
sys.path.insert(0, str(moecap_path))

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


def start_recording(server_url: str, model_name: str) -> bool:
    """Start batch recording on the server."""
    url = f"{server_url}/start_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Started batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to start recording: {e}")
        return False


def stop_recording(server_url: str, model_name: str) -> bool:
    """Stop batch recording on the server."""
    url = f"{server_url}/stop_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Stopped batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to stop recording: {e}")
        return False


def get_recording_status(server_url: str, model_name: str) -> Dict[str, Any]:
    """Get current recording status."""
    url = f"{server_url}/batch_recording_status"

    try:
        response = requests.get(url, params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Failed to get status: {e}")
        return {}


def dump_recording(server_url: str, model_name: str) -> Dict[str, Any]:
    """Dump recorded batch statistics."""
    url = f"{server_url}/dump_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Normalize the response - backend might return 'batches' or 'records'
        if "batches" in data and "records" not in data:
            data["records"] = data["batches"]
        elif "records" not in data:
            data["records"] = []

        print(f"✓ Dumped {len(data.get('records', []))} batch records")
        return data
    except Exception as e:
        print(f"✗ Failed to dump recording: {e}")
        return {}


def clear_recording(server_url: str, model_name: str) -> bool:
    """Clear all recorded statistics."""
    url = f"{server_url}/clear_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Cleared batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to clear recording: {e}")
        return False


def send_inference_request(
    server_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Send an inference request to ServerlessLLM.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model to use
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Response dictionary with generated text
    """
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
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
) -> Dict[str, Any]:
    """Send an inference request asynchronously."""
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
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


def calculate_moecap_metrics(
    batch_data: Dict[str, Any],
    model_name: str,
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """Calculate MoE-CAP metrics from batch statistics.

    Args:
        batch_data: Dictionary with 'records' list from dump_batch_recording
        model_name: Model identifier for loading model config
        num_gpus: Number of GPUs used for inference

    Returns:
        Dictionary with MoE-CAP metrics computed by _calculate_continuous_metrics
    """
    # The backend returns {'records': [...], 'status': 'success', ...}
    records = batch_data.get("records", [])
    if not records:
        return {"error": "No batch data available"}

    # Calculate MoE-CAP metrics using model info
    try:
        # Load model configuration
        config = CAPConfig(
            dataset_names=["gsm8k"],  # Dummy dataset for config
            metrics=["em"],
            model_id=model_name,
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

        # Get precision (precision should be bytes per element, not a string)
        precision_bits = (
            model_info.get_model_precision_bits()
        )  # Returns bytes (e.g., 2.0 for bfloat16)
        used_dtype = f"float{int(precision_bits * 8)}"  # String like "float16" for hardware specs
        precision = (
            precision_bits  # Use the numeric value (bytes) for calculations
        )

        # The records are already in the exact format expected by _calculate_continuous_metrics!
        # Each record has: batch_size, latency (seconds), seq_lens_sum, forward_mode, expert_activation
        output_data = records

        # Calculate advanced metrics using MoE-CAP's function
        hf_config = model_info.hf_config
        metrics = _calculate_continuous_metrics(
            n_layers=n_layers,
            d_model=d_model,
            n_attn_heads=n_attn_heads,
            d_head=d_head,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            hf_config=hf_config,
            num_gpus=num_gpus,
            model_name=model_name,
            used_dtype=used_dtype,
            precision=precision,
            output_data=output_data,
        )

        # Add total records count
        metrics["total_records"] = len(records)
        return metrics

    except Exception as e:
        print(f"Warning: Could not calculate MoE-CAP metrics: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "total_records": len(records)}


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


def compute_inference_accuracy(
    outputs: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    """Compute accuracy metrics for inference outputs using MoE-CAP utilities.

    Args:
        outputs: List of output dictionaries with 'response' and 'expected' fields
        dataset_name: Name of dataset for answer extraction

    Returns:
        Dictionary with accuracy metrics
    """
    predictions = []
    targets = []

    for output in outputs:
        # Extract generated text from response
        generated = extract_generated_text(output.get("response", {}))
        expected = output.get("expected", "")

        predictions.append(generated)
        targets.append(str(expected))

    # Use MoE-CAP's accuracy computation with answer extraction
    accuracy_metrics = compute_accuracy_metrics(
        predictions=predictions,
        targets=targets,
        dataset_name=dataset_name,
        extract_answers=True,
    )

    return accuracy_metrics


def save_results(
    output_dir: str, results: Dict[str, Any], backend: str = "vllm_moecap"
):
    """Save evaluation results to files.

    Files are saved in: {output_dir}/{model_name}/cap_metrics_{backend}_{dataset}.json
    """
    model_name = results.get("config", {}).get("model", "unknown_model")
    dataset_name = results.get("config", {}).get("dataset", "unknown_dataset")

    # Create model-specific directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save combined metrics with naming: cap_metrics_{backend}_{dataset}.json
    metrics_file = os.path.join(
        model_dir, f"cap_metrics_{backend}_{dataset_name}.json"
    )
    combined_metrics = {
        "batch_metrics": results.get("batch_metrics", {}),
        "accuracy_metrics": results.get("accuracy_metrics", {}),
    }
    with open(metrics_file, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")

    # Save raw batch data
    batch_file = os.path.join(
        model_dir, f"batch_data_{backend}_{dataset_name}.json"
    )
    with open(batch_file, "w") as f:
        json.dump(results["batch_data"], f, indent=2)
    print(f"✓ Saved batch data to {batch_file}")

    # Save inference outputs
    outputs_file = os.path.join(
        model_dir, f"inference_outputs_{dataset_name}.json"
    )
    with open(outputs_file, "w") as f:
        json.dump(results["outputs"], f, indent=2)
    print(f"✓ Saved inference outputs to {outputs_file}")

    # Save config
    config_file = os.path.join(
        model_dir, f"config_{backend}_{dataset_name}.json"
    )
    with open(config_file, "w") as f:
        json.dump(results.get("config", {}), f, indent=2)
    print(f"✓ Saved config to {config_file}")


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

    args = parser.parse_args()

    print("=" * 80)
    print("MoE-CAP Evaluation for ServerlessLLM")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Server: {args.server_url}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load dataset using MoE-CAP data loaders
    print("\n[1/6] Loading dataset...")
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
        print("\n[2/6] Clearing previous recordings...")
        clear_recording(args.server_url, args.model)
    else:
        print("\n[2/6] Skipping clear (use --clear-before to clear)")

    # Warmup
    warmup_model(args.server_url, args.model)

    # Start recording
    print("\n[3/6] Starting batch recording...")
    if not start_recording(args.server_url, args.model):
        print("Failed to start recording. Exiting.")
        return

    # Check status
    status = get_recording_status(args.server_url, args.model)
    print(f"Recording status: {status}")

    # Send inference requests
    print(f"\n[4/6] Sending {len(prompts)} inference requests...")
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
    print(f"✓ Completed {len(outputs)} inference requests")

    # Stop recording
    print("\n[5/6] Stopping batch recording...")
    if not stop_recording(args.server_url, args.model):
        print("Failed to stop recording.")

    # Dump and analyze results
    print("\n[6/6] Dumping and analyzing results...")
    batch_data = dump_recording(args.server_url, args.model)

    if not batch_data.get("records"):
        print("✗ No batch data collected. Check if backend is vllm_moecap.")
        return

    # Calculate batch performance metrics
    batch_metrics = calculate_moecap_metrics(
        batch_data,
        model_name=args.model,
        num_gpus=args.num_gpus,
    )

    # Calculate accuracy metrics using MoE-CAP utilities
    print("\n[7/7] Computing accuracy metrics...")
    accuracy_metrics = compute_inference_accuracy(outputs, args.dataset)

    print("\n" + "=" * 80)
    print("Performance Metrics (Batch Statistics)")
    print("=" * 80)
    for key, value in batch_metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    print("\n" + "=" * 80)
    print("Accuracy Metrics")
    print("=" * 80)
    print(format_accuracy_summary(accuracy_metrics))
    for key, value in accuracy_metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    print("=" * 80)

    # Save results
    results = {
        "batch_metrics": batch_metrics,
        "accuracy_metrics": accuracy_metrics,
        "batch_data": batch_data,
        "outputs": outputs,
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "num_samples": len(prompts),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }
    save_results(args.output_dir, results, backend="vllm_moecap")

    print(
        f"\n✓ Evaluation complete! Results saved to {args.output_dir}/{args.model}/"
    )


if __name__ == "__main__":
    main()
