import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
import torch
from moe_cap.configs import CAPConfig
from moe_cap.data_loader.loader_registry import get_loader_for_task
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.acc_metrics import (
    compute_accuracy_metrics,
    format_accuracy_summary,
)
from moe_cap.utils.continuous_batching_utils import (
    _calculate_continuous_metrics,
)
from moe_cap.utils.cost_utils import calculate_cost
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=100 * 60 * 60)


@dataclass
class RequestFuncInput:
    messages: List[Dict[str, str]]  # Chat messages format
    api_url: str
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]
    backend: Optional[str] = None  # Backend for SLLM routing


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list
    )  # List of inter-token latencies
    error: str = ""
    output_len: int = 0
    prompt_len: int = 0


def get_auth_headers() -> Dict[str, str]:
    """Get authorization headers from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}


def remove_prefix(text: str, prefix: str) -> str:
    """Remove prefix from text if it exists."""
    return text[len(prefix) :] if text.startswith(prefix) else text


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[async_tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> RequestFuncOutput:
    """Send async request to OpenAI-compatible chat completions API."""
    api_url = request_func_input.api_url
    messages = request_func_input.messages

    # Use provided session or create a new one
    should_close_session = session is None
    if session is None:
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        session = aiohttp.ClientSession(
            timeout=AIOHTTP_TIMEOUT, connector=connector
        )

    try:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": False,
            **request_func_input.extra_request_body,
        }
        # Add backend for SLLM routing if specified
        if request_func_input.backend:
            payload["backend"] = request_func_input.backend
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"

        output = RequestFuncOutput()
        st = time.perf_counter()

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    # Non-streaming response - parse complete JSON
                    data = await response.json()
                    latency = time.perf_counter() - st

                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        generated_text = message.get("content", "")
                        # Count tokens from usage if available, otherwise estimate
                        usage = data.get("usage", {})
                        output_len = usage.get(
                            "completion_tokens", len(generated_text.split())
                        )
                    else:
                        generated_text = ""
                        output_len = 0

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                    output.ttft = (
                        latency  # For non-streaming, TTFT = total latency
                    )
                else:
                    error_text = await response.text()
                    output.error = (
                        f"{response.status}: {response.reason} - {error_text}"
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    finally:
        if should_close_session:
            await session.close()

    if pbar:
        pbar.update(1)
    return output


class SLLMProfiler:
    """MoE-CAP Profiler for ServerlessLLM with SGLang backend."""

    def __init__(
        self,
        config: CAPConfig,
        api_url: str,
        output_dir: str = None,
        backend: Optional[str] = None,
        database_path: Optional[str] = None,
        num_samples: Optional[int] = None,
    ):
        """Initialize profiler from a CAPConfig object.

        Args:
            config: CAPConfig instance containing model and dataset info.
            api_url: OpenAI-compatible API endpoint URL (SLLM router).
            output_dir: optional output directory. If not provided, will use './output'.
            backend: Backend for SLLM routing (e.g., 'moe-cap-sglang').
            database_path: Path to SLLM database for auto-discovering backend port.
            num_samples: Number of samples to run. If None, run all.
        """
        self.config = config
        self.api_url = api_url
        self.backend = backend
        self.database_path = database_path or os.environ.get(
            "SLLM_DATABASE_PATH", os.path.expanduser("~/.sllm/state.db")
        )
        self.num_samples = num_samples
        self.backend_url = None  # Will be discovered from database

        # Extract base URL for control endpoints
        from urllib.parse import urlparse

        parsed = urlparse(api_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        # dataset names (can be multiple)
        self.dataset_names = config.dataset_names or ["gsm8k"]

        # output dir
        self.output_dir = output_dir or "./output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Build HF model info retriever using the CAPConfig API
        self.hf_model_name = config.model_id
        self.model_info = HFModelInfoRetriever(config=config)
        moe_info = self.model_info.get_moe_info()
        attn_info = self.model_info.get_attention_info()

        # precision and dtype
        self.precision = self.model_info.get_model_precision_bytes()
        self.used_dtype = config.precision or "bfloat16"

        # architecture info
        arch = self.model_info.get_architecture_info()
        self.d_model = arch.get("hidden_size")
        self.n_layers = arch.get("num_hidden_layers")
        self.n_vocab = arch.get("vocab_size")

        # moe/attention info
        self.d_ff = moe_info.get("ffn_dim")
        self.total_experts = moe_info.get("num_experts_per_layer")
        self.used_experts = moe_info.get("moe_top_k")
        self.n_kv_heads = attn_info.get("num_key_value_heads")
        self.n_attn_heads = attn_info.get(
            "num_attention_heads", self.n_kv_heads
        )
        self.d_head = attn_info.get("head_dim")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )

    def _load_data_for_task(self, task_name: str):
        """Load data for a single task name using the modern data loader APIs."""
        try:
            loader, max_new_tokens = get_loader_for_task(task_name, self.config)
        except KeyError:
            raise ValueError(
                f"Unsupported task '{task_name}'. No loader registered."
            )

        all_input_raw = loader.get_input()

        # Limit samples if num_samples is set
        if self.num_samples is not None:
            all_input_raw = all_input_raw[: self.num_samples]

        return all_input_raw, max_new_tokens, loader

    def _prepare_inputs(self, all_input_raw, max_new_tokens):
        """Prepare chat messages for the model."""
        system_prompt = (
            "You are an expert problem solver. Provide concise answers."
        )
        chat_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            for q in all_input_raw
        ]

        # Calculate prompt lengths using tokenizer
        chat_prompts_text = self.tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=False
        )
        prompt_lengths = [
            len(self.tokenizer.encode(p)) for p in chat_prompts_text
        ]

        return chat_messages, prompt_lengths, max_new_tokens

    def _get_deployment_id(self) -> str:
        """Get the deployment ID for this model and backend."""
        if self.backend:
            return f"{self.hf_model_name}:{self.backend}"
        return self.hf_model_name

    def _discover_backend_url(
        self, timeout: float = 300, poll_interval: float = 2.0
    ) -> Optional[str]:
        """Discover backend URL from SLLM database, waiting for endpoint to be available.

        Args:
            timeout: Maximum time to wait for endpoint (seconds)
            poll_interval: Time between database polls (seconds)

        Returns:
            Backend URL if found, None otherwise
        """
        deployment_id = self._get_deployment_id()
        print(
            f"Waiting for backend endpoint for deployment '{deployment_id}'..."
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT endpoint FROM deployment_endpoints WHERE deployment_id = ?",
                    (deployment_id,),
                )
                row = cursor.fetchone()
                conn.close()

                if row:
                    endpoint = row[0]
                    # Add http:// prefix if not present
                    if not endpoint.startswith(
                        "http://"
                    ) and not endpoint.startswith("https://"):
                        endpoint = f"http://{endpoint}"
                    print(f"Found backend endpoint: {endpoint}")
                    return endpoint
            except Exception as e:
                print(f"Warning: Could not query database: {e}")

            time.sleep(poll_interval)

        print(f"Warning: Timeout waiting for backend endpoint after {timeout}s")
        return None

    def _wait_for_backend_ready(
        self, timeout: float = 300, poll_interval: float = 2.0
    ) -> bool:
        """Wait for the backend server to be ready.

        Args:
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between health checks (seconds)

        Returns:
            True if backend is ready, False otherwise
        """
        if not self.backend_url:
            return False

        print(
            f"Waiting for backend server to be ready at {self.backend_url}..."
        )
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    print("Backend server is ready.")
                    return True
            except Exception:
                pass
            time.sleep(poll_interval)

        print(f"Warning: Backend server not ready after {timeout}s")
        return False

    def _start_expert_distribution_record(self):
        """Start expert distribution recording on the backend server."""
        if not self.backend_url:
            return False
        url = f"{self.backend_url}/start_expert_distribution_record"
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()
            print("Started expert distribution recording.")
            return True
        except Exception as e:
            print(
                f"Warning: Could not start expert distribution recording: {e}"
            )
            return False

    def _stop_expert_distribution_record(self):
        """Stop expert distribution recording on the backend server."""
        if not self.backend_url:
            return False
        url = f"{self.backend_url}/stop_expert_distribution_record"
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()
            print("Stopped expert distribution recording.")
            return True
        except Exception as e:
            print(f"Warning: Could not stop expert distribution recording: {e}")
            return False

    def _dump_expert_distribution_record(self):
        """Dump expert distribution record from the backend server."""
        if not self.backend_url:
            return False
        url = f"{self.backend_url}/dump_expert_distribution_record"
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()
            print("Dumped expert distribution record.")
            return True
        except Exception as e:
            print(f"Warning: Could not dump expert distribution record: {e}")
            return False

    def _load_expert_records(self, dataset_name: str) -> List[dict]:
        """Load expert distribution records from file."""
        if self.backend_url is None:
            return []

        # The server dumps the file to: ~/expert_records/{model_path}/expert_distribution_record.jsonl
        # Use expanduser to resolve ~ to the user's home directory
        server_output_base = os.environ.get(
            "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
        )
        if server_output_base is None:
            server_output_base = os.path.join(
                os.path.expanduser("~"), "expert_records"
            )
        record_path = os.path.join(
            server_output_base,
            self.hf_model_name,
            "expert_distribution_record.jsonl",
        )

        if not os.path.exists(record_path):
            print(f"Warning: Expert record file not found at {record_path}")
            return []

        try:
            with open(record_path, "r", encoding="utf-8") as f:
                records = [json.loads(line.strip()) for line in f]
            print(
                f"Loaded {len(records)} expert distribution records from {record_path}"
            )

            # Rename to dataset-specific file
            dest_record = os.path.join(
                server_output_base,
                self.hf_model_name,
                f"expert_distribution_record_{dataset_name}.jsonl",
            )
            try:
                os.replace(record_path, dest_record)
                print(f"Renamed expert records to {dest_record}")
            except Exception as e:
                print(f"Warning: Could not rename expert records: {e}")

            return records
        except Exception as e:
            print(f"Warning: Could not load expert records: {e}")
            return []

    def get_metrics(
        self, records: List[dict], num_gpus: int = 1
    ) -> Dict[str, Any]:
        """Calculate metrics from expert distribution records."""
        if not records:
            return {}

        gpu_raw_type = records[0].get("gpu_raw_type", None)
        try:
            res_dict = _calculate_continuous_metrics(
                n_layers=self.n_layers,
                d_model=self.d_model,
                gpu_raw_type=gpu_raw_type,
                n_attn_heads=self.n_attn_heads,
                d_head=self.d_head,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                hf_config=getattr(self.model_info, "hf_config", None),
                num_gpus=num_gpus,
                model_name=self.hf_model_name,
                used_dtype=self.used_dtype,
                precision=self.precision,
                output_data=records,
            )
            return res_dict
        except Exception as e:
            print(
                f"Warning: Could not calculate continuous batching metrics: {e}"
            )
            import traceback

            traceback.print_exc()
            return {}

    def get_model_simple_name(self) -> str:
        """Get simplified model name for output directory."""
        norm_path = os.path.normpath(self.hf_model_name)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return self.hf_model_name

    async def run_benchmark(
        self,
        chat_messages: List[List[Dict[str, str]]],
        max_output_len: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[RequestFuncOutput], float]:
        """Send all requests to the API and collect results.

        Args:
            chat_messages: List of chat message lists
            max_output_len: Maximum number of tokens to generate
            batch_size: Number of concurrent requests. If None, send all at once.

        Returns:
            Tuple of (results, total_time)
        """
        # Create a shared session with unlimited connection limits
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(
            timeout=AIOHTTP_TIMEOUT, connector=connector
        ) as session:
            if batch_size is None or batch_size >= len(chat_messages):
                # Send all at once
                tasks = []
                pbar = async_tqdm(
                    total=len(chat_messages), desc="Processing requests"
                )

                for messages in chat_messages:
                    request_input = RequestFuncInput(
                        messages=messages,
                        api_url=self.api_url,
                        output_len=max_output_len,
                        model=self.hf_model_name,
                        extra_request_body={},
                        backend=self.backend,
                    )
                    tasks.append(
                        async_request_openai_chat_completions(
                            request_input, pbar, session
                        )
                    )

                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks)
                total_time = time.perf_counter() - start_time
                pbar.close()
                return results, total_time

            # Batched execution with 50% overlap
            all_results = [None] * len(chat_messages)
            pbar = async_tqdm(
                total=len(chat_messages), desc="Processing requests"
            )

            start_time = time.perf_counter()

            batch_start_idx = 0
            active_tasks = {}  # Maps task to its original index

            while batch_start_idx < len(chat_messages) or active_tasks:
                # Launch new batch if we haven't processed all messages yet
                if batch_start_idx < len(chat_messages):
                    batch_end_idx = min(
                        batch_start_idx + batch_size, len(chat_messages)
                    )

                    for idx in range(batch_start_idx, batch_end_idx):
                        messages = chat_messages[idx]
                        request_input = RequestFuncInput(
                            messages=messages,
                            api_url=self.api_url,
                            output_len=max_output_len,
                            model=self.hf_model_name,
                            extra_request_body={},
                            backend=self.backend,
                        )
                        task = asyncio.create_task(
                            async_request_openai_chat_completions(
                                request_input, pbar, session
                            )
                        )
                        active_tasks[task] = idx

                    current_batch_size = batch_end_idx - batch_start_idx
                    threshold = current_batch_size // 2  # 50% of current batch
                    completed_in_batch = 0

                    # Wait until 50% of current batch is complete before launching next batch
                    while completed_in_batch < threshold and active_tasks:
                        done, pending = await asyncio.wait(
                            active_tasks.keys(),
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in done:
                            result = await task
                            idx = active_tasks.pop(task)
                            all_results[idx] = result

                            # Count if this task belongs to the current batch
                            if batch_start_idx <= idx < batch_end_idx:
                                completed_in_batch += 1

                    batch_start_idx = batch_end_idx

                else:
                    # No more batches to launch, just wait for remaining tasks
                    done, pending = await asyncio.wait(
                        active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        result = await task
                        idx = active_tasks.pop(task)
                        all_results[idx] = result

            total_time = time.perf_counter() - start_time
            pbar.close()

            return all_results, total_time

    async def run_async(self, batch_size: Optional[int] = None):
        """Run profiling for all configured datasets."""
        for dataset_name in self.dataset_names:
            print(f"\n{'='*60}")
            print(f"Running profiling for dataset: {dataset_name}")
            print(f"{'='*60}")

            # Load and prepare inputs
            all_input_raw, max_new_tokens, loader = self._load_data_for_task(
                dataset_name
            )
            chat_messages, prompt_lengths, max_output_len = (
                self._prepare_inputs(all_input_raw, max_new_tokens)
            )

            # Get ground truth targets for evaluation
            try:
                ground_truth = loader.get_target()
                # Limit ground truth to match num_samples
                if self.num_samples is not None:
                    ground_truth = ground_truth[: self.num_samples]
            except Exception as e:
                print(
                    f"Warning: Could not load ground truth for {dataset_name}: {e}"
                )
                ground_truth = None

            # Send a warmup request to trigger instance startup
            print("Sending warmup request to trigger instance startup...")
            warmup_input = RequestFuncInput(
                messages=[{"role": "user", "content": "Hello"}],
                api_url=self.api_url,
                output_len=1,
                model=self.hf_model_name,
                extra_request_body={},
                backend=self.backend,
            )
            await async_request_openai_chat_completions(warmup_input)

            # Discover backend URL from database and wait for it to be ready
            self.backend_url = self._discover_backend_url(timeout=300)
            if self.backend_url:
                self._wait_for_backend_ready(timeout=60)

            # Start expert distribution recording
            self._start_expert_distribution_record()

            # Run benchmark
            print(f"Sending {len(chat_messages)} requests to {self.api_url}")
            start_time = time.time()
            results, total_time = await self.run_benchmark(
                chat_messages=chat_messages,
                max_output_len=max_output_len,
                batch_size=batch_size,
            )
            e2e_time = time.time() - start_time

            # Stop and dump expert distribution recording
            self._stop_expert_distribution_record()
            self._dump_expert_distribution_record()

            # Load expert records
            expert_records = self._load_expert_records(dataset_name)

            # Determine num_gpus from records
            num_gpus = 1
            if expert_records and len(expert_records) > 0:
                first_record = expert_records[0]
                num_gpus = first_record.get("gpu_num", 1)
                print(f"Detected num_gpus from records: {num_gpus}")

            # Calculate metrics from expert records
            res_dict = self.get_metrics(expert_records, num_gpus=num_gpus)

            # Compute accuracy metrics if ground truth is available
            if ground_truth is not None:
                try:
                    predictions = [
                        r.generated_text for r in results if r.success
                    ]

                    accuracy_metrics = compute_accuracy_metrics(
                        predictions=predictions,
                        targets=ground_truth[: len(predictions)],
                        dataset_name=dataset_name,
                        extract_answers=True,
                    )
                    res_dict.update(accuracy_metrics)

                    summary = format_accuracy_summary(accuracy_metrics)
                    print(f"Accuracy for {dataset_name}: {summary}")
                except Exception as e:
                    print(f"Warning: Could not compute accuracy metrics: {e}")

            # Auto-detect GPU type from records
            gpu_raw_type = res_dict.get("gpu_raw_type", None)
            cost = calculate_cost(round(e2e_time, 2), gpu_raw_type, num_gpus)
            if cost is not None:
                res_dict["cost"] = cost

            if gpu_raw_type:
                gpu_name_pattern = re.compile(
                    r"NVIDIA[\s-]+(RTX[\s-]+)?([A-Z0-9]+)"
                )
                match = gpu_name_pattern.search(gpu_raw_type)
                if match:
                    gpu_type = "".join(filter(None, match.groups())).strip()
                else:
                    gpu_type = "Unknown"
            else:
                gpu_type = "Unknown"

            # Filter out gpu_raw_type from metrics
            if "gpu_raw_type" in res_dict:
                del res_dict["gpu_raw_type"]

            # Add metadata fields
            res_dict["model_name"] = self.hf_model_name
            res_dict["method"] = "sllm"
            res_dict["precision"] = self.used_dtype
            res_dict["e2e_s"] = round(e2e_time, 2)
            res_dict["batch_size"] = (
                batch_size  # None indicates all inputs sent at once
            )
            res_dict["gpu_type"] = f"{num_gpus}x{gpu_type}"
            res_dict["dataset"] = dataset_name
            res_dict["total_requests"] = len(results)
            res_dict["successful_requests"] = sum(
                1 for r in results if r.success
            )
            res_dict["failed_requests"] = sum(
                1 for r in results if not r.success
            )
            # Determine model type based on model name (heuristic)
            res_dict["model_type"] = (
                "instruct"
                if any(
                    x in self.hf_model_name.lower()
                    for x in ["instruct", "chat"]
                )
                else "thinking"
            )

            print(
                f"\nMetrics for {dataset_name}: {json.dumps(res_dict, indent=2)}"
            )

            # Save results
            dest_dir = os.path.join(
                self.output_dir, self.get_model_simple_name()
            )
            os.makedirs(dest_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                dest_dir, f"cap_metrics_{dataset_name}_{timestamp}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(res_dict, f, indent=4)
            print(f"Metrics written to {output_path}")

            # Save detailed results
            detailed_output_path = os.path.join(
                dest_dir, f"detailed_results_{dataset_name}_{timestamp}.jsonl"
            )
            with open(detailed_output_path, "w", encoding="utf-8") as f:
                for i, result in enumerate(results):
                    record = {
                        "index": i,
                        "prompt_length": prompt_lengths[i]
                        if i < len(prompt_lengths)
                        else 0,
                        "success": result.success,
                        "generated_text": result.generated_text,
                        "output_len": result.output_len,
                        "ttft": result.ttft,
                        "latency": result.latency,
                        "error": result.error,
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"Detailed results written to {detailed_output_path}")

    def run(self, batch_size: Optional[int] = None):
        """Synchronous wrapper for run_async."""
        asyncio.run(self.run_async(batch_size=batch_size))


def main():
    parser = argparse.ArgumentParser(
        description="MoE-CAP Profiler for ServerlessLLM"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model ID (required unless specified in config file)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="One or more dataset names (e.g. gsm8k), required unless specified in config file",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a JSON or YAML config file that contains CAPConfig fields",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="SLLM router API endpoint URL (e.g., http://localhost:8343/v1/chat/completions)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of requests per batch. If not set, all requests are sent at once.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend for SLLM routing (e.g., 'moe-cap-sglang')",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default=None,
        help="Path to SLLM database (default: ~/.sllm/state.db)",
    )
    args = parser.parse_args()

    # Load config file if provided (JSON or YAML). CLI args override file values.
    file_cfg = {}
    if args.config_file:
        cf = args.config_file
        if cf.endswith(".json"):
            with open(cf, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
        else:
            try:
                import yaml

                with open(cf, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f)
            except Exception:
                with open(cf, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)

    # Merge CLI args over file config
    merged = dict(file_cfg or {})
    merged["model_id"] = args.model_name or merged.get("model_id")
    merged["dataset_names"] = args.datasets or merged.get("dataset_names")

    # Validate required fields
    if not merged.get("model_id"):
        parser.error(
            "--model_name is required (or 'model_id' must be specified in the config file)"
        )
    if not merged.get("dataset_names"):
        parser.error(
            "--datasets is required (or 'dataset_names' must be specified in the config file)"
        )

    # Validate that all datasets have registered loaders
    from moe_cap.data_loader.loader_registry import _REGISTRY

    unsupported = [
        ds for ds in merged["dataset_names"] if ds.lower() not in _REGISTRY
    ]
    if unsupported:
        available = sorted(_REGISTRY.keys())
        parser.error(
            f"Unsupported dataset(s): {', '.join(unsupported)}. "
            f"Available datasets: {', '.join(available)}"
        )

    # Build CAPConfig
    cap_cfg = CAPConfig(
        dataset_names=merged.get("dataset_names"),
        metrics=merged.get("metrics", []),
        model_id=merged.get("model_id"),
        precision=merged.get("precision", "bfloat16"),
        dataset_subset=merged.get("dataset_subset"),
        dataset_split=merged.get("dataset_split", "test"),
    )

    profiler = SLLMProfiler(
        config=cap_cfg,
        api_url=args.api_url,
        output_dir=args.output_dir,
        backend=args.backend,
        database_path=args.database_path,
        num_samples=args.num_samples,
    )

    profiler.run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
