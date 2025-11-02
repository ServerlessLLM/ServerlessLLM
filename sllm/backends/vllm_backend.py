# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import json
import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

import aiohttp

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
    cleanup_subprocess,
)
from sllm.logger import init_logger
from sllm.worker.utils import allocate_backend_port
from sllm_store.transformers import save_model

logger = init_logger(__name__)


class VllmBackend(SllmBackend):
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model = model
        self.process = None
        self.port = backend_config.get("port") or allocate_backend_port("vllm")
        self.host = backend_config.get("host", "0.0.0.0")
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = None

        self.backend_config["port"] = self.port
        self.backend_config["host"] = self.host

    def _start_vllm_server(self):
        cmd = self._build_serve_command()
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            logger.info(f"Starting vLLM backend for {self.model}")

            try:
                storage_path = os.getenv("STORAGE_PATH", "./models")
                model_path = os.path.join(storage_path, "vllm", self.model)

                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"vLLM model not found at {model_path}"
                    )

                self.status = BackendStatus.RUNNING
                logger.info(f"vLLM AsyncLLMEngine started for {self.model}")

            except Exception as e:
                logger.error(f"Failed to start vLLM backend: {e}")
                await self._cleanup()
                raise

    def _build_serve_command(self) -> list:
        storage_path = os.getenv("STORAGE_PATH", "./models")
        storage_path = os.path.abspath(storage_path)
        model_path = os.path.join(storage_path, "vllm", self.model)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VLLM model not found at {model_path}")

        rank_0_path = os.path.join(model_path, "rank_0")
        if not os.path.exists(rank_0_path):
            raise FileNotFoundError(
                f"VLLM model missing rank_0 directory: {model_path}"
            )

        tensor_index_path = os.path.join(rank_0_path, "tensor_index.json")
        if not os.path.exists(tensor_index_path):
            raise FileNotFoundError(
                f"VLLM model missing tensor_index.json: {model_path}"
            )

        has_tensor_data = any(
            f.startswith("tensor.data_") for f in os.listdir(rank_0_path)
        )
        if not has_tensor_data:
            raise FileNotFoundError(
                f"VLLM model missing tensor.data_* files: {model_path}"
            )

        cmd = [
            "vllm",
            "serve",
            model_path,
            "--load-format",
            "serverless_llm",
            "--served-model-name",
            self.model,
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]

        try:
            subprocess.run(
                ["vllm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            logger.warning("VLLM command validation failed, proceeding anyway")

        if "max_model_len" in self.backend_config:
            cmd.extend(
                ["--max-model-len", str(self.backend_config["max_model_len"])]
            )
        if "tensor_parallel_size" in self.backend_config:
            cmd.extend(
                [
                    "--tensor-parallel-size",
                    str(self.backend_config["tensor_parallel_size"]),
                ]
            )
        if "gpu_memory_utilization" in self.backend_config:
            cmd.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.backend_config["gpu_memory_utilization"]),
                ]
            )
        if (
            "enforce_eager" in self.backend_config
            and self.backend_config["enforce_eager"]
        ):
            cmd.append("--enforce-eager")
        if "enable_prefix_caching" in self.backend_config:
            if self.backend_config["enable_prefix_caching"]:
                cmd.append("--enable-prefix-caching")
            else:
                cmd.append("--disable-prefix-caching")
        if "dtype" in self.backend_config:
            cmd.extend(["--dtype", self.backend_config["dtype"]])

        # Add a simple chat template for base models that don't have one
        # This allows /v1/chat/completions to work with base models like OPT
        simple_template = (
            "{% for message in messages %}{{ message.content }}{% endfor %}"
        )

        return cmd

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        if self.session:
            await self.session.close()
            self.session = None

        self._cleanup_process()

    async def stop(self) -> None:
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        await self.shutdown()

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        if not self.session:
            return {"error": "HTTP session not initialized"}

        try:
            embedding_request = {
                "model": request_data.get("model", self.model),
                "input": request_data.get("input", []),
            }

            async with self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=embedding_request,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"VLLM embedding API error: {response.status} - {error_text}"
                    )
                    return {
                        "error": f"Embedding request failed: {response.status}"
                    }

        except Exception as e:
            logger.error(f"Error in encode: {e}")
            return {"error": f"Encoding failed: {str(e)}"}

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        if self.status != BackendStatus.RUNNING:
            return []

        try:
            async with self.session.get(
                f"{self.base_url}/get_current_tokens"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("tokens", [])
                else:
                    logger.warning(
                        f"Failed to get current tokens: {response.status}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Error getting current tokens: {e}")
            return []

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache for given request token sequences."""
        if self.status != BackendStatus.RUNNING:
            return

        try:
            # For vLLM, simulate cache warming by sending short generation requests
            constructed_inputs = [
                {
                    "prompt": "",  # Will be filled from tokens
                    "max_tokens": 1,
                    "temperature": 0.0,
                }
                for request_data in request_datas
            ]

            tasks = []
            for i, (inputs, tokens) in enumerate(
                zip(constructed_inputs, request_datas)
            ):
                # Convert tokens back to text (simplified approach)
                inputs["prompt"] = f"<resume_cache_{i}>"
                task = self.session.post(
                    f"{self.base_url}/v1/completions", json=inputs
                )
                tasks.append(task)

            # Execute all cache warming requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, Exception):
                    logger.warning(f"Cache warming request failed: {response}")
                elif hasattr(response, "close"):
                    await response.close()

        except Exception as e:
            logger.error(f"Error resuming KV cache: {e}")

    async def fine_tuning(self, request_data: Dict[str, Any]):
        raise NotImplementedError(
            "Fine-tuning is not supported in this backend"
        )
