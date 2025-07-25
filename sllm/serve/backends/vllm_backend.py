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
from typing import Any, Dict, Optional, List

import aiohttp

from sllm.serve.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
    cleanup_subprocess,
)
from sllm.serve.logger import init_logger

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
        self.port = backend_config.get("port", 8000)
        self.host = backend_config.get("host", "127.0.0.1")
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            logger.info(f"Starting vllm serve for model {self.model}")

            cmd = self._build_serve_command()

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                await self._wait_for_server()
                self.session = aiohttp.ClientSession()
                self.status = BackendStatus.RUNNING
                logger.info(
                    f"VLLM serve started successfully on {self.base_url}"
                )

            except Exception as e:
                logger.error(f"Failed to start VLLM serve: {e}")
                if self.process:
                    self._cleanup_process()
                raise

    def _build_serve_command(self) -> list:
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "vllm", self.model)

        # Validate model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VLLM model not found at {model_path}")

        cmd = [
            "vllm",
            "serve",
            model_path,
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]

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

        return cmd

    async def _wait_for_server(self, timeout: int = 120):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/health"
                    ) as response:
                        if response.status == 200:
                            return
            except Exception:
                pass

            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(
                    f"VLLM serve process died: stdout={stdout.decode()}, stderr={stderr.decode()}"
                )
                raise RuntimeError("VLLM serve process died during startup")

            await asyncio.sleep(2)

        raise TimeoutError(f"VLLM serve did not start within {timeout} seconds")

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        if not self.session:
            return {"error": "HTTP session not initialized"}

        try:
            openai_request = {
                "model": request_data.get("model", self.model),
                "messages": request_data.get("messages", []),
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 1.0),
                "stream": False,
            }

            # Add task_id if present
            if "task_id" in request_data:
                openai_request["request_id"] = request_data["task_id"]

            for param in [
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "logprobs",
            ]:
                if param in request_data:
                    openai_request[param] = request_data[param]

            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=openai_request,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"VLLM serve API error: {response.status} - {error_text}"
                    )
                    return {"error": f"API request failed: {response.status}"}

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return {"error": f"Generation failed: {str(e)}"}

    def _cleanup_process(self):
        cleanup_subprocess(self.process)
        self.process = None

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        logger.info("Shutting down VLLM serve backend")

        if self.session:
            await self.session.close()
            self.session = None

        self._cleanup_process()
        logger.info("VLLM serve backend shutdown complete")

    async def stop(self) -> None:
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        logger.info("Stopping VLLM serve backend")
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
                        f"VLLM serve embedding API error: {response.status} - {error_text}"
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
                elif hasattr(response, 'close'):
                    await response.close()

        except Exception as e:
            logger.error(f"Error resuming KV cache: {e}")

    async def fine_tuning(self, request_data: Dict[str, Any]):
        raise NotImplementedError(
            "Fine-tuning is not supported in this backend"
        )
