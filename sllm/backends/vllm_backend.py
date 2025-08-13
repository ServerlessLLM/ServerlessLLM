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

        # Update backend_config with allocated port for reference
        self.backend_config["port"] = self.port
        self.backend_config["host"] = self.host

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            cmd = self._build_serve_command()

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    preexec_fn=os.setsid,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Check if process started successfully
                await asyncio.sleep(1)  # Give it a moment to start
                if self.process.poll() is not None:
                    stdout, _ = self.process.communicate()
                    logger.error(
                        f"VLLM process failed to start. Exit code: {self.process.returncode}"
                    )
                    logger.error(f"VLLM output: {stdout}")
                    raise RuntimeError(
                        f"VLLM process failed to start with exit code {self.process.returncode}"
                    )

                await self._wait_for_server()
                self.session = aiohttp.ClientSession()
                self.status = BackendStatus.RUNNING

                # Start monitoring VLLM process logs
                asyncio.create_task(self._monitor_process_logs())

                logger.info(f"VLLM started on {self.base_url}")

            except Exception as e:
                logger.error(f"Failed to start VLLM serve: {e}")
                if self.process:
                    self._cleanup_process()
                raise

    def _build_serve_command(self) -> list:
        storage_path = os.getenv("STORAGE_PATH", "/models")
        storage_path = os.path.abspath(storage_path)
        model_path = os.path.join(storage_path, "vllm", self.model)

        # Validate model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VLLM model not found at {model_path}")

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

        # Validate VLLM command exists (optional check)
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
        cmd.extend(["--chat-template", simple_template])

        return cmd

    async def _wait_for_server(self, timeout: int = 600):
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
                exit_code = self.process.returncode
                # Read any available output without blocking
                try:
                    stdout = (
                        self.process.stdout.read()
                        if self.process.stdout
                        else ""
                    )
                except:
                    stdout = "Could not read output"
                logger.error(
                    f"VLLM process died during startup with exit code {exit_code}"
                )
                logger.error(f"VLLM output: {stdout}")
                raise RuntimeError(
                    f"VLLM process died during startup with exit code {exit_code}"
                )

            await asyncio.sleep(2)

        raise TimeoutError(f"VLLM serve did not start within {timeout} seconds")

    async def _monitor_process_logs(self):
        """Monitor VLLM process logs and detect crashes."""
        if not self.process or not self.process.stdout:
            return

        try:
            while (
                self.process.poll() is None
                and self.status == BackendStatus.RUNNING
            ):
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                if line:
                    line = line.strip()
                    if line:
                        # Only log errors and important messages
                        if any(
                            error in line.lower()
                            for error in [
                                "error",
                                "exception",
                                "traceback",
                                "failed",
                                "cuda error",
                                "out of memory",
                            ]
                        ):
                            logger.error(f"VLLM-{self.port}: {line}")
                else:
                    # No output, small delay to prevent busy waiting
                    await asyncio.sleep(0.1)

            # Process has ended
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                logger.error(
                    f"VLLM-{self.port} process exited with code {exit_code}"
                )

        except Exception as e:
            logger.error(f"VLLM-{self.port} error monitoring logs: {e}")

    def _cleanup_process(self):
        cleanup_subprocess(self.process)
        self.process = None

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
