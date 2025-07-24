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
from typing import Any, Dict, Optional

import aiohttp

import sllm.serve.backends.transformers_server
from sllm.serve.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
    cleanup_subprocess,
)
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model_name = model_name
        self.process = None
        self.port = backend_config.get("port", 8001)
        self.host = backend_config.get("host", "127.0.0.1")
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            logger.info(
                f"Starting transformers HTTP server for model {self.model_name}"
            )

            # Build transformers server command
            cmd = self._build_serve_command()

            try:
                # Start transformers server process
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

                # Wait for server to be ready
                await self._wait_for_server()

                # Create HTTP session
                self.session = aiohttp.ClientSession()

                self.status = BackendStatus.RUNNING
                logger.info(
                    f"Transformers server started successfully on {self.base_url}"
                )

            except Exception as e:
                logger.error(f"Failed to start Transformers server: {e}")
                if self.process:
                    self._cleanup_process()
                raise

    def _build_serve_command(self) -> list:
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "transformers", self.model_name)

        # Validate model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Transformers model not found at {model_path}"
            )

        # Use the dedicated transformers server

        server_script_path = sllm.serve.worker.transformers_server.__file__

        cmd = [
            "python",
            server_script_path,
            "--model_name",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--device_map",
            self.backend_config.get("device_map", "auto"),
            "--torch_dtype",
            self.backend_config.get("torch_dtype", "float16"),
            "--hf_model_class",
            self.backend_config.get("hf_model_class", "AutoModelForCausalLM"),
        ]

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
                    f"Transformers server process died: stdout={stdout.decode()}, stderr={stderr.decode()}"
                )
                raise RuntimeError(
                    "Transformers server process died during startup"
                )

            await asyncio.sleep(2)

        raise TimeoutError(
            f"Transformers server did not start within {timeout} seconds"
        )

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        if not self.session:
            return {"error": "HTTP session not initialized"}

        try:
            # Prepare request for OpenAI-compatible API
            openai_request = {
                "model": request_data.get("model", self.model_name),
                "messages": request_data.get("messages", []),
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 1.0),
            }

            # Add task_id if present
            if "task_id" in request_data:
                openai_request["task_id"] = request_data["task_id"]

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
                        f"Transformers server API error: {response.status} - {error_text}"
                    )
                    return {"error": f"API request failed: {response.status}"}

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return {"error": f"Generation failed: {str(e)}"}

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        if not self.session:
            return {"error": "HTTP session not initialized"}

        try:
            # Prepare request for embeddings API
            embedding_request = {
                "model": request_data.get("model", self.model_name),
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
                        f"Transformers server embedding API error: {response.status} - {error_text}"
                    )
                    return {
                        "error": f"Embedding request failed: {response.status}"
                    }

        except Exception as e:
            logger.error(f"Error in encode: {e}")
            return {"error": f"Encoding failed: {str(e)}"}

    def _cleanup_process(self):
        cleanup_subprocess(self.process)
        self.process = None

    async def shutdown(self):
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        logger.info("Shutting down Transformers server backend")

        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None

        # Cleanup process
        self._cleanup_process()

        logger.info("Transformers server backend shutdown complete")

    async def stop(self) -> None:
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        logger.info("Stopping Transformers server backend")
        await self.shutdown()

    async def fine_tuning(self, request_data: Dict[str, Any]):
        raise NotImplementedError(
            "Fine-tuning is not supported in this HTTP backend version"
        )
