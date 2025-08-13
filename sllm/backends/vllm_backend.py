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
<<<<<<< HEAD
import json
import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

import aiohttp
=======
import gc
import inspect
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt
from vllm.utils import Counter
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
<<<<<<< HEAD
    cleanup_subprocess,
)
from sllm.logger import init_logger
from sllm.worker.utils import allocate_backend_port

logger = init_logger(__name__)


class VllmBackend(SllmBackend):
=======
)

logger = logging.getLogger("ray")


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
<<<<<<< HEAD
        self.model = model
        self.process = None
        self.port = backend_config.get("port") or allocate_backend_port("vllm")
        self.host = backend_config.get("host", "0.0.0.0")
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = None

        # Update backend_config with allocated port for reference
        self.backend_config["port"] = self.port
        self.backend_config["host"] = self.host
=======
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        self.task = backend_config.get("task", "auto")

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        # NOTE: Automatic enable prefix cachinging
        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
<<<<<<< HEAD

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
=======
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # TODO stream results

        # Non-stream case
        final_output = None
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

        return process_output(final_output, model_name)

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

<<<<<<< HEAD
        if self.session:
            await self.session.close()
            self.session = None

        self._cleanup_process()

    async def stop(self) -> None:
=======
        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
<<<<<<< HEAD

        await self.shutdown()

=======
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449
    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

<<<<<<< HEAD
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
=======
        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)
>>>>>>> 827126df3e5deadbd63032c989eae2545fed9449

    async def fine_tuning(self, request_data: Dict[str, Any]):
        raise NotImplementedError(
            "Fine-tuning is not supported in this backend"
        )
