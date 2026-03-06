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
import gc
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Union

import torch

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)

logger = logging.getLogger("ray")


def process_output(output: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Process SGLang output to match OpenAI API format."""
    # SGLang output format: {"text": str, "meta_info": {...}}
    text = output.get("text", "")
    meta_info = output.get("meta_info", {})

    choices: List[Dict[str, Any]] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "logprobs": meta_info.get("logprobs", None),
            "finish_reason": meta_info.get("finish_reason", "stop"),
        }
    ]

    api_response = {
        "id": meta_info.get("id", f"chatcmpl-{uuid.uuid4()}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "total_tokens": meta_info.get("prompt_tokens", 0)
            + meta_info.get("completion_tokens", 0),
        },
    }
    return api_response


def process_embedding_output(
    output: Dict[str, Any], model_name: str
) -> Dict[str, Any]:
    """Process SGLang embedding output to match OpenAI API format."""
    # SGLang embedding output format
    embeddings = output.get("embedding", [])
    if not isinstance(embeddings[0], list):
        embeddings = [embeddings]

    meta_info = output.get("meta_info", {})
    query_tokens = meta_info.get("prompt_tokens", 0)

    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": emb,
            }
            for i, emb in enumerate(embeddings)
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
        self.status_dict: Dict[str, Union[Dict, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[Dict, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            if request_id in self.status_dict:
                del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[Dict, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


class SglangBackend(SllmBackend):
    """SGLang backend for ServerlessLLM."""

    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)
        self.model_name = model

        # Import ServerArgs to get valid field names
        from sglang.srt.server_args import ServerArgs

        server_args_fields = {f.name for f in fields(ServerArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in server_args_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model_path"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join(storage_path, "sglang", model)
            filtered_engine_config["model_path"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        logger.info(
            f"Creating new SGLang engine with config: {filtered_engine_config}"
        )

        self.engine_config = filtered_engine_config
        self.engine = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            # Import here to avoid circular dependencies
            from sglang.srt.entrypoints.engine import Engine

            self.engine = Engine(**self.engine_config)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "sglang-model")
        messages: List[Dict[str, str]] = request_data.pop("messages", [])

        # Construct prompt from messages
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        prompt: str = request_data.pop("prompt", construct_prompt)

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        # Map request parameters to SGLang format
        sampling_params = {}
        if "temperature" in request_data:
            sampling_params["temperature"] = request_data["temperature"]
        if "top_p" in request_data:
            sampling_params["top_p"] = request_data["top_p"]
        if "max_tokens" in request_data:
            sampling_params["max_new_tokens"] = request_data["max_tokens"]
        if "frequency_penalty" in request_data:
            sampling_params["frequency_penalty"] = request_data[
                "frequency_penalty"
            ]
        if "presence_penalty" in request_data:
            sampling_params["presence_penalty"] = request_data[
                "presence_penalty"
            ]
        if "stop" in request_data:
            sampling_params["stop"] = request_data["stop"]

        try:
            # Call SGLang engine's async_generate
            output = await self.engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                rid=request_id,
            )

            await self.request_trace.update_status(request_id, output)

            if not self.trace_debug:
                await self.request_trace.delete_request(request_id)

            return process_output(output, model_name)

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return {"error": f"Generation failed: {e}"}

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        model_name = request_data.get("model", "sglang-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        try:
            # Call SGLang engine's async_encode
            output = await self.engine.async_encode(prompt=query)

            return process_embedding_output(output, model_name)

        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            return {"error": f"Encoding failed: {e}"}

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Clear request trace
        requests = await self.request_trace.return_all_request_ids()
        for request_id in requests:
            await self.request_trace.delete_request(request_id)

        if hasattr(self, "engine") and self.engine is not None:
            self.engine.shutdown()
            del self.engine

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

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
        ongoing_results: List[Dict] = [
            result for result in results if isinstance(result, dict)
        ]
        tokens: List[List[int]] = []
        for result in ongoing_results:
            # Extract token IDs from SGLang output
            meta_info = result.get("meta_info", {})
            prompt_token_ids = meta_info.get("prompt_token_ids", [])
            output_token_ids = meta_info.get("output_token_ids", [])
            # Concatenate prompt tokens and output tokens
            tokens.append(prompt_token_ids + output_token_ids)
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache for given requests by rerunning them."""
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
