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
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import torch
from serverless_llm_store.transformers import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from serverless_llm.serve.backends.backend_utils import SllmBackend
from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


class TransformersBackend(SllmBackend):
    def __init__(self, backend_config: Optional[Dict[str, Any]] = None) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend with config: {backend_config}"
        )
        self.model_name = backend_config.get("pretrained_model_name_or_path")
        self.model = None
        self.model_initialized = False
        self.model_status_lock = asyncio.Lock()
        self.tokenizer = None

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            # print(f"Failed to decode JSON string: {e}")
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    async def init_backend(self) -> None:
        async with self.model_status_lock:
            if self.model_initialized:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = Path(
                os.path.join(storage_path, "transformers", self.model_name)
            ).resolve()
            self.model = load_model(
                str(model_path),
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model_initialized = True
    
    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    async def generate(self, request_data: Optional[Dict[str, Any]]):
        async with self.model_status_lock:
            if not self.model_initialized:
                return {"error": "Model not initialized"}
        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)

        # Combine messages to form the prompt
        prompt = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        inputs = self._tokenize(prompt) 

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature
            )

        output_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        # Simulate token counts for the response
        prompt_tokens = len(self.tokenizer.tokenize(prompt))
        completion_tokens = len(self.tokenizer.tokenize(output_text))
        total_tokens = prompt_tokens + completion_tokens

        # Generate response compatible with OpenAI's API
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response

    async def shutdown(self):
        pass

    async def stop(self):
        pass

    async def get_current_tokens(self):
        logger.error("Not implemented")
        raise NotImplementedError

    async def resume_kv_cache(self, request_datas):
        logger.error("Not implemented")
        raise NotImplementedError
