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
import time
import uuid
from typing import Any, Dict, Optional

from serverless_llm.serve.backends.backend_utils import SllmBackend
from serverless_llm.serve.logger import init_logger


class DummyBackend(SllmBackend):
    def __init__(self, backend_config: Optional[Dict[str, Any]] = None) -> None:
        self.backend_config = backend_config

    def init_backend(self) -> None:
        # sleep to simulate model latency
        sleep_time = 5
        self.log(
            f"Sleeping for {sleep_time} seconds to simulate model init time."
        )
        time.sleep(sleep_time)

    def log(self, msg):
        logger = init_logger(__name__)
        logger.info(msg)

    async def generate(self, request_data):
        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        # temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        token_latency = request_data.get("token_latency", 0.1)

        # Combine messages to form the prompt
        prompt = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # Simulate model latency
        self.log(
            f"Sleeping for {max_tokens * token_latency} seconds to simulate model response time."
        )
        for i in range(max_tokens):
            await asyncio.sleep(token_latency)

        # Dummy response content
        response_content = f"Debug model received prompt: {prompt}"

        # Simulate token counts for the response
        prompt_tokens = len(prompt.split())
        completion_tokens = max_tokens
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
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
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
        self.log("Not implemented")
        raise NotImplementedError

    async def resume_kv_cache(self, request_datas):
        self.log("Not implemented")
        raise NotImplementedError
