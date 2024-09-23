# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from vllm import CompletionOutput, RequestOutput

from serverless_llm.serve.backends.vllm_backend import (BackendStatus,
                                                        VllmBackend)


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "test-model",
        "trace_debug": False,
        "load_format": "sharded_state",
        "torch_dtype": None,
    }


async def generate(
    inputs, sampling_params, request_id
) -> AsyncIterator[RequestOutput]:
    prompt = inputs
    tokens = [1, 2, 3]

    yield RequestOutput(
        request_id,
        prompt,
        tokens,
        None,
        [CompletionOutput(0, "test", [4], 0.5, None)],
        True,
    )
    await asyncio.sleep(1)
    yield RequestOutput(
        request_id,
        prompt,
        tokens,
        None,
        [CompletionOutput(0, "test output", [4, 5], 0.5, None)],
        True,
    )
    await asyncio.sleep(1)
    yield RequestOutput(
        request_id,
        prompt,
        tokens,
        None,
        [CompletionOutput(0, "test output result", [4, 5, 6], 0.5, None)],
        True,
    )


@pytest.fixture
def async_llm_engine():
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine"
    ) as MockAsyncLLMEngine:
        async_llm_engine_obj = MockAsyncLLMEngine.return_value
        async_llm_engine_obj.from_engine_args.return_value = (
            async_llm_engine_obj
        )
        async_llm_engine_obj.abort = AsyncMock()
        async_llm_engine_obj.generate.side_effect = generate
        yield async_llm_engine_obj


@pytest.fixture
def vllm_backend(backend_config, async_llm_engine):
    yield VllmBackend(backend_config)


def test_init(vllm_backend, backend_config):
    assert vllm_backend.backend_config == backend_config
    assert vllm_backend.status == BackendStatus.UNINITIALIZED


@pytest.mark.asyncio
async def test_init_backend(vllm_backend, async_llm_engine):
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    assert vllm_backend.status == BackendStatus.RUNNING


@pytest.mark.asyncio
async def test_generate_without_init(vllm_backend):
    request_data = {
        "model_name": "test-model",
        "prompt": "user: Hello",
        "request_id": "test-request-id",
    }
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=AsyncMock(),
    ):
        response = await vllm_backend.generate(request_data)
    assert "error" in response


@pytest.mark.asyncio
async def test_generate(vllm_backend, async_llm_engine):
    request_data = {
        "model_name": "test-model",
        "prompt": "user: Hello",
        "request_id": "test-request-id",
    }
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    response = await vllm_backend.generate(request_data)
    assert "error" not in response
    assert "model" in response and response["model"] == "test-model"
    assert "id" in response and response["id"] == "test-request-id"


@pytest.mark.asyncio
async def test_shutdown(backend_config, async_llm_engine):
    # Open trace debug to avoid clean the finished request in record map
    backend_config["trace_debug"] = True
    vllm_backend = VllmBackend(backend_config)
    request_data = {
        "model_name": "test-model",
        "prompt": "user: Hello",
        "request_id": "test-request-id",
    }
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    await vllm_backend.generate(request_data)
    await vllm_backend.shutdown()
    assert vllm_backend.status == BackendStatus.DELETING
    # Since open trace debug, will try to abort the already done task
    assert async_llm_engine.abort.call_count == 1


@pytest.mark.asyncio
async def test_stop(vllm_backend, async_llm_engine):
    request_data = {
        "model_name": "test-model",
        "prompt": "user: Hello",
        "request_id": "test-request-id",
    }
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    await vllm_backend.generate(request_data)
    with patch(
        "serverless_llm.serve.backends.vllm_backend.VllmBackend.shutdown",
        new_callable=AsyncMock,
    ) as mock_shutdown:
        await vllm_backend.stop()
        assert vllm_backend.status == BackendStatus.STOPPING
        # stop will not call abort
        assert async_llm_engine.abort.call_count == 0
        # stop will call shutdown after stop done
        assert mock_shutdown.call_count == 1


@pytest.mark.asyncio
async def test_get_current_tokens(backend_config, async_llm_engine):
    # Open trace debug to avoid clean the finished request in record map
    backend_config["trace_debug"] = True
    vllm_backend = VllmBackend(backend_config)
    request_data = [
        {
            "model_name": "test-model",
            "prompt": "user: Hello",
            "request_id": f"test-request-id{i}",
        }
        for i in range(3)
    ]
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    for request in request_data:
        await vllm_backend.generate(request)
    current_tokens = await vllm_backend.get_current_tokens()
    assert len(current_tokens) == 3


@pytest.mark.asyncio
async def test_resume_kv_cache(vllm_backend):
    with patch(
        "serverless_llm.serve.backends.vllm_backend.AsyncLLMEngine.from_engine_args",
        return_value=async_llm_engine,
    ):
        await vllm_backend.init_backend()
    vllm_backend.generate = AsyncMock()
    request_datas = [[1, 2, 3], [4, 5, 6]]
    await vllm_backend.resume_kv_cache(request_datas)
    assert vllm_backend.generate.call_count == len(request_datas)
