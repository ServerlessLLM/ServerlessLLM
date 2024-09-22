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
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from serverless_llm.serve.backends.transformers_backend import (
    TransformersBackend,
)


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
    }


@pytest.fixture
def transformers_backend(backend_config):
    yield TransformersBackend(backend_config)


def test_init(transformers_backend, backend_config):
    assert transformers_backend.backend_config == backend_config
    assert not transformers_backend.model_initialized


@pytest.mark.asyncio
async def test_init_backend(transformers_backend, backend_config):
    with patch(
        "serverless_llm.serve.backends.transformers_backend.load_model"
    ) as mock_load_model:
        await transformers_backend.init_backend()
        mock_load_model.assert_called_once()
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = Path(
            os.path.join(
                storage_path,
                "transformers",
                backend_config["pretrained_model_name_or_path"],
            )
        ).resolve()
        device_map = backend_config.get("device_map", "auto")
        torch_dtype = backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        mock_load_model.assert_called_once_with(
            str(model_path),
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
        )


@pytest.fixture
def model():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cpu")
    yield model


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    yield tokenizer("test_prompt", return_tensors="pt")

@pytest.fixture
def encoder():
    encoder = AutoModel.from_pretrained("BAAI/bge-en-icl").to("cpu")
    yield encoder

@pytest.fixture
def encoder_tokenizer():
    encoder_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-en-icl")
    yield encoder_tokenizer(["test_prompt"], max_length=4096, padding=True, truncation=True, return_tensors="pt")


@pytest.mark.asyncio
async def test_generate(transformers_backend, model, tokenizer):
    with patch(
        "serverless_llm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "serverless_llm.serve.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        await transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }
        result = await transformers_backend.generate(input)
        assert "error" not in result
        assert "choices" in result and len(result["choices"]) == 1

@pytest.mark.asyncio
async def test_encode(transformers_backend, encoder, encoder_tokenizer):
    with patch(
        "serverless_llm.serve.backends.transformers_backend.load_model",
        return_value=encoder,
    ), patch(
        "serverless_llm.serve.backends.transformers_backend.TransformersBackend._encoder_tokenize",
        return_value=encoder_tokenizer,
    ):
        await transformers_backend.init_backend()
        input = {
                "model": "BAAI/bge-en-icl",
                "task_instruct": "Given a question, retrieve passages that answer the question",
                "query": [
                "Hi, How are you?"
                ]
            }
        result = await transformers_backend.encode(input)
        assert "error" not in result
        assert "data" in result and len(result["data"]) == 1


@pytest.mark.asyncio
async def test_generate_without_init(transformers_backend):
    with patch(
        "serverless_llm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ):
        request_data = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }
        response = await transformers_backend.generate(request_data)
        assert "error" in response
