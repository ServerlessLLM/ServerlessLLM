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
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from sllm.serve.backends.transformers_backend import (
    TransformersBackend,
)


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
    }


@pytest.fixture
def encoder_config():
    return {
        "pretrained_model_name_or_path": "BAAI/bge-small-en-v1.5",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModel",
    }


@pytest.fixture
def transformers_backend(backend_config):
    yield TransformersBackend(backend_config)


@pytest.fixture
def encoder_backend(encoder_config):
    yield TransformersBackend(encoder_config)


def test_init(transformers_backend, backend_config):
    assert transformers_backend.backend_config == backend_config
    assert not transformers_backend.model_initialized


def test_init_encoder(encoder_backend, encoder_config):
    assert encoder_backend.backend_config == encoder_config
    assert not encoder_backend.model_initialized


@pytest.mark.asyncio
async def test_init_backend(transformers_backend, backend_config):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model"
    ) as mock_load_model:
        await transformers_backend.init_backend()
        mock_load_model.assert_called_once()
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(
            "transformers",
            backend_config["pretrained_model_name_or_path"],
        )
        device_map = backend_config.get("device_map", "auto")
        torch_dtype = backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        hf_model_class = backend_config.get("hf_model_class", None)
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
        )


@pytest.mark.asyncio
async def test_init_encoder_backend(encoder_backend, encoder_config):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model"
    ) as mock_load_model:
        await encoder_backend.init_backend()
        mock_load_model.assert_called_once()
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(
            "transformers",
            encoder_config["pretrained_model_name_or_path"],
        )
        device_map = encoder_config.get("device_map", "auto")
        torch_dtype = encoder_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        hf_model_class = encoder_config.get("hf_model_class", None)
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
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
    encoder = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to("cpu")
    yield encoder


@pytest.fixture
def encoder_tokenizer():
    encoder_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    yield encoder_tokenizer(
        ["test_prompt"],
        max_length=4096,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


@pytest.mark.asyncio
async def test_generate(transformers_backend, model, tokenizer):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
async def test_encode(encoder_backend, encoder, encoder_tokenizer):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=encoder,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._encoder_tokenize",
        return_value=encoder_tokenizer,
    ):
        await encoder_backend.init_backend()
        input = {
            "model": "BAAI/bge-small-en-v1.5",
            "task_instruct": "Given a question, retrieve passages that answer the question",
            "input": ["Hi, How are you?"],
        }
        result = await encoder_backend.encode(input)
        assert "error" not in result
        assert "data" in result and len(result["data"]) == 1


@pytest.mark.asyncio
async def test_generate_without_init(transformers_backend):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model",
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
