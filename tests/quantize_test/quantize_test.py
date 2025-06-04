import json
import os
import unittest

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from sllm_store.transformers import load_model, save_model


@pytest.fixture(scope="session")
def model_name():
    return "facebook/opt-1.3b"


@pytest.fixture(scope="session")
def storage_path():
    model_folder = os.getenv("MODEL_FOLDER")
    if model_folder:
        return model_folder
    return pytest.tmp_path_factory.mktemp("models")


@pytest.fixture
def model_path(model_name, storage_path):
    return os.path.join(storage_path, model_name)


@pytest.fixture(scope="session", autouse=True)
def setup_models(storage_path):
    """Save the original model before tests."""
    os.makedirs(storage_path, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    save_model(model, os.path.join(storage_path, "facebook/opt-1.3b"))


# quantization configs
@pytest.fixture(
    params=[
        BitsAndBytesConfig(load_in_4bit=True),
        BitsAndBytesConfig(load_in_8bit=True),
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
    ]
)
def get_quantization_config(request):
    return request.param


# model configs
@pytest.fixture
def hf_model(get_quantization_config, model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_quantization_config,
        device_map="auto",
    )
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(params=[True, False], ids=["fully_parallel", "best_effort"])
def sllm_model(get_quantization_config, model_name, storage_path, request):
    model = load_model(
        model_name,
        storage_path=storage_path,
        quantization_config=get_quantization_config,
        fully_parallel=request.param,
        device_map="auto",
    )
    yield model
    del model
    torch.cuda.empty_cache()


# tests
def compare_state_dicts(transformers_model, sllm_model):
    """Compares model state dicts with support for partial quantization."""
    transformers_params = transformers_model.state_dict()
    sllm_params = sllm_model.state_dict()

    # ignore lm_head
    ignore_candidates = {"lm_head", "lm_head.weight"}
    ignore_keys = {
        k
        for k in ignore_candidates
        if k in transformers_params and k in sllm_params
    }

    # get comparable keys
    transformers_keys = set(transformers_params.keys()) - ignore_keys
    sllm_keys = set(sllm_params.keys()) - ignore_keys
    assert (
        transformers_keys == sllm_keys
    ), f"Key mismatch. Diff: {transformers_keys.symmetric_difference(sllm_keys)}"

    for key in transformers_keys:
        t_param = transformers_params[key]
        s_param = sllm_params[key]

        # Shape check
        assert t_param.shape == s_param.shape, (
            f"Shape mismatch for {key}: "
            f"Transformers={t_param.shape}, SLLM={s_param.shape}"
        )

        # Dtype check (accounts for mixed quantization)
        assert s_param.dtype == t_param.dtype, (
            f"Dtype mismatch for {key}: "
            f"Transformers={t_param.dtype}, SLLM={s_param.dtype}"
        )

        # individual parameter check
        assert torch.allclose(t_param, s_param, rtol=1e-02, atol=1e-03), (
            f"Param mismatch for {key}: "
            f"Transformers={t_param.dtype}, SLLM={s_param.dtype}"
        )

def test_valid_quantization(hf_model, sllm_model):
    compare_state_dicts(hf_model, sllm_model)
