import os
import pytest
import unittest 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sllm_store import save_model, load_model
import json

@pytest.fixture
def model_name():
    return "facebook/opt-1.3b"

@pytest.fixture(scope="session")
def storage_path(tmp_path_factory):
    return tmp_path_factory.mktemp("models")

@pytest.fixture
def model_path(model_name, storage_path):
    return os.path.join(storage_path, model_name)


def save_hf_model(model_name, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    save_model(model, model_path)


@pytest.fixture(scope="session", autouse=True)
def setup_models(model_name, storage_path):
    """Save original model before tests"""
    os.makedirs(storage_path, exist_ok=True)
    save_hf_model(model_name, os.path.join(storage_path, model_name))


# quantization configs
@pytest.fixture(params=[
    BitsAndBytesConfig(load_in_4bit=True),
    BitsAndBytesConfig(load_in_8bit=True),
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
])
def valid_quantization_config(request):
    return request.param

@pytest.fixture(params=[
    BitsAndBytesConfig(load_in_4bit=True, load_in_8bit=True),  
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="invalid_type"),
    "not_a_config_object" 
])
def invalid_quantization_config(request):
    return request.param


# model configs
@pytest.fixture
def hf_model(valid_quantization_config, model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=valid_quantization_config,
        device_map="auto"
    )
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def sllm_model(valid_quantization_config, model_name, storage_path):
    model = load_model(
        model_name,
        storage_path=storage_path,
        quantization_config=valid_quantization_config,
        device_map="auto"
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
    ignore_candidates = {'lm_head', 'lm_head.weight'}
    ignore_keys = {
        k for k in ignore_candidates 
        if k in transformers_params and k in sllm_params
    }
    
    # get comparable keys 
    transformers_keys = set(transformers_params.keys()) - ignore_keys
    sllm_keys = set(sllm_params.keys()) - ignore_keys
    assert transformers_keys == sllm_keys, f"Key mismatch. Diff: {transformers_keys.symmetric_difference(sllm_keys)}"

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

def test_valid_quantization(hf_model, sllm_model):
    compare_state_dicts(hf_model, sllm_model)

def test_invalid_quantization(invalid_quantization_config, model_name, storage_path):
    """Test invalid configs raise errors"""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        load_model(
            model_name,
            storage_path=storage_path,
            quantization_config=invalid_quantization_config,
            device_map="auto"
        )
