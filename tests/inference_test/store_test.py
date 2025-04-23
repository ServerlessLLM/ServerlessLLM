import json
import os
import pathlib

import pytest
import torch
from transformers import AutoModelForCausalLM

from sllm_store.transformers import load_model, save_model

with open("supported_models.json") as fh:
    _MODELS = list(json.load(fh).keys())


@pytest.fixture(scope="session")
def storage_path(tmp_path_factory):
    env_path = os.getenv("MODEL_FOLDER")
    return (
        pathlib.Path(env_path)
        if env_path
        else tmp_path_factory.mktemp("models")
    )


@pytest.fixture(scope="session", params=_MODELS)
def model_name(request):
    return request.param


def _store_and_compare(model, storage_path):
    try:
        cached_path = storage_path / model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, trust_remote_code=True
        )
        save_model(hf_model, cached_path)
        test_model = load_model(
            model,
            storage_path=storage_path,
            device_map="auto",
            torch_dtype=torch.float16,
            fully_parallel=True,
        )
        for name, param in test_model.named_parameters():
            ref_param = hf_model.state_dict()[name]
            if param.dtype != ref_param.dtype:
                return f"dtype mismatch for {name}: {param.dtype} vs {ref_param.dtype}"
            if param.shape != ref_param.shape:
                return f"shape mismatch for {name}: {param.shape} vs {ref_param.shape}"
            if not torch.allclose(param.cpu(), ref_param.cpu(), atol=1e-6):
                return f"value mismatch for {name}"
        return None
    except Exception as exc:
        return str(exc)


def test_model_can_be_stored(model_name, storage_path, request):
    error = _store_and_compare(model_name, storage_path)
    if error:
        failures = request.session.__dict__.setdefault("_model_failures", [])
        failures.append({"model": model_name, "error": error})
    assert error is None, error


def pytest_sessionfinish(session, exitstatus):
    failures = session.__dict__.get("_model_failures", [])
    if failures:
        with open("failed_models.json", "w") as fh:
            json.dump(failures, fh, indent=2)
