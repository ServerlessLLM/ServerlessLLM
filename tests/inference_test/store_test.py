import json
import os
import pathlib

import pytest
import torch
from transformers import AutoModelForCausalLM

from sllm_store.transformers import load_model, save_model

with open("tests/inference_test/supported_models.json") as fh:
    models = list(json.load(fh).keys())


@pytest.fixture(scope="session", params=models, ids=models)
def model_name(request):
    return request.param


@pytest.fixture(scope="session")
def storage_path(tmp_path_factory):
    env = os.getenv("MODEL_FOLDER")
    return pathlib.Path(env) if env else tmp_path_factory.mktemp("models")


def store_and_compare(model_name, storage_path):
    try:
        os.makedirs(storage_path, exist_ok=True)
        cache_dir = storage_path / model_name
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        save_model(hf_model, cache_dir)

        test_model = load_model(
            model_name,
            storage_path=storage_path,
            device_map="auto",
            torch_dtype=torch.float16,
            fully_parallel=True,
        )

        for name, param in test_model.named_parameters():
            ref = hf_model.state_dict()[name]
            if param.dtype != ref.dtype:
                return (
                    f"dtype mismatch for {name}: {param.dtype} vs {ref.dtype}"
                )
            if param.shape != ref.shape:
                return (
                    f"shape mismatch for {name}: {param.shape} vs {ref.shape}"
                )
            if not torch.allclose(param.cpu(), ref.cpu(), atol=1e-6):
                return f"value mismatch for {name}"

        del hf_model, test_model
        torch.cuda.empty_cache()
        return None

    except Exception as exc:
        return str(exc)


def test_model_can_be_stored(model_name, storage_path, request):
    error = store_and_compare(model_name, storage_path)
    if error:
        failures = request.session.__dict__.setdefault("_model_failures", [])
        failures.append({"model": model_name, "error": error})
    assert error is None, error


def pytest_sessionfinish(session, exitstatus):
    failures = session.__dict__.get("_model_failures", [])
    if failures:
        with open("tests/inference_test/failed_models.json", "w") as fh:
            json.dump(failures, fh, indent=2)
