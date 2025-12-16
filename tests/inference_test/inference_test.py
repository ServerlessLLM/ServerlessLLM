import json
import os
import pathlib

import pytest
import torch
from transformers import AutoTokenizer

from sllm_store.transformers import load_model

with open("tests/inference_test/supported_models.json") as fh:
    _SUPPORTED_MODELS = json.load(fh)

models = list(_SUPPORTED_MODELS.keys())

try:
    with open("tests/inference_test/failed_models.json") as fh:
        _FAILED = {f["model"] for f in json.load(fh)}
except Exception:
    _FAILED = set()


@pytest.fixture(scope="session")
def storage_path(tmp_path_factory):
    env = os.getenv("MODEL_FOLDER")
    return pathlib.Path(env) if env else tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session", params=models, ids=models)
def model_name(request):
    return request.param


def test_inference(model_name, storage_path):
    if model_name in _FAILED:
        pytest.skip("storage failure")
    model_cfg = _SUPPORTED_MODELS.get(model_name, {})
    hf_model_class = model_cfg.get("hf_model_class", "AutoModelForCausalLM")
    model = load_model(
        model_name,
        storage_path=storage_path,
        device_map="auto",
        torch_dtype=torch.float16,
        fully_parallel=True,
        hf_model_class=hf_model_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    print(
        f"{model_name} output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}"
    )
