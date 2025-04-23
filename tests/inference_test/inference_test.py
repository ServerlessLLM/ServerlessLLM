import json
import os
import pathlib

import pytest
import torch
from transformers import AutoTokenizer

from sllm_store.transformers import load_model

with open("supported_models.json") as fh:
    _MODELS = list(json.load(fh).keys())

try:
    with open("failed_models.json") as fh:
        _FAILED = {f["model"] for f in json.load(fh)}
except Exception:
    _FAILED = set()


@pytest.fixture(scope="session")
def storage_path(tmp_path_factory):
    env = os.getenv("MODEL_FOLDER")
    return pathlib.Path(env) if env else tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session", params=_MODELS)
def model_name(request):
    return request.param


def test_inference(model_name, storage_path):
    if model_name in _FAILED:
        pytest.skip("storage failure")
    model = load_model(
        model_name,
        storage_path=storage_path,
        device_map="auto",
        torch_dtype=torch.float16,
        fully_parallel=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    inp = tok("Hello, my dog is cute", return_tensors="pt").to("cuda")
    model.generate(**inp)

