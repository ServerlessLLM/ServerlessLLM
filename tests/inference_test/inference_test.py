import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import requests
import torch
from transformers import AutoTokenizer

from sllm_store.transformers import load_model


def cleanup_models(models: List[str]) -> None:
    print("::group::Cleanup")
    for model in models:
        try:
            print(f"Cleaning up model: {model}")
            subprocess.run(
                ["sllm-cli", "delete", model],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"::warning::Failed to cleanup {model}: {e.stderr}")
    print("::endgroup::")


def test_inference(model_name: str) -> bool:
    try:
        model = load_model(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            storage_path=os.getenv("MODEL_FOLDER"),  
            fully_parallel=True,
        )
    except Exception as e:
        print(f"::error::Model {model_name} loading failed: {str(e)}")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(
            "cuda"
        )
        outputs = model.generate(**inputs)
        return True

    except Exception as e:
        print(f"::error::Model {model_name} inference failed: {str(e)}")
        return False


def main() -> int:
    failed_models: List[Dict[str, str]] = []

    try:
        with open("supported_models.json", "r") as f:
            models: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::error::Failed to read supported_models.json: {e}")
        return 1

    try:
        with open("failed_models.json", "r") as f:
            failed_storage = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::warning::Failed to read failed_models.json: {e}")
        failed_storage = []

    print("::group::Inference Testing Results")
    for model, model_info in models.items():
        if any(f["model"] == model for f in failed_storage):
            print(f"Skipping {model} - failed storage test")
            continue

        print(f"Testing inference: {model}")
        success = test_inference(model)

        if not success:
            failed_models.append({"model": model, "error": "Inference failed"})
    print("::endgroup::")

    if failed_models:
        print("::group::Failed Models Summary")
        for failure in failed_models:
            print(f"::error::❌ {failure['model']}: {failure['error']}")
        print("::endgroup::")
        return 1

    print("::notice::✅ All inference tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
