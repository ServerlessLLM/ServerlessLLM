import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import torch

from sllm_store.transformers import load_model, save_model
from transformers import AutoModelForCausalLM


def store_test(model: str, model_folder: str) -> Optional[str]:
    try:
        model_path = os.path.join(model_folder, model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, trust_remote_code=True
        )
        save_model(hf_model, model_path)

        serverless_llm_model = load_model(
            model,
            device_map="auto",
            torch_dtype=torch.float16,
            storage_path=model_folder,
            fully_parallel=True,
        )

        for name, param in serverless_llm_model.named_parameters():
            ground_truth_param = hf_model.state_dict()[name]

            if param.dtype != ground_truth_param.dtype:
                return f"param {name} dtype mismatch: {param.dtype} vs {ground_truth_param.dtype}"

            if param.shape != ground_truth_param.shape:
                return f"param {name} shape mismatch: {param.shape} vs {ground_truth_param.shape}"

            if not torch.allclose(
                param.cpu(), ground_truth_param.cpu(), atol=1e-3
            ):
                return f"param {name} value mismatch"

        return None

    except Exception as e:
        return str(e)


def main() -> int:
    failed_models: List[Dict[str, str]] = []
    MODEL_FOLDER = os.environ["MODEL_FOLDER"]

    try:
        with open("supported_models.json", "r") as f:
            models: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::error::Failed to read supported_models.json: {e}")
        return 1

    print("::group::Model Testing Results")
    for model, model_info in models.items():
        print(f"Testing: {model}")
        error = store_test(model, MODEL_FOLDER)

        if error:
            print(
                f"::error file=supported_models.json::Model {model} failed: {error}"
            )
            failed_models.append({"model": model, "error": error})
    print("::endgroup::")

    if failed_models:
        try:
            with open(
                "failed_models.json", "w"
            ) as f:  # save failed models to use in inference_test
                json.dump(failed_models, f, indent=2)
        except IOError as e:
            print(f"::warning::Failed to save failed_models.json: {e}")

        print("::group::Failed Models Summary")
        for failure in failed_models:
            print(f"::error::❌ {failure['model']}: {failure['error']}")
        print("::endgroup::")
        return 1

    print("::notice::✅ All models stored successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
