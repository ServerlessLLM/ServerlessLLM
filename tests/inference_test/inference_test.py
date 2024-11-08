import sys
import json
import requests
import subprocess
from typing import Optional, Dict, Any


def cleanup_models(models: List[str]) -> None:
    print("::group::Cleanup")
    for model in models:
        try:
            print(f"Cleaning up model: {model}")
            subprocess.run(
                ["sllm-cli", "delete", model],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"::warning::Failed to cleanup {model}: {e.stderr}")
    print("::endgroup::")


def test_inference(model: str) -> bool:
    url = 'http://127.0.0.1:8343/v1/chat/completions'
    headers = {"Content-Type": "application/json"}
    query = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=query
        )
        response.raise_for_status()
        return True

    except Exception as e:
        print(f"::error::Model {model} inference failed: {str(e)}")
        return False


def main() -> int:
    failed_models = []

    try:
        with open('supported_models.json', 'r') as f:
            models: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::error::Failed to read supported_models.json: {e}")
        return 1

    try:
        with open('failed_models.json', 'r') as f:
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
            failed_models.append({
                "model": model,
                "error": "Inference failed"
            })
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
