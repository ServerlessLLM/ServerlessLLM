import sys
import json
import subprocess
from typing import Optional, Dict, Any

from sllm_store.transformers import save_model

def store_test(model: str, model_path: str) -> Optional[str]:
    try: 
        save_model(model, model_path) 
        return None

    except Exception as e: 
        return str(e)


def main():
    failed_models = []
    MODEL_FOLDER = os.environ["MODEL_FOLDER"]

    try: 
        with open('supported_models.json', 'r') as f:
            models: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::error::Failed to read supported_models.json: {e}")
        return 1
         
    print("::group::Model Testing Results")
    for model, model_info in models.items():
        model_path = os.path.join(MODEL_FOLDER, model)

        print(f"Testing: {model}")
        error = store_test(model, model_path)

        if error: 
            print(f"::error file=supported_models.json::Model {model} failed: {error}")
            failed_models.append({
                "model": model,
                "error": error
            })
    print("::endgroup::")

    if failed_models:
        try:
            with open('failed_models.json', 'w') as f: # save failed models to use in inference_test
                json.dump(failed_models, f, indent=2)
        except IOError as e:
            print(f"::warning::Failed to save failed_models.json: {e}")

        print("::group::Failed Models Summary")
        for failure in failed_models:
            print(f"::error::❌ {failure['model']}: {failure['error']}")
        print("::endgroup::")
        return 1

    print("::notice::✅ All models tested successfully")
    return 0 


if __name__ == "__main__":
    sys.exit(main())
