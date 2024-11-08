import sys
import json
import subprocess
from typing import Optional, Dict, Any

def store_test(model: str) -> Optional[str]:
    try: 
        result = subprocess.run(
            f"sllm-cli deploy --model {model}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0: 
            return None

        return str(result.stderr) if result.stderr else str(result.stdout)

    except subprocess.TimeoutExpired:
        return "Deployment timed out after 10 minutes"
        
    except FileNotFoundError:
        return "sllm-cli command not found"
            
    except subprocess.SubprocessError as e:
        return str(e)

    except Exception as e: 
        return str(e)


def main():
    failed_models = []

    try: 
        with open('supported_models.json', 'r') as f:
            models: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"::error::Failed to read supported_models.json: {e}")
        return 1
         
    print("::group::Model Testing Results")
    for model, model_info in models.items():
        print(f"Testing: {model}")
        error = store_test(model)

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
