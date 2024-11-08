import sys
import subprocess
import json

def test_store(model):
    try: 
        result = subprocess.run(
            f"sllm-cli deploy --model {model}",
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode == 0: 
            return True

        if result.stderr:
            print(f"Error storing model {model}: {result.stderr}")
            return False

        return False

    except Exception as e: 
        print(f"Error in handling model {model}: {e}")
        return False

def main():
    with open('supported_models.json', 'r') as f:
        MODELS = json.load(f)

    for model, model_info in MODELS.items():
        can_store = test_model(model)
        MODELS[model]["can_store"] = can_store

    with open('supported_models.json', 'w') as f:
        json.dump(MODELS, f, indent=4)

if __name__ == "__main__":
    main()
