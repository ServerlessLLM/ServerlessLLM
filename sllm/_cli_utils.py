import json
import os
import subprocess
import sys

import click
import requests


# ----------------------------- START COMMAND ----------------------------- #
def start_server():
    """Start the SLLM server using docker-compose."""
    compose_file = os.getenv(
        "SLLM_COMPOSE_FILE", "examples/docker/docker-compose.yml"
    )
    compose_file = os.path.abspath(compose_file)

    if not os.path.exists(compose_file):
        click.echo(f"[❌] Cannot find docker-compose.yml at {compose_file}")
        return

    try:
        click.echo(f"[ℹ] Starting services using {compose_file}...")
        subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d"], check=True
        )
        click.echo(f"[🚀] SLLM server started successfully.")
    except Exception as e:
        click.echo(f"[❌] Failed to start services: {e}")


# ----------------------------- DEPLOY COMMAND ----------------------------- #
def read_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Config file {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] JSON decode error in config file {config_path}.")
        sys.exit(1)


def deep_update(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def deploy_model(
    model,
    config=None,
    backend=None,
    num_gpus=None,
    target=None,
    min_instances=None,
    max_instances=None,
    adapter_name=None,
    precision=None,
):
    default_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "default_config.json")
    )

    if not os.path.exists(default_config_path):
        print(f"[ERROR] Default config not found at {default_config_path}")
        return

    config_data = read_config(default_config_path)

    if config:
        # Try to find the config file in multiple locations
        config_path = None
        search_paths = [
            os.path.abspath(
                config
            ),  # Absolute or relative to current directory
            os.path.join(
                os.path.dirname(__file__), config
            ),  # Relative to sllm package
            os.path.expanduser(f"~/.sllm/{config}"),  # User config directory
            os.path.join("/etc/sllm", config),  # System config directory
        ]

        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

        if not config_path:
            print(
                f"[ERROR] Config file '{config}' not found in any of these locations:"
            )
            for path in search_paths:
                print(f"  - {path}")
            print("")
            print("Available config files:")
            # Show available config files in current directory
            current_dir_configs = [
                f for f in os.listdir(".") if f.endswith(".json")
            ]
            if current_dir_configs:
                for cf in current_dir_configs:
                    print(f"  - {cf}")
            else:
                print("  - No JSON config files found in current directory")

            # Show the default config location
            default_config_rel = os.path.join(
                os.path.dirname(__file__), "default_config.json"
            )
            if os.path.exists(default_config_rel):
                print(f"  - {default_config_rel} (default config)")
            sys.exit(1)  # Exit with error code

        user_config = read_config(config_path)
        config_data = deep_update(config_data, user_config)

    if model:
        config_data["model"] = model
        config_data.setdefault("backend_config", {})[
            "pretrained_model_name_or_path"
        ] = model

    # 验证模型名称是否存在
    if not config_data.get("model"):
        print("[❌ ERROR] Model name is required!")
        print("Please specify the model in one of these ways:")
        print("  1. Command line: --model MODEL_NAME")
        print("  2. Config file: include 'model' field in your JSON config")
        print("  3. Both: --config CONFIG_FILE --model MODEL_NAME")
        print("")
        print("Examples:")
        print("  sllm deploy --model microsoft/DialoGPT-small")
        print("  sllm deploy --config my_config.json")
        print("  sllm deploy --config my_config.json --model facebook/opt-1.3b")
        sys.exit(1)

    if backend:
        config_data["backend"] = backend
    if num_gpus is not None:
        config_data["num_gpus"] = num_gpus
    if target is not None:
        config_data.setdefault("auto_scaling_config", {})["target"] = target
    if min_instances is not None:
        config_data["auto_scaling_config"]["min_instances"] = min_instances
    if max_instances is not None:
        config_data["auto_scaling_config"]["max_instances"] = max_instances
    if adapter_name:
        config_data.setdefault("backend_config", {})["adapter_name"] = (
            adapter_name
        )
    if precision:
        config_data.setdefault("backend_config", {})["precision"] = precision

    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    url = f"{base_url.rstrip('/')}/register"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=config_data)
        if response.status_code == 200:
            print(
                f"[✅ SUCCESS] Model '{config_data['model']}' deployed successfully."
            )
        else:
            print(
                f"[❌ ERROR] Deploy failed with status {response.status_code}: {response.text}"
            )
            sys.exit(1)  # Exit with error code to indicate failure
    except Exception as e:
        print(f"[EXCEPTION] Failed to deploy: {str(e)}")
        sys.exit(1)  # Exit with error code to indicate failure


# ----------------------------- DELETE COMMAND ----------------------------- #
def delete_model(models):
    if not models:
        print("[⚠️ WARNING] No model names provided for deletion.")
        return

    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    url = f"{base_url.rstrip('/')}/delete"
    headers = {"Content-Type": "application/json"}

    for model in models:
        data = {"model": model}
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print(f"[✅ SUCCESS] Model '{model}' deleted successfully.")
            else:
                print(
                    f"[❌ ERROR] Failed to delete model '{model}'. Status: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            print(f"[EXCEPTION] Failed to delete model '{model}': {str(e)}")


# ----------------------------- STATUS COMMAND ----------------------------- #
def show_status():
    """Query the information of registered models."""
    endpoint = "v1/models"
    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343/")
    url = base_url.rstrip("/") + "/" + endpoint
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json()
                models = data.get("models", [])
                if not models:
                    click.echo("[ℹ] No models currently deployed.")
                else:
                    click.echo("[✅ SUCCESS] Model status retrieved:")
                    for model in models:
                        model_id = model.get("id", "<unknown>")
                        click.echo(f"- {model_id}")
            except ValueError:
                click.echo("[❌ ERROR] Invalid JSON received from server.")
        else:
            click.echo(
                f"[❌ ERROR] Failed with status {response.status_code}: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        click.echo(f"[EXCEPTION] Failed to query status: {str(e)}")
