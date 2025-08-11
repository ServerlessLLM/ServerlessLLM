# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
import subprocess
import sys

import click
import ray
import requests
import uvicorn

from sllm.app_lib import create_app
from sllm.controller import SllmController
from sllm.logger import init_logger

logger = init_logger(__name__)


# ----------------------------- START COMMAND ----------------------------- #
def start_server(
    host="0.0.0.0",
    port=8343,
    enable_storage_aware=False,
    enable_migration=False,
):
    """Start the SLLM server using Ray and uvicorn."""
    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            click.echo("[ℹ] Initializing Ray...")
            ray.init()
        else:
            click.echo("[ℹ] Ray already initialized")

        # Create the FastAPI app
        click.echo("[ℹ] Creating FastAPI application...")
        app = create_app()

        # Create and start the controller
        click.echo("[ℹ] Starting SLLM controller...")
        controller_cls = ray.remote(SllmController)
        controller = controller_cls.options(
            name="controller", num_cpus=1, resources={"control_node": 0.1}
        ).remote(
            {
                "enable_storage_aware": enable_storage_aware,
                "enable_migration": enable_migration,
            }
        )

        # Start the controller
        ray.get(controller.start.remote())
        click.echo("[✅] SLLM controller started successfully")

        # Start the uvicorn server
        click.echo(f"[🚀] Starting SLLM server on {host}:{port}...")
        uvicorn.run(app, host=host, port=port)

    except KeyboardInterrupt:
        click.echo("[ℹ] Shutting down SLLM server...")
        try:
            if "controller" in locals():
                ray.get(controller.shutdown.remote())
            click.echo("[✅] SLLM server shut down successfully")
        except Exception as e:
            click.echo(f"[⚠️] Warning during shutdown: {e}")
    except Exception as e:
        click.echo(f"[❌] Failed to start SLLM server: {e}")
        sys.exit(1)


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
    enable_lora=None,
    lora_adapters=None,
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
    if lora_adapters:
        # Only parse if not already a dict
        if isinstance(lora_adapters, dict):
            adapters_dict = lora_adapters
        else:
            adapters_dict = {}
            if isinstance(lora_adapters, str):
                items = lora_adapters.replace(",", " ").split()
            elif isinstance(lora_adapters, (list, tuple)):
                items = []
                for item in lora_adapters:
                    items.extend(item.replace(",", " ").split())
            else:
                items = [str(lora_adapters)]
            for module in items:
                module = module.strip()
                if not module:
                    continue
                if "=" not in module:
                    print(
                        f"[ERROR] Invalid LoRA module format: {module}. Expected <name>=<path>."
                    )
                    sys.exit(1)
                name, path = module.split("=", 1)
                adapters_dict[name] = path
        config_data.setdefault("backend_config", {})["lora_adapters"] = (
            adapters_dict
        )
    if enable_lora is not None:
        config_data.setdefault("backend_config", {})["enable_lora"] = (
            enable_lora
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
def delete_model(models, lora_adapters=None):
    if not models:
        print("[⚠️ WARNING] No model names provided for deletion.")
        return

    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    headers = {"Content-Type": "application/json"}

    if lora_adapters is not None and len(models) > 1:
        print(
            "[❌ ERROR] You can only delete one model when using --lora-adapters."
        )
        return

    for model in models:
        url = f"{base_url.rstrip('/')}/delete"
        data = {"model": model}
        # Robust lora_adapters parsing (same as deploy)
        if lora_adapters is not None:
            # Accept: demo-lora1 demo-lora2 OR demo-lora1=path ...
            if isinstance(lora_adapters, dict):
                adapters = lora_adapters
            else:
                # flatten and split
                if isinstance(lora_adapters, str):
                    items = lora_adapters.replace(",", " ").split()
                elif isinstance(lora_adapters, (list, tuple)):
                    items = []
                    for item in lora_adapters:
                        items.extend(item.replace(",", " ").split())
                else:
                    items = [str(lora_adapters)]
                # If all items have '=', parse as dict; else, treat as list
                if all("=" in module for module in items if module.strip()):
                    adapters = {}
                    for module in items:
                        module = module.strip()
                        if not module:
                            continue
                        name, path = module.split("=", 1)
                        adapters[name] = path
                else:
                    # Only adapter names
                    adapters = [
                        module.strip() for module in items if module.strip()
                    ]
            data["lora_adapters"] = adapters
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print(
                    f"[✅ SUCCESS] Delete request for '{model}' sent successfully."
                )
            else:
                print(
                    f"[❌ ERROR] Failed to delete '{model}'. Status: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            print(f"[EXCEPTION] Failed to delete '{model}': {str(e)}")


# ----------------------------- STATUS COMMAND ----------------------------- #
def show_status():
    """Query the information of registered models."""
    endpoint = "/v1/models"
    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    url = base_url.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json()
                models = data.get("models", [])
                if models:
                    print("Model status retrieved successfully:")
                    for model in models:
                        if isinstance(model, dict) and "id" in model:
                            print(f"- {model['id']}")
                        else:
                            print(f"- {model}")
                else:
                    print("No models currently deployed.")
            except ValueError:
                print("[❌ ERROR] Invalid JSON received from server.")
        else:
            print(
                f"[❌ ERROR] Failed with status {response.status_code}: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Failed to query status: {str(e)}")
