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
import asyncio
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import click
import requests
import uvicorn
from uvicorn import Config, Server

from sllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class HeadConfig:
    """Configuration for v1-beta head node."""

    host: str = "0.0.0.0"
    port: int = 8343
    pylet_endpoint: str = "http://localhost:8000"
    database_path: str = "/var/lib/sllm/state.db"
    storage_path: str = "/models"


# Global config instance
_head_config: Optional[HeadConfig] = None


def get_head_config() -> HeadConfig:
    """Get the global head config instance."""
    global _head_config
    if _head_config is None:
        _head_config = HeadConfig()
    return _head_config


def get_advertise_ip() -> str:
    """Get the IP address to advertise to other services.

    Uses NODE_IP environment variable if set, otherwise auto-detects
    the container/host IP address.
    """
    # Allow explicit override for multi-machine setups
    node_ip = os.environ.get("NODE_IP")
    if node_ip:
        return node_ip

    try:
        # Auto-detect container IP by connecting to a remote address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            # Fallback: use hostname resolution
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            # Last resort: localhost (not ideal for multi-container)
            logger.warning("Could not auto-detect IP address, using localhost")
            return "127.0.0.1"


# ----------------------------- START COMMAND ----------------------------- #
def start_head(
    host: str = "0.0.0.0",
    port: int = 8343,
    pylet_endpoint: str = "http://localhost:8000",
    database_path: str = "/var/lib/sllm/state.db",
    storage_path: str = "/models",
):
    """Start the SLLM head node (control plane) - v1-beta with Pylet."""
    global _head_config
    _head_config = HeadConfig(
        host=host,
        port=port,
        pylet_endpoint=pylet_endpoint,
        database_path=database_path,
        storage_path=storage_path,
    )

    logger.info("=" * 60)
    logger.info("ServerlessLLM v1-beta Head Node")
    logger.info("=" * 60)
    logger.info(f"Pylet endpoint: {pylet_endpoint}")
    logger.info(f"Database path: {database_path}")
    logger.info(f"Storage path: {storage_path}")
    logger.info("=" * 60)

    try:
        asyncio.run(_run_head_node_v1beta())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Failed to start head node")
        click.echo(f"Failed to start head node: {e}")
        sys.exit(1)


async def _run_head_node_v1beta():
    """Async implementation of v1-beta head node startup."""
    config = get_head_config()
    logger.info("Starting head node (v1-beta)...")

    # Import v1-beta components
    from sllm.api_gateway import create_app as create_head_app
    from sllm.autoscaler import init_autoscaler
    from sllm.database import init_database
    from sllm.lb_registry import get_lb_registry
    from sllm.pylet_client import init_pylet_client
    from sllm.reconciler import init_reconciler
    from sllm.storage_manager import init_storage_manager

    # Initialize SQLite database
    logger.info(f"Initializing database at {config.database_path}")
    db = init_database(config.database_path)

    # Initialize Pylet client
    logger.info(f"Connecting to Pylet at {config.pylet_endpoint}")
    pylet_client = None
    try:
        pylet_client = await init_pylet_client(config.pylet_endpoint)
        logger.info("Connected to Pylet successfully")
    except Exception as e:
        logger.warning(f"Failed to connect to Pylet: {e}")
        logger.warning(
            "Starting without Pylet - some features will be unavailable"
        )

    # Create API Gateway app (this also initializes the lb_registry)
    app = create_head_app(
        database=db,
        pylet_client=pylet_client,
        config=config,
    )

    # Get the LB registry from app state (initialized by create_app)
    # We need to wait for app lifespan to run, so we'll get it from app.state later
    # For now, use the global registry

    # Configure uvicorn server
    uvicorn_config = Config(
        app, host=config.host, port=config.port, log_level="info"
    )
    uvicorn_server = Server(uvicorn_config)

    # Initialize background components
    autoscaler = None
    reconciler = None
    storage_manager = None
    background_tasks = []

    if pylet_client:
        # Initialize StorageManager
        # Use advertise IP (not bind address) for external accessibility
        advertise_ip = get_advertise_ip()
        head_url = f"http://{advertise_ip}:{config.port}"
        storage_manager = init_storage_manager(
            database=db,
            pylet_client=pylet_client,
            storage_path=config.storage_path,
            head_url=head_url,
        )
        await storage_manager.recover_from_db()
        logger.info("StorageManager initialized")

    # Start HTTP server first (so LB registry is initialized)
    server_task = asyncio.create_task(uvicorn_server.serve())

    # Give the server a moment to start and initialize the LB registry
    await asyncio.sleep(0.5)

    # Now initialize autoscaler and reconciler with the LB registry
    lb_registry = get_lb_registry()

    # Initialize Autoscaler
    autoscaler = init_autoscaler(database=db, lb_registry=lb_registry)
    autoscaler_task = asyncio.create_task(autoscaler.run())
    background_tasks.append(autoscaler_task)
    logger.info("Autoscaler started")

    # Initialize Reconciler (only if Pylet is available)
    if pylet_client and storage_manager:
        reconciler = init_reconciler(
            database=db,
            pylet_client=pylet_client,
            lb_registry=lb_registry,
            storage_manager=storage_manager,
            storage_path=config.storage_path,
        )
        await reconciler.start()
        reconciler_task = asyncio.create_task(reconciler.run())
        background_tasks.append(reconciler_task)
        logger.info("Reconciler started")

    try:
        logger.info(f"Head node started on {config.host}:{config.port}")
        logger.info("Head node services started (v1-beta).")
        await server_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Shutting down head node...")

        # Shutdown components
        if autoscaler:
            autoscaler.shutdown()
        if reconciler:
            await reconciler.stop()

        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        uvicorn_server.should_exit = True
        await asyncio.sleep(1)

        # Shutdown LB registry
        if lb_registry:
            await lb_registry.shutdown()

        # Close database connection
        db.close()

        logger.info("Head node shutdown complete.")


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
def delete_model(models, backend=None, lora_adapters=None):
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

        # Add backend to request if specified
        if backend is not None:
            data["backend"] = backend

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
