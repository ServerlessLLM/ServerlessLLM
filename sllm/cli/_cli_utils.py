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

import click
import requests
import uvicorn
from uvicorn import Config, Server

from sllm.api_gateway import create_app as create_head_app
from sllm.autoscaler import AutoScaler
from sllm.dispatcher import Dispatcher
from sllm.kv_store import RedisStore
from sllm.logger import init_logger
from sllm.model_manager import ModelManager
from sllm.worker.api import create_worker_app
from sllm.worker.heartbeat import run_heartbeat_loop
from sllm.worker.instance_manager import InstanceManager
from sllm.worker.utils import benchmark_static_hardware
from sllm.worker_manager import WorkerManager

logger = init_logger(__name__)


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
    host="0.0.0.0",
    port=8343,
    redis_host=None,
    redis_port=None,
):
    """Start the SLLM head node (control plane)."""
    # Use environment variables if not explicitly provided
    if redis_host is None:
        redis_host = os.environ.get("REDIS_HOST", "redis")
    if redis_port is None:
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))

    logger.info(f"Using Redis host {redis_host}")
    logger.info(f"Using Redis port {redis_port}")

    try:
        asyncio.run(_run_head_node(host, port, redis_host, redis_port))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        click.echo(f"Failed to start head node: {e}")
        sys.exit(1)


async def _run_head_node(host, port, redis_host, redis_port):
    """Async implementation of head node startup."""
    logger.info("Starting head node...")

    store = RedisStore(host=redis_host, port=redis_port)
    await store.initialize_store(reset_on_start=True, full_reset=True)
    model_manager = ModelManager(store)
    worker_manager = WorkerManager(store, config={"prune_interval": 15})
    autoscaler = AutoScaler(store=store)
    dispatcher = Dispatcher(store)

    app = create_head_app(
        worker_manager=worker_manager,
        model_manager=model_manager,
        dispatcher=dispatcher,
    )

    uvicorn_config = Config(app, host=host, port=port, log_level="info")
    uvicorn_server = Server(uvicorn_config)

    worker_manager.start()
    dispatcher.start()
    autoscaler_task = asyncio.create_task(autoscaler.run_scaling_loop())
    dispatcher_task = asyncio.create_task(dispatcher.run_consumer_loop())
    server_task = asyncio.create_task(uvicorn_server.serve())

    try:
        logger.info("Head node services started.")
        await asyncio.gather(autoscaler_task, dispatcher_task, server_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Shutting down head node...")
        autoscaler.shutdown()
        await dispatcher.shutdown()
        await worker_manager.shutdown()
        uvicorn_server.should_exit = True
        await asyncio.sleep(2)
        await store.close()
        logger.info("Head node shutdown complete.")


# ----------------------------- START WORKER ----------------------------- #
def start_worker(host, port, head_node_url):
    """Start the SLLM worker node."""
    if not head_node_url:
        click.echo("Error: --head-node-url is required for worker mode")
        sys.exit(1)

    try:
        asyncio.run(_run_worker_node(host, port, head_node_url))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        click.echo(f"Failed to start worker node: {e}")
        sys.exit(1)


async def _run_worker_node(host, port, head_node_url):
    """Async implementation of worker node startup."""
    logger.info("Starting worker node...")

    # Separate bind address from advertise address
    advertise_ip = get_advertise_ip()
    logger.info(f"Worker binding to {host}:{port}, advertising as {advertise_ip}:{port}")

    static_hardware_info = benchmark_static_hardware()
    instance_manager = InstanceManager(node_ip=advertise_ip)
    worker_app = create_worker_app(instance_manager)
    uvicorn_config = Config(worker_app, host=host, port=port, log_level="info")
    uvicorn_server = Server(uvicorn_config)

    server_task = asyncio.create_task(uvicorn_server.serve())
    heartbeat_task = asyncio.create_task(
        run_heartbeat_loop(
            instance_manager=instance_manager,
            head_node_url=head_node_url,
            node_ip=advertise_ip,
            static_hardware_info=static_hardware_info,
            app_state=worker_app.state,
            worker_port=port,
        )
    )

    try:
        logger.info(f"Worker node started on {host}:{port}.")
        await asyncio.gather(server_task, heartbeat_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Shutting down worker node...")
        server_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(
            server_task, heartbeat_task, return_exceptions=True
        )
        logger.info("Worker node shutdown complete.")


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
                print(f"Model status: {data}")
            except ValueError:
                print("[❌ ERROR] Invalid JSON received from server.")
        else:
            print(
                f"[❌ ERROR] Failed with status {response.status_code}: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Failed to query status: {str(e)}")
