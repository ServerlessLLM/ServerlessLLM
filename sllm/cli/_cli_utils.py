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
from typing import Dict, List, Optional, Union

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


def parse_lora_adapters(
    lora_adapters,
) -> Optional[Union[Dict[str, str], List[str]]]:
    """Parse LoRA adapters from various input formats.

    Accepts:
    - dict: returned as-is
    - str: "name1=path1 name2=path2" or "name1=path1,name2=path2"
    - list/tuple: each item parsed as above
    - None: returns None

    Returns:
    - Dict[str, str] if all items have name=path format
    - List[str] if items are just names (for deletion)
    - None if input is None
    """
    if lora_adapters is None:
        return None
    if isinstance(lora_adapters, dict):
        return lora_adapters

    # Normalize to list of strings
    if isinstance(lora_adapters, str):
        items = lora_adapters.replace(",", " ").split()
    elif isinstance(lora_adapters, (list, tuple)):
        items = []
        for item in lora_adapters:
            items.extend(str(item).replace(",", " ").split())
    else:
        items = [str(lora_adapters)]

    items = [m.strip() for m in items if m.strip()]

    if not items:
        return None

    # If all items have "=", return dict; otherwise return list
    if all("=" in item for item in items):
        return {
            name: path for item in items for name, path in [item.split("=", 1)]
        }
    else:
        return items


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
    from sllm.pylet_client import init_pylet_client
    from sllm.reconciler import init_reconciler
    from sllm.router import init_router
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

    # Initialize Router (single global instance)
    router = init_router(database=db)
    logger.info("Router initialized")

    # Initialize Autoscaler
    autoscaler = init_autoscaler(database=db)
    logger.info("Autoscaler initialized")

    # Connect Router to Autoscaler for metrics push
    router.set_autoscaler(autoscaler)

    # Create API Gateway app
    app = create_head_app(
        database=db,
        pylet_client=pylet_client,
        router=router,
        autoscaler=autoscaler,
        config=config,
    )

    # Configure uvicorn server
    uvicorn_config = Config(
        app, host=config.host, port=config.port, log_level="info"
    )
    uvicorn_server = Server(uvicorn_config)

    # Initialize background components
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
        app.state.storage_manager = storage_manager
        logger.info("StorageManager initialized")

        # Start sllm-store on all worker nodes (expensive, do it eagerly)
        logger.info("Starting sllm-store on all worker nodes...")
        init_success = await storage_manager.initialize()
        if init_success:
            logger.info("All sllm-store instances ready")
        else:
            logger.warning(
                "Some sllm-store instances failed to start. "
                "Model loading may be slower on affected nodes."
            )

    # Start Router
    await router.start()

    # Start HTTP server
    server_task = asyncio.create_task(uvicorn_server.serve())

    # Start Autoscaler
    autoscaler_task = asyncio.create_task(autoscaler.run())
    background_tasks.append(autoscaler_task)
    logger.info("Autoscaler started")

    # Initialize Reconciler (only if Pylet is available)
    if pylet_client and storage_manager:
        reconciler = init_reconciler(
            database=db,
            pylet_client=pylet_client,
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

        # Shutdown Router
        if router:
            await router.drain(timeout=10.0)
            await router.stop()

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
        adapters_dict = parse_lora_adapters(lora_adapters)
        if adapters_dict is not None:
            if isinstance(adapters_dict, list):
                print(
                    "[ERROR] LoRA adapters must be in <name>=<path> format for deploy."
                )
                sys.exit(1)
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
    url = f"{base_url.rstrip('/')}/deployments"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=config_data)
        if response.status_code == 200:
            data = response.json()
            deployment_id = data.get(
                "deployment_id",
                f"{config_data['model']}:{config_data.get('backend', 'vllm')}",
            )
            print(f"Deployment created: {deployment_id}")
            print(f"  View status: sllm status {deployment_id}")
        else:
            print(
                f"[❌ ERROR] Deploy failed with status {response.status_code}: {response.text}"
            )
            sys.exit(1)  # Exit with error code to indicate failure
    except Exception as e:
        print(f"[EXCEPTION] Failed to deploy: {str(e)}")
        sys.exit(1)  # Exit with error code to indicate failure


# ----------------------------- DELETE COMMAND ----------------------------- #
def delete_deployment(models, backend, lora_adapters=None):
    """Delete deployments or LoRA adapters.

    Args:
        models: List of model names
        backend: Backend framework (required)
        lora_adapters: Optional list of LoRA adapters to delete
    """
    if not models:
        print("[⚠️ WARNING] No model names provided for deletion.")
        return

    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    headers = {"Content-Type": "application/json"}

    if lora_adapters is not None and len(models) > 1:
        print(
            "[❌ ERROR] You can only delete one deployment when using "
            "--lora-adapters."
        )
        return

    for model in models:
        # Construct deployment_id from model and backend
        deployment_id = f"{model}:{backend}"

        try:
            if lora_adapters is not None:
                url = (
                    f"{base_url.rstrip('/')}/deployments/{deployment_id}"
                    "/adapters"
                )
                adapters = parse_lora_adapters(lora_adapters)
                response = requests.delete(
                    url, headers=headers, json={"lora_adapters": adapters}
                )
            else:
                url = f"{base_url.rstrip('/')}/deployments/{deployment_id}"
                response = requests.delete(url, headers=headers)

            if response.status_code in (200, 202):
                if lora_adapters is not None:
                    print(
                        f"[✅ SUCCESS] LoRA adapters for deployment "
                        f"'{deployment_id}' deleted successfully."
                    )
                else:
                    print(
                        f"[✅ SUCCESS] Deployment '{deployment_id}' "
                        "deletion initiated."
                    )
            else:
                print(
                    f"[❌ ERROR] Failed to delete deployment '{deployment_id}'. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            print(
                f"[EXCEPTION] Failed to delete deployment '{deployment_id}': "
                f"{str(e)}"
            )


# ----------------------------- STATUS COMMAND ----------------------------- #
def _fetch_status():
    """Fetch cluster status from /status endpoint."""
    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    url = f"{base_url.rstrip('/')}/status"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def show_status(deployment_id: Optional[str] = None, show_nodes: bool = False):
    """Show cluster status."""
    try:
        data = _fetch_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to server: {e}")
        sys.exit(1)

    if show_nodes:
        _show_nodes(data.get("nodes", []))
    elif deployment_id:
        _show_deployment_detail(data, deployment_id)
    else:
        _show_deployments_table(data.get("deployments", []))


def _show_deployments_table(deployments: List[dict]):
    """Show deployments in table format."""
    if not deployments:
        print("No deployments.")
        return

    # Column widths
    id_width = max(len("DEPLOYMENT"), max(len(d["id"]) for d in deployments))
    status_width = len("STATUS")
    replica_width = len("REPLICAS")

    # Header
    print(
        f"{'DEPLOYMENT':<{id_width}}  "
        f"{'STATUS':<{status_width}}  "
        f"{'REPLICAS':<{replica_width}}"
    )

    # Rows
    for d in deployments:
        ready = d.get("ready_replicas", 0)
        desired = d.get("desired_replicas", 0)
        replica_str = f"{ready}/{desired}"
        print(
            f"{d['id']:<{id_width}}  "
            f"{d['status']:<{status_width}}  "
            f"{replica_str:<{replica_width}}"
        )


def _show_deployment_detail(data: dict, deployment_id: str):
    """Show detailed info for a single deployment."""
    deployments = data.get("deployments", [])
    deployment = next(
        (d for d in deployments if d["id"] == deployment_id), None
    )

    if not deployment:
        print(f"Deployment '{deployment_id}' not found.")
        sys.exit(1)

    ready = deployment.get("ready_replicas", 0)
    desired = deployment.get("desired_replicas", 0)

    print(f"Deployment: {deployment['id']}")
    print(f"Status:     {deployment['status']}")
    print(f"Replicas:   {ready}/{desired}")
    print()

    instances = deployment.get("instances", [])
    if instances:
        print("Instances:")
        # Column widths
        id_width = max(len("ID"), max(len(i["id"]) for i in instances))
        node_width = max(
            len("NODE"), max(len(i.get("node", "-")) for i in instances)
        )
        ep_width = max(
            len("ENDPOINT"), max(len(i.get("endpoint", "-")) for i in instances)
        )
        status_width = max(
            len("STATUS"), max(len(i.get("status", "-")) for i in instances)
        )

        print(
            f"  {'ID':<{id_width}}  "
            f"{'NODE':<{node_width}}  "
            f"{'ENDPOINT':<{ep_width}}  "
            f"{'STATUS':<{status_width}}"
        )

        for inst in instances:
            print(
                f"  {inst['id']:<{id_width}}  "
                f"{inst.get('node', '-'):<{node_width}}  "
                f"{inst.get('endpoint', '-'):<{ep_width}}  "
                f"{inst.get('status', '-'):<{status_width}}"
            )
    else:
        print("Instances: (none)")


def _show_nodes(nodes: List[dict]):
    """Show nodes in table format."""
    if not nodes:
        print("No nodes.")
        return

    # Column widths
    name_width = max(len("NODE"), max(len(n.get("name", "-")) for n in nodes))
    status_width = max(
        len("STATUS"), max(len(n.get("status", "-")) for n in nodes)
    )
    gpu_width = len("GPUS")

    # Header
    print(
        f"{'NODE':<{name_width}}  "
        f"{'STATUS':<{status_width}}  "
        f"{'GPUS':<{gpu_width}}"
    )

    # Rows
    for n in nodes:
        available = n.get("available_gpus", 0)
        total = n.get("total_gpus", 0)
        gpu_str = f"{available}/{total}"
        print(
            f"{n.get('name', '-'):<{name_width}}  "
            f"{n.get('status', '-'):<{status_width}}  "
            f"{gpu_str:<{gpu_width}}"
        )
