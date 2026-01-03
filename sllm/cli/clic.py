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
import os
import sys
from typing import Optional

import click
import requests

from sllm.cli._cli_utils import (
    delete_deployment,
    deploy_model,
    parse_lora_adapters,
    show_status,
    start_head,
)


@click.group()
def cli():
    """Unified CLI for ServerlessLLM."""
    pass


@cli.command()
@click.option(
    "--model",
    help="Model name from HuggingFace model hub (required if not specified in config file)",
)
@click.option("--config", help="Path to configuration file")
@click.option("--backend", help="Backend framework (e.g., vllm, transformers)")
@click.option(
    "--num-gpus", type=int, help="Number of GPUs to use for the model"
)
@click.option("--target", type=int, help="Target number of requests per second")
@click.option(
    "--min-instances", type=int, help="Minimum number of model instances"
)
@click.option(
    "--max-instances", type=int, help="Maximum number of model instances"
)
@click.option(
    "--lora-adapters",
    help=(
        'List of LoRA adapters, e.g. "demo_lora1=... demo_lora2=..." or "demo_lora1=...,demo_lora2=...". '
        "Must be wrapped in quotes if using spaces or commas. Each adapter must be in <name>=<path> format."
    ),
)
@click.option(
    "--enable-lora",
    is_flag=True,
    default=False,
    help="Enable LoRA support for the model",
)
@click.option(
    "--precision",
    help="Model precision for quantization (e.g., int8, fp4, nf4)",
)
def deploy(
    model,
    config,
    backend,
    num_gpus,
    target,
    min_instances,
    max_instances,
    lora_adapters,
    enable_lora,
    precision,
):
    """Deploy a model using a config file or model name.

    Either --model or a config file with a model specified is required.
    Command line options override values from the config file.
    """
    adapters_dict = parse_lora_adapters(lora_adapters)
    if adapters_dict is not None and isinstance(adapters_dict, list):
        click.echo(
            "[ERROR] LoRA adapters must be in <name>=<path> format for deploy."
        )
        return

    deploy_model(
        model=model,
        config=config,
        backend=backend,
        num_gpus=num_gpus,
        target=target,
        min_instances=min_instances,
        max_instances=max_instances,
        lora_adapters=adapters_dict,
        enable_lora=enable_lora,
        precision=precision,
    )


@cli.command()
@click.argument("models", nargs=-1)
@click.option(
    "--backend",
    required=True,
    help="Backend framework (e.g., vllm, sglang). Required to identify the deployment.",
)
@click.option("--lora-adapters", help="LoRA adapters to delete.")
def delete(models, backend, lora_adapters):
    """Delete deployments, or remove only the LoRA adapters."""
    delete_deployment(
        models,
        backend=backend,
        lora_adapters=lora_adapters if lora_adapters else None,
    )


@cli.command()
@click.option("--model", required=True, help="Model name from HuggingFace model hub")
@click.option("--backend", default="vllm", help="Backend framework (e.g., vllm, sglang)")
@click.option("--num-nodes", default=1, type=int, help="Number of nodes to download to")
def pull(model, backend, num_nodes):
    """Pre-download a model to cluster nodes."""
    base_url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
    url = f"{base_url.rstrip('/')}/pull"
    click.echo(f"Pulling {model} to {num_nodes} node(s)...")
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"model": model, "backend": backend, "num_nodes": num_nodes},
        )
        if response.status_code == 200:
            nodes = response.json().get("nodes", [])
            click.echo(f"Model pulled to {len(nodes)} node(s): {', '.join(nodes)}")
        else:
            click.echo(f"[ERROR] Pull failed ({response.status_code}): {response.text}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"[ERROR] Failed to pull model: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    type=str,
    help="Host IP for the API Gateway.",
)
@click.option(
    "--port", default=8343, type=int, help="Port for the API Gateway."
)
@click.option(
    "--pylet-endpoint",
    default=lambda: os.getenv("PYLET_ENDPOINT", "http://localhost:8000"),
    type=str,
    help="Pylet head endpoint. Default: http://localhost:8000",
)
@click.option(
    "--database-path",
    default=lambda: os.getenv("SLLM_DATABASE_PATH", "/var/lib/sllm/state.db"),
    type=str,
    help="SQLite database path. Default: /var/lib/sllm/state.db",
)
@click.option(
    "--storage-path",
    default=lambda: os.getenv("STORAGE_PATH", "/models"),
    type=str,
    help="Model storage path. Default: /models",
)
def start(
    host,
    port,
    pylet_endpoint,
    database_path,
    storage_path,
):
    """Start the SLLM head node (control plane)."""
    start_head(
        host=host,
        port=port,
        pylet_endpoint=pylet_endpoint,
        database_path=database_path,
        storage_path=storage_path,
    )


@cli.command()
@click.argument("deployment_id", required=False)
@click.option(
    "--nodes",
    is_flag=True,
    help="Show cluster nodes instead of deployments.",
)
def status(deployment_id: Optional[str], nodes: bool):
    """Show cluster status.

    With no arguments, shows all deployments.
    With DEPLOYMENT_ID, shows detailed info for that deployment.
    With --nodes, shows cluster nodes.
    """
    show_status(deployment_id=deployment_id, show_nodes=nodes)


@cli.command()
@click.argument("instance_id")
@click.option("-f", "--follow", is_flag=True, help="Follow log output.")
def logs(instance_id: str, follow: bool):
    """View instance logs."""
    import asyncio

    try:
        from pylet.client import PyletClient
    except ImportError:
        click.echo("Error: pylet not installed.")
        sys.exit(1)

    endpoint = os.getenv("PYLET_ENDPOINT", "http://localhost:8000")

    async def fetch_logs():
        client = PyletClient(api_server_url=endpoint)
        offset = 0
        try:
            while True:
                try:
                    result = await client.get_logs(
                        instance_id, offset=offset, limit=10 * 1024 * 1024
                    )
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)

                data = result.get("data", b"")
                if data:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                    offset += len(data)

                if not follow:
                    break

                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            await client.client.aclose()

    asyncio.run(fetch_logs())


if __name__ == "__main__":
    cli()
