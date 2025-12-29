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
from typing import Optional

import click

from sllm.cli._cli_utils import (
    delete_model,
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
    help="Backend framework (e.g., vllm, transformers). Use 'all' to delete all backends. If not specified, deletes all backends for the model.",
)
@click.option("--lora-adapters", help="LoRA adapters to delete.")
def delete(models, backend, lora_adapters):
    """Delete deployed models, or remove only the LoRA adapters."""
    delete_model(
        models,
        backend=backend,
        lora_adapters=lora_adapters if lora_adapters else None,
    )


@cli.group()
def start():
    """Start SLLM head or worker node."""
    pass


@start.command()
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
    help="Pylet head endpoint (v1-beta). Default: http://localhost:8000",
)
@click.option(
    "--database-path",
    default=lambda: os.getenv("SLLM_DATABASE_PATH", "/var/lib/sllm/state.db"),
    type=str,
    help="SQLite database path (v1-beta). Default: /var/lib/sllm/state.db",
)
@click.option(
    "--storage-path",
    default=lambda: os.getenv("STORAGE_PATH", "/models"),
    type=str,
    help="Model storage path. Default: /models",
)
@click.option(
    "--redis-host",
    default=None,
    type=str,
    help="[DEPRECATED] Redis is no longer used in v1-beta.",
)
@click.option(
    "--redis-port",
    default=None,
    type=int,
    help="[DEPRECATED] Redis is no longer used in v1-beta.",
)
def head(
    host,
    port,
    pylet_endpoint,
    database_path,
    storage_path,
    redis_host,
    redis_port,
):
    """Start the head node (control plane).

    v1-beta uses Pylet for instance management and SQLite for state persistence.
    Redis is no longer required.
    """
    # Warn about deprecated options
    if redis_host is not None or redis_port is not None:
        click.echo(
            "[WARNING] --redis-host and --redis-port are deprecated in v1-beta. "
            "Redis is no longer used. These options will be ignored."
        )

    start_head(
        host=host,
        port=port,
        pylet_endpoint=pylet_endpoint,
        database_path=database_path,
        storage_path=storage_path,
    )


@cli.command()
def status():
    """Show all deployed models."""
    show_status()


if __name__ == "__main__":
    cli()
