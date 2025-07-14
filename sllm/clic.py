import click

from sllm._cli_utils import (
    delete_model,
    deploy_model,
    show_status,
    start_server,
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
    # ...existing code...
    adapters_dict = None
    if lora_adapters:
        adapters_dict = {}
        # If it's a string, split by comma or space
        if isinstance(lora_adapters, str):
            items = lora_adapters.replace(",", " ").split()
        elif isinstance(lora_adapters, (list, tuple)):
            items = []
            for item in lora_adapters:
                items.extend(item.replace(",", " ").split())
        else:
            items = [str(lora_adapters)]
        for module in items:
            if "=" not in module:
                click.echo(
                    f"[ERROR] Invalid LoRA module format: {module}. Expected <name>=<path>."
                )
                continue
            name, path = module.split("=", 1)
            adapters_dict[name] = path

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
@click.option("--lora-adapters", multiple=True, help="LoRA adapters to delete.")
def delete(models, lora_adapters):
    delete_model(models, lora_adapters=lora_adapters if lora_adapters else None)


@cli.command()
def start():
    """Start the head node of the SLLM cluster."""
    start_server()


@cli.command()
def status():
    """Show all deployed models."""
    show_status()


if __name__ == "__main__":
    cli()
