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
    multiple=True,
    help="Name of the LoRA adapter to use with the base model",
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
def deploy(**kwargs):
    """Deploy a model using a config file or model name.

    Either --model or a config file with a model specified is required.
    Command line options override values from the config file.
    """
    deploy_model(**kwargs)


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
