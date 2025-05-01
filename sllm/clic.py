import click
from sllm._cli_utils import deploy_model, delete_model, start_server,show_status

@click.group()
def cli():
    """Unified CLI for ServerlessLLM."""
    pass

@cli.command()
@click.option('--model', required=True, help="Model to deploy")
@click.option('--config', help="Path to config file")
@click.option('--backend')
@click.option('--num-gpus', type=int)
@click.option('--target', type=int)
@click.option('--min-instances', type=int)
@click.option('--max-instances', type=int)
def deploy(**kwargs):
    """Deploy a model using a config file or model name."""
    deploy_model(**kwargs)

@cli.command()
@click.argument('models', nargs=-1)
def delete(models):
    """Delete deployed models by name."""
    delete_model(models)

@cli.command()
def start():
    """Start the head node of the SLLM cluster."""
    start_server()

@cli.command()
def status():
    """Show all deployed models."""
    show_status()

if __name__ == '__main__':
    cli()
