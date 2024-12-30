import asyncio
import sys
import logging
import click

from sllm_store.server import serve
from sllm_store.logger import init_logger

logger = init_logger(__name__)


@click.group()
def cli():
    """sllm-store CLI"""
    pass


@cli.command()
@click.option("--storage-path", default="./models", help="Storage path")
@click.option("--server-port", default=8073, help="Server port")
@click.option("--num-thread", default=4, help="Number of I/O threads")
@click.option("--chunk-size", default=32, help="Chunk size in MB")
@click.option("--mem-pool-size", default=4, help="Memory pool size in GB")
@click.option("--disk-size", default=128, help="Disk size in GB")
@click.option(
    "--registration-required",
    default=False,
    help="Require registration before loading model",
)
def start(
    storage_path,
    server_port,
    num_thread,
    chunk_size,
    mem_pool_size,
    disk_size,
    registration_required,
):
    """Start the gRPC server"""
    try:
        logger.info("Starting gRPC server")
        asyncio.run(
            serve(
                storage_path=storage_path,
                port=server_port,
                num_thread=num_thread,
                chunk_size=chunk_size,
                mem_pool_size=mem_pool_size,
                # disk size is not used
                # disk_size=disk_size,
                registration_required=registration_required,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)


# Entry point for the 'sllm-store start' command
def main():
    cli()
