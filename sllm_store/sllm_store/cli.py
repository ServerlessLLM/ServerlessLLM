import asyncio
import sys
import logging
import click

from sllm_store.server import serve
from sllm_store.logger import init_logger
from sllm_store.utils import to_num_bytes

logger = init_logger(__name__)


@click.group()
def cli():
    """sllm-store CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8073, help="Port")
@click.option("--storage-path", default="./models", help="Storage path")
@click.option("--num-thread", default=4, help="Number of I/O threads")
@click.option(
    "--chunk-size", default="32MB", help="Chunk size, e.g., 4KB, 1MB, 1GB"
)
@click.option(
    "--mem-pool-size",
    default="4GB",
    help="Memory pool size, e.g., 1GB, 4GB, 1TB",
)
@click.option(
    "--disk-size", default="128GB", help="Disk size, e.g., 1GB, 4GB, 1TB"
)
@click.option(
    "--registration-required",
    default=False,
    help="Require registration before loading model",
)
def start(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    disk_size,
    registration_required,
):
    # Convert the chunk size to bytes
    chunk_size = to_num_bytes(chunk_size)

    # Convert the memory pool size to bytes
    mem_pool_size = to_num_bytes(mem_pool_size)

    """Start the gRPC server"""
    try:
        logger.info("Starting gRPC server")
        asyncio.run(
            serve(
                host=host,
                port=port,
                storage_path=storage_path,
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
