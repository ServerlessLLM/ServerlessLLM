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
def start():
    """Start the gRPC server"""
    try:
        logger.info("Starting gRPC server")
        asyncio.run(serve())
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)


# Entry point for the 'sllm-store start' command
def main():
    cli()
