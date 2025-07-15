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
import argparse
import asyncio
import sys

import uvicorn
from uvicorn import Config, Server

from sllm.serve.api_gateway import create_app
from sllm.serve.autoscaler import AutoScaler
from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger
from sllm.serve.model_manager import ModelManager
from sllm.serve.worker_manager import WorkerManager

logger = init_logger(__name__)


async def main():
    """
    The main entry point for the Sllm control plane.
    Initializes and runs all head-node services concurrently.
    """
    parser = argparse.ArgumentParser(
        description="ServerlessLLM CLI for the control plane."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host IP for the API Gateway.",
    )
    parser.add_argument(
        "--port", default=8343, type=int, help="Port for the API Gateway."
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        type=str,
        help="Hostname of the Redis server.",
    )
    parser.add_argument(
        "--redis-port",
        default=6379,
        type=int,
        help="Port of the Redis server.",
    )
    parser.add_argument(
        "--enable-storage-aware",
        action="store_true",
        help="Enable storage-aware scheduling (Not yet implemented).",
    )
    parser.add_argument(
        "--enable-migration",
        action="store_true",
        help="Enable live migration of model instances (Not yet implemented).",
    )
    args = parser.parse_args()

    logger.info(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    store = RedisStore(host=args.redis_host, port=args.redis_port)
    model_manager = ModelManager(store)
    worker_manager = WorkerManager(store, config={"prune_interval": 15})
    autoscaler = AutoScaler(
        store=store, model_manager=model_manager, worker_manager=worker_manager
    )

    app = create_app(
        worker_manager=worker_manager,
        model_manager=model_manager,
    )

    uvicorn_config = Config(app, host=args.host, port=args.port, log_level="info")
    uvicorn_server = Server(uvicorn_config)

    worker_manager.start()

    autoscaler_task = asyncio.create_task(autoscaler.run_scaling_loop())
    server_task = asyncio.create_task(uvicorn_server.serve())

    try:
        logger.info("Sllm control plane started. All services are running.")
        await asyncio.gather(autoscaler_task, server_task)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        logger.info("Initiating graceful shutdown...")

        autoscaler.shutdown()
        await worker_manager.shutdown()

        await asyncio.sleep(2) 
        
        await store.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
