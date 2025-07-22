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

from sllm.serve.api_gateway import create_app as create_head_app
from sllm.serve.autoscaler import AutoScaler
from sllm.serve.dispatcher import Dispatcher
from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger  # logger my beloved
from sllm.serve.model_manager import ModelManager
from sllm.serve.worker.api import create_worker_app
from sllm.serve.worker.hardware_utils import benchmark_static_hardware
from sllm.serve.worker.heartbeat import run_heartbeat_loop
from sllm.serve.worker.instance_manager import InstanceManager
from sllm.serve.worker_manager import WorkerManager

logger = init_logger(__name__)


async def run_head_node(args: argparse.Namespace):
    """Initializes and runs all head-node services concurrently."""
    logger.info("Starting Sllm in HEAD mode...")

    logger.info(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    store = RedisStore(host=args.redis_host, port=args.redis_port)
    model_manager = ModelManager(store)
    worker_manager = WorkerManager(store, config={"prune_interval": 15})
    autoscaler = AutoScaler(
        store=store, model_manager=model_manager, worker_manager=worker_manager
    )
    dispatcher = Dispatcher(store)

    app = create_head_app(
        worker_manager=worker_manager,
        model_manager=model_manager,
        dispatcher=dispatcher,
    )

    uvicorn_config = Config(
        app, host=args.host, port=args.port, log_level="info"
    )
    uvicorn_server = Server(uvicorn_config)

    worker_manager.start()
    dispatcher.start()
    autoscaler_task = asyncio.create_task(autoscaler.run_scaling_loop())
    dispatcher_task = asyncio.create_task(dispatcher.run_consumer_loop())
    server_task = asyncio.create_task(uvicorn_server.serve())

    try:
        logger.info("Sllm control plane started. All services are running.")
        await asyncio.gather(autoscaler_task, server_task)
        await asyncio.gather(autoscaler_task, dispatcher_task, server_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received for head node.")
    finally:
        logger.info("Initiating graceful shutdown for head node...")
        autoscaler.shutdown()
        await dispatcher.shutdown()
        await worker_manager.shutdown()
        uvicorn_server.should_exit = True
        await asyncio.sleep(2)
        await store.close()
        logger.info("Head node shutdown complete.")


async def run_worker_node(args: argparse.Namespace):
    """Initializes and runs all worker-node services concurrently."""
    logger.info("Starting Sllm in WORKER mode...")

    static_hardware_info = benchmark_static_hardware()

    instance_manager = InstanceManager()
    worker_app = create_worker_app(instance_manager)
    uvicorn_config = Config(
        worker_app, host=args.host, port=args.port, log_level="info"
    )
    uvicorn_server = Server(uvicorn_config)

    server_task = asyncio.create_task(uvicorn_server.serve())
    heartbeat_task = asyncio.create_task(
        run_heartbeat_loop(
            instance_manager=instance_manager,
            head_node_url=args.head_node_url,
            node_id=args.node_id,
            node_ip=args.host,
            static_hardware_info=static_hardware_info,
        )
    )

    try:
        logger.info(
            f"Sllm worker started. API running on {args.host}:{args.port}."
        )
        await asyncio.gather(server_task, heartbeat_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received for worker node.")
    finally:
        logger.info("Initiating graceful shutdown for worker node...")
        server_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(
            server_task, heartbeat_task, return_exceptions=True
        )
        logger.info("Worker node shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="ServerlessLLM (Sllm) main entry point."
    )
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="The mode to run in: 'head' or 'worker'.",
    )

    # --- Arguments for HEAD mode ---
    head_parser = subparsers.add_parser(
        "head", help="Run the control plane (head node)."
    )
    head_parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host IP for the API Gateway.",
    )
    head_parser.add_argument(
        "--port", default=8343, type=int, help="Port for the API Gateway."
    )
    head_parser.add_argument(
        "--redis-host",
        default="localhost",
        type=str,
        help="Hostname of the Redis server.",
    )
    head_parser.add_argument(
        "--redis-port", default=6379, type=int, help="Port of the Redis server."
    )

    # --- Arguments for WORKER mode ---
    worker_parser = subparsers.add_parser("worker", help="Run a worker node.")
    worker_parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host for the worker's API server.",
    )
    worker_parser.add_argument(
        "--port",
        default=8001,
        type=int,
        help="Port for the worker's API server.",
    )
    worker_parser.add_argument(
        "--node-id",
        type=str,
        required=True,
        help="A unique identifier for this worker node.",
    )
    worker_parser.add_argument(
        "--head-node-url",
        type=str,
        required=True,
        help="Full URL of the head node API Gateway (e.g., http://192.168.1.100:8343).",
    )

    args = parser.parse_args()

    if args.mode == "head":
        asyncio.run(run_head_node(args))
    elif args.mode == "worker":
        asyncio.run(run_worker_node(args))


if __name__ == "__main__":
    main()
