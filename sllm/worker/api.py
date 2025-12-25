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

import asyncio
import os
import shutil

from fastapi import FastAPI, HTTPException, Request

from sllm.logger import init_logger
from sllm.worker.instance_manager import InstanceManager

logger = init_logger(__name__)


async def _start_instance_background(
    instance_manager: InstanceManager, model_config: dict, instance_id: str
):
    """Background task to actually start the instance."""
    try:
        logger.debug(f"Starting instance {instance_id}")
        started_instance_id = await instance_manager.start_instance(
            model_config, instance_id
        )
        logger.debug(f"Started {started_instance_id}")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Model validation/loading failed for {instance_id}: {e}")
        logger.info(
            f"Triggering model re-download for {model_config.get('model')}"
        )

        model = model_config.get("model")
        backend = model_config.get("backend")
        storage_path = os.getenv("STORAGE_PATH", "./models")

        if backend == "vllm":
            model_path = os.path.join(storage_path, "vllm", model)
        elif backend == "transformers":
            model_path = os.path.join(storage_path, "transformers", model)
        else:
            logger.error(f"Unknown backend {backend} for cleanup")
            return

        if os.path.exists(model_path):
            logger.warning(f"Removing corrupted model directory: {model_path}")
            shutil.rmtree(model_path)

        try:
            await instance_manager._ensure_model_downloaded(model_config)
            logger.info(
                f"Re-download completed, retrying instance start for {instance_id}"
            )
            started_instance_id = await instance_manager.start_instance(
                model_config, instance_id
            )
            logger.debug(f"Started {started_instance_id} after re-download")
        except Exception as retry_e:
            logger.error(
                f"Failed to start instance {instance_id} even after re-download: {retry_e}"
            )
    except Exception as e:
        logger.error(
            f"Failed to start instance {instance_id} in background: {e}"
        )


def create_worker_app(instance_manager: InstanceManager) -> FastAPI:
    app = FastAPI()
    app.state.node_id = None

    @app.get("/health")
    async def health_check():
        """Health check endpoint for container orchestration."""
        return {
            "status": "HEALTHY",
            "node_id": getattr(app.state, "node_id", None),
            "running_instances": len(
                instance_manager.get_running_instances_info()
            ),
        }

    @app.post("/workers/confirmation")
    async def confirmation_handler(request: Request):
        """Handle confirmation from WorkerManager with node_id assignment."""
        try:
            payload = await request.json()
            node_id = payload.get("node_id")

            if not node_id:
                raise HTTPException(status_code=400, detail="Missing node_id")

            app.state.node_id = node_id

            return {"message": f"Node {node_id} confirmed successfully"}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Confirmation failed: {str(e)}"
            )

    @app.post("/instances")
    async def start_instance_api(request: Request):
        payload = await request.json()
        model_config = payload.get("model_config")
        instance_id = payload.get("instance_id")

        if not model_config:
            raise HTTPException(status_code=400, detail="Missing model_config")

        logger.debug(f"Start request: {instance_id}")

        response = {
            "status": "RECEIVED",
            "instance_id": instance_id,
            "message": f"Instance {instance_id} start request received and processing",
        }

        asyncio.create_task(
            _start_instance_background(
                instance_manager, model_config, instance_id
            )
        )

        return response

    @app.delete("/instances/{instance_id}")
    async def stop_instance_api(instance_id: str):
        success = await instance_manager.stop_instance(instance_id)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to stop model instance"
            )
        return {"message": f"Instance {instance_id} stopped successfully"}

    return app
