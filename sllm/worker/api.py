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

from fastapi import FastAPI, HTTPException, Request

from sllm.logger import init_logger
from sllm.worker.instance_manager import InstanceManager

logger = init_logger(__name__)


async def _start_instance_background(
    instance_manager: InstanceManager, model_config: dict, instance_id: str
):
    """Background task to actually start the instance."""
    try:
        logger.info(
            f"[WORKER_API] Starting background instance creation for {instance_id}"
        )
        started_instance_id = await instance_manager.start_instance(
            model_config, instance_id
        )
        logger.info(
            f"[WORKER_API] Successfully started instance {started_instance_id}"
        )
    except Exception as e:
        logger.error(
            f"[WORKER_API] Failed to start instance {instance_id} in background: {e}"
        )


def create_worker_app(instance_manager: InstanceManager) -> FastAPI:
    app = FastAPI()

    # Store node_id once assigned
    app.state.node_id = None

    @app.get("/health")
    async def health_check():
        """Health check endpoint for container orchestration."""
        return {
            "status": "healthy",
            "node_id": getattr(app.state, "node_id", None),
            "running_instances": len(instance_manager.get_running_instances_info())
        }

    @app.post("/confirmation")
    async def confirmation_handler(request: Request):
        """Handle confirmation from WorkerManager with node_id assignment."""
        try:
            payload = await request.json()
            node_id = payload.get("node_id")

            if not node_id:
                raise HTTPException(status_code=400, detail="Missing node_id")

            # Store the assigned node_id
            app.state.node_id = node_id

            return {"message": f"Node {node_id} confirmed successfully"}

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Confirmation failed: {str(e)}"
            )

    @app.post("/start_instance")
    async def start_instance_api(request: Request):
        payload = await request.json()
        model_config = payload.get("model_config")
        instance_id = payload.get("instance_id")

        if not model_config:
            raise HTTPException(status_code=400, detail="Missing model_config")

        logger.info(
            f"[WORKER_API] Received start instance request for {instance_id}"
        )

        # Send immediate confirmation that request was received
        response = {
            "status": "received",
            "instance_id": instance_id,
            "message": f"Instance {instance_id} start request received and processing",
        }

        # Start instance in background (don't await)
        asyncio.create_task(
            _start_instance_background(
                instance_manager, model_config, instance_id
            )
        )

        return response

    @app.post("/stop_instance")
    async def stop_instance_api(request: Request):
        payload = await request.json()
        instance_id = payload.get("instance_id")
        if not instance_id:
            raise HTTPException(status_code=400, detail="Missing instance_id")

        success = await instance_manager.stop_instance(instance_id)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to stop model instance"
            )
        return {"message": f"Instance {instance_id} stopped successfully"}

    @app.post("/invoke")
    async def invoke_handler(request: Request):
        body = await request.json()
        instance_id = body.get("instance_id")
        payload = body.get("payload")

        if not instance_id or not payload:
            raise HTTPException(
                status_code=400,
                detail="Internal invoke requires instance_id and payload",
            )

        try:
            result = await instance_manager.run_inference(instance_id, payload)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {str(e)}"
            )

    return app
