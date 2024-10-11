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
from contextlib import asynccontextmanager

import ray
import ray.exceptions
from fastapi import FastAPI, HTTPException, Request

from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await controller.register.remote(body)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Cannot register model, please contact the administrator")

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await controller.update.remote(model_name, body)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to delete model {model_name}")
        await controller.delete.remote(model_name)

        return {"status": f"deleted model {model_name}"}
    
    async def inference_handler(request: Request, action: str):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.inference.remote(body, action)
        return await result

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")
    
    return app