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
from fastapi.responses import StreamingResponse

from sllm.serve.logger import init_logger
from sllm.serve.openai_api_protocol import chat_completion_stream_generator

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
            raise HTTPException(
                status_code=500,
                detail="Cannot register model, please contact the administrator",
            )

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

        if body.get("stream", False):
            generator = request_router.generate_stream.remote(body)
            chat_generator = chat_completion_stream_generator(
                model_name, generator
            )
            return StreamingResponse(
                chat_generator, media_type="text/event-stream"
            )

        result = request_router.inference.remote(body, action)
        return await result

    async def fine_tuning_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.fine_tuning.remote(body)
        return await result

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/fine-tuning")
    async def fine_tuning(request: Request):
        return await fine_tuning_handler(request)

    @app.get("/v1/models")
    async def get_models():
        logger.info("Attempting to retrieve the controller actor")
        try:
            controller = ray.get_actor("controller")
            if not controller:
                logger.error("Controller not initialized")
                raise HTTPException(
                    status_code=500, detail="Controller not initialized"
                )
            logger.info("Controller actor found")
            result = await controller.status.remote()
            logger.info("Controller status retrieved successfully")
            return result
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve models"
            )

    return app
