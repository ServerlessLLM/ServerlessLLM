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

from sllm.logger import init_logger

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
        lora_adapters = body.get("lora_adapters", None)

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        if lora_adapters is not None:
            logger.info(
                f"Received request to delete LoRA adapters {lora_adapters} on model {model_name}"
            )
            await controller.delete.remote(model_name, lora_adapters)
        else:
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

    async def fine_tuning_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received fine tuning request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )
        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        job_id = await controller.register_ft_job.remote(body)
        return {"job_id": job_id}

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/completions")
    async def completions_handler(request: Request):
        """Handle legacy /v1/completions endpoint by converting to chat format."""
        body = await request.json()

        # Convert prompt-based request to chat format
        prompt = body.get("prompt", "")

        # OpenAI completions API supports both string and array of strings
        # For simplicity, we only handle single prompt (string or single-element list)
        # Multiple prompts in one request would require multiple inference calls
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise HTTPException(status_code=400, detail="Empty prompt list")
            elif len(prompt) > 1:
                raise HTTPException(
                    status_code=400,
                    detail="Multiple prompts in single request not supported. Please send separate requests for each prompt.",
                )
            prompt = prompt[0]

        # Create chat-style messages
        chat_body = {
            "model": body.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": body.get("max_tokens", 100),
            "temperature": body.get("temperature", 0.0),
            "stream": body.get("stream", False),
        }

        # Copy any extra parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty"]:
            if key in body:
                chat_body[key] = body[key]

        # Create a new request with modified body
        from fastapi import Request
        from starlette.datastructures import Headers

        # Call the chat completions handler with modified body
        class ModifiedRequest:
            def __init__(self, original_request, new_body):
                self._original = original_request
                self._body = new_body

            async def json(self):
                return self._body

            def __getattr__(self, name):
                return getattr(self._original, name)

        modified_request = ModifiedRequest(request, chat_body)
        return await inference_handler(modified_request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/v1/fine-tuning/jobs")
    async def fine_tuning(request: Request):
        return await fine_tuning_handler(request)

    @app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}")
    async def get_job_status(fine_tuning_job_id: str):
        if not fine_tuning_job_id:
            raise HTTPException(
                status_code=400, detail="Missing fine_tuning_job_id parameter"
            )
        controller = ray.get_actor("controller")
        status = await controller.get_ft_job_status.remote(fine_tuning_job_id)
        return {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",
            "status": status,
        }

    @app.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel")
    async def cancel_job(fine_tuning_job_id: str):
        if not fine_tuning_job_id:
            raise HTTPException(
                status_code=400, detail="Missing fine_tuning_job_id parameter"
            )
        controller = ray.get_actor("controller")
        await controller.cancel_ft_job.remote(fine_tuning_job_id)
        return {
            "id": fine_tuning_job_id,
            "object": "fine_tuning.job",
            "status": "cancelled",
        }

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
