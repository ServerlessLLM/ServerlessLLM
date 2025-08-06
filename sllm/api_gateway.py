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
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from sllm.dispatcher import Dispatcher
from sllm.kv_store import RedisStore
from sllm.logger import init_logger
from sllm.model_manager import ModelManager
from sllm.worker_manager import WorkerManager

INFERENCE_REQUEST_TIMEOUT = 300  # 5 minutes for VLLM cold starts

logger = init_logger(__name__)


def create_app(
    worker_manager: WorkerManager,
    model_manager: ModelManager,
    dispatcher: Dispatcher,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_manager = worker_manager
        app.state.model_manager = model_manager
        app.state.dispatcher = dispatcher
        app.state.redis_store = worker_manager.store

        yield

    app = FastAPI(lifespan=lifespan, title="SLLM API Gateway")

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": {"message": str(exc)}}
        )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON payload: {str(e)}"
            )

        if not body.get("model"):
            raise HTTPException(
                status_code=400, detail="Missing required field: model"
            )

        try:
            await request.app.state.model_manager.register(body)
            model_name = body.get("model")
            return {"message": f"Model {model_name} registered successfully"}
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Model registration validation failed: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Cannot register model: {e}", exc_info=True)
            raise RuntimeError("Model registration failed")

    @app.post("/update")
    async def update_handler(request: Request):
        try:
            body = await request.json()
            model = body.get("model")
            backend = body.get("backend")

            if not all([model, backend]):
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'model' or 'backend' in request body.",
                )

            await request.app.state.model_manager.update(model, backend, body)
            return {"status": f"updated model {model}:{backend}"}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error updating model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/delete")
    async def delete_model_handler(request: Request):
        try:
            body = await request.json()
            model = body.get("model")
            if not model:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'model' in request body.",
                )

            backend = body.get("backend", None)
            lora_adapters = body.get("lora_adapters", None)
            model_manager = request.app.state.model_manager

            if lora_adapters:
                await model_manager.delete(model, "transformers", lora_adapters)
                return {"status": f"deleted LoRA adapters from {model}"}

            elif backend:
                logger.info(f"Deleting model '{model}:{backend}'")
                await model_manager.delete(model, backend)
                return {"status": f"deleted model {model}:{backend}"}

            else:
                logger.info(f"Deleting all backends for model '{model}'")
                await model_manager.delete(model, "all")
                return {"status": f"deleted all backends for model {model}"}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(
                f"Error during delete operation for '{model}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="An internal server error occurred."
            )

    @app.post("/heartbeat")
    async def handle_heartbeat(request: Request):
        try:
            payload = await request.json()
            await request.app.state.worker_manager.process_heartbeat(payload)
            return {
                "status": "ok",
                "message": "Heartbeat received and processed.",
            }
        except Exception as e:
            logger.error(f"Failed to process heartbeat: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing the heartbeat.",
            )

    async def inference_handler(request: Request, action: str) -> Response:
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON body: {e}")
            raise HTTPException(
                status_code=400, detail="Invalid JSON in request body"
            )

        model_identifier = body.get("model")
        if not model_identifier:
            raise HTTPException(
                status_code=400,
                detail="Request body must include a 'model' field",
            )

        explicit_backend = body.get("backend")

        if ":" in model_identifier:
            model, backend = model_identifier.split(":", 1)
            if explicit_backend and explicit_backend != backend:
                raise HTTPException(
                    status_code=400,
                    detail=f"Backend mismatch: model specifies '{backend}' but request body specifies '{explicit_backend}'",
                )
        else:
            model = model_identifier
            backend = explicit_backend

            if not backend:
                all_models = (
                    await request.app.state.model_manager.get_all_models()
                )
                available_backends = [
                    m.get("backend")
                    for m in all_models
                    if m.get("model") == model
                    and m.get("status") != "excommunicado"
                ]

                if not available_backends:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No available backends found for model '{model}'",
                    )

                backend = available_backends[0]

        if not await request.app.state.model_manager.get_model(model, backend):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_identifier}' not found or not registered.",
            )

        # Generate OpenAI-compatible task_id based on action type
        if body.get("task_id") is None:
            if action == "generate":
                task_id = f"chatcmpl-{uuid.uuid4()}"
            elif action == "completions":
                task_id = f"cmpl-{uuid.uuid4()}"
            elif action == "encode":
                task_id = f"embedding-{uuid.uuid4()}"
            elif action == "fine-tuning":
                task_id = f"ftjob-{uuid.uuid4()}"
            else:
                task_id = f"task-{uuid.uuid4()}"
        else:
            task_id = body["task_id"]

        # Fix model field in payload to contain only model name, not model:backend
        payload = body.copy()
        payload["model"] = model  # Use parsed model name instead of full identifier
        
        task_package = {"task_id": task_id, "action": action, "payload": payload}

        store: RedisStore = request.app.state.redis_store
        result_queue = asyncio.Queue()

        async def _result_listener(task_id: str):
            try:
                async for message in store.subscribe_to_result(
                    task_id, timeout=INFERENCE_REQUEST_TIMEOUT + 5
                ):
                    await result_queue.put(message)
                    break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(
                    f"Error in result listener for task {task_id}: {e}"
                )
                await result_queue.put(
                    {"status": "error", "message": "Result listener failed."}
                )

        listener_task = asyncio.create_task(_result_listener(task_id))

        await store.enqueue_task(model, backend, task_package)

        try:
            result = await asyncio.wait_for(
                result_queue.get(), timeout=INFERENCE_REQUEST_TIMEOUT
            )
            return JSONResponse(content=result)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Request timed out waiting for inference result.",
            )
        finally:
            listener_task.cancel()

    @app.post("/v1/chat/completions")
    async def chat_completions_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/completions")
    async def completions_handler(request: Request):
        return await inference_handler(request, "completions")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/fine-tuning")
    async def fine_tuning_handler(request: Request):
        return await inference_handler(request, "fine-tuning")

    @app.get("/v1/models")
    async def get_models(request: Request):
        try:
            # Use KV store's OpenAI-compliant get_all_models method  
            store: RedisStore = request.app.state.redis_store
            model_statuses = await store.get_all_models()
            return model_statuses
        except Exception as e:
            logger.error(f"Error retrieving models: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve models from the store.",
            )

    return app
