import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from sllm.serve.dispatcher import Dispatcher
from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger
from sllm.serve.model_manager import ModelManager
from sllm.serve.worker_manager import WorkerManager

INFERENCE_REQUEST_TIMEOUT = 120

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
                await model_manager.delete(model, backend)
                return {"status": f"deleted model {model}:{backend}"}

            else:
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

        task_id = (
            str(uuid.uuid4())
            if body.get("task_id") == None
            else body["task_id"]
        )

        task_package = {"task_id": task_id, "action": action, "payload": body}

        store: RedisStore = request.app.state.redis_store
        result_queue = asyncio.Queue()

        async def _result_listener(channel_name: str):
            try:
                async for message in store.subscribe_to_result(
                    channel_name, timeout=INFERENCE_REQUEST_TIMEOUT + 5
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

        listener_task = asyncio.create_task(
            _result_listener(f"result-channel:{task_id}")
        )

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
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.post("/fine-tuning")
    async def fine_tuning_handler(request: Request):
        return await inference_handler(request, "fine-tuning")

    @app.get("/v1/models")
    async def get_models(request: Request):
        try:
            model_statuses = (
                await request.app.state.model_manager.get_all_models_status()
            )
            return model_statuses
        except Exception as e:
            logger.error(f"Error retrieving models: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve models from the store.",
            )

    return app
