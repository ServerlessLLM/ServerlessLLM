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
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sllm.dispatcher import Dispatcher
from sllm.kv_store import RedisStore
from sllm.lb_manager import LoadBalancerManager
from sllm.logger import init_logger
from sllm.model_manager import ModelManager
from sllm.worker_manager import WorkerManager

INFERENCE_REQUEST_TIMEOUT = 300  # 5 minutes for VLLM cold starts

logger = init_logger(__name__)
origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin for origin in origins_env.split(",") if origin]
origins += ["http://localhost", "http://localhost:3000"]


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

        # Initialize load balancer manager
        app.state.lb_manager = LoadBalancerManager(worker_manager.store)

        # Initialize HTTP session for forwarding requests
        app.state.http_session = aiohttp.ClientSession()

        yield

        # Cleanup
        await app.state.http_session.close()
        await app.state.lb_manager.shutdown_all()

    app = FastAPI(lifespan=lifespan, title="SLLM API Gateway")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=86400,
    )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": {"message": str(exc)}}
        )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/models")
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
            backend = body.get("backend", "vllm")

            # Start load balancer for this model
            lb_config = body.get("lb_config", {})
            await request.app.state.lb_manager.start_lb(
                model_name, backend, lb_config
            )

            return {
                "message": f"Model {model_name}:{backend} registered successfully"
            }
        except ValueError as e:
            error_msg = str(e)
            if "already registered" in error_msg:
                raise HTTPException(
                    status_code=409,
                    detail=error_msg,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model registration validation failed: {error_msg}",
                )
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Model registration validation failed: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Cannot register model: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Model registration failed due to internal error",
            )

    @app.put("/models/{model_id}")
    async def update_handler(model_id: str, request: Request):
        try:
            body = await request.json()
            backend = body.get("backend")

            if not backend:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'backend' in request body.",
                )

            await request.app.state.model_manager.update(model_id, backend, body)
            return {"status": f"updated model {model_id}:{backend}"}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error updating model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/models/{model_id}")
    async def delete_model_handler(
        model_id: str, request: Request, backend: str = None
    ):
        try:
            model_manager = request.app.state.model_manager

            if backend:
                logger.info(f"Deleting model '{model_id}:{backend}'")
                await request.app.state.lb_manager.stop_lb(model_id, backend)
                await model_manager.delete(model_id, backend)
                return {"status": f"deleted model {model_id}:{backend}"}
            else:
                logger.info(f"Deleting all backends for model '{model_id}'")
                all_models = await model_manager.get_all_models()
                for m in all_models:
                    if m.get("model") == model_id:
                        backend_to_delete = m.get("backend")
                        await request.app.state.lb_manager.stop_lb(
                            model_id, backend_to_delete
                        )
                await model_manager.delete(model_id, "all")
                return {"status": f"deleted all backends for model {model_id}"}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(
                f"Error during delete operation for '{model_id}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="An internal server error occurred."
            )

    @app.delete("/models/{model_id}/adapters")
    async def delete_adapters_handler(model_id: str, request: Request):
        try:
            body = await request.json()
            lora_adapters = body.get("lora_adapters")
            if not lora_adapters:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'lora_adapters' in request body.",
                )
            model_manager = request.app.state.model_manager
            await model_manager.delete(model_id, "transformers", lora_adapters)
            return {"status": f"deleted LoRA adapters from {model_id}"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(
                f"Error deleting adapters for '{model_id}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="An internal server error occurred."
            )

    @app.post("/workers/heartbeat")
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

        # Get load balancer endpoint for this model
        lb_endpoint = await request.app.state.lb_manager.get_lb_endpoint(
            model, backend
        )
        if not lb_endpoint:
            raise HTTPException(
                status_code=404,
                detail=f"No load balancer available for model '{model}:{backend}'",
            )

        # Map action to endpoint
        endpoint_map = {
            "generate": "/v1/chat/completions",
            "completions": "/v1/completions",
            "encode": "/v1/embeddings",
            "fine-tuning": "/fine-tuning",
        }
        endpoint = endpoint_map.get(action, "/v1/chat/completions")
        url = f"http://{lb_endpoint}{endpoint}"

        logger.info(f"Forwarding {action} request to LB at {url}")

        # Forward request to load balancer
        try:
            async with request.app.state.http_session.post(
                url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=INFERENCE_REQUEST_TIMEOUT),
            ) as resp:
                result = await resp.json()
                return JSONResponse(content=result, status_code=resp.status)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Request timed out",
            )
        except Exception as e:
            logger.error(f"Failed to forward request to load balancer: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to communicate with load balancer: {str(e)}",
            )

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
            store: RedisStore = request.app.state.redis_store
            model_statuses = await store.get_all_models()
            return model_statuses
        except Exception as e:
            logger.error(f"Error retrieving models: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve models from the store.",
            )

    @app.get("/v1/status/{request_id}")
    async def get_request_status(request_id: str, request: Request):
        try:
            store: RedisStore = request.app.state.redis_store
            status = await store.get_request_status(request_id)

            if status is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Request {request_id} not found or has expired",
                )

            return {"request_id": request_id, "status": status}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving request status: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve request status.",
            )

    @app.get("/v1/workers")
    async def get_workers(request: Request):
        try:
            store: RedisStore = request.app.state.redis_store
            all_workers = await store.get_all_workers()

            sanitized_workers = []
            for worker in all_workers:
                sanitized_worker = worker.copy()
                sanitized_worker.pop("node_ip", None)
                sanitized_workers.append(sanitized_worker)

            return {"object": "list", "data": sanitized_workers}
        except Exception as e:
            logger.error(f"Error retrieving workers: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve workers from the store.",
            )

    return app
