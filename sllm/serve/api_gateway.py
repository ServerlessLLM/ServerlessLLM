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
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from sllm.serve.kv_store import RedisStore
from sllm.serve.model_manager import ModelManager
from sllm.serve.worker_manager import WorkerManager
from sllm.serve.schema import .
from sllm.serve.logger import init_logger

INFERENCE_REQUEST_TIMEOUT = 120 

logger = init_logger(__name__)

def create_app(worker_manager: WorkerManager, model_manager: ModelManager, dispatcher: Dispatcher) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_manager = worker_manager
        app.state.model_manager = model_manager
        app.state.dispatcher = dispatcher
        app.state.redis_store = worker_manager.store 
        
        yield
        
        logger.info("API Gateway is shutting down.")

    app = FastAPI(lifespan=lifespan, title="SLLM API Gateway")

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok"}

    @app.get("/v1/models", tags=["Models"])
    async def get_models(request: Request):
        try:
            model_statuses = await request.app.state.model_manager.get_all_models_status()
            return model_statuses
        except Exception as e:
            logger.error(f"Error retrieving models: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve models from the store.")

    @app.post("/register", tags=["Admin"])
    async def register_handler(request: Request):
        try:
            body = await request.json()
            await request.app.state.model_manager.register_model(body)
            return {"status": "ok", "message": f"Model '{body.get('model_name')}' registered successfully."}
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Cannot register model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="An internal error occurred during model registration.")

    @app.post("/update", tags=["Admin"])
    async def update_handler(request: Request):
        try:
            body = await request.json()
            model_name = body.get("model_name")
            backend = body.get("backend")
            
            if not all([model_name, backend]):
                raise HTTPException(status_code=400, detail="Missing 'model_name' or 'backend' in request body.")
                
            await request.app.state.model_manager.update_model(model_name, backend, body)
            return {"status": f"updated model {model_name}:{backend}"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error updating model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/delete", tags=["Admin"])
    async def delete_model_handler(request: Request):
        try:
            body = await request.json()
            model_name = body.get("model_name")
            backend = body.get("backend")

            if not all([model_name, backend]):
                raise HTTPException(status_code=400, detail="Missing 'model_name' or 'backend' in request body.")
            
            await request.app.state.model_manager.delete_model(model_name, backend)
            return {"status": f"deleted model {model_name}:{backend}"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error deleting model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/heartbeat", tags=["Status"])
    async def handle_heartbeat(request: Request):
        try:
            payload = await request.json()
            await request.app.state.worker_manager.process_heartbeat(payload)
            return {"status": "ok", "message": "Heartbeat received and processed."}
        except Exception as e: 
            logger.error(f"Failed to process heartbeat: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail="An internal error occurred while processing the heartbeat.",
            )

    async def inference_handler(request: Request) -> Response:
        body = await request.json()
        model_identifier = body.get("model")
        
        if not model_identifier or ':' not in model_identifier:
            raise HTTPException(status_code=400, detail="Request body must include a 'model' field in 'model_name:backend' format.")
        
        model_name, backend = model_identifier.split(":", 1)
        
        if not await request.app.state.model_manager.get_model(model_name, backend):
            raise HTTPException(status_code=404, detail=f"Model '{model_identifier}' not found or not registered.")

        task_id = str(uuid.uuid4()) if request["task_id"] == None else request["task_id"]
        task_package = {"task_id": task_id, "payload": body}
        
        store: RedisStore = request.app.state.redis_store
        result_queue = asyncio.Queue()

        async def _result_listener(channel_name: str):
            try:
                async for message in store.subscribe_to_result(channel_name, timeout=INFERENCE_REQUEST_TIMEOUT + 5):
                    await result_queue.put(message)
                    break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in result listener for task {task_id}: {e}")
                await result_queue.put({"status": "error", "message": "Result listener failed."})

        listener_task = asyncio.create_task(_result_listener(f"result-channel:{task_id}"))

        await store.enqueue_task(model_name, backend, task_package)

        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=INFERENCE_REQUEST_TIMEOUT)
            return JSONResponse(content=result)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timed out waiting for inference result.")
        finally:
            listener_task.cancel()

    @app.post("/v1/chat/completions", tags=["Inference"])
    async def generate_handler(request: Request):
        return await inference_handler(request)

    @app.post("/v1/embeddings", tags=["Inference"])
    async def embeddings_handler(request: Request):
        return await inference_handler(request)

    @app.post("/fine-tuning", tags=["Fine-tuning"], status_code=202)
    async def fine_tuning_handler(request: Request):
        return await inference_handler(request)

    return app
