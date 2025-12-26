# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
"""
API Gateway for ServerlessLLM v1-beta.

Stateless HTTP router with:
- OpenAI-compatible inference endpoints
- Model registration and deletion
- Direct load balancer calls (in-process)
- SQLite for model configuration
- Pylet for instance information
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sllm.database import Database, Model
from sllm.lb_registry import (
    LoadBalancerRegistry,
    get_lb_registry,
    init_lb_registry,
)
from sllm.load_balancer import LBConfig
from sllm.logger import init_logger
from sllm.pylet_client import PyletClient

logger = init_logger(__name__)

# CORS origins
origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin for origin in origins_env.split(",") if origin]
origins += ["http://localhost", "http://localhost:3000"]


def create_app(
    database: Optional[Database] = None,
    pylet_client: Optional[PyletClient] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """
    Create the SLLM API Gateway FastAPI application.

    Args:
        database: SQLite database instance
        pylet_client: Pylet client instance (may be None if Pylet unavailable)
        config: Head configuration

    Returns:
        FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store dependencies in app state
        app.state.database = database
        app.state.pylet_client = pylet_client
        app.state.config = config

        # Initialize load balancer registry
        app.state.lb_registry = init_lb_registry()

        # Restore LBs for existing models and recover running instances
        if database:
            for model in database.get_active_models():
                lb_config = LBConfig()
                if model.backend_config:
                    lb_config.cold_start_timeout = model.backend_config.get(
                        "cold_start_timeout", 120.0
                    )
                lb = await app.state.lb_registry.get_or_create(
                    model.id, lb_config
                )
                logger.info(f"Restored LoadBalancer for {model.id}")

                # Query Pylet for existing running instances to recover endpoints
                if pylet_client:
                    try:
                        instances = await pylet_client.get_model_instances(
                            model.id
                        )
                        for inst in instances:
                            # Only add endpoints for running instances
                            if inst.status == "RUNNING" and inst.endpoint:
                                await lb.add_endpoint(inst.endpoint)
                                logger.info(
                                    f"Recovered endpoint {inst.endpoint} for {model.id}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to recover instances for {model.id}: {e}"
                        )

        logger.info("API Gateway started")
        yield

        # Cleanup
        await app.state.lb_registry.shutdown()
        logger.info("API Gateway shutdown")

    app = FastAPI(
        lifespan=lifespan,
        title="ServerlessLLM API Gateway",
        version="1.0.0-beta",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=86400,
    )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": {"message": str(exc)}}
        )

    # -------------------------------------------------------------------------
    # Health Endpoints
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health_check(request: Request):
        """Health check endpoint."""
        pylet_healthy = False
        if request.app.state.pylet_client:
            pylet_healthy = await request.app.state.pylet_client.is_healthy()

        return {
            "status": "ok",
            "version": "v1-beta",
            "pylet_connected": pylet_healthy,
        }

    # -------------------------------------------------------------------------
    # Model Management Endpoints
    # -------------------------------------------------------------------------

    @app.post("/register")
    async def register_handler(request: Request):
        """Register a new model."""
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON payload: {str(e)}"
            )

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing required field: model"
            )

        backend = body.get("backend", "vllm")
        model_id = f"{model_name}:{backend}"

        # Check if already exists
        db: Database = request.app.state.database
        if db.get_model(model_id):
            raise HTTPException(
                status_code=409,
                detail=f"Model {model_id} is already registered",
            )

        # Parse configuration
        backend_config = body.get("backend_config", {})
        auto_scaling_config = body.get("auto_scaling_config", {})

        try:
            # Create model in database
            model = db.create_model(
                model_id=model_id,
                model_name=model_name,
                backend=backend,
                min_replicas=auto_scaling_config.get("min_instances", 0),
                max_replicas=auto_scaling_config.get("max_instances", 1),
                target_pending_requests=auto_scaling_config.get(
                    "target_ongoing_requests", 5
                ),
                keep_alive_seconds=auto_scaling_config.get(
                    "keep_alive_seconds", 0
                ),
                backend_config=backend_config,
            )

            # Create load balancer
            lb_registry: LoadBalancerRegistry = request.app.state.lb_registry
            lb_config = LBConfig(
                cold_start_timeout=backend_config.get(
                    "cold_start_timeout", 120.0
                ),
            )
            await lb_registry.get_or_create(model_id, lb_config)

            logger.info(f"Registered model {model_id}")

            return {
                "model_id": model_id,
                "status": "active",
                "message": f"Model {model_id} registered successfully",
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to register model: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Model registration failed due to internal error",
            )

    @app.delete("/delete/{model_id:path}")
    async def delete_model_handler(model_id: str, request: Request):
        """Delete a model.

        Returns 202 Accepted immediately. The Reconciler will:
        1. Stop all instances via Pylet
        2. Drain and remove the LoadBalancer
        3. Delete the model from the database
        """
        db: Database = request.app.state.database

        model = db.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not found"
            )

        if model.status == "deleting":
            # Already deleting
            return JSONResponse(
                status_code=202,
                content={
                    "status": "deleting",
                    "model_id": model_id,
                    "message": "Deletion already in progress",
                },
            )

        try:
            # Mark as deleting and set desired=0
            # The Reconciler will handle the actual cleanup:
            # - Cancel instances in Pylet
            # - Drain and remove LoadBalancer
            # - Delete from database
            db.update_model_status(model_id, "deleting")
            db.update_desired_replicas(model_id, 0)

            logger.info(f"Model {model_id} marked for deletion")

            return JSONResponse(
                status_code=202,
                content={
                    "status": "deleting",
                    "model_id": model_id,
                    "message": "Deletion in progress. Instances will be stopped by reconciler.",
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to mark model {model_id} for deletion: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model: {str(e)}",
            )

    @app.post("/delete")
    async def delete_model_post_handler(request: Request):
        """Delete a model (POST variant for backward compatibility)."""
        try:
            body = await request.json()
            model = body.get("model")
            if not model:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'model' in request body.",
                )

            backend = body.get("backend", "vllm")
            model_id = f"{model}:{backend}"

            # Delegate to DELETE handler
            return await delete_model_handler(model_id, request)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in delete: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Inference Endpoints
    # -------------------------------------------------------------------------

    async def inference_handler(
        request: Request,
        path: str = "/v1/chat/completions",
    ) -> JSONResponse:
        """Forward inference request to load balancer."""
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

        # Parse model:backend format
        if ":" in model_identifier:
            model_name, backend = model_identifier.rsplit(":", 1)
            model_id = model_identifier
        else:
            model_name = model_identifier
            backend = body.get("backend", "vllm")
            model_id = f"{model_name}:{backend}"

        # Verify model exists
        db: Database = request.app.state.database
        model = db.get_model(model_id)
        if not model or model.status != "active":
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found or not active",
            )

        # Get load balancer
        lb_registry: LoadBalancerRegistry = request.app.state.lb_registry
        lb = lb_registry.get(model_id)
        if not lb:
            raise HTTPException(
                status_code=503,
                detail=f"Load balancer not available for '{model_id}'",
            )

        # Forward to load balancer
        try:
            result = await lb.forward(body, path)
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=str(e),
            )

    @app.post("/v1/chat/completions")
    async def chat_completions_handler(request: Request):
        """OpenAI-compatible chat completions."""
        return await inference_handler(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions_handler(request: Request):
        """OpenAI-compatible completions."""
        return await inference_handler(request, "/v1/completions")

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        """OpenAI-compatible embeddings."""
        return await inference_handler(request, "/v1/embeddings")

    # -------------------------------------------------------------------------
    # Status Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/models")
    async def get_models(request: Request):
        """List all registered models."""
        db: Database = request.app.state.database
        lb_registry: LoadBalancerRegistry = request.app.state.lb_registry

        models = db.get_all_models()
        model_list = []

        for model in models:
            lb = lb_registry.get(model.id)
            ready_endpoints = lb.endpoint_count if lb else 0

            model_list.append(
                {
                    "id": model.id,
                    "model": model.model_name,
                    "backend": model.backend,
                    "status": model.status,
                    "desired_replicas": model.desired_replicas,
                    "ready_replicas": ready_endpoints,
                    "min_replicas": model.min_replicas,
                    "max_replicas": model.max_replicas,
                }
            )

        return {"object": "list", "models": model_list}

    @app.get("/status")
    async def cluster_status(request: Request):
        """Get comprehensive cluster status."""
        db: Database = request.app.state.database
        lb_registry: LoadBalancerRegistry = request.app.state.lb_registry
        pylet_client: Optional[PyletClient] = request.app.state.pylet_client

        # Get models
        models = db.get_all_models()
        model_status = []

        for model in models:
            lb = lb_registry.get(model.id)
            endpoints = lb.get_endpoints() if lb else []

            # Get instances from Pylet if available
            instances = []
            if pylet_client:
                try:
                    pylet_instances = await pylet_client.get_model_instances(
                        model.id
                    )
                    instances = [
                        {
                            "id": inst.instance_id,
                            "node": inst.node,
                            "endpoint": inst.endpoint,
                            "status": inst.status.lower(),
                        }
                        for inst in pylet_instances
                    ]
                except Exception as e:
                    logger.warning(f"Failed to get instances from Pylet: {e}")

            model_status.append(
                {
                    "id": model.id,
                    "status": model.status,
                    "desired_replicas": model.desired_replicas,
                    "ready_replicas": len(endpoints),
                    "starting_replicas": len(
                        [
                            i
                            for i in instances
                            if i["status"] in ("pending", "assigned")
                        ]
                    ),
                    "instances": instances,
                }
            )

        # Get nodes from Pylet
        nodes = []
        if pylet_client:
            try:
                workers = await pylet_client.list_workers()
                for worker in workers:
                    node_storage = db.get_node_storage(worker.worker_id)
                    nodes.append(
                        {
                            "name": worker.worker_id,
                            "host": worker.host,
                            "status": worker.status.lower(),
                            "total_gpus": worker.total_gpus,
                            "available_gpus": worker.available_gpus,
                            "sllm_store_endpoint": (
                                node_storage.sllm_store_endpoint
                                if node_storage
                                else None
                            ),
                            "cached_models": (
                                node_storage.cached_models
                                if node_storage
                                else []
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get workers from Pylet: {e}")

        return {
            "models": model_status,
            "nodes": nodes,
        }

    # -------------------------------------------------------------------------
    # Internal Endpoints (for sllm-store)
    # -------------------------------------------------------------------------

    @app.post("/internal/storage-report")
    async def storage_report_handler(request: Request):
        """Receive storage report from sllm-store.

        Updates both the in-memory StorageManager cache (for fast placement)
        and the SQLite database (for persistence).
        """
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {str(e)}"
            )

        node_name = body.get("node_name")
        if not node_name:
            raise HTTPException(status_code=400, detail="Missing 'node_name'")

        # Try to use StorageManager for in-memory cache + DB update
        storage_manager = getattr(request.app.state, "storage_manager", None)
        if storage_manager:
            try:
                from sllm.storage_manager import StorageReport

                report = StorageReport(
                    node_name=node_name,
                    sllm_store_endpoint=body.get("sllm_store_endpoint"),
                    cached_models=body.get("cached_models", []),
                )
                await storage_manager.handle_storage_report(report)
            except Exception as e:
                logger.warning(f"StorageManager update failed: {e}")
                # Fall back to direct DB update
                db: Database = request.app.state.database
                db.upsert_node_storage(
                    node_name=node_name,
                    sllm_store_endpoint=body.get("sllm_store_endpoint"),
                    cached_models=body.get("cached_models", []),
                )
        else:
            # No StorageManager, direct DB update
            db: Database = request.app.state.database
            db.upsert_node_storage(
                node_name=node_name,
                sllm_store_endpoint=body.get("sllm_store_endpoint"),
                cached_models=body.get("cached_models", []),
            )

        logger.debug(
            f"Received storage report from {node_name}: "
            f"{len(body.get('cached_models', []))} models cached"
        )

        return {"status": "ok"}

    return app
