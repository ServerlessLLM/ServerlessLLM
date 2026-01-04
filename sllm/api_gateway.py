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
- Deployment registration and deletion
- Single global Router for load balancing
- SQLite for deployment configuration
- Pylet for instance information

Terminology:
- Deployment: A (model_name, backend) pair - the basic scheduling unit
- deployment_id: Unique identifier (format: "{model_name}:{backend}")
- model_name: HuggingFace model name (what users specify in requests)
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sllm.autoscaler import AutoScaler
from sllm.database import Database, Deployment
from sllm.logger import init_logger
from sllm.pylet_client import PyletClient
from sllm.router import Router, RouterConfig

logger = init_logger(__name__)

# CORS origins
origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin for origin in origins_env.split(",") if origin]
origins += ["http://localhost", "http://localhost:3000"]


def create_app(
    database: Optional[Database] = None,
    pylet_client: Optional[PyletClient] = None,
    router: Optional[Router] = None,
    autoscaler: Optional[AutoScaler] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """
    Create the SLLM API Gateway FastAPI application.

    Args:
        database: SQLite database instance
        pylet_client: Pylet client instance (may be None if Pylet unavailable)
        router: Global Router instance for request routing
        autoscaler: AutoScaler instance (for connecting Router to it)
        config: Head configuration

    Returns:
        FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store dependencies in app state
        app.state.database = database
        app.state.pylet_client = pylet_client
        app.state.router = router
        app.state.autoscaler = autoscaler
        app.state.config = config

        # Connect Router to Autoscaler for metrics push
        if router and autoscaler:
            router.set_autoscaler(autoscaler)

        # Start router if provided
        if router:
            await router.start()

        logger.info("API Gateway started")
        yield

        # Cleanup
        if router:
            await router.drain(timeout=10.0)
            await router.stop()
        logger.info("API Gateway shutdown")

    app = FastAPI(
        lifespan=lifespan,
        title="ServerlessLLM API Gateway",
        version="1.0.0-beta",
    )

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
    # Deployment Management Endpoints
    # -------------------------------------------------------------------------

    @app.post("/deployments")
    async def register_handler(request: Request):
        """Register a new deployment."""
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
        deployment_id = Deployment.make_id(model_name, backend)

        # Check if already exists
        db: Database = request.app.state.database
        if db.get_deployment(model_name, backend):
            raise HTTPException(
                status_code=409,
                detail=f"Deployment {deployment_id} is already registered",
            )

        # Parse configuration
        backend_config = body.get("backend_config", {})
        auto_scaling_config = body.get("auto_scaling_config", {})

        try:
            deployment = db.create_deployment(
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

            logger.info(f"Registered deployment {deployment_id}")

            return {
                "deployment_id": deployment_id,
                "status": "active",
                "message": f"Deployment {deployment_id} registered successfully",
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to register deployment: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Deployment registration failed due to internal error",
            )

    @app.delete("/deployments/{deployment_id:path}")
    async def delete_deployment_handler(deployment_id: str, request: Request):
        """Delete a deployment.

        Returns 202 Accepted immediately. The Reconciler will:
        1. Stop all instances via Pylet
        2. Remove endpoints from deployment_endpoints table
        3. Delete the deployment from the database
        """
        db: Database = request.app.state.database

        deployment = db.get_deployment_by_id(deployment_id)
        if not deployment:
            raise HTTPException(
                status_code=404, detail=f"Deployment {deployment_id} not found"
            )

        if deployment.status == "deleting":
            # Already deleting
            return JSONResponse(
                status_code=202,
                content={
                    "status": "deleting",
                    "deployment_id": deployment_id,
                    "message": "Deletion already in progress",
                },
            )

        try:
            # Mark as deleting and set desired=0
            # The Reconciler will handle the actual cleanup:
            # - Cancel instances in Pylet
            # - Remove endpoints from deployment_endpoints table
            # - Delete from database
            db.update_deployment_status(deployment_id, "deleting")
            db.update_desired_replicas(deployment_id, 0)

            logger.info(f"Deployment {deployment_id} marked for deletion")

            return JSONResponse(
                status_code=202,
                content={
                    "status": "deleting",
                    "deployment_id": deployment_id,
                    "message": "Deletion in progress. Instances will be stopped by reconciler.",
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to mark deployment {deployment_id} for deletion: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete deployment: {str(e)}",
            )

    # -------------------------------------------------------------------------
    # Inference Endpoints
    # -------------------------------------------------------------------------
    async def inference_handler(
        request: Request,
        path: str = "/v1/chat/completions",
    ) -> JSONResponse:
        """Forward inference request to Router."""
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON body: {e}")
            raise HTTPException(
                status_code=400, detail="Invalid JSON in request body"
            )

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Request body must include a 'model' field",
            )

        # Pop backend field (non-standard, remove before forwarding)
        backend = body.pop("backend", "vllm")
        deployment_id = Deployment.make_id(model_name, backend)

        # Verify deployment exists
        db: Database = request.app.state.database
        deployment = db.get_deployment(model_name, backend)
        if not deployment or deployment.status != "active":
            raise HTTPException(
                status_code=404,
                detail=f"Deployment '{deployment_id}' not found or not active",
            )

        # Get router
        router: Router = request.app.state.router
        if not router:
            raise HTTPException(
                status_code=503,
                detail="Router not available",
            )

        # Forward to router
        try:
            result = await router.handle_request(
                body, path, deployment_id=deployment_id
            )
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
    async def get_deployments(request: Request):
        """List all registered deployments (OpenAI-compatible endpoint)."""
        db: Database = request.app.state.database
        router: Router = request.app.state.router

        deployments = db.get_all_deployments()
        deployment_list = []

        for deployment in deployments:
            ready_endpoints = (
                router.get_endpoint_count(deployment.id) if router else 0
            )

            deployment_list.append(
                {
                    "id": deployment.id,
                    "model": deployment.model_name,
                    "backend": deployment.backend,
                    "status": deployment.status,
                    "desired_replicas": deployment.desired_replicas,
                    "ready_replicas": ready_endpoints,
                    "min_replicas": deployment.min_replicas,
                    "max_replicas": deployment.max_replicas,
                }
            )

        return {"object": "list", "data": deployment_list}

    @app.get("/status")
    async def cluster_status(request: Request):
        """Get comprehensive cluster status."""
        db: Database = request.app.state.database
        router: Router = request.app.state.router
        pylet_client: Optional[PyletClient] = request.app.state.pylet_client

        # Get deployments
        deployments = db.get_all_deployments()
        deployment_status = []

        for deployment in deployments:
            endpoints = db.get_deployment_endpoints(deployment.id)

            # Get instances from Pylet if available
            instances = []
            if pylet_client:
                try:
                    pylet_instances = (
                        await pylet_client.get_deployment_instances(
                            deployment.id
                        )
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

            deployment_status.append(
                {
                    "id": deployment.id,
                    "status": deployment.status,
                    "desired_replicas": deployment.desired_replicas,
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
            "deployments": deployment_status,
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
