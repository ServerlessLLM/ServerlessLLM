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
import random
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
from sllm.storage_manager import StorageManager

logger = init_logger(__name__)


async def _select_download_node(pylet: PyletClient) -> str:
    """Select a random online worker for model download.

    Args:
        pylet: Pylet client for querying workers

    Returns:
        Worker ID of selected node

    Raises:
        HTTPException: If no workers are available
    """
    workers = await pylet.get_online_workers()
    if not workers:
        raise HTTPException(
            status_code=503,
            detail="No worker nodes available for model download",
        )
    return random.choice(workers).worker_id


# CORS origins
origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin for origin in origins_env.split(",") if origin]
origins += ["http://localhost", "http://localhost:3000"]


def create_app(
    database: Optional[Database] = None,
    pylet_client: Optional[PyletClient] = None,
    router: Optional[Router] = None,
    autoscaler: Optional[AutoScaler] = None,
    storage_manager: Optional[StorageManager] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """
    Create the SLLM API Gateway FastAPI application.

    Args:
        database: SQLite database instance
        pylet_client: Pylet client instance (may be None if Pylet unavailable)
        router: Global Router instance for request routing
        autoscaler: AutoScaler instance (for connecting Router to it)
        storage_manager: StorageManager for model downloads and cache info
        config: Head configuration

    Returns:
        FastAPI application
    """

    async def _recover_stale_downloads(
        db: Database,
        sm: Optional[StorageManager],
        pylet: Optional[PyletClient],
    ):
        """Recover deployments stuck in 'downloading' status after restart.

        On startup, check for deployments that were downloading when the
        server stopped. Either mark them ready (if model is now cached)
        or retry the download.
        """
        downloading = db.get_downloading_deployments()
        if not downloading:
            return

        logger.info(
            f"Recovering {len(downloading)} stale downloading deployments"
        )

        for deployment in downloading:
            try:
                # Check if model is now cached
                if sm:
                    nodes_with_model = sm.get_nodes_with_model(
                        deployment.model_name
                    )
                    if nodes_with_model:
                        # Model is available, mark as active
                        db.update_deployment_download_status(
                            deployment.id,
                            status="active",
                            download_node=nodes_with_model[0],
                        )
                        logger.info(
                            f"Recovered {deployment.id}: model found on {nodes_with_model[0]}"
                        )
                        continue

                # Model not cached, try to retry download
                if pylet and sm:
                    workers = await pylet.get_online_workers()
                    if not workers:
                        db.update_deployment_download_status(
                            deployment.id,
                            status="failed",
                            failure_reason="No workers available after restart",
                        )
                        logger.warning(
                            f"Recovery failed for {deployment.id}: no workers available"
                        )
                        continue

                    # Check if original download_node is online
                    online_ids = {w.worker_id for w in workers}
                    if (
                        deployment.download_node
                        and deployment.download_node in online_ids
                    ):
                        download_node = deployment.download_node
                    else:
                        download_node = random.choice(workers).worker_id

                    logger.info(
                        f"Retrying download for {deployment.id} on {download_node}"
                    )

                    # Trigger async download
                    asyncio.create_task(
                        _trigger_model_download(
                            sm,
                            db,
                            deployment.id,
                            deployment.model_name,
                            deployment.backend,
                            download_node,
                        )
                    )
                else:
                    # No pylet/sm, mark as failed
                    db.update_deployment_download_status(
                        deployment.id,
                        status="failed",
                        failure_reason="Cannot retry download: missing dependencies after restart",
                    )
                    logger.warning(
                        f"Recovery failed for {deployment.id}: no pylet/storage_manager"
                    )

            except Exception as e:
                logger.error(f"Error recovering {deployment.id}: {e}")
                db.update_deployment_download_status(
                    deployment.id,
                    status="failed",
                    failure_reason=f"Recovery error: {e}",
                )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store dependencies in app state
        # Note: Some may already be set on app.state before lifespan runs
        # (e.g., storage_manager is created after create_app but before serve)
        app.state.database = database
        app.state.pylet_client = pylet_client
        app.state.router = router
        app.state.autoscaler = autoscaler
        # Only set storage_manager if passed (may be set later externally)
        if storage_manager is not None:
            app.state.storage_manager = storage_manager
        elif not hasattr(app.state, "storage_manager"):
            app.state.storage_manager = None
        app.state.config = config

        # Connect Router to Autoscaler for metrics push
        if router and autoscaler:
            router.set_autoscaler(autoscaler)

        # Recover stale downloads from previous run
        if database:
            await _recover_stale_downloads(
                database, storage_manager, pylet_client
            )

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
        """Register a new deployment.

        Checks if model is available on any node. If not, triggers async
        download and returns status='downloading'. Otherwise returns 'ready'.
        """
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

        db: Database = request.app.state.database
        sm: Optional[StorageManager] = request.app.state.storage_manager
        pylet: Optional[PyletClient] = request.app.state.pylet_client

        # Check if model is cached on any node
        model_cached = False
        download_node = None
        nodes_with_model = []

        if sm:
            nodes_with_model = sm.get_nodes_with_model(model_name)
            if nodes_with_model:
                model_cached = True
                logger.info(
                    f"Model {model_name} already cached on nodes: {nodes_with_model}"
                )

        # Check if deployment already exists
        existing = db.get_deployment(model_name, backend)
        if existing:
            # Deployment exists - check if we need to trigger download
            if model_cached:
                # Model is available, truly a conflict
                raise HTTPException(
                    status_code=409,
                    detail=f"Deployment {deployment_id} is already registered",
                )
            elif existing.status == "downloading":
                # Already downloading
                return JSONResponse(
                    status_code=202,
                    content={
                        "deployment_id": deployment_id,
                        "status": "downloading",
                        "download_node": existing.download_node,
                        "message": f"Model download already in progress on {existing.download_node}",
                    },
                )
            else:
                # Deployment exists but model not cached - trigger download
                if pylet:
                    download_node = await _select_download_node(pylet)

                    # Update deployment to downloading status
                    db.update_deployment_download_status(
                        deployment_id,
                        status="downloading",
                        download_node=download_node,
                    )

                    # Trigger async download
                    asyncio.create_task(
                        _trigger_model_download(
                            sm,
                            db,
                            deployment_id,
                            model_name,
                            backend,
                            download_node,
                        )
                    )

                    logger.info(
                        f"Deployment {deployment_id} exists but model not cached, "
                        f"triggering download to {download_node}"
                    )

                    return JSONResponse(
                        status_code=202,
                        content={
                            "deployment_id": deployment_id,
                            "status": "downloading",
                            "download_node": download_node,
                            "message": f"Model download started on {download_node}. Check status with 'sllm status'.",
                        },
                    )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="No Pylet client available to check workers",
                    )

        # New deployment - select node for download if model not cached
        if not model_cached:
            if pylet:
                download_node = await _select_download_node(pylet)
                logger.info(
                    f"Model {model_name} not cached, will download to {download_node}"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Model not cached and Pylet client is unavailable to orchestrate a download.",
                )

        # Parse configuration
        backend_config = body.get("backend_config", {})
        auto_scaling_config = body.get("auto_scaling_config", {})

        try:
            # Determine initial status based on model availability
            if model_cached:
                initial_status = "active"
            else:
                initial_status = "downloading"

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
                initial_status=initial_status,
                download_node=download_node,
            )

            logger.info(
                f"Registered deployment {deployment_id} with status={initial_status}"
            )

            # If downloading, trigger async download task
            if initial_status == "downloading" and sm and download_node:
                asyncio.create_task(
                    _trigger_model_download(
                        sm,
                        db,
                        deployment_id,
                        model_name,
                        backend,
                        download_node,
                    )
                )

            # Return appropriate response
            if initial_status == "active":
                return {
                    "deployment_id": deployment_id,
                    "status": "active",
                    "message": f"Deployment {deployment_id} registered successfully",
                }
            else:
                return JSONResponse(
                    status_code=202,
                    content={
                        "deployment_id": deployment_id,
                        "status": "downloading",
                        "download_node": download_node,
                        "message": f"Model download started on {download_node}. Check status with 'sllm status'.",
                    },
                )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to register deployment: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Deployment registration failed due to internal error",
            )

    async def _update_status_with_retry(
        db: Database,
        deployment_id: str,
        status: str,
        download_node: Optional[str] = None,
        failure_reason: Optional[str] = None,
        max_retries: int = 3,
    ) -> bool:
        """Update deployment status with retry logic.

        Retries on database errors to handle transient failures.
        Returns True if update succeeded, False if deployment was deleted.
        """
        for attempt in range(max_retries):
            try:
                updated = db.update_deployment_download_status_if_not_deleting(
                    deployment_id,
                    status=status,
                    download_node=download_node,
                    failure_reason=failure_reason,
                )
                return updated
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Status update failed for {deployment_id}, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.critical(
                        f"CRITICAL: Failed to update status for {deployment_id} after "
                        f"{max_retries} attempts. Manual intervention may be required. "
                        f"Deployment may be stuck in 'downloading' status. Error: {e}"
                    )
                    return False

    async def _trigger_model_download(
        sm: StorageManager,
        db: Database,
        deployment_id: str,
        model_name: str,
        backend: str,
        node_name: str,
    ):
        """Background task to download model and update deployment status."""
        try:
            logger.info(
                f"Starting model download for {deployment_id} on {node_name}"
            )
            success = await sm.download_model_on_node(
                node_name, model_name, backend
            )

            if success:
                # Use conditional update with retry to handle transient DB errors
                updated = await _update_status_with_retry(
                    db, deployment_id, status="active", download_node=node_name
                )
                if updated:
                    logger.info(f"Deployment {deployment_id} is now active")
                else:
                    logger.info(
                        f"Deployment {deployment_id} was deleted during download, "
                        "skipping status update"
                    )
            else:
                updated = await _update_status_with_retry(
                    db,
                    deployment_id,
                    status="failed",
                    download_node=node_name,
                    failure_reason=f"Download failed on {node_name}",
                )
                if updated:
                    logger.error(f"Deployment {deployment_id} download failed")
                else:
                    logger.info(
                        f"Deployment {deployment_id} was deleted during download"
                    )

        except Exception as e:
            logger.error(f"Error downloading model for {deployment_id}: {e}")
            await _update_status_with_retry(
                db,
                deployment_id,
                status="failed",
                download_node=node_name,
                failure_reason=str(e),
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

        # Verify deployment exists and is ready for inference
        db: Database = request.app.state.database
        deployment = db.get_deployment(model_name, backend)
        if not deployment or deployment.status != "active":
            status_msg = f" (status: {deployment.status})" if deployment else ""
            raise HTTPException(
                status_code=404,
                detail=f"Deployment '{deployment_id}' not found or not active{status_msg}",
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
