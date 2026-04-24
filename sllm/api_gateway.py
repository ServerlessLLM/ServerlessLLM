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


from sllm.batch_scheduler import BatchScheduler

# ...

def create_app(
    database: Optional[Database] = None,
    pylet_client: Optional[PyletClient] = None,
    router: Optional[Router] = None,
    autoscaler: Optional[AutoScaler] = None,
    config: Optional[Any] = None,
) -> FastAPI:
    """
    Create the SLLM API Gateway FastAPI application.
    """
    
    # Initialize Scheduler if database and router are present and enabled
    scheduler: Optional[BatchScheduler] = None
    if database and router:
        if os.getenv("ENABLE_BATCH_SCHEDULER", "1").lower() in ("1", "true", "yes"):
            scheduler = BatchScheduler(database, router)
            logger.info("BatchScheduler enabled via config")
        else:
            logger.info("BatchScheduler disabled via config")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store dependencies in app state
        app.state.database = database
        app.state.pylet_client = pylet_client
        app.state.router = router
        app.state.autoscaler = autoscaler
        app.state.config = config
        app.state.scheduler = scheduler

        # Connect Router to Autoscaler for metrics push
        if router and autoscaler:
            router.set_autoscaler(autoscaler)

        # Connect Scheduler to Autoscaler for proactive scaling
        if scheduler and autoscaler:
            scheduler.set_autoscaler(autoscaler)

        # Connect Scheduler to StorageManager for checkpoint prefetch
        storage_manager = getattr(app.state, "storage_manager", None)
        if scheduler and storage_manager:
            scheduler.set_storage_manager(storage_manager)

        # Connect Scheduler to PyletClient for GPU-aware scaling
        if scheduler and pylet_client:
            scheduler.set_pylet_client(pylet_client)

        # Start router if provided
        if router:
            await router.start()

        # Start Scheduler if initialized (default: BatchScheduler)
        if scheduler:
            await scheduler.start()

        logger.info("API Gateway started")
        yield

        # Cleanup
        if scheduler and scheduler.running:
            await scheduler.stop()

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
        existing = db.get_deployment(model_name, backend)
        if existing:
            if existing.status == "deleting":
                # Following K8s behavior: do nothing while deletion is in progress
                logger.info(
                    f"Deployment {deployment_id} is being deleted, "
                    "ignoring create request"
                )
                return {
                    "deployment_id": deployment_id,
                    "status": existing.status,
                    "message": (
                        f"Deployment {deployment_id} is currently being deleted. "
                        "Please wait for deletion to complete and retry."
                    ),
                }
            else:
                # Deployment exists and is active
                logger.warning(f"Deployment {deployment_id} already exists")
                return {
                    "deployment_id": deployment_id,
                    "status": existing.status,
                    "message": f"Deployment {deployment_id} already exists",
                }

        # Parse configuration
        backend_config = body.get("backend_config", {})
        auto_scaling_config = body.get("auto_scaling_config", {})

        try:
            # Create deployment in database
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
    # File Management Endpoints
    # -------------------------------------------------------------------------

    MAX_UPLOAD_SIZE = int(os.getenv("SLLM_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100MB default
    MAX_TASKS_PER_BATCH = int(os.getenv("SLLM_MAX_TASKS_PER_BATCH", "50000"))

    @app.post("/v1/files")
    async def upload_file_handler(request: Request):
        """Upload a file that contains batch requests."""
        # Check Content-Length header early to reject oversized uploads
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE} bytes"
            )

        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid form data: {str(e)}")

        file = form.get("file")
        purpose = form.get("purpose", "batch")

        if not file or not hasattr(file, "filename"):
            raise HTTPException(status_code=400, detail="Missing file in form data")

        import uuid
        import os as _os
        file_id = f"file_{uuid.uuid4().hex[:12]}"

        # Save file to disk
        upload_dir = "sllm_files"
        _os.makedirs(upload_dir, exist_ok=True)
        file_path = _os.path.join(upload_dir, f"{file_id}.jsonl")

        file_content = await file.read()

        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE} bytes"
            )

        with open(file_path, "wb") as f:
            f.write(file_content)
            
        db: Database = request.app.state.database
        file_obj = db.create_file(
            file_id=file_id,
            filename=file.filename,
            bytes_size=len(file_content),
            purpose=purpose
        )
        
        return {
            "id": file_obj.id,
            "object": "file",
            "bytes": file_obj.bytes,
            "created_at": file_obj.created_at,
            "filename": file_obj.filename,
            "purpose": file_obj.purpose
        }

    @app.get("/v1/files")
    async def list_files_handler(request: Request):
        db: Database = request.app.state.database
        files = db.get_all_files()
        return {
            "object": "list",
            "data": [
                {
                    "id": f.id,
                    "object": "file",
                    "bytes": f.bytes,
                    "created_at": f.created_at,
                    "filename": f.filename,
                    "purpose": f.purpose
                } for f in files
            ]
        }

    # -------------------------------------------------------------------------
    # Batch Job Endpoints
    # -------------------------------------------------------------------------

    @app.post("/v1/batches")
    async def create_batch_handler(request: Request):
        """Create a new batch job."""
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON payload: {str(e)}"
            )

        input_file_id = body.get("input_file_id")
        tasks = body.get("tasks")
        if not input_file_id and not tasks:
            raise HTTPException(
                status_code=400,
                detail="Request body must include either 'tasks' list or 'input_file_id'",
            )

        import uuid
        import json
        import os

        db: Database = request.app.state.database
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        # Handle file-based tasks
        if input_file_id:
            logger.info(f"Processing batch from file: {input_file_id}")
            file_obj = db.get_file(input_file_id)
            if not file_obj:
                raise HTTPException(status_code=404, detail=f"File {input_file_id} not found")
                
            file_path = f"sllm_files/{input_file_id}.jsonl"
            if not os.path.exists(file_path):
                raise HTTPException(status_code=500, detail="File content missing on disk")
                
            tasks = []
            line_idx = 0
            try:
                with open(file_path, "r") as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        task_data = json.loads(line)
                        tasks.append(task_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse jsonl file at line {line_idx+1}: {e}")

        # Validate tasks
        if len(tasks) > MAX_TASKS_PER_BATCH:
            raise HTTPException(
                status_code=400,
                detail=f"Too many tasks ({len(tasks)}). Maximum is {MAX_TASKS_PER_BATCH} per batch."
            )

        for task in tasks:
            if not all(
                k in task for k in ("custom_id", "method", "url", "body")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Each task must have custom_id, method, url, and body",
                )

        try:
            # Create batch job
            metadata = body.get("metadata", {})

            db.create_batch_job(batch_id, metadata=metadata, input_file_id=input_file_id)

            # Bulk-insert tasks in a single transaction (off event loop)
            task_rows = [
                {
                    "task_id": f"task_{uuid.uuid4().hex[:16]}",
                    "custom_id": task["custom_id"],
                    "method": task["method"],
                    "url": task["url"],
                    "body": task["body"],
                }
                for task in tasks
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, db.create_batch_tasks_bulk, task_rows, batch_id
            )

            logger.info(f"Created batch job {batch_id} with {len(tasks)} tasks")

            return {
                "id": batch_id,
                "object": "batch",
                "status": "pending",
                "request_counts": {
                    "total": len(tasks),
                    "completed": 0,
                    "failed": 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to create batch job: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal error creating batch job",
            )

    @app.get("/v1/batches/{batch_id}")
    async def get_batch_handler(batch_id: str, request: Request):
        """Get batch job status."""
        db: Database = request.app.state.database
        batch_job = db.get_batch_job(batch_id)

        if not batch_job:
            raise HTTPException(
                status_code=404, detail=f"Batch job {batch_id} not found"
            )

        batch_tasks = db.get_batch_tasks(batch_id)

        completed_count = sum(
            1 for t in batch_tasks if t.status == "completed"
        )
        failed_count = sum(1 for t in batch_tasks if t.status == "failed")

        return {
            "id": batch_job.id,
            "object": "batch",
            "status": batch_job.status,
            "metadata": batch_job.metadata,
            "created_at": batch_job.created_at,
            "request_counts": {
                "total": len(batch_tasks),
                "completed": completed_count,
                "failed": failed_count,
            },
            "tasks": [
                {
                    "id": t.id,
                    "custom_id": t.custom_id,
                    "status": t.status,
                    "body": t.body,
                    "output": t.output,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                }
                for t in batch_tasks
            ],
        }

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

    # ========================================================================
    # Admin Endpoints for Experiment Control
    # ========================================================================

    @app.post("/admin/set_strategy")
    async def set_batch_strategy(request: Request):
        """Set batch scheduling strategy at runtime (for experiments).

        Body:
            strategy: "sync", "chunked", or "semaphore"
            buffer_limit: Concurrency limit (default: 10)
            enable_model_grouping: Whether to sort tasks by model (default: true)
            enable_johnsons_rule: Whether to reorder groups via Johnson's Rule (default: true)
            enable_prefetch: Whether to prefetch next model checkpoint (default: current)
            prefetch_threshold: Fraction of group done before prefetch fires (default: current)
        """
        scheduler = request.app.state.scheduler
        if not scheduler:
            raise HTTPException(status_code=503, detail="Batch scheduler not available")

        body = await request.json()
        strategy = body.get("strategy", "semaphore")
        buffer_limit = body.get("buffer_limit", 10)
        enable_model_grouping = body.get("enable_model_grouping", True)
        enable_johnsons_rule = body.get("enable_johnsons_rule", True)
        enable_prefetch = body.get("enable_prefetch", scheduler.enable_prefetch)
        prefetch_threshold = body.get("prefetch_threshold", scheduler.prefetch_threshold)
        forced_group_order = body.get("force_group_order", [])

        try:
            scheduler.set_strategy(strategy, buffer_limit, enable_model_grouping, enable_johnsons_rule, forced_group_order)
            scheduler.enable_prefetch = enable_prefetch
            scheduler.prefetch_threshold = prefetch_threshold

            # Tensor parallelism config: {"model_name": tp_size, ...}
            tp_config = body.get("tp_config")
            if tp_config and isinstance(tp_config, dict):
                scheduler.set_tp_config(tp_config)

            if not scheduler.running:
                await scheduler.start()

            logger.info(
                f"Scheduler: prefetch={enable_prefetch}, threshold={prefetch_threshold}, johnsons_rule={enable_johnsons_rule}"
            )

            return {
                "status": "ok",
                "strategy": strategy,
                "buffer_limit": buffer_limit,
                "enable_model_grouping": enable_model_grouping,
                "enable_johnsons_rule": enable_johnsons_rule,
                "enable_prefetch": enable_prefetch,
                "prefetch_threshold": prefetch_threshold,
                "tp_config": {**scheduler._auto_tp_cache, **scheduler._tp_config},
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/admin/get_strategy")
    async def get_batch_strategy(request: Request):
        """Get current batch scheduling strategy."""
        scheduler = request.app.state.scheduler
        if not scheduler:
            raise HTTPException(status_code=503, detail="Batch scheduler not available")

        return {
            "strategy": scheduler.strategy,
            "buffer_limit": scheduler.buffer_limit,
            "enable_model_grouping": scheduler.enable_model_grouping,
            "enable_johnsons_rule": scheduler.enable_johnsons_rule,
            "enable_prefetch": scheduler.enable_prefetch,
            "prefetch_threshold": scheduler.prefetch_threshold,
            "tp_config": {**scheduler._auto_tp_cache, **scheduler._tp_config},
        }

    return app
