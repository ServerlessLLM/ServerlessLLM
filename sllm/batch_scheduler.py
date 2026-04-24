
import asyncio
import math
import os
import subprocess
from typing import Optional, List, Dict, Set
from datetime import datetime, timezone
from sllm.database import Database, BatchTask, BatchJob
from sllm.logger import init_logger
from sllm.router import Router

logger = init_logger(__name__)


class BatchScheduler:
    """Offline batch scheduler for ServerlessLLM.

    Designed for offline workloads where all tasks within a batch are
    submitted before execution begins.  The scheduler processes one batch
    at a time: it collects all pending tasks, groups them by model name
    (and optionally by prefix hash), then executes each group sequentially
    with aggressive checkpoint prefetching to overlap I/O with compute.

    Multi-GPU utilisation is handled by the autoscaler and router — the
    scheduler creates deployments with ``max_replicas`` equal to the
    number of available GPUs, and the router load-balances inference
    requests across all live endpoints.
    """

    def __init__(self, database: Database, router: Router):
        self.database = database
        self.router = router
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
        self.autoscaler = None
        self.pylet_client = None

        # Track active batches to prevent premature scale-down
        self._active_batch_jobs: set = set()

        # Prevent concurrent processing of same batch
        self._processing_batches: set = set()
        self._batch_lock = asyncio.Lock()

        # Runtime-configurable scheduling strategy
        self.strategy = "semaphore"
        self._router_buffer_size = getattr(self.router.config, 'max_buffer_size', 100)
        self.buffer_limit = self._router_buffer_size
        self.enable_model_grouping = True

        # Checkpoint prefetch configuration
        self.storage_manager = None
        self.enable_prefetch = True
        self.prefetch_threshold = 0.0  # Eager: prefetch all upcoming models immediately

        # Johnson's Rule: reorder model groups to minimise I/O idle
        self.enable_johnsons_rule = True

        # Forced group ordering for experiments (overrides both alphabetical and JR)
        self.forced_group_order: list = []

        # Tensor parallelism: model_name → number of GPUs required.
        # Auto-computed from model size vs GPU memory; manual overrides
        # via set_tp_config() take priority.
        self._tp_config: Dict[str, int] = {}
        self._auto_tp_cache: Dict[str, int] = {}
        self._gpu_mem_bytes: int = 0  # per-GPU memory, detected lazily

        # Per-batch concurrency limit
        self.max_concurrent_tasks_per_batch = max(10, self._router_buffer_size // 2)

        # Global concurrency limit across ALL batches
        self._global_semaphore = asyncio.Semaphore(self._router_buffer_size)

        # Models confirmed ready in current processing cycle
        self._models_ready: set = set()

        # ----- Memory-aware prefetch -----
        self.cpu_mem_pool_bytes: int = 0  # 0 = unknown, skip check
        self._estimated_model_sizes: Dict[str, int] = {}
        self._cpu_cached_models: Set[str] = set()

    # ------------------------------------------------------------------ #
    #  Setters (called by API Gateway during lifespan)                    #
    # ------------------------------------------------------------------ #

    def set_autoscaler(self, autoscaler):
        self.autoscaler = autoscaler
        logger.info("BatchScheduler connected to AutoScaler")

    def set_pylet_client(self, pylet_client):
        self.pylet_client = pylet_client
        logger.info("BatchScheduler connected to PyletClient")

    def set_storage_manager(self, storage_manager):
        self.storage_manager = storage_manager
        logger.info("BatchScheduler connected to StorageManager for prefetch")

    def set_tp_config(self, tp_config: Dict[str, int]):
        """Set manual tensor parallelism overrides per model.

        Args:
            tp_config: mapping of model_name → tensor_parallel_size.
                       Models not listed are auto-computed from model
                       size vs GPU memory.
        """
        self._tp_config = dict(tp_config)
        logger.info(f"BatchScheduler TP manual overrides: {self._tp_config}")

    def _detect_gpu_memory(self) -> int:
        """Detect per-GPU memory in bytes (cached after first call)."""
        if self._gpu_mem_bytes > 0:
            return self._gpu_mem_bytes
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
                logger.info(
                    f"Detected GPU memory: {self._gpu_mem_bytes / 1e9:.1f} GB"
                )
                return self._gpu_mem_bytes
        except Exception:
            pass
        # Fallback: parse nvidia-smi
        try:
            # Treat empty-string CUDA_VISIBLE_DEVICES like unset (startup script
            # may export it empty when caller hasn't set it).
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
            gpu_ids = cvd.split(",")[0]
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_ids}",
                 "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            mem_mib = int(result.stdout.strip().split("\n")[0])
            self._gpu_mem_bytes = mem_mib * 1024 * 1024
            logger.info(f"Detected GPU memory (nvidia-smi): {self._gpu_mem_bytes / 1e9:.1f} GB")
            return self._gpu_mem_bytes
        except Exception as e:
            logger.warning(f"Cannot detect GPU memory: {e}")
            return 0

    # vLLM reserves ~10% for KV cache overhead and runtime buffers.
    _VLLM_WEIGHT_FRACTION = 0.80

    def _compute_tp(self, model_name: str) -> int:
        """Auto-compute tensor_parallel_size from model size vs GPU memory.

        Returns the smallest power-of-2 TP that fits the model weights
        within the usable fraction of GPU memory. Falls back to 1 if
        model size is unknown or GPU memory cannot be detected.
        """
        model_bytes = self._get_checkpoint_size_bytes(model_name)
        if model_bytes <= 0:
            return 1
        gpu_mem = self._detect_gpu_memory()
        if gpu_mem <= 0:
            return 1
        usable = gpu_mem * self._VLLM_WEIGHT_FRACTION
        if model_bytes <= usable:
            return 1
        # TP must be a power of 2 for NCCL efficiency
        raw_tp = math.ceil(model_bytes / usable)
        tp = 1
        while tp < raw_tp:
            tp *= 2
        return tp

    def _get_tp(self, model_name: str) -> int:
        """Return tensor_parallel_size for a model.

        Priority: manual override > auto-cache > compute from model size.
        """
        if model_name in self._tp_config:
            return self._tp_config[model_name]
        if model_name in self._auto_tp_cache:
            return self._auto_tp_cache[model_name]
        tp = self._compute_tp(model_name)
        self._auto_tp_cache[model_name] = tp
        if tp > 1:
            logger.info(
                f"Auto-detected TP={tp} for {model_name} "
                f"(weights={self._estimated_model_sizes.get(model_name, 0) / 1e9:.1f}GB, "
                f"GPU={self._gpu_mem_bytes / 1e9:.1f}GB)"
            )
        return tp

    # ------------------------------------------------------------------ #
    #  Strategy                                                           #
    # ------------------------------------------------------------------ #

    def set_strategy(self, strategy: str, buffer_limit: int = 10, enable_model_grouping: bool = True, enable_johnsons_rule: bool = True, forced_group_order: list = None):
        valid_strategies = ["sync", "chunked", "semaphore"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        self.strategy = strategy
        effective_limit = min(buffer_limit, self._router_buffer_size)
        if effective_limit != buffer_limit:
            logger.info(f"buffer_limit {buffer_limit} capped to Router max_buffer_size {self._router_buffer_size}")
        self.buffer_limit = effective_limit
        self.enable_model_grouping = enable_model_grouping
        self.enable_johnsons_rule = enable_johnsons_rule
        self.forced_group_order = forced_group_order or []
        logger.info(f"Strategy updated: {strategy}, buffer_limit={self.buffer_limit}, model_grouping={enable_model_grouping}, johnsons_rule={enable_johnsons_rule}, forced_group_order={self.forced_group_order}")

    def _unregister_batch(self, batch_id: str, tasks: List[BatchTask]):
        """Remove batch from active tracking and unregister from autoscaler."""
        if batch_id in self._active_batch_jobs:
            self._active_batch_jobs.remove(batch_id)
            logger.info(f"Batch {batch_id} completed, allowing scale-down")
            if self.autoscaler and tasks:
                models = set(t.body.get("model") for t in tasks if t.body.get("model"))
                for model_name in models:
                    self.autoscaler.unregister_active_batch(f"{model_name}:vllm", batch_id)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def start(self):
        if self.running:
            return
        self._recover_stale_batches()
        self.running = True
        self._loop_task = asyncio.create_task(self._schedule_loop())
        logger.info("BatchScheduler started")

    async def stop(self):
        logger.info("Stopping BatchScheduler...")
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped")

    def _recover_stale_batches(self):
        stale_ids = [
            bid for bid in self.database.get_pending_batch_ids()
            if self.database.get_batch_job(bid)
            and self.database.get_batch_job(bid).status == "in_progress"
        ]
        for bid in stale_ids:
            tasks = self.database.get_batch_tasks(bid)
            if tasks and all(t.status in ("completed", "failed") for t in tasks):
                self.database.update_batch_job_status(bid, "completed")
                logger.info(f"Recovered stale batch {bid} -> completed")
            elif not tasks:
                self.database.update_batch_job_status(bid, "completed")
                logger.info(f"Recovered empty stale batch {bid} -> completed")
            else:
                self.database.update_batch_job_status(bid, "pending")
                logger.info(f"Recovered stale batch {bid} -> pending (has unfinished tasks)")

    # ------------------------------------------------------------------ #
    #  Schedule Loop                                                      #
    # ------------------------------------------------------------------ #

    async def _schedule_loop(self):
        background_tasks: set = set()

        while self.running:
            try:
                pending_batch_ids = self.database.get_pending_batch_ids()

                for batch_id in pending_batch_ids:
                    async with self._batch_lock:
                        if batch_id not in self._processing_batches:
                            self._processing_batches.add(batch_id)
                            task = asyncio.create_task(self._process_batch_safe(batch_id))
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        if background_tasks:
            logger.info(f"Waiting for {len(background_tasks)} in-flight batches to finish...")
            await asyncio.gather(*background_tasks, return_exceptions=True)

    # ------------------------------------------------------------------ #
    #  Shared: Deployment Readiness                                       #
    # ------------------------------------------------------------------ #

    async def _get_available_gpu_count(self) -> int:
        """Return the total number of GPUs across all online workers.

        Uses *total* GPUs (not currently-free GPUs) because the batch
        scheduler owns model lifecycle: it creates and tears down
        deployments as needed.  If we used ``available_gpus`` instead,
        warm replicas left over from earlier batches would make the
        count artificially low, forcing the scheduler onto the
        single-GPU sequential path even on a multi-GPU machine.
        """
        if not self.pylet_client:
            return 1
        try:
            workers = await self.pylet_client.get_online_workers()
            total = sum(w.total_gpus for w in workers)
            return max(1, total)
        except Exception as e:
            logger.warning(f"Failed to query GPU count: {e}")
            return 1

    async def _ensure_deployment_ready(self, model: str, timeout: float = 300.0):
        """Ensure a model's deployment exists and has live endpoints."""
        if model in self._models_ready:
            return f"{model}:vllm"

        deployment_id = f"{model}:vllm"

        deployment = self.database.get_deployment_by_id(deployment_id)
        tp = self._get_tp(model)
        # Use 0.85 to leave headroom for other processes on the same GPU.
        gpu_mem_util = float(os.environ.get("SLLM_GPU_MEM_UTIL", "0.85"))
        if tp > 1:
            backend_config = {"tensor_parallel_size": tp,
                              "gpu_memory_utilization": gpu_mem_util}
        else:
            backend_config = {"gpu_memory_utilization": gpu_mem_util}
        # enforce_eager=True disables CUDA-graph capture, reducing startup time
        # from ~50s to ~10-15s per model switch. Slightly slower per-token
        # throughput but amortised over large batches.
        backend_config["enforce_eager"] = True
        if not deployment:
            available_gpus = await self._get_available_gpu_count()
            logger.info(f"Auto-creating deployment for {model} (max_replicas={available_gpus}, tp={tp})")
            try:
                self.database.create_deployment(
                    model_name=model,
                    backend="vllm",
                    min_replicas=0,
                    max_replicas=available_gpus,
                    backend_config=backend_config,
                )
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Deployment {deployment_id} created by concurrent caller, continuing")
                else:
                    raise
        else:
            # Deployment exists — fix any state that would prevent scaling.
            available_gpus = await self._get_available_gpu_count()
            needs_update = False

            # Resurrect "deleting" deployments left by previous batches.
            if deployment.status == "deleting":
                logger.info(f"[RESURRECT] {deployment_id}: was 'deleting', setting back to 'active'")
                self.database.update_deployment_status(deployment_id, "active")
                needs_update = True

            # A previous batch's _prepare_gpu_allocations may have set
            # max_replicas=0, which blocks the autoscaler.
            if deployment.max_replicas < 1:
                logger.info(
                    f"[SCALE-UP] {deployment_id}: max_replicas was "
                    f"{deployment.max_replicas}, raising to {available_gpus}"
                )
                needs_update = True

            if needs_update:
                self.database.update_max_replicas(deployment_id, available_gpus)
                self.database.update_desired_replicas(deployment_id, 1)

        if self.autoscaler:
            self.autoscaler.receive_metrics(deployment_id, buffer_len=1, in_flight=0)

        endpoints = self.database.get_deployment_endpoints(deployment_id)
        if endpoints:
            logger.info(f"[READY] {model} already has {len(endpoints)} endpoint(s)")
            self._models_ready.add(model)
            return deployment_id

        logger.info(f"[WAIT] Waiting for {model} endpoints (timeout={timeout}s)...")
        poll_interval = 1.0
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            endpoints = self.database.get_deployment_endpoints(deployment_id)
            if endpoints:
                logger.info(f"[READY] {model} endpoint(s) live after {elapsed:.1f}s")
                self._models_ready.add(model)
                return deployment_id
            poll_interval = min(poll_interval * 1.2, 5.0)

        raise TimeoutError(
            f"Model {model} not ready after {timeout}s -- no endpoints appeared. "
            f"Check reconciler/pylet logs."
        )

    # ------------------------------------------------------------------ #
    #  Memory-Aware Prefetch Guard                                        #
    # ------------------------------------------------------------------ #

    async def _can_prefetch(self, model_name: str) -> bool:
        """Check if we have enough CPU pinned memory for prefetch."""
        if self.cpu_mem_pool_bytes <= 0:
            return True  # Unknown pool size, optimistically proceed

        model_size = self._estimated_model_sizes.get(model_name, 0)
        if model_size <= 0:
            return True  # Unknown model size, optimistically proceed

        used = sum(
            self._estimated_model_sizes.get(m, 0)
            for m in self._cpu_cached_models
        )

        if used + model_size > self.cpu_mem_pool_bytes:
            logger.warning(
                f"[PREFETCH] Skipping prefetch for {model_name}: "
                f"estimated {model_size / 1e9:.1f}GB needed, "
                f"{used / 1e9:.1f}GB/{self.cpu_mem_pool_bytes / 1e9:.1f}GB used. "
                f"Will rely on C++ LRU eviction at load time."
            )
            return False
        return True

    async def _do_prefetch(self, model_name: str):
        """Prefetch with memory guard. Updates _cpu_cached_models on success."""
        if not await self._can_prefetch(model_name):
            return
        try:
            result = await self.storage_manager.prefetch_to_cpu(model_name)
            if result:
                self._cpu_cached_models.add(model_name)
        except Exception as e:
            logger.warning(f"[PREFETCH] prefetch_to_cpu({model_name}) failed: {e}")

    async def _do_unload(self, model_name: str):
        """Proactively release GPU and CPU resources for a finished model.

        Immediately cancels Pylet instances and removes DB endpoints so the
        GPU is freed *now*, without waiting for the reconciler's 3-second
        loop.  The reconciler will later see desired_replicas=0 and no live
        instances, and simply skip the deployment — harmless.

        Without this, the next model group is submitted to Pylet while the
        previous instance is still alive (exclusive GPU allocation), causing
        "0 free GPUs, waiting..." and eventual timeout.
        """
        deployment_id = f"{model_name}:vllm"

        # 1. Signal reconciler: no replicas wanted
        try:
            self.database.update_desired_replicas(deployment_id, 0)
            self.database.update_max_replicas(deployment_id, 0)
        except Exception as e:
            logger.warning(f"[UNLOAD] Failed to update replicas for {deployment_id}: {e}")

        # 2. Immediately remove endpoints from DB so the Router stops routing
        try:
            self.database.remove_deployment_endpoints(deployment_id)
        except Exception as e:
            logger.warning(f"[UNLOAD] Failed to remove endpoints for {deployment_id}: {e}")

        # 3. Invalidate ready-cache so next model group re-waits for readiness
        self._models_ready.discard(model_name)

        # 4. Directly cancel Pylet instances and wait for GPU memory to drop.
        # pylet.cancel() marks the instance CANCELLED and releases the logical
        # GPU slot, but the vllm EngineCore child process keeps the CUDA context
        # alive until it fully exits.  _allocate_gpus_for_batch polls
        # w.available_gpus (logical) which returns free immediately — but the
        # physical GPU memory is still occupied.  The next model's vllm instance
        # then fails with "Free memory ... less than desired utilization".
        # Fix: after cancel, poll nvidia-smi until no process holds >1 GiB on
        # any GPU that was used by this deployment.
        if self.pylet_client:
            try:
                instances = await self.pylet_client.get_deployment_instances(deployment_id)
                active = [
                    inst for inst in instances
                    if inst.status not in ("CANCELLED", "FAILED", "UNKNOWN", "COMPLETED")
                ]
                for inst in active:
                    try:
                        await self.pylet_client.cancel_instance(inst.instance_id)
                        logger.info(
                            f"[UNLOAD] Cancelled instance {inst.instance_id} "
                            f"for {model_name}"
                        )
                    except Exception as e:
                        logger.warning(f"[UNLOAD] cancel_instance({inst.instance_id}) failed: {e}")

                if active:
                    # Wait for all cancelled vLLM instances to fully exit.
                    # Poll pylet instance status: once CANCELLED/COMPLETED/FAILED
                    # the process has truly exited and GPU memory is freed.
                    # This is more reliable than nvidia-smi (which can show
                    # unrelated processes or CUDA-registered pinned memory).
                    TERMINAL = {"CANCELLED", "COMPLETED", "FAILED", "UNKNOWN"}
                    for inst in active:
                        for _attempt in range(60):
                            await asyncio.sleep(1)
                            try:
                                info = await self.pylet_client.get_instance(inst.instance_id)
                                if info is None or info.status in TERMINAL:
                                    logger.info(
                                        f"[UNLOAD] GPU memory released after "
                                        f"{_attempt + 1}s for {model_name} "
                                        f"(instance {inst.instance_id} → {info.status if info else 'gone'})"
                                    )
                                    break
                            except Exception as _e:
                                logger.debug(f"[UNLOAD] instance status poll failed: {_e}")
                                break  # can't poll → assume done
                        else:
                            logger.warning(
                                f"[UNLOAD] GPU memory still held after 60s for "
                                f"{model_name} — proceeding anyway"
                            )
            except Exception as e:
                logger.warning(f"[UNLOAD] Failed to query/cancel instances for {deployment_id}: {e}")

        # 5. Evict from pinned CPU memory
        if self.storage_manager:
            try:
                result = await self.storage_manager.unload_from_cpu(model_name)
                if result:
                    self._cpu_cached_models.discard(model_name)
                    logger.info(f"[UNLOAD] {model_name} evicted from pinned memory")
            except Exception as e:
                logger.warning(f"[UNLOAD] unload_from_cpu({model_name}) failed: {e}")

        logger.info(f"[UNLOAD] {model_name} resources released (GPU + CPU)")

    # ------------------------------------------------------------------ #
    #  GPU Allocation for Multi-GPU Batches                                #
    # ------------------------------------------------------------------ #

    async def _prepare_gpu_allocations(
        self,
        batch_id: str,
        num_replicas_per_model: Dict[str, int],
        num_gpus: int,
        peak_gpus: int = None,
    ):
        """Ensure GPU allocations match batch requirements before launching.

        Without this step, leftover replicas from previous batches or
        baselines can hog all GPUs, deadlocking new model deployments.

        Steps:
        1. For models NOT in this batch: scale to 0 and unregister from
           autoscaler so the batch-aware scale-down guard lets them go.
        2. For models IN this batch but over-provisioned: cap max_replicas
           and desired_replicas to the assigned count.
        3. Wait for the reconciler to tear down excess instances so GPUs
           are actually freed.
        """
        batch_models = set(num_replicas_per_model.keys())
        all_deployments = self.database.get_all_deployments()
        need_to_wait = False

        for deployment in all_deployments:
            if deployment.status == "deleting":
                continue

            deployment_id = deployment.id
            model_name = deployment.model_name
            current_endpoints = self.database.get_deployment_endpoints(deployment_id)
            current_count = len(current_endpoints)

            if model_name not in batch_models:
                # Not needed for this batch — scale to 0
                if current_count > 0:
                    logger.info(
                        f"[GPU-ALLOC] {deployment_id}: not needed for batch, "
                        f"scaling {current_count} → 0"
                    )
                    # Unregister from autoscaler so batch-aware guard allows scale-down
                    if self.autoscaler:
                        for bid in list(self.autoscaler._active_batches.get(deployment_id, set())):
                            self.autoscaler.unregister_active_batch(deployment_id, bid)
                    self.database.update_max_replicas(deployment_id, 0)
                    self.database.update_desired_replicas(deployment_id, 0)
                    need_to_wait = True
            else:
                # Needed, but may be over-provisioned
                needed = num_replicas_per_model[model_name]
                if current_count > needed:
                    logger.info(
                        f"[GPU-ALLOC] {deployment_id}: has {current_count} replicas, "
                        f"batch needs {needed} — scaling down"
                    )
                    # Unregister stale batch refs so autoscaler permits scale-down
                    if self.autoscaler:
                        for bid in list(self.autoscaler._active_batches.get(deployment_id, set())):
                            if bid != batch_id:
                                self.autoscaler.unregister_active_batch(deployment_id, bid)
                    self.database.update_max_replicas(deployment_id, needed)
                    self.database.update_desired_replicas(deployment_id, needed)
                    need_to_wait = True

        if not need_to_wait:
            return

        # Clear models_ready cache — replica counts changed
        self._models_ready.clear()

        # Wait for reconciler to actually free the GPUs
        if self.pylet_client:
            # peak_gpus reflects the maximum SIMULTANEOUS GPU usage (models on the
            # same sequential lane don't stack).  Fall back to the naive sum if not
            # provided (e.g. called from outside _process_with_prefetch).
            total_needed = peak_gpus if peak_gpus is not None else sum(
                reps * self._get_tp(model)
                for model, reps in num_replicas_per_model.items()
            )
            logger.info(
                f"[GPU-ALLOC] Waiting for GPUs (need {total_needed} free "
                f"out of {num_gpus} total)..."
            )
            for attempt in range(180):  # Up to 3 minutes
                try:
                    workers = await self.pylet_client.get_online_workers()
                    free = sum(w.available_gpus for w in workers)
                    if free >= total_needed:
                        logger.info(
                            f"[GPU-ALLOC] {free}/{num_gpus} GPUs available "
                            f"after {attempt}s"
                        )
                        return
                except Exception as e:
                    logger.debug(f"[GPU-ALLOC] Poll failed: {e}")
                await asyncio.sleep(1)

            logger.warning(
                "[GPU-ALLOC] Timed out waiting for GPU reclamation, "
                "proceeding anyway (may deadlock)"
            )

    # ------------------------------------------------------------------ #
    #  Batch Processing                                                    #
    # ------------------------------------------------------------------ #

    async def _process_batch_safe(self, batch_id: str):
        try:
            await self._process_batch(batch_id)
        except Exception as e:
            logger.error(f"Batch {batch_id} failed with: {e}")
            # Mark remaining pending tasks as failed so the batch
            # can be finalised instead of stuck at in_progress forever.
            try:
                tasks = self.database.get_batch_tasks(batch_id)
                pending = [t for t in tasks if t.status == "pending"]
                for t in pending:
                    self.database.update_batch_task_result(
                        task_id=t.id,
                        status="failed",
                        output={"error": str(e)},
                    )
                self.database.update_batch_job_status(batch_id, "failed")
                self._unregister_batch(batch_id, tasks)
                logger.info(
                    f"Batch {batch_id} marked as failed "
                    f"({len(pending)} pending tasks marked failed)"
                )
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up batch {batch_id}: {cleanup_err}")
        finally:
            async with self._batch_lock:
                self._processing_batches.discard(batch_id)

    async def _process_batch(self, batch_id: str):
        tasks = self.database.get_batch_tasks(batch_id)
        retries = 0
        while not tasks and retries < 10:
            await asyncio.sleep(0.5)
            tasks = self.database.get_batch_tasks(batch_id)
            retries += 1
        if not tasks:
            logger.warning(f"Batch {batch_id} has no tasks after waiting, skipping")
            return

        self._models_ready.clear()

        pending_tasks = [t for t in tasks if t.status == 'pending']

        if not pending_tasks:
            if tasks and all(t.status in ('completed', 'failed') for t in tasks):
                self.database.update_batch_job_status(batch_id, 'completed')
                self._unregister_batch(batch_id, tasks)
            return

        self.database.update_batch_job_status(batch_id, 'in_progress')
        self._active_batch_jobs.add(batch_id)

        # Register batch with autoscaler to prevent premature scale-down
        if self.autoscaler:
            models_in_batch = set(t.body.get("model") for t in pending_tasks if t.body.get("model"))
            for model_name in models_in_batch:
                self.autoscaler.register_active_batch(f"{model_name}:vllm", batch_id)

        if self.enable_model_grouping:
            if self.forced_group_order:
                order_map = {m: i for i, m in enumerate(self.forced_group_order)}
                pending_tasks.sort(key=lambda t: order_map.get(t.body.get("model", ""), 999))
                logger.info(f"Model grouping enabled: tasks sorted by forced order {self.forced_group_order}")
            else:
                pending_tasks.sort(key=lambda t: t.body.get("model", ""))
                logger.info(f"Model grouping enabled: tasks sorted by model name (alphabetical)")
        else:
            logger.info(f"Model grouping disabled: preserving original task order")

        use_prefetch = (
            self.enable_prefetch
            and self.enable_model_grouping
            and self.storage_manager is not None
        )

        if self.enable_model_grouping:
            await self._process_with_prefetch(pending_tasks, do_prefetch=use_prefetch)
        else:
            await self._process_without_prefetch(pending_tasks)

        all_tasks = self.database.get_batch_tasks(batch_id)
        if all_tasks and all(t.status in ('completed', 'failed') for t in all_tasks):
            self.database.update_batch_job_status(batch_id, 'completed')
            self._unregister_batch(batch_id, all_tasks)

    # ------------------------------------------------------------------ #
    #  Shared: Non-prefetch execution                                     #
    # ------------------------------------------------------------------ #

    async def _process_without_prefetch(self, pending_tasks: List[BatchTask]):
        if self.strategy == "sync":
            logger.info(f"Processing {len(pending_tasks)} tasks with SYNC strategy")
            current_model = None
            for task in pending_tasks:
                task_model = task.body.get("model")
                if task_model and current_model and task_model != current_model:
                    logger.info(f"Model switch: {current_model} -> {task_model}, unloading current")
                    await self._do_unload(current_model)
                current_model = task_model
                await self._execute_task(task)

        elif self.strategy == "chunked":
            chunk_size = max(1, self.max_concurrent_tasks_per_batch)
            logger.info(f"Processing {len(pending_tasks)} tasks with CHUNKED strategy (chunk_size: {chunk_size})")
            for i in range(0, len(pending_tasks), chunk_size):
                chunk = pending_tasks[i : i + chunk_size]
                await asyncio.gather(*[self._execute_task(task) for task in chunk])

        else:  # semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

            async def _sem_execute(task):
                async with semaphore:
                    await self._execute_task(task)

            logger.info(f"Processing {len(pending_tasks)} tasks with SEMAPHORE strategy (limit: {self.max_concurrent_tasks_per_batch})")
            await asyncio.gather(*(_sem_execute(task) for task in pending_tasks))

    # ------------------------------------------------------------------ #
    #  Shared: Prefetch path                                              #
    # ------------------------------------------------------------------ #

    def _estimate_group_time(self, model_name: str, num_tasks: int, model_load_time_s: float = 90.0) -> tuple:
        """Return (i_time, p_time) for a model group."""
        size_bytes = self._get_checkpoint_size_bytes(model_name)
        if size_bytes > 0:
            i_time = size_bytes / self._NVME_READ_BPS
            per_task = self._BASE_TASK_TIME_S * (size_bytes / self._BASE_MODEL_BYTES)
        else:
            i_time = model_load_time_s
            per_task = self._BASE_TASK_TIME_S
        p_time = num_tasks * per_task
        return i_time, p_time

    def _assign_groups_to_gpus(
        self,
        groups: List[tuple],
        num_gpus: int,
    ) -> List[List[tuple]]:
        """Assign model groups to execution lanes using capacity-aware packing.

        Each lane is a logical execution slot that processes its groups
        sequentially.  All lanes run in parallel.  A lane's *budget* is
        the number of physical GPUs it occupies while running.

        **TP lane packing** (first-fit decreasing):
          - Each TP>1 group gets its own lane (budget = tp) when GPU
            budget allows, enabling parallel execution of TP models.
          - When budget is insufficient, groups with the same TP size
            share a lane (sequential within lane).
          - At least 1 GPU is reserved for TP=1 groups when they exist.

        **TP=1 lanes:** remaining GPU slots are filled with single-GPU
        lanes.  Groups are assigned via greedy LPT, then rebalanced.

        Returns a list of per-lane queues (variable length, not
        necessarily equal to ``num_gpus``).
        """
        if num_gpus <= 1:
            return [groups]

        # ----- Separate TP>1 and TP=1 groups -----
        tp_groups = []
        single_groups = []
        for model_name, tasks in groups:
            tp = self._get_tp(model_name)
            if tp > 1:
                i_time, p_time = self._estimate_group_time(model_name, len(tasks))
                tp_groups.append((model_name, tasks, tp, i_time + self._MODEL_SWITCH_OVERHEAD_S + p_time))
            else:
                single_groups.append((model_name, tasks))

        # Sort TP groups: largest TP first, then longest first (FFD)
        tp_groups.sort(key=lambda x: (-x[2], -x[3]))

        # ----- Pack TP groups into lanes -----
        # Each lane: [budget, [(model, tasks), ...], load]
        tp_lanes: List[List] = []       # [[budget, groups_list, load], ...]
        used_budget = 0
        reserve = 1 if single_groups else 0  # keep ≥1 GPU for TP=1

        for model_name, tasks, tp, total_time in tp_groups:
            if tp > num_gpus:
                logger.warning(
                    f"[MULTI-GPU] TP={tp} for {model_name} exceeds available "
                    f"GPUs ({num_gpus}) — creating lane anyway"
                )
                tp_lanes.append([tp, [(model_name, tasks)], total_time])
                used_budget += tp
                continue

            # Try to create a new parallel lane if budget allows
            if used_budget + tp <= num_gpus - reserve:
                tp_lanes.append([tp, [(model_name, tasks)], total_time])
                used_budget += tp
                logger.info(
                    f"[MULTI-GPU] TP={tp} model {model_name} → lane {len(tp_lanes)-1} "
                    f"(budget={tp}, parallel)"
                )
            else:
                # Not enough GPUs — add to existing lane with same TP size
                candidates = [l for l in tp_lanes if l[0] == tp]
                if candidates:
                    target = min(candidates, key=lambda l: l[2])
                    target[1].append((model_name, tasks))
                    target[2] += total_time
                    lane_idx = tp_lanes.index(target)
                    logger.info(
                        f"[MULTI-GPU] TP={tp} model {model_name} → lane {lane_idx} "
                        f"(budget={tp}, sequential, shared)"
                    )
                else:
                    # No matching lane — create one anyway (over-subscription)
                    tp_lanes.append([tp, [(model_name, tasks)], total_time])
                    used_budget += tp
                    logger.warning(
                        f"[MULTI-GPU] TP={tp} model {model_name} → lane {len(tp_lanes)-1} "
                        f"(budget={tp}, over-subscribed)"
                    )

        # ----- TP=1 lanes -----
        remaining_gpus = num_gpus - used_budget

        # Annotate and sort single groups by LPT
        annotated = []
        for model_name, tasks in single_groups:
            i_time, p_time = self._estimate_group_time(model_name, len(tasks))
            annotated.append((model_name, tasks, i_time + self._MODEL_SWITCH_OVERHEAD_S + p_time))
        annotated.sort(key=lambda x: x[2], reverse=True)

        if remaining_gpus <= 0 and single_groups:
            # All GPUs consumed by TP lanes — no room for parallel TP=1 lanes.
            # Append TP=1 groups sequentially into the least-loaded TP lane so
            # they run after the TP model finishes (once GPUs are freed).
            logger.info(
                f"[MULTI-GPU] No free GPUs for TP=1 models "
                f"(used_budget={used_budget}/{num_gpus}); "
                f"scheduling TP=1 groups sequentially in TP lane(s)"
            )
            for model_name, tasks, total_time in annotated:
                if tp_lanes:
                    target = min(tp_lanes, key=lambda l: l[2])
                    target[1].append((model_name, tasks))
                    target[2] += total_time
                    lane_idx = tp_lanes.index(target)
                    logger.info(
                        f"[MULTI-GPU] TP=1 model {model_name} → lane {lane_idx} "
                        f"(sequential after TP={target[0]}, no free GPUs)"
                    )
                else:
                    tp_lanes.append([1, [(model_name, tasks)], total_time])
                    used_budget += 1
            single_groups = []
            annotated = []

        remaining_gpus = max(0, remaining_gpus)
        single_queues: List[List[tuple]] = [[] for _ in range(remaining_gpus)]
        single_loads = [0.0] * remaining_gpus

        for model_name, tasks, total_time in annotated:
            target = single_loads.index(min(single_loads))
            single_queues[target].append((model_name, tasks))
            single_loads[target] += total_time

        # ----- Rebalance TP=1 lanes only -----
        IMBALANCE_THRESHOLD = 0.15
        MAX_REBALANCE_ITERS = remaining_gpus * max(len(single_groups), 1)

        for _ in range(MAX_REBALANCE_ITERS):
            if remaining_gpus < 2:
                break
            slowest = max(range(remaining_gpus), key=lambda i: single_loads[i])
            fastest = min(range(remaining_gpus), key=lambda i: single_loads[i])

            if slowest == fastest:
                break

            imbalance = (
                (single_loads[slowest] - single_loads[fastest])
                / max(single_loads[slowest], 1e-9)
            )
            if imbalance <= IMBALANCE_THRESHOLD:
                break

            if not single_queues[slowest]:
                break
            heaviest_idx = max(
                range(len(single_queues[slowest])),
                key=lambda j: len(single_queues[slowest][j][1]),
            )
            h_model, h_tasks = single_queues[slowest][heaviest_idx]

            if len(h_tasks) < 2:
                break

            i_time, _ = self._estimate_group_time(h_model, 1)
            # Add irreducible GPU init cost: creating a new replica on a GPU requires
            # vLLM CUDA init + warmup even when weights are pre-loaded into CPU memory.
            i_time += self._MODEL_SWITCH_OVERHEAD_S
            _, p_all = self._estimate_group_time(h_model, len(h_tasks))
            per_task_time = p_all / len(h_tasks)

            gap = single_loads[slowest] - single_loads[fastest] - i_time
            if gap <= 0:
                break

            n_move = int(gap / (2 * per_task_time))
            n_move = max(1, min(n_move, len(h_tasks) - 1))

            keep_tasks = h_tasks[:-n_move]
            move_tasks = h_tasks[-n_move:]
            single_queues[slowest][heaviest_idx] = (h_model, keep_tasks)

            existing_idx = None
            for j, (m, _) in enumerate(single_queues[fastest]):
                if m == h_model:
                    existing_idx = j
                    break

            if existing_idx is not None:
                _, existing_tasks = single_queues[fastest][existing_idx]
                single_queues[fastest][existing_idx] = (h_model, existing_tasks + move_tasks)
            else:
                single_queues[fastest].append((h_model, move_tasks))

            for i in [slowest, fastest]:
                single_loads[i] = sum(
                    self._estimate_group_time(m, len(t))[0]
                    + self._MODEL_SWITCH_OVERHEAD_S
                    + self._estimate_group_time(m, len(t))[1]
                    for m, t in single_queues[i]
                )

            logger.info(
                f"[REBALANCE] Moved {n_move} {h_model} tasks "
                f"(loads: {', '.join(f'Lane {i}={single_loads[i]:.1f}s' for i in range(remaining_gpus))})"
            )

        # ----- Combine: TP lanes first, then non-empty TP=1 lanes -----
        all_queues: List[List[tuple]] = [lane[1] for lane in tp_lanes]
        all_loads = [lane[2] for lane in tp_lanes]
        tp_lane_count = len(tp_lanes)

        for i, q in enumerate(single_queues):
            if q:
                all_queues.append(q)
                all_loads.append(single_loads[i])

        # Summary logging
        parts = []
        for i, q in enumerate(all_queues):
            if i < tp_lane_count:
                budget = tp_lanes[i][0]
                parts.append(
                    f"Lane {i} [TP={budget}]: "
                    f"[{', '.join(f'{m}({len(t)})' for m, t in q)}] "
                    f"load={all_loads[i]:.1f}s"
                )
            else:
                parts.append(
                    f"Lane {i}: "
                    f"[{', '.join(f'{m}({len(t)})' for m, t in q)}] "
                    f"load={all_loads[i]:.1f}s"
                )
        total_budget = used_budget + sum(1 for q in single_queues if q)
        logger.info(
            f"[MULTI-GPU] Assigned {len(groups)} groups to {len(all_queues)} lanes "
            f"({total_budget} GPU slots): " + ", ".join(parts)
        )
        return all_queues

    async def _process_gpu_queue(
        self,
        gpu_id: int,
        queue: List[tuple],
        do_prefetch: bool,
        num_replicas_per_model: Dict[str, int],
        model_lane_refs: Dict[str, int] = None,
        model_ref_lock: asyncio.Lock = None,
    ):
        """Process a single GPU's queue of model groups sequentially."""
        # Track which models still have future groups in THIS queue
        remaining_in_queue = {}
        for idx, (m, _) in enumerate(queue):
            remaining_in_queue.setdefault(m, []).append(idx)

        # Eager prefetch: kick off all subsequent models at the start
        if do_prefetch and self.storage_manager and len(queue) > 1:
            subsequent_models = list(dict.fromkeys(m for m, _ in queue[1:]))
            for m in subsequent_models:
                if m != queue[0][0] and await self._can_prefetch(m):
                    logger.info(f"[GPU {gpu_id}] Eagerly prefetching {m} to CPU")
                    asyncio.create_task(self._do_prefetch(m))

        for group_idx, (model_name, group_tasks) in enumerate(queue):
            replicas = num_replicas_per_model.get(model_name, 1)
            await self._ensure_deployment_ready_with_replicas(model_name, replicas)

            logger.info(
                f"[GPU {gpu_id}] Starting group {group_idx+1}/{len(queue)}: "
                f"{model_name} ({len(group_tasks)} tasks, {replicas} replica(s))"
            )

            await self._execute_group(group_tasks)

            # Proactive eviction: unload model if no future groups need it
            # in THIS queue AND no other lanes still need it.
            remaining_in_queue[model_name].pop(0)
            if not remaining_in_queue[model_name]:
                if model_lane_refs is not None and model_ref_lock is not None:
                    async with model_ref_lock:
                        model_lane_refs[model_name] -= 1
                        should_unload = model_lane_refs[model_name] <= 0
                    if should_unload:
                        await self._do_unload(model_name)
                    else:
                        logger.info(
                            f"[GPU {gpu_id}] {model_name} done on this lane, "
                            f"{model_lane_refs[model_name]} lane(s) still active"
                        )
                else:
                    # Single-lane or legacy path — safe to unload
                    await self._do_unload(model_name)

    async def _ensure_deployment_ready_with_replicas(self, model: str, max_replicas: int, timeout: float = 300.0):
        """Like _ensure_deployment_ready but with a specific replica count."""
        if model in self._models_ready:
            return f"{model}:vllm"

        deployment_id = f"{model}:vllm"
        tp = self._get_tp(model)
        gpu_mem_util = float(os.environ.get("SLLM_GPU_MEM_UTIL", "0.85"))
        if tp > 1:
            backend_config = {"tensor_parallel_size": tp,
                              "gpu_memory_utilization": gpu_mem_util}
        else:
            backend_config = {"gpu_memory_utilization": gpu_mem_util}
        backend_config["enforce_eager"] = True
        deployment = self.database.get_deployment_by_id(deployment_id)
        if not deployment:
            logger.info(f"Auto-creating deployment for {model} (max_replicas={max_replicas}, tp={tp})")
            try:
                self.database.create_deployment(
                    model_name=model,
                    backend="vllm",
                    min_replicas=0,
                    max_replicas=max_replicas,
                    backend_config=backend_config,
                )
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Deployment {deployment_id} created by concurrent caller, continuing")
                else:
                    raise
        else:
            # Deployment exists — fix any state that would prevent scaling.
            needs_update = False

            # Resurrect "deleting" deployments left by previous batches.
            if deployment.status == "deleting":
                logger.info(f"[RESURRECT] {deployment_id}: was 'deleting', setting back to 'active'")
                self.database.update_deployment_status(deployment_id, "active")
                needs_update = True

            # A previous batch's _prepare_gpu_allocations may have set
            # max_replicas=0, which blocks the autoscaler.
            if deployment.max_replicas < max_replicas:
                logger.info(
                    f"[SCALE-UP] {deployment_id}: max_replicas was "
                    f"{deployment.max_replicas}, raising to {max_replicas}"
                )
                needs_update = True

            if needs_update:
                self.database.update_max_replicas(deployment_id, max_replicas)
                self.database.update_desired_replicas(deployment_id, max_replicas)

        if self.autoscaler:
            self.autoscaler.receive_metrics(deployment_id, buffer_len=1, in_flight=0)

        endpoints = self.database.get_deployment_endpoints(deployment_id)
        if endpoints:
            logger.info(f"[READY] {model} already has {len(endpoints)} endpoint(s)")
            self._models_ready.add(model)
            return deployment_id

        logger.info(f"[WAIT] Waiting for {model} endpoints (timeout={timeout}s)...")
        poll_interval = 1.0
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            endpoints = self.database.get_deployment_endpoints(deployment_id)
            if endpoints:
                logger.info(f"[READY] {model} endpoint(s) live after {elapsed:.1f}s")
                self._models_ready.add(model)
                return deployment_id
            poll_interval = min(poll_interval * 1.2, 5.0)

        raise TimeoutError(
            f"Model {model} not ready after {timeout}s -- no endpoints appeared. "
            f"Check reconciler/pylet logs."
        )

    async def _process_with_prefetch(self, pending_tasks: List[BatchTask], do_prefetch: bool = True):
        groups = self._extract_model_groups(pending_tasks)
        num_gpus = await self._get_available_gpu_count()

        if num_gpus <= 1:
            # Single GPU: original path — Johnson's Rule on the global queue
            if self.enable_johnsons_rule and not self.forced_group_order and len(groups) > 1:
                groups = self._apply_johnsons_rule(groups)
            logger.info(
                f"[PREFETCH] Processing {len(pending_tasks)} tasks "
                f"(prefetch={'ON' if do_prefetch else 'OFF'}, threshold={self.prefetch_threshold}) in {len(groups)} "
                f"model groups (1 GPU): {[g[0] for g in groups]}"
            )
            await self._run_single_gpu_queue(groups, do_prefetch)
            return

        # --- Multi-GPU path ---
        # 1. Assign groups to GPUs (greedy least-load).
        #    Exception: when a forced group order is set, honour it exactly by
        #    placing all groups in a single sequential lane in that order.
        #    _assign_groups_to_gpus would otherwise reorder by TP and LPT,
        #    ignoring the forced sequence (breaks JR ordering experiments).
        if self.forced_group_order:
            gpu_queues = [groups]
            logger.info(
                f"[FORCED-ORDER] Using single lane for forced group order "
                f"{[g[0] for g in groups]}"
            )
        else:
            gpu_queues = self._assign_groups_to_gpus(groups, num_gpus)

        # 2. Per-GPU: apply Johnson's Rule to each queue independently
        if self.enable_johnsons_rule and not self.forced_group_order:
            for i, queue in enumerate(gpu_queues):
                if len(queue) > 1:
                    gpu_queues[i] = self._apply_johnsons_rule(queue)
                    logger.info(f"[GPU {i}] Johnson's order: {[g[0] for g in gpu_queues[i]]}")

        # 3. Count how many GPUs each model is assigned to (for replica count)
        num_replicas_per_model: Dict[str, int] = {}
        for queue in gpu_queues:
            for model_name, _tasks in queue:
                num_replicas_per_model[model_name] = num_replicas_per_model.get(model_name, 0) + 1

        logger.info(
            f"[MULTI-GPU] Processing {len(pending_tasks)} tasks across {num_gpus} GPUs, "
            f"replicas: {num_replicas_per_model}"
        )

        # 4. Reclaim GPUs from over-provisioned / unneeded deployments
        #    Must happen BEFORE launching GPU queues to avoid deadlock
        #    where leftover replicas block new model deployments.
        batch_id = pending_tasks[0].batch_id if pending_tasks else ""
        # Peak simultaneous GPU demand: models in the SAME lane run sequentially,
        # so only count the heaviest model per lane (not the sum of all models).
        peak_gpus_needed = sum(
            max(self._get_tp(m) for m, _ in queue)
            for queue in gpu_queues
            if queue
        )
        await self._prepare_gpu_allocations(batch_id, num_replicas_per_model, num_gpus, peak_gpus=peak_gpus_needed)

        # 5. Build shared reference counter for cross-lane eviction safety.
        #    Each lane that has a model increments the counter.  Only the
        #    last lane to finish with a model calls _do_unload.
        model_lane_refs: Dict[str, int] = {}
        for queue in gpu_queues:
            seen = set()
            for model_name, _ in queue:
                if model_name not in seen:
                    model_lane_refs[model_name] = model_lane_refs.get(model_name, 0) + 1
                    seen.add(model_name)
        model_ref_lock = asyncio.Lock()

        # 6. Execute all GPU queues in parallel
        gpu_coroutines = []
        for gpu_id, queue in enumerate(gpu_queues):
            if queue:
                gpu_coroutines.append(
                    self._process_gpu_queue(
                        gpu_id, queue, do_prefetch, num_replicas_per_model,
                        model_lane_refs=model_lane_refs,
                        model_ref_lock=model_ref_lock,
                    )
                )
        await asyncio.gather(*gpu_coroutines)

    async def _run_single_gpu_queue(self, groups: List[tuple], do_prefetch: bool):
        """Original single-GPU execution path."""
        # Build remaining-group tracker for proactive eviction
        remaining = {}
        for idx, (m, _) in enumerate(groups):
            remaining.setdefault(m, []).append(idx)

        if do_prefetch and self.prefetch_threshold == 0.0:
            # === EAGER PREFETCH MODE ===
            if self.storage_manager and len(groups) > 1:
                subsequent_models = list(dict.fromkeys(g[0] for g in groups[1:]))
                for m in subsequent_models:
                    if m != groups[0][0]:
                        if await self._can_prefetch(m):
                            logger.info(f"[PREFETCH_ALL] Eagerly prefetching {m} to CPU right now")
                            asyncio.create_task(self._do_prefetch(m))

            for group_idx, (model_name, group_tasks) in enumerate(groups):
                await self._ensure_deployment_ready(model_name)
                logger.info(
                    f"Starting group {group_idx+1}/{len(groups)}: "
                    f"{model_name} ({len(group_tasks)} tasks) (Eager Mode)"
                )
                await self._execute_group(group_tasks)

                # Proactive eviction
                remaining[model_name].pop(0)
                if not remaining[model_name]:
                    await self._do_unload(model_name)

        else:
            # === OVERLAPPING PREFETCH MODE or NO-PREFETCH GROUP MODE ===
            for group_idx, (model_name, group_tasks) in enumerate(groups):
                await self._ensure_deployment_ready(model_name)
                next_model = (
                    groups[group_idx + 1][0]
                    if group_idx + 1 < len(groups)
                    else None
                )

                logger.info(
                    f"[PREFETCH] Starting group {group_idx+1}/{len(groups)}: "
                    f"{model_name} ({len(group_tasks)} tasks)"
                )

                if do_prefetch and next_model and next_model != model_name:
                    await self._execute_group_with_prefetch(
                        group_tasks, next_model, self.prefetch_threshold
                    )
                else:
                    await self._execute_group(group_tasks)

                # Proactive eviction
                remaining[model_name].pop(0)
                if not remaining[model_name]:
                    await self._do_unload(model_name)

    async def _execute_group_with_prefetch(
        self,
        tasks: List[BatchTask],
        next_model: str,
        threshold: float,
    ):
        prefetch_at = max(1, int(len(tasks) * threshold))
        prefetch_triggered = False
        completed_count = 0
        lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

        async def _execute_and_track(task):
            nonlocal completed_count, prefetch_triggered

            async with semaphore:
                await self._execute_task(task)

            async with lock:
                completed_count += 1

                if (
                    not prefetch_triggered
                    and completed_count >= prefetch_at
                    and self.storage_manager
                ):
                    prefetch_triggered = True
                    if await self._can_prefetch(next_model):
                        logger.info(
                            f"[PREFETCH] {completed_count}/{len(tasks)} done "
                            f"(threshold {threshold:.0%}), prefetching {next_model}"
                        )
                        asyncio.create_task(self._do_prefetch(next_model))
                    else:
                        logger.info(
                            f"[PREFETCH] {completed_count}/{len(tasks)} done "
                            f"but skipping prefetch for {next_model} (memory)"
                        )

        await asyncio.gather(*[_execute_and_track(t) for t in tasks])

    async def _execute_group(self, tasks: List[BatchTask]):
        if self.strategy == "sync":
            for task in tasks:
                await self._execute_task(task)
        elif self.strategy == "chunked":
            chunk_size = max(1, self.max_concurrent_tasks_per_batch)
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i : i + chunk_size]
                await asyncio.gather(*[self._execute_task(t) for t in chunk])
        else:
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

            async def _sem(task):
                async with semaphore:
                    await self._execute_task(task)

            await asyncio.gather(*[_sem(t) for t in tasks])

    def _extract_model_groups(self, tasks: List[BatchTask]) -> List[tuple]:
        groups = []
        current_model = None
        current_group = []
        for task in tasks:
            model = task.body.get("model", "")
            if model != current_model:
                if current_group:
                    groups.append((current_model, current_group))
                current_model = model
                current_group = [task]
            else:
                current_group.append(task)
        if current_group:
            groups.append((current_model, current_group))
        return groups

    def _get_checkpoint_size_bytes(self, model_name: str) -> int:
        """Return total checkpoint size in bytes by scanning the model directory.

        Falls back to ``_estimated_model_sizes`` cache, then 0 if unavailable.
        """
        cached = self._estimated_model_sizes.get(model_name, 0)
        if cached > 0:
            return cached

        if self.storage_manager is None:
            return 0

        model_dir = os.path.join(self.storage_manager.storage_path, model_name)
        if not os.path.isdir(model_dir):
            return 0

        # Prefer sllm-store native format (tensor.data_*) if present; fall back
        # to HuggingFace checkpoint files.  Never double-count both formats.
        sllm_total = 0
        hf_total = 0
        for root, _dirs, files in os.walk(model_dir):
            for fname in files:
                if fname.startswith("tensor.data"):
                    sllm_total += os.path.getsize(os.path.join(root, fname))
                elif fname.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                    hf_total += os.path.getsize(os.path.join(root, fname))
        total = sllm_total if sllm_total > 0 else hf_total

        if total > 0:
            self._estimated_model_sizes[model_name] = total
        return total

    # Baseline per-task inference time for a ~1B parameter model (seconds).
    _BASE_TASK_TIME_S = 0.5
    # Reference model size in bytes (~1B params in fp16 ≈ 2 GB).
    _BASE_MODEL_BYTES = 2e9
    # NVMe sequential read throughput for checkpoint I/O to pinned CPU memory (bytes/s).
    # Raw NVMe bandwidth; used for the overlappable SSD→CPU portion only (Machine A in
    # the flow-shop formulation). Actual sllm-store read times match size/bandwidth
    # to within ~10% (Experiment 5, Table 6).
    _NVME_READ_BPS = 3e9
    # Non-overlappable GPU model-switch overhead: PCIe DMA + vLLM CUDA init + warmup (s).
    # Measured: 32.6 s average on the test cluster with prefetch enabled (Experiment 2).
    # This overhead is irreducible even when weights are pre-loaded into CPU pinned memory,
    # because GPU memory allocation and vLLM engine warm-up cannot be parallelised with
    # the previous group's inference. Must be included in all lane-load estimates and in
    # the LPT rebalancing cost check (but NOT in Johnson's Rule, which only models the
    # overlappable I/O phase).
    _MODEL_SWITCH_OVERHEAD_S: float = 32.6

    def _apply_johnsons_rule(
        self,
        groups: List[tuple],
        model_load_time_s: float = 90.0,
    ) -> List[tuple]:
        """Reorder model groups using Johnson's Rule to minimise I/O idle.

        Each model group is treated as a two-stage job:
          - Stage 1 (I/O): load the model checkpoint from SSD to CPU
          - Stage 2 (compute): run all tasks on GPU

        I/O time is estimated from actual checkpoint size on disk (bytes / NVMe
        throughput).  Compute time scales with both task count and model size
        (larger models have proportionally longer per-token latency).

        Johnson's Rule splits groups into:
          S1: groups where compute >= load  (P >= I), sorted by I ascending
          S2: groups where compute <  load  (P <  I), sorted by I descending
        Then concatenates S1 + S2.
        """
        if len(groups) <= 1:
            return groups

        s1 = []  # P >= I
        s2 = []  # P < I

        for model_name, group_tasks in groups:
            size_bytes = self._get_checkpoint_size_bytes(model_name)

            if size_bytes > 0:
                i_time = size_bytes / self._NVME_READ_BPS
                # Scale per-task time linearly with model size
                per_task = self._BASE_TASK_TIME_S * (size_bytes / self._BASE_MODEL_BYTES)
            else:
                i_time = model_load_time_s
                per_task = self._BASE_TASK_TIME_S

            p_time = len(group_tasks) * per_task

            if p_time >= i_time:
                s1.append((model_name, group_tasks, i_time, p_time))
            else:
                s2.append((model_name, group_tasks, i_time, p_time))

        # S1: sort by load time ascending (shortest load first)
        s1.sort(key=lambda x: x[2])
        # S2: sort by compute time descending (longest compute first)
        s2.sort(key=lambda x: x[3], reverse=True)

        ordered = [(m, t) for m, t, _, _ in s1] + [(m, t) for m, t, _, _ in s2]

        logger.info(
            f"[JOHNSON] Reordered {len(groups)} model groups: "
            f"S1={[(g[0], f'I={g[2]:.1f}s P={g[3]:.1f}s') for g in s1]} "
            f"S2={[(g[0], f'I={g[2]:.1f}s P={g[3]:.1f}s') for g in s2]} -> "
            f"{[g[0] for g in ordered]}"
        )
        return ordered

    # ------------------------------------------------------------------ #
    #  Task Execution                                                     #
    # ------------------------------------------------------------------ #

    async def _execute_task(self, task: BatchTask):
        async with self._global_semaphore:
            await self._execute_task_inner(task)

    async def _execute_task_inner(self, task: BatchTask):
        logger.info(f"Executing task {task.id} (Batch: {task.batch_id})")

        started_at = datetime.now(timezone.utc).isoformat()

        try:
            model = task.body.get("model")
            if not model:
                raise ValueError("Model not specified in task body")

            deployment_id = f"{model}:vllm"

            if self.enable_model_grouping:
                await self._ensure_deployment_ready(model)
            else:
                deployment = self.database.get_deployment_by_id(deployment_id)
                available_gpus = await self._get_available_gpu_count()
                tp = self._get_tp(model)
                bc: dict | None = {"enforce_eager": True}
                if tp > 1:
                    bc["tensor_parallel_size"] = tp
                # Each replica of a TP>1 model occupies `tp` GPUs, so the number
                # of replicas that fit is floor(available_gpus / tp).
                replicas = max(1, available_gpus // max(tp, 1))
                if not deployment:
                    try:
                        self.database.create_deployment(
                            model_name=model, backend="vllm",
                            min_replicas=0, max_replicas=replicas,
                            backend_config=bc,
                        )
                    except Exception as e:
                        if "already exists" not in str(e):
                            raise
                elif deployment.max_replicas < 1:
                    # Restore max_replicas after a previous _do_unload set it to 0
                    logger.info(
                        f"[SCALE-UP] {deployment_id}: max_replicas was "
                        f"{deployment.max_replicas}, raising to {replicas}"
                    )
                    self.database.update_max_replicas(deployment_id, replicas)
                    self.database.update_desired_replicas(deployment_id, replicas)

            result = await self.router.handle_request(
                payload=task.body,
                path=task.url,
                deployment_id=deployment_id
            )

            completed_at = datetime.now(timezone.utc).isoformat()
            self.database.upsert_batch_task(
                task_id=task.id,
                batch_id=task.batch_id,
                custom_id=task.custom_id,
                method=task.method,
                url=task.url,
                body=task.body,
                status="completed",
                output=result,
                started_at=started_at,
                completed_at=completed_at
            )
            logger.info(f"Task {task.id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            completed_at = datetime.now(timezone.utc).isoformat()
            self.database.upsert_batch_task(
                task_id=task.id,
                batch_id=task.batch_id,
                custom_id=task.custom_id,
                method=task.method,
                url=task.url,
                body=task.body,
                status="failed",
                output={"error": str(e)},
                started_at=started_at,
                completed_at=completed_at
            )
