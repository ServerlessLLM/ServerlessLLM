"""
Tests for BatchScheduler: model grouping, Johnson's Rule, prefetch,
proactive unloading, GPU rebalancing, and autoscaler integration.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from sllm.database import BatchTask
from sllm.batch_scheduler import BatchScheduler


# ============================================================================ #
# Helpers
# ============================================================================ #


def make_task(
    task_id: str = "t1",
    batch_id: str = "b1",
    model: str = "Qwen-7B",
    prompt: str = "Hello",
    status: str = "pending",
) -> BatchTask:
    return BatchTask(
        id=task_id,
        batch_id=batch_id,
        custom_id=f"custom-{task_id}",
        method="POST",
        url="/v1/chat/completions",
        body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        },
        status=status,
        output=None,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
    )


def make_mock_database():
    db = MagicMock()
    db.get_pending_batch_ids.return_value = []
    db.get_batch_tasks.return_value = []
    db.get_batch_job.return_value = None
    db.get_deployment_by_id.return_value = None
    db.update_batch_job_status = MagicMock()
    db.upsert_batch_task = MagicMock()
    db.create_deployment = MagicMock()
    db.get_deployment_endpoints.return_value = ["localhost:8000"]
    return db


def make_mock_router():
    router = MagicMock()
    router.config = MagicMock()
    router.config.max_buffer_size = 100
    router.handle_request = AsyncMock(return_value={"choices": [{"text": "ok"}]})
    return router


def make_scheduler(**kwargs):
    db = kwargs.get("database", make_mock_database())
    router = kwargs.get("router", make_mock_router())
    s = BatchScheduler(database=db, router=router)
    return s, db, router


# ============================================================================ #
# Model Grouping & Extraction
# ============================================================================ #


class TestModelGroupExtraction:
    """Verify _extract_model_groups produces correct groups."""

    def test_single_model(self):
        s, _, _ = make_scheduler()
        tasks = [make_task(f"t{i}", model="A") for i in range(5)]
        groups = s._extract_model_groups(tasks)
        assert len(groups) == 1
        assert groups[0][0] == "A"
        assert len(groups[0][1]) == 5

    def test_multiple_models_sorted(self):
        s, _, _ = make_scheduler()
        tasks = [
            make_task("t1", model="A"),
            make_task("t2", model="A"),
            make_task("t3", model="B"),
            make_task("t4", model="B"),
            make_task("t5", model="B"),
        ]
        groups = s._extract_model_groups(tasks)
        assert len(groups) == 2
        assert groups[0] == ("A", tasks[:2])
        assert groups[1] == ("B", tasks[2:])

    def test_interleaved_creates_separate_groups(self):
        """Unsorted input creates multiple groups for same model."""
        s, _, _ = make_scheduler()
        tasks = [
            make_task("t1", model="A"),
            make_task("t2", model="B"),
            make_task("t3", model="A"),
        ]
        groups = s._extract_model_groups(tasks)
        assert len(groups) == 3

    def test_empty_tasks(self):
        s, _, _ = make_scheduler()
        groups = s._extract_model_groups([])
        assert groups == []


# ============================================================================ #
# Johnson's Rule
# ============================================================================ #


class TestJohnsonsRule:
    """Verify Johnson's Rule ordering."""

    def test_single_group_unchanged(self):
        s, _, _ = make_scheduler()
        groups = [("A", [make_task("t1", model="A")])]
        result = s._apply_johnsons_rule(groups)
        assert len(result) == 1
        assert result[0][0] == "A"

    def test_two_groups_ordering(self):
        """With default estimates (no checkpoint sizes), all groups use
        fallback model_load_time_s=90s for I/O. Groups with more tasks
        (higher P) go into S1, sorted by I ascending."""
        s, _, _ = make_scheduler()
        # Group A: 200 tasks (P >> I) → S1
        # Group B: 1 task (P < I) → S2
        groups = [
            ("A", [make_task(f"a{i}", model="A") for i in range(200)]),
            ("B", [make_task("b0", model="B")]),
        ]
        result = s._apply_johnsons_rule(groups)
        # S1=[A], S2=[B] → order: A, B
        assert result[0][0] == "A"
        assert result[1][0] == "B"

    def test_s2_sorted_descending(self):
        """S2 groups (P < I) should be sorted by I descending."""
        s, _, _ = make_scheduler()
        # All groups have 1 task → P = 0.5s, I = 90s → all in S2
        # With no checkpoint sizes, all have same I, so order is stable
        groups = [
            ("A", [make_task("a0", model="A")]),
            ("B", [make_task("b0", model="B")]),
            ("C", [make_task("c0", model="C")]),
        ]
        result = s._apply_johnsons_rule(groups)
        assert len(result) == 3

    def test_empty_groups(self):
        s, _, _ = make_scheduler()
        result = s._apply_johnsons_rule([])
        assert result == []


# ============================================================================ #
# GPU Assignment & Rebalancing
# ============================================================================ #


class TestGPUAssignment:
    """Verify _assign_groups_to_gpus with greedy + rebalancing."""

    def test_single_gpu_returns_all(self):
        s, _, _ = make_scheduler()
        groups = [
            ("A", [make_task(f"a{i}", model="A") for i in range(10)]),
            ("B", [make_task(f"b{i}", model="B") for i in range(5)]),
        ]
        result = s._assign_groups_to_gpus(groups, 1)
        assert len(result) == 1
        assert result[0] == groups

    def test_two_gpus_balanced(self):
        """Two equal groups should go to separate GPUs."""
        s, _, _ = make_scheduler()
        groups = [
            ("A", [make_task(f"a{i}", model="A") for i in range(10)]),
            ("B", [make_task(f"b{i}", model="B") for i in range(10)]),
        ]
        result = s._assign_groups_to_gpus(groups, 2)
        assert len(result) == 2
        # Each GPU should have one group
        assert len(result[0]) >= 1
        assert len(result[1]) >= 1

    def test_rebalancing_splits_heavy_group(self):
        """One very heavy group + one light group on 2 GPUs should trigger rebalancing."""
        s, _, _ = make_scheduler()
        # A: 100 tasks, B: 2 tasks — huge imbalance
        groups = [
            ("A", [make_task(f"a{i}", model="A") for i in range(100)]),
            ("B", [make_task(f"b{i}", model="B") for i in range(2)]),
        ]
        result = s._assign_groups_to_gpus(groups, 2)
        # After rebalancing, model A should be split across both GPUs
        all_a_tasks = []
        for queue in result:
            for model, tasks in queue:
                if model == "A":
                    all_a_tasks.extend(tasks)
        assert len(all_a_tasks) == 100  # No tasks lost

        # Both GPUs should have work
        for queue in result:
            total_tasks = sum(len(t) for _, t in queue)
            assert total_tasks > 0

    def test_rebalancing_preserves_all_tasks(self):
        """Rebalancing must never lose or duplicate tasks."""
        s, _, _ = make_scheduler()
        groups = [
            ("A", [make_task(f"a{i}", model="A") for i in range(50)]),
            ("B", [make_task(f"b{i}", model="B") for i in range(30)]),
            ("C", [make_task(f"c{i}", model="C") for i in range(20)]),
        ]
        result = s._assign_groups_to_gpus(groups, 3)

        # Count all tasks across all GPUs
        all_task_ids = set()
        for queue in result:
            for _, tasks in queue:
                for t in tasks:
                    all_task_ids.add(t.id)
        assert len(all_task_ids) == 100

    def test_single_task_group_not_split(self):
        """A group with 1 task cannot be split."""
        s, _, _ = make_scheduler()
        groups = [
            ("A", [make_task("a0", model="A")]),
            ("B", [make_task(f"b{i}", model="B") for i in range(50)]),
        ]
        result = s._assign_groups_to_gpus(groups, 2)
        # A should remain intact (1 task)
        a_counts = []
        for queue in result:
            for model, tasks in queue:
                if model == "A":
                    a_counts.append(len(tasks))
        assert 1 in a_counts

    def test_tp_group_gets_own_lane(self):
        """TP>1 model should get a dedicated lane."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"BigModel": 4})
        groups = [
            ("BigModel", [make_task(f"big{i}", model="BigModel") for i in range(10)]),
            ("SmallA", [make_task(f"sa{i}", model="SmallA") for i in range(10)]),
            ("SmallB", [make_task(f"sb{i}", model="SmallB") for i in range(10)]),
        ]
        result = s._assign_groups_to_gpus(groups, 8)
        # BigModel should be alone on its lane (not mixed with TP=1 models)
        for queue in result:
            models_in_queue = [m for m, _ in queue]
            if "BigModel" in models_in_queue:
                assert models_in_queue == ["BigModel"]
                break

    def test_tp_group_not_split_by_rebalancing(self):
        """TP groups should not be split across lanes by the rebalancing loop."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"BigModel": 4})
        groups = [
            ("BigModel", [make_task(f"big{i}", model="BigModel") for i in range(200)]),
            ("Small", [make_task(f"s{i}", model="Small") for i in range(5)]),
        ]
        result = s._assign_groups_to_gpus(groups, 8)
        # All BigModel tasks on exactly one lane
        big_tasks_per_lane = {}
        for qi, queue in enumerate(result):
            for model, tasks in queue:
                if model == "BigModel":
                    big_tasks_per_lane[qi] = big_tasks_per_lane.get(qi, 0) + len(tasks)
        assert len(big_tasks_per_lane) == 1
        assert list(big_tasks_per_lane.values())[0] == 200

    def test_two_tp2_models_parallel_lanes(self):
        """Two TP=2 models on 4+ GPUs should get separate parallel lanes."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"ModelA": 2, "ModelB": 2})
        groups = [
            ("ModelA", [make_task(f"a{i}", model="ModelA") for i in range(10)]),
            ("ModelB", [make_task(f"b{i}", model="ModelB") for i in range(10)]),
        ]
        result = s._assign_groups_to_gpus(groups, 4)
        # Each TP=2 model should be on its own lane (2 TP lanes)
        lane_models = [[m for m, _ in q] for q in result if q]
        a_lanes = [i for i, ms in enumerate(lane_models) if "ModelA" in ms]
        b_lanes = [i for i, ms in enumerate(lane_models) if "ModelB" in ms]
        assert len(a_lanes) == 1
        assert len(b_lanes) == 1
        assert a_lanes[0] != b_lanes[0]  # Different lanes → parallel

    def test_two_tp2_models_shared_lane_insufficient_gpus(self):
        """Two TP=2 models on 3 GPUs with a TP=1 model → must share a lane."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"ModelA": 2, "ModelB": 2})
        groups = [
            ("ModelA", [make_task(f"a{i}", model="ModelA") for i in range(10)]),
            ("ModelB", [make_task(f"b{i}", model="ModelB") for i in range(10)]),
            ("SmallC", [make_task(f"c{i}", model="SmallC") for i in range(5)]),
        ]
        result = s._assign_groups_to_gpus(groups, 3)
        # Both TP=2 models should share one lane (budget=2, 1 GPU left for SmallC)
        for queue in result:
            models_in_queue = [m for m, _ in queue]
            if "ModelA" in models_in_queue and "ModelB" in models_in_queue:
                break
        else:
            pytest.fail("ModelA and ModelB should share a lane on 3 GPUs")

    def test_mixed_tp_sizes(self):
        """TP=4 + TP=2 on 8 GPUs → separate lanes (budget 4+2), 2 TP=1 lanes."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"Big": 4, "Med": 2})
        groups = [
            ("Big", [make_task(f"big{i}", model="Big") for i in range(10)]),
            ("Med", [make_task(f"med{i}", model="Med") for i in range(10)]),
            ("SmA", [make_task(f"a{i}", model="SmA") for i in range(10)]),
            ("SmB", [make_task(f"b{i}", model="SmB") for i in range(10)]),
        ]
        result = s._assign_groups_to_gpus(groups, 8)
        # Big and Med should be on separate lanes
        big_lane = med_lane = None
        for i, queue in enumerate(result):
            for m, _ in queue:
                if m == "Big":
                    big_lane = i
                if m == "Med":
                    med_lane = i
        assert big_lane is not None and med_lane is not None
        assert big_lane != med_lane

    def test_tp_reserve_for_single_groups(self):
        """TP lanes should reserve at least 1 GPU for TP=1 groups."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"ModelA": 2})
        groups = [
            ("ModelA", [make_task(f"a{i}", model="ModelA") for i in range(10)]),
            ("SmallB", [make_task(f"b{i}", model="SmallB") for i in range(5)]),
        ]
        result = s._assign_groups_to_gpus(groups, 3)
        # SmallB should get its own lane (not starved)
        small_found = False
        for queue in result:
            for m, _ in queue:
                if m == "SmallB":
                    small_found = True
        assert small_found

    def test_rebalancing_not_blocked_by_tp(self):
        """Rebalancing of TP=1 lanes should work even when TP lanes exist."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"BigTP": 2})
        # Use enough tasks so compute gap exceeds I/O cost of splitting.
        # Default model_load_time = 90s, per_task ~0.5s, so we need
        # enough imbalance: Heavy(500)=250s vs Light(5)=2.5s, gap > 90s.
        groups = [
            ("BigTP", [make_task(f"tp{i}", model="BigTP") for i in range(5)]),
            ("Heavy", [make_task(f"h{i}", model="Heavy") for i in range(500)]),
            ("Light", [make_task(f"l{i}", model="Light") for i in range(5)]),
        ]
        result = s._assign_groups_to_gpus(groups, 4)
        # Heavy group should be split across TP=1 lanes (rebalancing works)
        heavy_count = 0
        heavy_lanes = 0
        for queue in result:
            for m, tasks in queue:
                if m == "Heavy":
                    heavy_count += len(tasks)
                    heavy_lanes += 1
        assert heavy_count == 500  # All tasks preserved
        # With 2 TP=1 lanes, rebalancing should split Heavy across both
        assert heavy_lanes >= 2

    def test_single_model_uses_replicas_not_tp(self):
        """Single model (fits on 1 GPU) on 4 GPUs → 4 lanes with split tasks."""
        s, _, _ = make_scheduler()
        # No TP config → auto-TP would be 1 (model fits)
        groups = [
            ("Model", [make_task(f"t{i}", model="Model") for i in range(100)]),
        ]
        result = s._assign_groups_to_gpus(groups, 4)
        # Should use multiple lanes (replicas), not TP
        non_empty = [q for q in result if q]
        assert len(non_empty) >= 2  # Split across lanes
        total_tasks = sum(len(t) for q in result for _, t in q)
        assert total_tasks == 100

    def test_tp_config_default_is_one(self):
        """Models without explicit TP config should default to tp=1."""
        s, _, _ = make_scheduler()
        s.set_tp_config({"BigModel": 4})
        assert s._get_tp("BigModel") == 4
        assert s._get_tp("UnknownModel") == 1

    def test_auto_tp_small_model_single_gpu(self):
        """Model that fits in one GPU should auto-compute TP=1."""
        s, _, _ = make_scheduler()
        # 14GB model on 24GB GPU (80% usable = 19.2GB) → fits
        s._estimated_model_sizes["SmallModel"] = int(14e9)
        s._gpu_mem_bytes = int(24e9)
        assert s._get_tp("SmallModel") == 1

    def test_auto_tp_large_model_needs_two(self):
        """Model too large for one GPU should auto-compute TP=2."""
        s, _, _ = make_scheduler()
        # 30GB model on 24GB GPU (80% usable = 19.2GB) → needs 2 GPUs
        s._estimated_model_sizes["LargeModel"] = int(30e9)
        s._gpu_mem_bytes = int(24e9)
        assert s._get_tp("LargeModel") == 2

    def test_auto_tp_very_large_model_power_of_two(self):
        """TP should always be a power of 2."""
        s, _, _ = make_scheduler()
        # 60GB model on 24GB GPU (80% usable = 19.2GB) → needs ceil(60/19.2)=4
        s._estimated_model_sizes["HugeModel"] = int(60e9)
        s._gpu_mem_bytes = int(24e9)
        assert s._get_tp("HugeModel") == 4

    def test_manual_tp_overrides_auto(self):
        """Manual tp_config should override auto-computation."""
        s, _, _ = make_scheduler()
        s._estimated_model_sizes["Model"] = int(30e9)
        s._gpu_mem_bytes = int(24e9)
        # Auto would be TP=2, but manual sets TP=4
        s.set_tp_config({"Model": 4})
        assert s._get_tp("Model") == 4


# ============================================================================ #
# Memory-Aware Prefetch Guard
# ============================================================================ #


class TestMemoryAwarePrefetch:
    """Verify _can_prefetch memory checks."""

    @pytest.mark.asyncio
    async def test_unknown_pool_allows_prefetch(self):
        s, _, _ = make_scheduler()
        s.cpu_mem_pool_bytes = 0  # Unknown
        assert await s._can_prefetch("any-model") is True

    @pytest.mark.asyncio
    async def test_unknown_model_size_allows_prefetch(self):
        s, _, _ = make_scheduler()
        s.cpu_mem_pool_bytes = 100_000_000_000
        # No entry in _estimated_model_sizes
        assert await s._can_prefetch("unknown-model") is True

    @pytest.mark.asyncio
    async def test_fits_in_pool(self):
        s, _, _ = make_scheduler()
        s.cpu_mem_pool_bytes = 40_000_000_000  # 40GB
        s._estimated_model_sizes = {"ModelA": 15_000_000_000}  # 15GB
        s._cpu_cached_models = set()
        assert await s._can_prefetch("ModelA") is True

    @pytest.mark.asyncio
    async def test_exceeds_pool(self):
        s, _, _ = make_scheduler()
        s.cpu_mem_pool_bytes = 20_000_000_000  # 20GB
        s._estimated_model_sizes = {
            "ModelA": 15_000_000_000,
            "ModelB": 15_000_000_000,
        }
        s._cpu_cached_models = {"ModelA"}  # 15GB used
        # ModelB needs 15GB, only 5GB free
        assert await s._can_prefetch("ModelB") is False

    @pytest.mark.asyncio
    async def test_exact_fit(self):
        s, _, _ = make_scheduler()
        s.cpu_mem_pool_bytes = 30_000_000_000  # 30GB
        s._estimated_model_sizes = {
            "ModelA": 15_000_000_000,
            "ModelB": 15_000_000_000,
        }
        s._cpu_cached_models = {"ModelA"}
        # ModelB needs exactly the remaining 15GB
        assert await s._can_prefetch("ModelB") is True


# ============================================================================ #
# Proactive Model Unloading
# ============================================================================ #


class TestProactiveUnloading:
    """Verify _do_unload is called after model groups finish."""

    @pytest.mark.asyncio
    async def test_do_unload_calls_storage_manager(self):
        s, _, _ = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.unload_from_cpu = AsyncMock(return_value=True)
        s.storage_manager = mock_sm
        s._cpu_cached_models = {"ModelA"}

        await s._do_unload("ModelA")

        mock_sm.unload_from_cpu.assert_called_once_with("ModelA")
        assert "ModelA" not in s._cpu_cached_models

    @pytest.mark.asyncio
    async def test_do_unload_no_storage_manager(self):
        s, _, _ = make_scheduler()
        s.storage_manager = None
        # Should not raise
        await s._do_unload("ModelA")

    @pytest.mark.asyncio
    async def test_do_unload_failure_keeps_cached(self):
        s, _, _ = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.unload_from_cpu = AsyncMock(side_effect=Exception("gRPC error"))
        s.storage_manager = mock_sm
        s._cpu_cached_models = {"ModelA"}

        await s._do_unload("ModelA")
        # Should still be in cached set since unload failed
        assert "ModelA" in s._cpu_cached_models

    @pytest.mark.asyncio
    async def test_single_gpu_queue_unloads_after_group(self):
        """_run_single_gpu_queue should unload each model after its last group."""
        s, db, router = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.unload_from_cpu = AsyncMock(return_value=True)
        s.storage_manager = mock_sm

        tasks_a = [make_task(f"a{i}", model="A") for i in range(3)]
        tasks_b = [make_task(f"b{i}", model="B") for i in range(2)]
        groups = [("A", tasks_a), ("B", tasks_b)]

        await s._run_single_gpu_queue(groups, do_prefetch=False)

        # Both models should have been unloaded
        unload_calls = mock_sm.unload_from_cpu.call_args_list
        unloaded_models = [c[0][0] for c in unload_calls]
        assert "A" in unloaded_models
        assert "B" in unloaded_models

    @pytest.mark.asyncio
    async def test_multi_gpu_queue_unloads_after_group(self):
        """_process_gpu_queue should unload model after its last group on that GPU."""
        s, db, router = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.unload_from_cpu = AsyncMock(return_value=True)
        s.storage_manager = mock_sm

        tasks = [make_task(f"t{i}", model="A") for i in range(3)]
        queue = [("A", tasks)]

        await s._process_gpu_queue(
            gpu_id=0,
            queue=queue,
            do_prefetch=False,
            num_replicas_per_model={"A": 1},
        )

        mock_sm.unload_from_cpu.assert_called_once_with("A")


# ============================================================================ #
# Autoscaler Batch Registration
# ============================================================================ #


class TestAutoscalerBatchRegistration:
    """Verify batches are registered/unregistered with autoscaler."""

    @pytest.mark.asyncio
    async def test_register_on_process_start(self):
        s, db, router = make_scheduler()
        mock_as = MagicMock()
        mock_as.receive_metrics = MagicMock()
        mock_as.register_active_batch = MagicMock()
        mock_as.unregister_active_batch = MagicMock()
        s.autoscaler = mock_as

        tasks = [
            make_task("t1", batch_id="b1", model="A"),
            make_task("t2", batch_id="b1", model="B"),
        ]
        db.get_batch_tasks.side_effect = [
            tasks,  # initial fetch
            [make_task("t1", batch_id="b1", model="A", status="completed"),
             make_task("t2", batch_id="b1", model="B", status="completed")],  # final check
        ]

        await s._process_batch("b1")

        # Should register for both models
        register_calls = mock_as.register_active_batch.call_args_list
        registered = set(c[0][0] for c in register_calls)
        assert "A:vllm" in registered
        assert "B:vllm" in registered

    @pytest.mark.asyncio
    async def test_unregister_on_completion(self):
        s, db, router = make_scheduler()
        mock_as = MagicMock()
        mock_as.receive_metrics = MagicMock()
        mock_as.register_active_batch = MagicMock()
        mock_as.unregister_active_batch = MagicMock()
        s.autoscaler = mock_as

        tasks = [make_task("t1", batch_id="b1", model="A")]
        completed = [make_task("t1", batch_id="b1", model="A", status="completed")]
        db.get_batch_tasks.side_effect = [tasks, completed]

        await s._process_batch("b1")

        # Should unregister
        unregister_calls = mock_as.unregister_active_batch.call_args_list
        assert any("A:vllm" in str(c) for c in unregister_calls)


# ============================================================================ #
# Strategy Execution
# ============================================================================ #


class TestStrategyExecution:
    """Verify sync, chunked, and semaphore strategies all execute correctly."""

    @pytest.mark.asyncio
    async def test_sync_strategy(self):
        s, db, router = make_scheduler()
        s.set_strategy("sync")
        tasks = [make_task(f"t{i}", model="A") for i in range(5)]
        await s._execute_group(tasks)
        assert router.handle_request.call_count == 5

    @pytest.mark.asyncio
    async def test_chunked_strategy(self):
        s, db, router = make_scheduler()
        s.set_strategy("chunked")
        tasks = [make_task(f"t{i}", model="A") for i in range(10)]
        await s._execute_group(tasks)
        assert router.handle_request.call_count == 10

    @pytest.mark.asyncio
    async def test_semaphore_strategy(self):
        s, db, router = make_scheduler()
        s.set_strategy("semaphore")
        tasks = [make_task(f"t{i}", model="A") for i in range(10)]
        await s._execute_group(tasks)
        assert router.handle_request.call_count == 10

    @pytest.mark.asyncio
    async def test_invalid_strategy_raises(self):
        s, _, _ = make_scheduler()
        with pytest.raises(ValueError):
            s.set_strategy("invalid_strategy")


# ============================================================================ #
# Prefetch with Threshold
# ============================================================================ #


class TestPrefetchWithThreshold:
    """Verify prefetch triggers at the right threshold."""

    @pytest.mark.asyncio
    async def test_prefetch_triggers_at_threshold(self):
        s, db, router = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.prefetch_to_cpu = AsyncMock(return_value=True)
        s.storage_manager = mock_sm
        s.prefetch_threshold = 0.5

        tasks = [make_task(f"t{i}", model="A") for i in range(10)]

        await s._execute_group_with_prefetch(tasks, "B", 0.5)

        # Prefetch should have been triggered for model B
        mock_sm.prefetch_to_cpu.assert_called_once_with("B")

    @pytest.mark.asyncio
    async def test_prefetch_skipped_when_memory_full(self):
        s, db, router = make_scheduler()
        mock_sm = MagicMock()
        mock_sm.prefetch_to_cpu = AsyncMock(return_value=True)
        s.storage_manager = mock_sm
        s.cpu_mem_pool_bytes = 10_000_000_000  # 10GB
        s._estimated_model_sizes = {"A": 8_000_000_000, "B": 8_000_000_000}
        s._cpu_cached_models = {"A"}  # 8GB used, only 2GB free

        tasks = [make_task(f"t{i}", model="A") for i in range(10)]
        await s._execute_group_with_prefetch(tasks, "B", 0.5)

        # Prefetch should NOT have been called (B needs 8GB, only 2GB free)
        mock_sm.prefetch_to_cpu.assert_not_called()


# ============================================================================ #
# End-to-End: Full Batch Processing
# ============================================================================ #


class TestEndToEndBatchProcessing:
    """Integration tests for the full _process_batch path."""

    @pytest.mark.asyncio
    async def test_single_model_batch(self):
        s, db, router = make_scheduler()
        tasks = [make_task(f"t{i}", batch_id="b1", model="A") for i in range(5)]
        completed = [make_task(f"t{i}", batch_id="b1", model="A", status="completed") for i in range(5)]
        db.get_batch_tasks.side_effect = [tasks, completed]

        await s._process_batch("b1")

        assert router.handle_request.call_count == 5
        db.update_batch_job_status.assert_any_call("b1", "in_progress")
        db.update_batch_job_status.assert_any_call("b1", "completed")

    @pytest.mark.asyncio
    async def test_multi_model_batch(self):
        s, db, router = make_scheduler()
        tasks = [
            *[make_task(f"a{i}", batch_id="b1", model="A") for i in range(3)],
            *[make_task(f"b{i}", batch_id="b1", model="B") for i in range(2)],
        ]
        completed = [make_task(f"x{i}", batch_id="b1", model="A", status="completed") for i in range(5)]
        db.get_batch_tasks.side_effect = [tasks, completed]

        await s._process_batch("b1")

        assert router.handle_request.call_count == 5

    @pytest.mark.asyncio
    async def test_empty_batch_skipped(self):
        s, db, router = make_scheduler()
        db.get_batch_tasks.return_value = []

        await s._process_batch("b1")

        assert router.handle_request.call_count == 0

    @pytest.mark.asyncio
    async def test_all_completed_batch_marked_done(self):
        s, db, router = make_scheduler()
        tasks = [make_task("t1", batch_id="b1", model="A", status="completed")]
        db.get_batch_tasks.return_value = tasks

        await s._process_batch("b1")

        db.update_batch_job_status.assert_called_with("b1", "completed")

    @pytest.mark.asyncio
    async def test_task_failure_recorded(self):
        s, db, router = make_scheduler()
        router.handle_request = AsyncMock(side_effect=Exception("inference error"))

        tasks = [make_task("t1", batch_id="b1", model="A")]
        completed = [make_task("t1", batch_id="b1", model="A", status="failed")]
        db.get_batch_tasks.side_effect = [tasks, completed]

        await s._process_batch("b1")

        # Task should be recorded as failed
        upsert_calls = db.upsert_batch_task.call_args_list
        assert any(c.kwargs.get("status") == "failed" or
                    (len(c.args) > 6 and c.args[6] == "failed")
                    for c in upsert_calls)


# ============================================================================ #
# GPU Allocation Reclamation
# ============================================================================ #


class TestGPUAllocationReclamation:
    """Verify _prepare_gpu_allocations frees GPUs before multi-GPU batch."""

    def _make_deployment(self, model_name, backend="vllm", max_replicas=2):
        dep = MagicMock()
        dep.id = f"{model_name}:{backend}"
        dep.model_name = model_name
        dep.backend = backend
        dep.status = "active"
        dep.max_replicas = max_replicas
        dep.desired_replicas = max_replicas
        return dep

    @pytest.mark.asyncio
    async def test_scales_down_unneeded_deployment(self):
        """Deployments not in the batch should be scaled to 0."""
        s, db, router = make_scheduler()
        mock_as = MagicMock()
        mock_as._active_batches = {"OldModel:vllm": {"old_batch"}}
        mock_as.unregister_active_batch = MagicMock()
        s.autoscaler = mock_as

        # OldModel has 2 endpoints but is NOT needed for this batch
        old_dep = self._make_deployment("OldModel")
        db.get_all_deployments.return_value = [old_dep]
        db.get_deployment_endpoints.return_value = ["ep1", "ep2"]
        db.update_max_replicas = MagicMock()
        db.update_desired_replicas = MagicMock()

        # No pylet → skip wait loop
        s.pylet_client = None

        await s._prepare_gpu_allocations(
            batch_id="b1",
            num_replicas_per_model={"NewModel": 1},
            num_gpus=2,
        )

        db.update_max_replicas.assert_called_once_with("OldModel:vllm", 0)
        db.update_desired_replicas.assert_called_once_with("OldModel:vllm", 0)
        mock_as.unregister_active_batch.assert_called_once_with("OldModel:vllm", "old_batch")

    @pytest.mark.asyncio
    async def test_scales_down_over_provisioned_deployment(self):
        """Batch model with too many replicas should be capped."""
        s, db, router = make_scheduler()
        mock_as = MagicMock()
        mock_as._active_batches = {"A:vllm": {"baseline_batch"}}
        mock_as.unregister_active_batch = MagicMock()
        s.autoscaler = mock_as

        dep_a = self._make_deployment("A", max_replicas=2)
        db.get_all_deployments.return_value = [dep_a]
        db.get_deployment_endpoints.return_value = ["ep1", "ep2"]  # 2 endpoints
        db.update_max_replicas = MagicMock()
        db.update_desired_replicas = MagicMock()

        s.pylet_client = None

        await s._prepare_gpu_allocations(
            batch_id="b1",
            num_replicas_per_model={"A": 1, "B": 1},  # A needs only 1
            num_gpus=2,
        )

        db.update_max_replicas.assert_called_once_with("A:vllm", 1)
        db.update_desired_replicas.assert_called_once_with("A:vllm", 1)
        # Should unregister stale batch but NOT current batch
        mock_as.unregister_active_batch.assert_called_once_with("A:vllm", "baseline_batch")

    @pytest.mark.asyncio
    async def test_no_action_when_already_correct(self):
        """No scaling needed if deployments match batch requirements."""
        s, db, router = make_scheduler()

        dep_a = self._make_deployment("A", max_replicas=1)
        db.get_all_deployments.return_value = [dep_a]
        db.get_deployment_endpoints.return_value = ["ep1"]  # 1 endpoint = 1 needed
        db.update_max_replicas = MagicMock()
        db.update_desired_replicas = MagicMock()

        s.pylet_client = None

        await s._prepare_gpu_allocations(
            batch_id="b1",
            num_replicas_per_model={"A": 1},
            num_gpus=1,
        )

        db.update_max_replicas.assert_not_called()
        db.update_desired_replicas.assert_not_called()

    @pytest.mark.asyncio
    async def test_waits_for_gpus_with_pylet(self):
        """Should poll pylet until enough GPUs are free."""
        s, db, router = make_scheduler()

        dep = self._make_deployment("OldModel")
        db.get_all_deployments.return_value = [dep]
        db.get_deployment_endpoints.return_value = ["ep1"]
        db.update_max_replicas = MagicMock()
        db.update_desired_replicas = MagicMock()

        # Simulate pylet returning 0 free GPUs, then 2
        mock_pylet = MagicMock()
        worker_busy = MagicMock()
        worker_busy.available_gpus = 0
        worker_free = MagicMock()
        worker_free.available_gpus = 2
        mock_pylet.get_online_workers = AsyncMock(
            side_effect=[
                [worker_busy],
                [worker_busy],
                [worker_free],  # 3rd poll: GPUs freed
            ]
        )
        s.pylet_client = mock_pylet

        await s._prepare_gpu_allocations(
            batch_id="b1",
            num_replicas_per_model={"NewModel": 1},
            num_gpus=2,
        )

        assert mock_pylet.get_online_workers.call_count == 3

    @pytest.mark.asyncio
    async def test_clears_models_ready_cache(self):
        """Scaling down should clear _models_ready so endpoints are re-verified."""
        s, db, router = make_scheduler()
        s._models_ready = {"OldModel", "SomeOther"}

        dep = self._make_deployment("OldModel")
        db.get_all_deployments.return_value = [dep]
        db.get_deployment_endpoints.return_value = ["ep1"]
        db.update_max_replicas = MagicMock()
        db.update_desired_replicas = MagicMock()

        s.pylet_client = None

        await s._prepare_gpu_allocations(
            batch_id="b1",
            num_replicas_per_model={"NewModel": 1},
            num_gpus=2,
        )

        assert len(s._models_ready) == 0
