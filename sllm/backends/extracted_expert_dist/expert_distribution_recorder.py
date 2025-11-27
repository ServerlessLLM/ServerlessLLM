# SPDX-License-Identifier: Apache-2.0
# vLLM Expert Distribution Recorder
# Adapted from sglang's expert_distribution.py
#
# This module provides comprehensive expert distribution recording for MoE models.
# It supports multiple recording modes:
# - per_token: Records per-token expert selections in detail
# - per_pass: Records per-forward-pass expert activation metrics
# - stat: Records aggregate statistics across tokens
# - stat_approx: Approximate statistics for distributed expert dispatch

from __future__ import annotations

import logging
import math
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed


# torch.compile compatible expert selection recording
def record_expert_selections_atomic(
    expert_counts: torch.Tensor,  # Shape: (num_layers, num_experts)
    layer_idx: int,
    topk_ids: torch.Tensor,  # Shape: (num_tokens, topk)
) -> None:
    """
    Record expert selections using atomic operations that are torch.compile compatible.
    This function uses only PyTorch operations and can be traced by torch.compile.
    CUDA graph compatible: avoids CPU-side conditionals that break graph capture.
    """
    # Flatten and filter valid expert IDs (-1 indicates padding/invalid)
    # Follow SGLang's approach: use masked_fill instead of conditional to avoid CPU sync
    topk_ids_flat = topk_ids.flatten()

    # Ensure tensors are on the same device (important for CUDA graph compatibility)
    if topk_ids_flat.device != expert_counts.device:
        topk_ids_flat = topk_ids_flat.to(expert_counts.device)

    mask = topk_ids_flat != -1

    # Use scatter_add unconditionally - masked_fill ensures invalid indices don't affect results
    # This avoids CPU-side conditionals that break CUDA graph capture
    index = topk_ids_flat.masked_fill(~mask, 0).long()
    src = mask.int()

    # Safety check for layer_idx
    if layer_idx < 0 or layer_idx >= expert_counts.shape[0]:
        return

    # Atomic add to the expert counts - this is torch.compile and CUDA graph compatible
    # Clamp index to ensure we don't write out of bounds (which causes hard crash)
    max_expert_idx = expert_counts.shape[1] - 1
    index = index.clamp(0, max_expert_idx)
    expert_counts[layer_idx].scatter_add_(dim=0, index=index, src=src)


logger = logging.getLogger(__name__)

# Global buffer for torch.compile compatible recording
_global_expert_counts_buffer: torch.Tensor | None = None


def get_global_expert_counts_buffer() -> torch.Tensor | None:
    """Get the global expert counts buffer for torch.compile compatible recording."""
    return _global_expert_counts_buffer


def set_global_expert_counts_buffer(buffer: torch.Tensor | None) -> None:
    """Set the global expert counts buffer."""
    global _global_expert_counts_buffer
    _global_expert_counts_buffer = buffer


# --------------------------------------- Recording Modes -----------------------------------------

_RecordingMode = str  # "per_token", "stat", "stat_approx", or None


class ExpertLocationMetadata:
    """Metadata about expert locations in the model."""

    def __init__(
        self,
        num_layers: int,
        num_logical_experts: int,
        num_physical_experts: int,
        num_local_physical_experts: int,
        ep_size: int,
        physical_to_logical_map: Optional[torch.Tensor] = None,
    ):
        self.num_layers = num_layers
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.ep_size = ep_size
        # physical_to_logical_map: maps physical expert indices to logical expert indices
        self.physical_to_logical_map = physical_to_logical_map or torch.arange(
            num_logical_experts
        )


# --------------------------------------- Entrypoint -----------------------------------------


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording interface."""

    @staticmethod
    def init_new(
        recording_mode: Optional[_RecordingMode],
        expert_location_metadata: Optional[ExpertLocationMetadata] = None,
        rank: int = 0,
        device: str = "cuda",
        buffer_size: int = -1,
        enable_metrics: bool = False,
    ) -> "ExpertDistributionRecorder":
        """Initialize a new expert distribution recorder.

        Args:
            recording_mode: One of "per_token", "per_pass", "stat", "stat_approx", or None
            expert_location_metadata: Metadata about expert locations
            rank: Distributed rank
            device: Device to use (cuda or cpu)
            buffer_size: Size of recording buffer (-1 for unlimited)
            enable_metrics: Whether to compute and log metrics

        Returns:
            An ExpertDistributionRecorder instance
        """
        if recording_mode is not None:
            assert (
                expert_location_metadata is not None
            ), "ExpertLocationMetadata is required for expert distribution recording"
            return _ExpertDistributionRecorderReal(
                recording_mode,
                expert_location_metadata,
                rank,
                device,
                buffer_size,
                enable_metrics,
            )
        else:
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx: int):
        """Context manager for tracking current layer."""
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int):
        """Context manager for tracking a forward pass."""
        yield

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Called when experts are selected.

        Args:
            layer_idx: Index of the current layer
            topk_ids: Selected expert indices (shape: [num_tokens, topk])
        """
        pass

    def on_expert_dispatch(
        self,
        layer_idx: int,
        num_tokens_per_expert: torch.Tensor,
    ) -> None:
        """Called during expert dispatch/load balancing.

        Args:
            layer_idx: Index of the current layer
            num_tokens_per_expert: Token count per expert
        """
        pass

    def start_record(self) -> None:
        """Start recording expert distributions."""
        self._on_not_implemented()

    def stop_record(self) -> None:
        """Stop recording expert distributions."""
        self._on_not_implemented()

    def dump_record(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump the recorded expert distribution data.

        Args:
            output_path: Path to save the recording (optional)

        Returns:
            Dictionary containing recorded data
        """
        self._on_not_implemented()

    @property
    def recording(self) -> bool:
        """Whether recording is currently active."""
        return False

    def _on_not_implemented(self):
        raise NotImplementedError(
            "Please set recording_mode to use ExpertDistributionRecorder"
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    """No-op implementation for when recording is disabled."""

    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    """Full implementation of expert distribution recording."""

    def __init__(
        self,
        recording_mode: _RecordingMode,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
        buffer_size: int,
        enable_metrics: bool,
    ):
        self._recording_mode = recording_mode
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank
        self._device = device
        self._buffer_size = buffer_size
        self._enable_metrics = enable_metrics

        self._recording = False
        # Use Tensor for current_layer_idx to be traceable by Dynamo/CUDAGraph
        self._current_layer_idx = torch.tensor(
            [-1], dtype=torch.long, device=self._device
        )
        self._current_forward_pass_id = None

        self._gatherer = _SinglePassGatherer.init_new(
            recording_mode, expert_location_metadata, rank, device
        )
        self._accumulator = _Accumulator.init_new(
            recording_mode,
            expert_location_metadata,
            rank,
            device,
            buffer_size,
            enable_metrics,
        )

        # Set up global buffer early so it's available during CUDA graph capture
        # This ensures the buffer pointer is captured into CUDA graphs
        if hasattr(self._gatherer, "get_expert_counts_buffer"):
            set_global_expert_counts_buffer(
                self._gatherer.get_expert_counts_buffer()
            )

        if enable_metrics:
            logger.debug(
                "ExpertDistributionRecorder auto-starting due to enable_metrics=True"
            )
            self.start_record()

    @contextmanager
    def with_current_layer(self, layer_idx: int):
        """Context manager for tracking current layer.
        Updates the tensor in-place so it's captured by Dynamo/CUDAGraph.
        """
        # We don't support nested layers with tensor state restoration efficiently in graph
        # But for sequential layers, this is fine.
        self._current_layer_idx.fill_(layer_idx)
        try:
            yield
        finally:
            self._current_layer_idx.fill_(-1)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int):
        """Context manager for tracking a forward pass."""
        prev_pass_id = self._current_forward_pass_id
        self._current_forward_pass_id = forward_pass_id

        if self._recording:
            self._gatherer.reset()

        try:
            yield
        finally:
            if self._recording:
                single_pass_data = self._gatherer.collect()
                self._accumulator.append(forward_pass_id, single_pass_data)

            self._current_forward_pass_id = prev_pass_id

    def on_select_experts(
        self, layer_idx: int | torch.Tensor | None, topk_ids: torch.Tensor
    ) -> None:
        """Record expert selection.

        prefers _current_layer_idx from context manager,
        falls back to layer_idx parameter. Allows operations during CUDA graph capture.
        """
        is_capturing = torch.get_device_module().is_current_stream_capturing()

        if not (self._recording or is_capturing):
            return

        # Prefer _current_layer_idx from context manager (matches SGLang pattern)
        # If layer_idx is explicitly passed, use it. Otherwise use context.
        if layer_idx is None:
            layer_idx = self._current_layer_idx

        # We assume layer_idx is valid (either int or Tensor).
        # If it's a Tensor, it might be -1 (invalid), but we rely on patched_qwen_forward to set it.

        # During CUDA graph capture, only allow CUDA graph compatible gatherers
        if is_capturing and not self._recording:
            if not hasattr(self._gatherer, "_expert_count"):
                return

        self._gatherer.on_select_experts(layer_idx, topk_ids)

    def get_expert_counts_buffer(self) -> Optional[torch.Tensor]:
        """Get the GPU buffer for expert counts (for torch.compile compatibility)."""
        if hasattr(self._gatherer, "get_expert_counts_buffer"):
            return self._gatherer.get_expert_counts_buffer()
        return None

    def on_expert_dispatch(
        self,
        layer_idx: int,
        num_tokens_per_expert: torch.Tensor,
    ) -> None:
        """Record expert dispatch information."""
        is_capturing = torch.cuda.is_current_stream_capturing()
        if not (self._recording or is_capturing):
            return
        if self._current_layer_idx is None and layer_idx is None:
            return

        effective_layer_idx = (
            layer_idx if layer_idx is not None else self._current_layer_idx
        )
        self._gatherer.on_expert_dispatch(
            effective_layer_idx, num_tokens_per_expert
        )

    def start_record(self) -> None:
        """Start recording."""
        if self._recording:
            logger.warning("Expert distribution recorder already recording")
            return
        self._gatherer.reset()
        self._accumulator.reset()
        self._recording = True

        # Set global buffer for torch.compile compatibility
        if hasattr(self._gatherer, "get_expert_counts_buffer"):
            set_global_expert_counts_buffer(
                self._gatherer.get_expert_counts_buffer()
            )

        logger.debug(
            f"Expert distribution recording started (mode={self._recording_mode})"
        )

    def stop_record(self) -> None:
        """Stop recording."""
        if not self._recording:
            logger.warning(
                "Expert distribution recorder not currently recording"
            )
            return
        self._recording = False

        # Reset gatherer but preserve accumulator data for dumping
        self._gatherer.reset()
        # self._accumulator.reset()  # Don't reset accumulator on stop, so we can dump results

        # Clear global buffer
        set_global_expert_counts_buffer(None)

        logger.debug("Expert distribution recording stopped")

    def dump_record(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump recorded data to file or return as dict."""
        if not hasattr(self._accumulator, "dump"):
            raise NotImplementedError(
                f"dump_record not supported for mode {self._recording_mode}"
            )

        # For STAT mode, collect data from gatherer first since it accumulates directly
        if self._recording_mode in ("stat", "stat_approx"):
            single_pass_data = self._gatherer.collect()
            self._accumulator.append(
                0, single_pass_data
            )  # Use dummy forward_pass_id

        output = self._accumulator.dump(output_path=output_path)
        # Add recording_mode to output for easier identification
        output["recording_mode"] = self._recording_mode
        # Don't reset here - reset happens in stop_record
        return output

    @property
    def recording(self) -> bool:
        return self._recording


# Global singleton
_global_expert_distribution_recorder: ExpertDistributionRecorder = (
    _ExpertDistributionRecorderNoop()
)


def get_global_expert_distribution_recorder() -> ExpertDistributionRecorder:
    """Get the global expert distribution recorder instance."""
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(
    recorder: ExpertDistributionRecorder,
) -> None:
    """Set the global expert distribution recorder instance."""
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = recorder


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    """Gathers expert distribution data for a single forward pass."""

    @staticmethod
    def init_new(
        recording_mode: _RecordingMode,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ) -> "_SinglePassGatherer":
        """Create appropriate gatherer based on recording mode."""
        if recording_mode == "per_token":
            return _PerTokenGatherer(expert_location_metadata, rank, device)
        elif recording_mode == "stat":
            return _StatGatherer(expert_location_metadata, rank, device)
        elif recording_mode == "stat_approx":
            return _StatApproxGatherer(expert_location_metadata, rank, device)
        elif recording_mode == "per_pass":
            return _PerPassGatherer(expert_location_metadata, rank, device)
        else:
            raise ValueError(f"Unknown recording mode: {recording_mode}")

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank
        self._device = device

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Record expert selection."""
        pass

    def on_expert_dispatch(
        self,
        layer_idx: int,
        num_tokens_per_expert: torch.Tensor,
    ) -> None:
        """Record expert dispatch information."""
        pass

    def reset(self) -> None:
        """Reset gatherer state."""
        raise NotImplementedError

    def collect(self) -> Dict[str, Any]:
        """Collect gathered data."""
        raise NotImplementedError


class _PerTokenGatherer(_SinglePassGatherer):
    """Records per-token expert selections in detail."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        super().__init__(expert_location_metadata, rank, device)
        # Convert device string to torch.device if needed
        self._device = (
            torch.device(device)
            if isinstance(device, str)
            else torch.device(device)
        )
        self._num_layers = expert_location_metadata.num_layers
        self._reset_storage()

    def on_select_experts(
        self, layer_idx: int | torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        """Store expert selection."""
        if topk_ids.numel() == 0:
            return

        # Handle layer_idx as Tensor or int
        if isinstance(layer_idx, torch.Tensor):
            layer_idx_val = layer_idx.item()
        else:
            layer_idx_val = layer_idx

        # Validate layer_idx
        if layer_idx_val < 0 or layer_idx_val >= self._num_layers:
            return  # Invalid layer, skip recording

        if topk_ids.device != self._device:
            topk_ids = topk_ids.to(self._device)

        stored = self._layer_records[layer_idx_val]
        topk_ids = topk_ids.detach().clone()
        if stored is None:
            self._layer_records[layer_idx_val] = topk_ids
        else:
            self._layer_records[layer_idx_val] = torch.cat(
                (stored, topk_ids), dim=0
            )

        updated = self._layer_records[layer_idx_val]
        self._max_tokens = max(self._max_tokens, updated.shape[0])
        self._max_topk = max(self._max_topk, updated.shape[1])

    def reset(self) -> None:
        """Reset gatherer."""
        self._reset_storage()

    def collect(self) -> Dict[str, Any]:
        """Collect per-token data."""
        if self._max_tokens == 0 or self._max_topk == 0:
            topk_tensor = torch.empty(
                (self._num_layers, 0, 0),
                dtype=torch.int32,
                device=self._device,
            )
        else:
            topk_tensor = torch.full(
                (self._num_layers, self._max_tokens, self._max_topk),
                fill_value=-1,
                dtype=torch.int32,
                device=self._device,
            )

            for layer_idx, record in enumerate(self._layer_records):
                if record is None:
                    continue
                tokens, topk = record.shape
                topk_tensor[layer_idx, :tokens, :topk] = record

        return {
            "type": "per_token",
            "topk_ids": topk_tensor.cpu(),
            "num_tokens": self._max_tokens,
            "topk": self._max_topk,
        }

    def _reset_storage(self) -> None:
        self._layer_records: list[torch.Tensor | None] = [
            None
        ] * self._num_layers
        self._max_tokens = 0
        self._max_topk = 0


class _StatGatherer(_SinglePassGatherer):
    """Records aggregate expert selection statistics."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        super().__init__(expert_location_metadata, rank, device)
        # Convert device string to torch.device if needed
        device_obj = torch.device(device) if isinstance(device, str) else device
        # Shape: (num_layers, num_physical_experts)
        self._expert_count = torch.zeros(
            (
                expert_location_metadata.num_layers,
                expert_location_metadata.num_physical_experts,
            ),
            dtype=torch.int32,
            device=device_obj,
        )
        self._num_layers = expert_location_metadata.num_layers

    def on_select_experts(
        self, layer_idx: int | torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        """Accumulate expert selection counts.

        CUDA Graph compatible: All operations are pure tensor ops.
        When layer_idx is invalid (-1 or None), we use layer 0 as fallback
        to still count expert activations (just without per-layer granularity).
        """
        num_layers = self._expert_count.shape[0]
        num_experts = self._expert_count.shape[1]
        topk_ids_flat = topk_ids.flatten()

        # Create mask for valid topk_ids (not -1)
        valid_topk_mask = topk_ids_flat != -1

        # safe_topk ensures we don't have negative indices
        safe_topk = topk_ids_flat.masked_fill(~valid_topk_mask, 0).long()

        # Handle layer_idx - use tensor operations for CUDA graph compatibility
        if isinstance(layer_idx, torch.Tensor):
            # Tensor case: clamp to valid range, use 0 for invalid (-1)
            # This is CUDA graph compatible
            safe_layer_idx = layer_idx.clamp(0, num_layers - 1)
            flat_indices = safe_layer_idx * num_experts + safe_topk
        elif (
            layer_idx is not None
            and isinstance(layer_idx, int)
            and layer_idx >= 0
        ):
            # Valid int layer_idx
            safe_layer_idx = min(layer_idx, num_layers - 1)
            flat_indices = safe_layer_idx * num_experts + safe_topk
        else:
            # Invalid layer_idx (None or negative) - use layer 0 as fallback
            flat_indices = safe_topk

        src = valid_topk_mask.int()

        # Clamp to valid range (safety)
        max_flat_idx = num_layers * num_experts - 1
        flat_indices = flat_indices.clamp(0, max_flat_idx)

        # Atomic add - CUDA graph compatible
        self._expert_count.view(-1).scatter_add_(
            dim=0, index=flat_indices, src=src
        )

    def get_expert_counts_buffer(self) -> torch.Tensor:
        """Get the GPU buffer for expert counts (for torch.compile compatibility)."""
        return self._expert_count

    def reset(self) -> None:
        """Reset gatherer."""
        self._expert_count.fill_(0)

    def collect(self) -> Dict[str, Any]:
        """Collect aggregated statistics."""
        return {
            "type": "stat",
            "expert_count": self._expert_count.clone(),
        }


class _StatApproxGatherer(_SinglePassGatherer):
    """Records approximate statistics via expert dispatch information."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        super().__init__(expert_location_metadata, rank, device)
        # Convert device string to torch.device if needed
        device_obj = torch.device(device) if isinstance(device, str) else device
        self._expert_count = torch.zeros(
            (
                expert_location_metadata.num_layers,
                expert_location_metadata.num_local_physical_experts,
            ),
            dtype=torch.int32,
            device=device_obj,
        )

    def on_expert_dispatch(
        self,
        layer_idx: int | torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> None:
        """Accumulate from dispatch information."""
        # Handle layer_idx as Tensor or int
        if isinstance(layer_idx, torch.Tensor):
            layer_idx_val = layer_idx.item()
        else:
            layer_idx_val = layer_idx

        # Validate layer_idx
        num_layers = self._expert_count.shape[0]
        if layer_idx_val < 0 or layer_idx_val >= num_layers:
            return  # Invalid layer, skip recording

        # Ensure num_tokens_per_expert is on the same device and dtype
        num_tokens_per_expert = num_tokens_per_expert.to(
            device=self._expert_count.device, dtype=self._expert_count.dtype
        )
        # Accumulate instead of replace
        self._expert_count[layer_idx_val] += num_tokens_per_expert

    def reset(self) -> None:
        """Reset gatherer."""
        self._expert_count.fill_(0)

    def collect(self) -> Dict[str, Any]:
        """Collect approximate statistics."""
        return {
            "type": "stat_approx",
            "expert_count": self._expert_count.clone(),
        }


class _PerPassGatherer(_StatGatherer):
    """Gatherer for per-pass mode - collects expert counts per forward pass."""

    def collect(self) -> Dict[str, Any]:
        """Collect expert counts for this forward pass."""
        result = super().collect()
        result["type"] = "per_pass"  # Override type
        return result


# --------------------------------------- Accumulator -----------------------------------------


class _Accumulator(ABC):
    """Accumulates expert distribution data across multiple forward passes."""

    @staticmethod
    def init_new(
        recording_mode: _RecordingMode,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
        buffer_size: int,
        enable_metrics: bool,
    ) -> "_Accumulator":
        """Create appropriate accumulator based on recording mode."""
        if recording_mode in ("stat", "stat_approx"):
            return _StatAccumulator(
                expert_location_metadata,
                rank,
                device,
                buffer_size,
                enable_metrics,
            )
        elif recording_mode == "per_token":
            return _PerTokenAccumulator(expert_location_metadata, rank, device)
        elif recording_mode == "per_pass":
            return _PerPassAccumulator(expert_location_metadata, rank, device)
        else:
            raise ValueError(f"Unknown recording mode: {recording_mode}")

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank
        self._device = device

    def append(
        self, forward_pass_id: int, single_pass_data: Dict[str, Any]
    ) -> None:
        """Append data from a single forward pass."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset accumulator."""
        raise NotImplementedError

    def dump(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump accumulated data."""
        raise NotImplementedError


class _PerTokenAccumulator(_Accumulator):
    """Accumulates per-token expert selection data."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        super().__init__(expert_location_metadata, rank, device)
        self._records = []

    def append(
        self, forward_pass_id: int, single_pass_data: Dict[str, Any]
    ) -> None:
        """Append per-token data."""
        self._records.append(
            {
                "forward_pass_id": forward_pass_id,
                "rank": self._rank,
                **single_pass_data,
            }
        )

    def reset(self) -> None:
        """Reset accumulator."""
        self._records.clear()

    def dump(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump per-token data to file."""
        processed_records = []
        for record in self._records:
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, torch.Tensor):
                    cleaned[key] = value.detach().cpu().tolist()
                else:
                    cleaned[key] = value
            processed_records.append(cleaned)

        output = {
            "records": processed_records,
            "rank": self._rank,
            "num_layers": self._expert_location_metadata.num_layers,
            "num_physical_experts": self._expert_location_metadata.num_physical_experts,
        }

        if output_path:
            logger.info(
                f"Saving expert distribution recording to {output_path}"
            )
            torch.save(output, output_path)

        return output


class _PerPassAccumulator(_Accumulator):
    """Accumulates per-forward-pass expert activation metrics."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
    ):
        super().__init__(expert_location_metadata, rank, device)
        self._pass_records = []

    def append(
        self, forward_pass_id: int, single_pass_data: Dict[str, Any]
    ) -> None:
        """Store per-pass expert activation data."""
        # Calculate expert activation metrics per forward pass
        if "expert_count" in single_pass_data:
            expert_counts = single_pass_data["expert_count"]

            # Only append if there are actual expert activations
            total_sum = expert_counts.sum().item()
            if total_sum > 0:
                # Calculate activated experts per layer
                # expert_counts shape: (num_layers, num_experts)
                activated_experts = (expert_counts > 0).float()
                activated_per_layer = activated_experts.sum(
                    dim=1
                )  # Sum across experts per layer
                total_activated = (
                    activated_per_layer.sum().item()
                )  # Total activated experts
                avg_activated_per_layer = activated_per_layer.mean().item()

                # Calculate expert utilization
                total_possible_experts = expert_counts.numel()
                expert_utilization = (
                    total_activated / total_possible_experts
                    if total_possible_experts > 0
                    else 0
                )

                # Store only summary statistics (not full expert_counts to save memory/bandwidth)
                record = {
                    "forward_pass_id": forward_pass_id,
                    "rank": self._rank,
                    "total_activated_experts": total_activated,
                    "avg_activated_per_layer": round(
                        avg_activated_per_layer, 3
                    ),
                    "expert_utilization": round(expert_utilization, 4),
                    "total_tokens": int(total_sum),
                    "timestamp": time.time(),
                }

                # Add forward_mode if available (injected by vLLM integration)
                if "forward_mode" in single_pass_data:
                    record["forward_mode"] = single_pass_data["forward_mode"]

                self._pass_records.append(record)

    def reset(self) -> None:
        """Reset accumulator."""
        self._pass_records.clear()

    def dump(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump per-pass records with summary statistics."""
        num_records = len(self._pass_records)

        # Compute aggregate statistics
        if num_records > 0:
            avg_activated = (
                sum(
                    r.get("avg_activated_per_layer", 0)
                    for r in self._pass_records
                )
                / num_records
            )
            avg_utilization = (
                sum(r.get("expert_utilization", 0) for r in self._pass_records)
                / num_records
            )
            total_tokens = sum(
                r.get("total_tokens", 0) for r in self._pass_records
            )
        else:
            avg_activated = 0
            avg_utilization = 0
            total_tokens = 0

        output = {
            "rank": self._rank,
            "num_layers": self._expert_location_metadata.num_layers,
            "num_experts": self._expert_location_metadata.num_logical_experts,
            "num_physical_experts": self._expert_location_metadata.num_physical_experts,
            "total_forward_passes": num_records,
            # Summary statistics (fast to serialize)
            "avg_activated_per_layer": round(avg_activated, 3),
            "avg_expert_utilization": round(avg_utilization, 4),
            "total_tokens_processed": total_tokens,
        }

        if output_path:
            # Only save full records to file, not return them
            full_output = {**output, "records": self._pass_records}
            logger.info(
                f"Saving expert distribution recording to {output_path}"
            )
            torch.save(full_output, output_path)

        return output


class _StatAccumulator(_Accumulator):
    """Accumulates aggregated expert statistics."""

    def __init__(
        self,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
        device: str,
        buffer_size: int,
        enable_metrics: bool,
    ):
        super().__init__(expert_location_metadata, rank, device)
        self._buffer_size = buffer_size
        self._enable_metrics = enable_metrics
        self._expert_counts = []
        self._forward_pass_ids = []

        if enable_metrics:
            self._history = _DequeCollection(maxlens=[10, 100, 1000])

    def append(
        self, forward_pass_id: int, single_pass_data: Dict[str, Any]
    ) -> None:
        """Append aggregated data."""
        expert_count = single_pass_data["expert_count"]
        self._expert_counts.append(expert_count)
        self._forward_pass_ids.append(forward_pass_id)

        # Enforce buffer size
        if (
            self._buffer_size > 0
            and len(self._expert_counts) > self._buffer_size
        ):
            self._expert_counts.pop(0)
            self._forward_pass_ids.pop(0)

        # Compute metrics if enabled
        if self._enable_metrics:
            self._compute_and_log_metrics(forward_pass_id, expert_count)

    def _compute_and_log_metrics(
        self,
        forward_pass_id: int,
        expert_count: torch.Tensor,
    ) -> None:
        """Compute and log expert balancedness metrics."""
        if not torch.distributed.is_initialized():
            return

        # Compute balancedness: avg_load / max_load
        gpu_load = self._compute_gpu_load(expert_count)
        balancedness = self._compute_balancedness(gpu_load)
        avg_balancedness = balancedness.mean().item()

        if self._enable_metrics:
            self._history.append(avg_balancedness)

            if self._rank == 0:
                history_means = self._history.mean()
                logger.debug(
                    f"[Expert Balancedness] forward_pass={forward_pass_id} "
                    f"current={avg_balancedness:.3f} "
                    f"history={history_means}"
                )

    def _compute_gpu_load(self, expert_count: torch.Tensor) -> torch.Tensor:
        """Compute per-GPU load from expert counts."""
        # expert_count shape: (num_layers, num_physical_experts)
        # Reshape to (num_layers, num_gpus, num_experts_per_gpu)
        num_layers, num_physical_experts = expert_count.shape
        num_gpus = self._expert_location_metadata.ep_size

        if num_physical_experts % num_gpus != 0:
            # Fallback: return sum per layer
            return expert_count.sum(dim=1, keepdim=True)

        num_experts_per_gpu = num_physical_experts // num_gpus
        gpu_count = expert_count.view(
            num_layers, num_gpus, num_experts_per_gpu
        ).sum(dim=2)
        return gpu_count

    def _compute_balancedness(self, gpu_load: torch.Tensor) -> torch.Tensor:
        """Compute balancedness per layer."""
        avg_load = gpu_load.mean(dim=1, keepdim=True)
        max_load = gpu_load.max(dim=1, keepdim=True)[0]
        balancedness = (avg_load + 1e-5) / (max_load + 1e-5)
        return balancedness.squeeze(-1)

    def reset(self) -> None:
        """Reset accumulator."""
        self._expert_counts.clear()
        self._forward_pass_ids.clear()
        if self._enable_metrics:
            self._history.clear()

    def dump(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Dump aggregated statistics."""
        # Aggregate expert counts across all ranks
        if torch.distributed.is_initialized():
            aggregated_counts = self._aggregate_counts_distributed()
        elif self._expert_counts:
            aggregated_counts = torch.stack(self._expert_counts)
        else:
            aggregated_counts = torch.empty(0, dtype=torch.int32)

        aggregated_counts_cpu = aggregated_counts.cpu()
        aggregated_counts_list = aggregated_counts_cpu.tolist()

        output = {
            "aggregated_expert_counts": aggregated_counts_list,
            "forward_pass_ids": list(self._forward_pass_ids),
            "rank": self._rank,
            "num_layers": self._expert_location_metadata.num_layers,
            "num_physical_experts": self._expert_location_metadata.num_physical_experts,
        }

        if output_path:
            logger.info(
                f"Saving expert distribution recording to {output_path}"
            )
            torch.save(output, output_path)

        return output

    def _aggregate_counts_distributed(self) -> torch.Tensor:
        """Aggregate counts across all ranks."""
        if not self._expert_counts:
            return torch.empty(0, device=self._device)

        local_counts = torch.stack(self._expert_counts)
        aggregated = local_counts.clone()
        torch.distributed.all_reduce(aggregated)
        return aggregated.cpu()


class _DequeCollection:
    """Collection of deques with different window sizes for metrics."""

    def __init__(self, maxlens: List[int]):
        self._deques = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value: float) -> None:
        """Append value to all deques."""
        for d in self._deques:
            d.append(value)

    def clear(self) -> None:
        """Clear all deques."""
        for d in self._deques:
            d.clear()

    def mean(self) -> Dict[int, float]:
        """Get mean for each deque."""
        return {d.maxlen: sum(d) / len(d) if d else 0.0 for d in self._deques}
