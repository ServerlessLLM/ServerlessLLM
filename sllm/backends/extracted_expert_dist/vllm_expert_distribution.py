#!/usr/bin/env python3
"""
vLLM Expert Distribution Easy Integration Wrapper

This module provides a simple, import-and-use wrapper for expert distribution
recording in vLLM, similar to sglang.py. Just import this module and use
LLM.generate() normally - expert distribution will be automatically recorded.

Usage:
    from vllm import LLM
    from vllm_expert_distribution import enable_expert_distribution

    enable_expert_distribution(mode="stat")
    llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B")
    outputs = llm.generate(["prompt"])  # Automatically records!
    # outputs[0].expert_stats contains expert distribution data
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add current directory to path for imports
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir))

# Import monkey patching function
from vllm_integration import apply_vllm_monkey_patching

logger = logging.getLogger(__name__)

# Global configuration state
_config = {
    "enabled": os.environ.get("VLLM_EXPERT_DIST_ENABLED", "true").lower()
    == "true",
    "mode": os.environ.get("VLLM_EXPERT_DIST_MODE", "stat"),
    "output_dir": os.environ.get("VLLM_EXPERT_DIST_OUTPUT_DIR", "./outputs"),
    "verbose": os.environ.get("VLLM_EXPERT_DIST_VERBOSE", "false").lower()
    == "true",
    "auto_record": True,  # Auto-start/stop recording per generation
}

# Track which LLM instances have been configured
_configured_llms = set()


def enable_expert_distribution(
    mode: str = "stat",
    enabled: bool = True,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    auto_record: bool = True,
):
    """
    Enable expert distribution recording with specified configuration.

    Args:
        mode: Recording mode - "stat", "per_token", or "per_pass" (default: "stat")
        enabled: Enable/disable recording (default: True)
        output_dir: Directory for output files (default: "./outputs")
        verbose: Enable verbose logging (default: False)
        auto_record: Auto-start/stop recording per generation (default: True)
    """
    _config["enabled"] = enabled
    _config["mode"] = mode
    if output_dir:
        _config["output_dir"] = output_dir
    _config["verbose"] = verbose
    _config["auto_record"] = auto_record

    # Ensure output directory exists
    if enabled:
        os.makedirs(_config["output_dir"], exist_ok=True)

    logger.info(
        f"Expert distribution enabled: mode={mode}, enabled={enabled}, auto_record={auto_record}"
    )


def disable_expert_distribution():
    """Disable expert distribution recording."""
    _config["enabled"] = False
    logger.info("Expert distribution disabled")


def set_mode(mode: str):
    """Set the recording mode."""
    if mode not in ["stat", "per_token", "per_pass"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of: stat, per_token, per_pass"
        )
    _config["mode"] = mode
    logger.info(f"Recording mode set to: {mode}")


def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return _config.copy()


def _configure_llm_recording(llm):
    """Configure expert distribution recording for an LLM instance."""
    if id(llm) in _configured_llms:
        return

    if not _config["enabled"]:
        return

    try:
        # Configure recording
        llm.collective_rpc(
            "configure_expert_distribution_recording",
            args=(_config["mode"], _config["verbose"]),
        )
        _configured_llms.add(id(llm))
        logger.debug(
            f"Configured expert distribution recording for LLM instance (mode={_config['mode']})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to configure expert distribution recording: {e}"
        )


def _extract_expert_stats(
    all_data: List[Dict], mode: str
) -> Optional[Dict[str, Any]]:
    """Extract expert statistics from worker data."""
    if not all_data:
        return None

    # Filter out empty data dictionaries
    valid_data = [d for d in all_data if d and isinstance(d, dict)]
    if not valid_data:
        return None

    # For stat mode, use first worker's data (rank 0)
    if mode == "stat":
        data = valid_data[0]
        if "aggregated_expert_counts" in data:
            counts = data["aggregated_expert_counts"]
            forward_pass_ids = data.get("forward_pass_ids", [])

            # Handle empty counts or None
            if not counts:
                return None

            # Ensure counts is a list of lists (even if single entry)
            if isinstance(counts[0], (int, float)) if counts else False:
                # Single list of counts, wrap it
                counts = [counts]

            if counts and len(counts) > 0 and len(counts[0]) > 0:
                import torch

                counts_tensor = torch.tensor(counts)
                expert_totals = (
                    counts_tensor.sum(dim=0)
                    if counts_tensor.ndim > 1
                    else counts_tensor
                )

                active_experts = (
                    (expert_totals > 0).sum().item()
                    if isinstance(expert_totals, torch.Tensor)
                    else sum(1 for c in expert_totals if c > 0)
                )
                total_experts = (
                    len(expert_totals)
                    if isinstance(expert_totals, torch.Tensor)
                    else len(expert_totals)
                )

                return {
                    "mode": "stat",
                    "active_experts": active_experts,
                    "total_experts": total_experts,
                    "tokens_per_expert": expert_totals.tolist()
                    if isinstance(expert_totals, torch.Tensor)
                    else expert_totals,
                    "forward_passes": len(forward_pass_ids),
                    "forward_pass_ids": forward_pass_ids,
                }

    # For per_token and per_pass modes
    elif mode in ["per_token", "per_pass"]:
        all_records = []
        for worker_data in all_data:
            if isinstance(worker_data, dict) and "records" in worker_data:
                all_records.extend(worker_data["records"])

        if all_records:
            if mode == "per_token":
                return {
                    "mode": "per_token",
                    "num_records": len(all_records),
                    "sample_routing": all_records[0].get("topk_ids")
                    if all_records
                    else None,
                }
            else:  # per_pass
                # Calculate average experts activated
                avg_activated = []
                for record in all_records:
                    avg = record.get("avg_activated_per_layer")
                    if avg is not None:
                        avg_activated.append(avg)

                return {
                    "mode": "per_pass",
                    "num_records": len(all_records),
                    "avg_experts_activated": sum(avg_activated)
                    / len(avg_activated)
                    if avg_activated
                    else None,
                    "forward_pass_ids": [
                        r.get("forward_pass_id") for r in all_records
                    ],
                }

    return None


# Auto-apply monkey patching on import
try:
    apply_vllm_monkey_patching()
    logger.info("vLLM expert distribution monkey patching applied")
except Exception as e:
    logger.warning(f"Failed to apply vLLM monkey patching: {e}")

# Patch LLM.generate() if vLLM is available
try:
    from vllm import LLM

    _original_generate = LLM.generate

    def generate_with_expert_dist(
        self, prompts, sampling_params=None, **kwargs
    ):
        """
        Wrapped LLM.generate() that automatically handles expert distribution recording.
        """
        # Check if expert distribution is enabled
        if not _config["enabled"]:
            return _original_generate(self, prompts, sampling_params, **kwargs)

        # Configure recording if not already configured
        _configure_llm_recording(self)

        # Auto-start recording if enabled
        if _config["auto_record"]:
            try:
                self.collective_rpc("start_expert_distribution_recording")
            except Exception as e:
                logger.warning(f"Failed to start recording: {e}")

        # Call original generate
        try:
            outputs = _original_generate(
                self, prompts, sampling_params, **kwargs
            )
        finally:
            # Auto-stop and dump recording if enabled
            if _config["auto_record"]:
                try:
                    all_data = self.collective_rpc(
                        "dump_expert_distribution_record"
                    )
                    expert_stats = _extract_expert_stats(
                        all_data, _config["mode"]
                    )

                    # Add expert_stats to each output
                    if expert_stats and hasattr(outputs, "__iter__"):
                        for output in outputs:
                            # Attach to top-level output object
                            output.expert_stats = expert_stats
                            # Also attach to individual request outputs for compatibility
                            if hasattr(output, "outputs") and output.outputs:
                                for req_output in output.outputs:
                                    req_output.expert_stats = expert_stats

                    # Stop recording
                    self.collective_rpc("stop_expert_distribution_recording")
                except Exception as e:
                    logger.warning(f"Failed to dump/stop recording: {e}")

        return outputs

    # Only patch if not already patched
    if not hasattr(LLM.generate, "_expert_dist_patched"):
        LLM.generate = generate_with_expert_dist
        LLM.generate._expert_dist_patched = True
        logger.info("LLM.generate() patched with expert distribution recording")

except ImportError:
    logger.debug("vLLM not available, skipping LLM.generate() patching")
except Exception as e:
    logger.warning(f"Failed to patch LLM.generate(): {e}")


@contextmanager
def with_expert_recording(mode: Optional[str] = None, verbose: bool = False):
    """
    Context manager for temporary expert distribution recording.

    Usage:
        with with_expert_recording(mode="stat"):
            outputs = llm.generate(["prompt"])
    """
    old_mode = _config["mode"]
    old_verbose = _config["verbose"]
    old_enabled = _config["enabled"]

    try:
        if mode:
            set_mode(mode)
        _config["verbose"] = verbose
        _config["enabled"] = True
        yield
    finally:
        _config["mode"] = old_mode
        _config["verbose"] = old_verbose
        _config["enabled"] = old_enabled


def get_expert_stats(llm) -> Optional[Dict[str, Any]]:
    """
    Get current expert distribution statistics without stopping recording.

    Note: This function calls dump_expert_distribution_record() which reads
    the current accumulated data. The data is not reset, but if recording
    is not active, it may return empty data.

    Args:
        llm: vLLM LLM instance

    Returns:
        Dictionary with expert statistics or None if not available
    """
    try:
        # Check if recording is active first
        # We can't easily check this without RPC, so just try to dump
        all_data = llm.collective_rpc("dump_expert_distribution_record")
        print(all_data)
        # Filter out empty dictionaries from workers
        if not all_data:
            return None

        # Extract stats
        stats = _extract_expert_stats(all_data, _config["mode"])

        # If stats are None or show 0 active experts, recording might not be active
        # or no data has been accumulated yet
        if stats is None:
            logger.debug(
                "No expert stats available - recording may not be active or no data accumulated"
            )

        return stats
    except Exception as e:
        logger.warning(f"Failed to get expert stats: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def reset_recording(llm):
    """
    Reset the recording state (stop and clear current records).

    Args:
        llm: vLLM LLM instance
    """
    try:
        llm.collective_rpc("stop_expert_distribution_recording")
        # Reconfigure to reset state
        _configured_llms.discard(id(llm))
        _configure_llm_recording(llm)
    except Exception as e:
        logger.warning(f"Failed to reset recording: {e}")
