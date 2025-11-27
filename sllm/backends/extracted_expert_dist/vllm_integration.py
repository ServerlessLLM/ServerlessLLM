#!/usr/bin/env python3
"""
vLLM Integration for Expert Distribution Recording

This module provides the monkey patching logic to integrate expert distribution
recording functionality into vLLM. It patches vLLM's Worker, GPUModelRunner,
and MoE model classes to enable automatic expert distribution recording.

MONKEY PATCHING COVERAGE:
✅ Module replacement: Replaces vLLM's expert_distribution_recorder with our custom one
✅ Worker RPC methods: Adds configure/start/stop/dump expert distribution methods
✅ Recorder initialization: Automatic MoE model detection and recorder setup
✅ Model layer context: Patches Qwen2MoeDecoderLayer and DeepseekV2DecoderLayer
✅ FusedMoE integration: Uses built-in recording logic in vLLM's FusedMoE.select_experts

REQUIREMENTS:
- vLLM v0.11.0 or compatible version
- MoE model (Qwen/Qwen1.5-MoE-A2.7B, DeepSeek-V2, etc.)
- GPU with sufficient memory

Usage:
    from vllm_integration import apply_vllm_monkey_patching
    apply_vllm_monkey_patching()
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Helper Functions for Common Patching Logic
# ============================================================================


def _create_select_experts_patcher(original_func, debug_prefix=""):
    """Create a patched select_experts function with expert distribution recording.

    This helper function creates the patching logic that's shared between
    FusedMoE.select_experts and SharedFusedMoE.select_experts, and between
    main process and worker process patching.
    """

    def patched_select_experts(*args, **kwargs):
        # Call original function
        result = original_func(*args, **kwargs)

        # Report expert selection for distribution recording
        try:
            from expert_distribution_recorder import (
                get_global_expert_distribution_recorder,
            )

            recorder = get_global_expert_distribution_recorder()

            if recorder is not None:
                # Extract topk_ids from result
                topk_weights, topk_ids, _ = result

                # Pass layer_idx as fallback, but on_select_experts will prefer _current_layer_idx
                recorder.on_select_experts(layer_idx=None, topk_ids=topk_ids)
        except Exception as e:
            # Silently fail - don't crash if recording fails
            pass

        return result

    return patched_select_experts


def _patch_execute_model(runner_class):
    """Patch execute_model to collect expert distribution data after execution.

    This is required for Cuda Graph compatibility, as the Python-side forward method
    is not executed during graph replay. We must collect data in execute_model instead.
    """
    if hasattr(runner_class, "_expert_dist_patched_execute_model"):
        return

    original_execute_model = runner_class.execute_model

    def patched_execute_model(self, *args, **kwargs):
        import torch

        # Call original execute_model (this runs the graph replay if enabled)
        result = original_execute_model(self, *args, **kwargs)

        # Post-execution collection for per_pass/per_token modes
        try:
            if (
                hasattr(self, "expert_distribution_recorder")
                and self.expert_distribution_recorder
            ):
                recorder = self.expert_distribution_recorder
                # Only collect if recording is active and mode requires per-pass collection
                # (stat mode uses atomic accumulation so it doesn't need per-pass collection)
                if getattr(recorder, "_recording", False):
                    mode = getattr(recorder, "_recording_mode", None)

                    if mode in ["per_pass", "per_token"]:
                        # Collect data from gatherer
                        # Note: collect() calls clone() which is a GPU operation.
                        # append() calls item() which triggers synchronization.
                        # This is safe here (after graph replay) but adds a small overhead.
                        collected_data = recorder._gatherer.collect()

                        recorder._gatherer.reset()

                        # Append to accumulator
                        if not hasattr(recorder, "_forward_pass_counter"):
                            recorder._forward_pass_counter = 0
                        recorder._accumulator.append(
                            recorder._forward_pass_counter, collected_data
                        )
                        recorder._forward_pass_counter += 1
        except Exception as e:
            # Don't crash if collection fails
            pass

        return result

    runner_class.execute_model = patched_execute_model
    runner_class._expert_dist_patched_execute_model = True
    # print(f"[ExpertDist-Worker] Patched {runner_class.__name__}.execute_model")


def _patch_decoder_layer_init(
    layer_class, extract_layer_idx_fn, layer_name, is_worker=False
):
    """Patch decoder layer __init__ to extract and store layer_idx.

    Args:
        layer_class: The decoder layer class to patch
        extract_layer_idx_fn: Function to extract layer_idx from prefix
        layer_name: Name of the layer class for logging
        is_worker: Whether this is being patched in a worker process
    """
    if hasattr(
        layer_class,
        "_expert_dist_patched_init" + ("_worker" if is_worker else ""),
    ):
        return  # Already patched

    original_init = layer_class.__init__

    def patched_init(self, *args, **kwargs):
        # Extract layer_idx from prefix
        layer_idx = None
        if len(args) >= 4:  # Qwen: config, layer_id, quant_config, prefix
            layer_idx = extract_layer_idx_fn(args[3])
        elif len(args) >= 2:  # DeepSeek: vllm_config, prefix
            layer_idx = extract_layer_idx_fn(args[1])
        elif "prefix" in kwargs:
            layer_idx = extract_layer_idx_fn(kwargs["prefix"])

        # Call original __init__
        original_init(self, *args, **kwargs)

        # Store layer_idx as instance attribute
        if layer_idx is not None:
            self.layer_idx = layer_idx

    layer_class.__init__ = patched_init
    setattr(
        layer_class,
        "_expert_dist_patched_init" + ("_worker" if is_worker else ""),
        True,
    )

    if is_worker:
        # import os
        # print(f"[ExpertDist-Worker PID {os.getpid()}] Patched {layer_name}.__init__ BEFORE model load", flush=True)
        pass
    else:
        # print(f"[ExpertDist] Patched {layer_name}.__init__ in main process")
        pass


def apply_vllm_monkey_patching():
    """Apply vLLM monkey patching for expert distribution recording."""
    try:
        # Try to import vLLM and apply monkey patching
        import vllm

        # print("Using real vLLM")
        # Import Worker early and store original __init__
        from vllm.v1.worker.gpu_worker import Worker

        # Store the original __init__ before we patch it (only once)
        if not hasattr(Worker, "_original_init"):
            Worker._original_init = Worker.__init__
        _original_worker_init = Worker._original_init

        # Monkey patch vLLM to use our custom expert_distribution_recorder
        # Replace the module in sys.modules BEFORE any imports happen
        import expert_distribution_recorder as custom_recorder
        import vllm.distributed.eplb as eplb_module

        sys.modules["vllm.distributed.eplb.expert_distribution_recorder"] = (
            custom_recorder
        )
        eplb_module.expert_distribution_recorder = custom_recorder

        # Also ensure our moe_hooks is used
        import moe_hooks as custom_hooks

        sys.modules["vllm.distributed.eplb.moe_hooks"] = custom_hooks
        eplb_module.moe_hooks = custom_hooks

        # print("Successfully monkey-patched vLLM with custom expert_distribution_recorder")

        # CRITICAL: Patch decoder layer __init__ methods BEFORE Worker.__init__ is called
        # This ensures layer_idx is set when layers are created during model loading
        try:
            from vllm.model_executor.models.utils import extract_layer_index

            # Patch Qwen2 models if available
            try:
                from vllm.model_executor.models.qwen2_moe import (
                    Qwen2MoeDecoderLayer,
                )

                _patch_decoder_layer_init(
                    Qwen2MoeDecoderLayer,
                    extract_layer_index,
                    "Qwen2MoeDecoderLayer",
                    is_worker=False,
                )
            except ImportError:
                pass

            # Patch Qwen3 MoE models if available (for Qwen3-30B-A3B and similar)
            try:
                from vllm.model_executor.models.qwen3_moe import (
                    Qwen3MoeDecoderLayer,
                )

                _patch_decoder_layer_init(
                    Qwen3MoeDecoderLayer,
                    extract_layer_index,
                    "Qwen3MoeDecoderLayer",
                    is_worker=False,
                )
            except ImportError:
                pass

            # Patch DeepSeek models if available
            try:
                from vllm.model_executor.models.deepseek_v2 import (
                    DeepseekV2DecoderLayer,
                )

                _patch_decoder_layer_init(
                    DeepseekV2DecoderLayer,
                    extract_layer_index,
                    "DeepseekV2DecoderLayer",
                    is_worker=False,
                )
            except ImportError:
                pass
        except Exception as e:
            print(
                f"[ExpertDist] Warning: Could not patch decoder layer __init__ in main process: {e}"
            )

        # Monkey patch Worker to add expert distribution methods
        try:
            from vllm.v1.worker.gpu_worker import Worker

            # Add expert distribution recorder attribute if it doesn't exist
            if not hasattr(Worker, "expert_distribution_recorder"):
                Worker.expert_distribution_recorder = None

            # Define the worker monkey patching method
            def _apply_worker_monkey_patching(self):
                """Apply monkey patching in the worker process for MoE model classes."""
                try:
                    import sys

                    # Apply module replacements in worker process
                    import expert_distribution_recorder as custom_recorder

                    sys.modules[
                        "vllm.distributed.eplb.expert_distribution_recorder"
                    ] = custom_recorder

                    import moe_hooks as custom_hooks

                    sys.modules["vllm.distributed.eplb.moe_hooks"] = (
                        custom_hooks
                    )

                    # Also monkey patch GPUModelRunner in worker process
                    try:
                        from vllm.v1.worker.gpu_model_runner import (
                            GPUModelRunner,
                        )

                        # Add expert_distribution_recorder attribute
                        if not hasattr(
                            GPUModelRunner, "expert_distribution_recorder"
                        ):
                            GPUModelRunner.expert_distribution_recorder = None

                        # Always ensure all methods are available in worker process
                        def _get_expert_location_metadata(self):
                            try:
                                model = self.get_model()
                                hf_config = self.model_config.hf_config

                                def is_mixture_of_experts(model):
                                    return hasattr(
                                        model, "num_logical_experts"
                                    ) and hasattr(model, "num_expert_layers")

                                if is_mixture_of_experts(model):
                                    num_logical_experts = (
                                        model.num_logical_experts
                                    )
                                    num_layers = model.num_expert_layers
                                    ep_size = getattr(model, "ep_size", 1)
                                    num_physical_experts = num_logical_experts
                                    num_local_physical_experts = (
                                        num_logical_experts // ep_size
                                        if ep_size > 1
                                        else num_logical_experts
                                    )
                                else:
                                    # Try to detect from hf_config with support for DeepSeek (n_routed_experts)
                                    num_experts = getattr(
                                        hf_config, "num_experts", None
                                    )
                                    if num_experts is None:
                                        num_experts = getattr(
                                            hf_config, "n_routed_experts", 60
                                        )

                                    num_layers = getattr(
                                        hf_config,
                                        "num_hidden_layers",
                                        getattr(hf_config, "n_layer", 24),
                                    )

                                    num_physical_experts = num_experts
                                    num_local_physical_experts = num_experts
                                    ep_size = 1

                                from expert_distribution_recorder import (
                                    ExpertLocationMetadata,
                                )

                                return ExpertLocationMetadata(
                                    num_layers=num_layers,
                                    num_logical_experts=num_experts,
                                    num_physical_experts=num_physical_experts,
                                    num_local_physical_experts=num_local_physical_experts,
                                    ep_size=ep_size,
                                )
                            except Exception:
                                return None

                        def configure_expert_distribution_recording(
                            self,
                            recording_mode=None,
                            enable_metrics=False,
                            buffer_size=-1,
                        ):
                            from expert_distribution_recorder import (
                                ExpertDistributionRecorder,
                                set_global_expert_distribution_recorder,
                            )

                            expert_location_metadata = (
                                self._get_expert_location_metadata()
                            )
                            if expert_location_metadata is None:
                                return

                            import os
                            import tempfile

                            import torch

                            rank = (
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            )

                            # Check for auto-start flag (similar to sglang.py's enable_expert_distribution_metrics)
                            EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE = (
                                os.path.join(
                                    tempfile.gettempdir(),
                                    "vllm_expert_distribution_auto_start.flag",
                                )
                            )
                            auto_start_enabled = os.path.exists(
                                EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE
                            )

                            # Auto-configure if flag is set but no mode specified
                            if auto_start_enabled and recording_mode is None:
                                recording_mode = "per_pass"  # Default to per_pass mode (CUDA graph compatible)
                                enable_metrics = True
                                if rank == 0:
                                    # print(f"[ExpertDist-Worker PID {os.getpid()}] Auto-starting expert distribution recording (mode={recording_mode})", flush=True)
                                    pass

                            self.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                                recording_mode=recording_mode,
                                expert_location_metadata=expert_location_metadata,
                                rank=rank,
                                device=str(self.device),
                                buffer_size=buffer_size,
                                enable_metrics=enable_metrics,
                            )
                            set_global_expert_distribution_recorder(
                                self.expert_distribution_recorder
                            )

                            # Auto-start recording if flag is set (similar to sglang.py)
                            if auto_start_enabled:
                                self.expert_distribution_recorder.start_record()
                                if rank == 0:
                                    # print(f"[ExpertDist-Worker PID {os.getpid()}] Expert distribution recording auto-started", flush=True)
                                    pass

                        def start_expert_distribution_recording(self):
                            if self.expert_distribution_recorder:
                                self.expert_distribution_recorder.start_record()

                        def stop_expert_distribution_recording(self):
                            if self.expert_distribution_recorder:
                                self.expert_distribution_recorder.stop_record()

                        def dump_expert_distribution_record(
                            self, output_path=None
                        ):
                            if self.expert_distribution_recorder:
                                return self.expert_distribution_recorder.dump_record(
                                    output_path
                                )
                            return {}

                        def check_expert_buffer(self):
                            if self.expert_distribution_recorder:
                                buffer = self.expert_distribution_recorder.get_expert_counts_buffer()
                                return {
                                    "buffer_available": buffer is not None,
                                    "buffer_shape": buffer.shape
                                    if buffer is not None
                                    else None,
                                }
                            return {"buffer_available": False}

                        # Always apply the methods in worker process (overwrite any existing)
                        GPUModelRunner._get_expert_location_metadata = (
                            _get_expert_location_metadata
                        )
                        GPUModelRunner.configure_expert_distribution_recording = configure_expert_distribution_recording
                        GPUModelRunner.start_expert_distribution_recording = (
                            start_expert_distribution_recording
                        )
                        GPUModelRunner.stop_expert_distribution_recording = (
                            stop_expert_distribution_recording
                        )
                        GPUModelRunner.dump_expert_distribution_record = (
                            dump_expert_distribution_record
                        )
                        GPUModelRunner.check_expert_buffer = check_expert_buffer

                    except ImportError:
                        pass

                    # Patch decoder layer forward methods for per_token/per_pass mode data collection
                    from vllm.model_executor.models.utils import (
                        extract_layer_index,
                    )

                    # Patch Qwen models forward method
                    try:
                        from vllm.model_executor.models.qwen2_moe import (
                            Qwen2MoeDecoderLayer,
                        )

                        if not hasattr(
                            Qwen2MoeDecoderLayer, "_expert_dist_patched_forward"
                        ):
                            original_forward = Qwen2MoeDecoderLayer.forward

                            def patched_qwen_forward(self, *args, **kwargs):
                                layer_idx = getattr(self, "layer_idx", None)

                                if layer_idx is not None:
                                    from expert_distribution_recorder import (
                                        get_global_expert_distribution_recorder,
                                    )

                                    recorder = get_global_expert_distribution_recorder()

                                    if recorder is not None and getattr(
                                        recorder, "_recording", False
                                    ):
                                        with recorder.with_current_layer(
                                            layer_idx
                                        ):
                                            result = original_forward(
                                                self, *args, **kwargs
                                            )

                                        # For per_pass and per_token modes, collection is now done in execute_model
                                        # to support Cuda Graph (where forward is skipped during replay)
                                        return result
                                return original_forward(self, *args, **kwargs)

                            Qwen2MoeDecoderLayer.forward = patched_qwen_forward
                            Qwen2MoeDecoderLayer._expert_dist_patched_forward = True
                            # print(f"[ExpertDist-Worker] Patched Qwen2MoeDecoderLayer.forward successfully", flush=True)
                    except ImportError:
                        # print(f"[ExpertDist-Worker] Failed to import Qwen2MoeDecoderLayer for forward patching", flush=True)
                        pass

                    # Patch Qwen3MoE models forward method (for Qwen3-30B-A3B and similar)
                    try:
                        from vllm.model_executor.models.qwen3_moe import (
                            Qwen3MoeDecoderLayer,
                        )

                        if not hasattr(
                            Qwen3MoeDecoderLayer, "_expert_dist_patched_forward"
                        ):
                            original_qwen3_forward = (
                                Qwen3MoeDecoderLayer.forward
                            )

                            def patched_qwen3_forward(self, *args, **kwargs):
                                layer_idx = getattr(self, "layer_idx", None)

                                if layer_idx is not None:
                                    from expert_distribution_recorder import (
                                        get_global_expert_distribution_recorder,
                                    )

                                    recorder = get_global_expert_distribution_recorder()

                                    if recorder is not None and getattr(
                                        recorder, "_recording", False
                                    ):
                                        with recorder.with_current_layer(
                                            layer_idx
                                        ):
                                            result = original_qwen3_forward(
                                                self, *args, **kwargs
                                            )

                                        # For per_pass and per_token modes, collection is now done in execute_model
                                        # to support Cuda Graph (where forward is skipped during replay)
                                        return result
                                return original_qwen3_forward(
                                    self, *args, **kwargs
                                )

                            Qwen3MoeDecoderLayer.forward = patched_qwen3_forward
                            Qwen3MoeDecoderLayer._expert_dist_patched_forward = True
                            print(
                                f"[ExpertDist-Worker] Patched Qwen3MoeDecoderLayer.forward successfully",
                                flush=True,
                            )
                    except ImportError:
                        pass

                    # Patch DeepSeek models forward method
                    try:
                        from vllm.model_executor.models.deepseek_v2 import (
                            DeepseekV2DecoderLayer,
                        )

                        if not hasattr(
                            DeepseekV2DecoderLayer,
                            "_expert_dist_patched_forward",
                        ):
                            original_forward = DeepseekV2DecoderLayer.forward

                            def patched_deepseek_forward(self, *args, **kwargs):
                                layer_idx = getattr(self, "layer_idx", None)
                                if layer_idx is not None:
                                    from expert_distribution_recorder import (
                                        get_global_expert_distribution_recorder,
                                    )

                                    recorder = get_global_expert_distribution_recorder()
                                    if recorder is not None and getattr(
                                        recorder, "_recording", False
                                    ):
                                        with recorder.with_current_layer(
                                            layer_idx
                                        ):
                                            result = original_forward(
                                                self, *args, **kwargs
                                            )

                                        # For per_pass and per_token modes, collection is now done in execute_model
                                        # to support Cuda Graph (where forward is skipped during replay)
                                        return result
                                return original_forward(self, *args, **kwargs)

                            DeepseekV2DecoderLayer.forward = (
                                patched_deepseek_forward
                            )
                            DeepseekV2DecoderLayer._expert_dist_patched_forward = True
                            # print(f"[ExpertDist-Worker] Patched DeepseekV2DecoderLayer.forward successfully", flush=True)
                    except ImportError:
                        pass

                    # IMPORTANT: Also patch FusedMoE.select_experts in worker process
                    # This is needed because worker processes might reload modules
                    try:
                        from vllm.model_executor.layers.fused_moe import (
                            FusedMoE,
                        )

                        if not hasattr(
                            FusedMoE.select_experts,
                            "_expert_dist_patched_worker_after_init",
                        ):
                            original_select_experts_worker = (
                                FusedMoE.select_experts
                            )
                            patched_func = _create_select_experts_patcher(
                                original_select_experts_worker
                            )
                            FusedMoE.select_experts = staticmethod(patched_func)
                            FusedMoE.select_experts._expert_dist_patched_worker_after_init = True
                            # print(f"[ExpertDist-Worker] Patched FusedMoE.select_experts in worker process")
                    except Exception as e:
                        print(
                            f"[ExpertDist-Worker] Could not patch FusedMoE.select_experts in worker: {e}"
                        )

                except Exception as e:
                    # Don't crash if monkey patching fails in worker
                    pass

            # Add the method to the Worker class
            Worker._apply_worker_monkey_patching = _apply_worker_monkey_patching

            # IMPORTANT: Also add the RPC methods to Worker class so they're available for collective_rpc
            # These need to be added BEFORE patching __init__ so they're available when workers are created
            def configure_expert_distribution_recorder(
                self,
                recording_mode: str | None = None,
                enable_metrics: bool = False,
                buffer_size: int = -1,
            ):
                """Configure the expert distribution recorder on the worker."""
                # Ensure model_runner exists (it should be initialized by now)
                if (
                    not hasattr(self, "model_runner")
                    or self.model_runner is None
                ):
                    # If model_runner not ready yet, return error
                    return {
                        "success": False,
                        "error": "model_runner not initialized",
                    }

                # Validate and normalize recording mode
                valid_modes = {"stat", "per_token", "per_pass"}
                if recording_mode is not None:
                    mode_lower = recording_mode.lower().strip()
                    # Normalize common typos
                    mode_normalizations = {
                        "per_path": "per_pass",
                        "stats": "stat",
                        "per-token": "per_token",
                        "per-pass": "per_pass",
                    }
                    if mode_lower in mode_normalizations:
                        mode_lower = mode_normalizations[mode_lower]
                    if mode_lower not in valid_modes:
                        return {
                            "success": False,
                            "error": f"Invalid recording mode: '{recording_mode}'. Valid modes are: {', '.join(sorted(valid_modes))}",
                        }
                    recording_mode = mode_lower
                else:
                    # Default to "per_pass" if None (CUDA graph compatible)
                    recording_mode = "per_pass"

                # Call the GPUModelRunner method directly to avoid delegation issues
                from expert_distribution_recorder import (
                    ExpertDistributionRecorder,
                    set_global_expert_distribution_recorder,
                )

                # Get expert location metadata inline (avoid method call issues)
                try:
                    model = self.model_runner.get_model()
                    hf_config = self.model_runner.model_config.hf_config

                    def is_mixture_of_experts(model):
                        return hasattr(
                            model, "num_logical_experts"
                        ) and hasattr(model, "num_expert_layers")

                    if is_mixture_of_experts(model):
                        num_logical_experts = model.num_logical_experts
                        num_layers = model.num_expert_layers
                        ep_size = getattr(model, "ep_size", 1)
                        num_physical_experts = num_logical_experts
                        num_local_physical_experts = (
                            num_logical_experts // ep_size
                            if ep_size > 1
                            else num_logical_experts
                        )
                    else:
                        # Try to detect from hf_config with support for DeepSeek (n_routed_experts)
                        num_experts = getattr(hf_config, "num_experts", None)
                        if num_experts is None:
                            num_experts = getattr(
                                hf_config, "n_routed_experts", 60
                            )

                        num_layers = getattr(
                            hf_config,
                            "num_hidden_layers",
                            getattr(hf_config, "n_layer", 24),
                        )

                        num_physical_experts = num_experts
                        num_local_physical_experts = num_experts
                        ep_size = 1

                    from expert_distribution_recorder import (
                        ExpertLocationMetadata,
                    )

                    expert_location_metadata = ExpertLocationMetadata(
                        num_layers=num_layers,
                        num_logical_experts=num_experts,
                        num_physical_experts=num_physical_experts,
                        num_local_physical_experts=num_local_physical_experts,
                        ep_size=ep_size,
                    )
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to get expert metadata: {e}",
                    }

                import torch

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )

                try:
                    self.model_runner.expert_distribution_recorder = (
                        ExpertDistributionRecorder.init_new(
                            recording_mode=recording_mode,
                            expert_location_metadata=expert_location_metadata,
                            rank=rank,
                            device=str(self.model_runner.device),
                            buffer_size=buffer_size,
                            enable_metrics=enable_metrics,
                        )
                    )
                    set_global_expert_distribution_recorder(
                        self.model_runner.expert_distribution_recorder
                    )
                except Exception as e:
                    # Return error instead of raising to prevent worker crash
                    return {"success": False, "error": str(e)}

                recorder = self.model_runner.expert_distribution_recorder
                return {
                    "success": True,
                    "recording_mode": getattr(
                        recorder, "_recording_mode", None
                    ),
                    "recording": recorder.recording,
                }

            # Backward compatibility alias for older RPC name
            def configure_expert_distribution_recording(
                self, mode: str, verbose: bool = False
            ):
                """Configure expert distribution recording mode."""
                return self.configure_expert_distribution_recorder(
                    recording_mode=mode, enable_metrics=verbose
                )

            def start_expert_distribution_recording(self):
                """Start recording expert distributions."""
                if (
                    hasattr(self, "model_runner")
                    and self.model_runner is not None
                ):
                    self.model_runner.start_expert_distribution_recording()

            def dump_expert_distribution_record(self, output_path=None):
                """Dump recorded expert distribution data."""
                if (
                    hasattr(self, "model_runner")
                    and self.model_runner is not None
                ):
                    return self.model_runner.dump_expert_distribution_record(
                        output_path
                    )
                return {}

            def stop_expert_distribution_recording(self):
                """Stop recording expert distributions."""
                if (
                    hasattr(self, "model_runner")
                    and self.model_runner is not None
                ):
                    self.model_runner.stop_expert_distribution_recording()

            # Add methods to Worker class BEFORE patching __init__
            Worker.configure_expert_distribution_recorder = (
                configure_expert_distribution_recorder
            )
            Worker.configure_expert_distribution_recording = (
                configure_expert_distribution_recording
            )
            Worker.start_expert_distribution_recording = (
                start_expert_distribution_recording
            )
            Worker.dump_expert_distribution_record = (
                dump_expert_distribution_record
            )
            Worker.stop_expert_distribution_recording = (
                stop_expert_distribution_recording
            )

            def patched_init(self, *args, **kwargs):
                import os
                import sys

                # CRITICAL: Apply ALL patching BEFORE Worker.__init__ loads the model
                # This ensures patching is applied BEFORE torch.compile and CUDA graph capture
                # This must happen BEFORE model loading, especially for spawned workers
                try:
                    # Patch decoder layers BEFORE model is loaded
                    import inspect

                    from vllm.model_executor.models.utils import (
                        extract_layer_index,
                    )

                    # Patch Qwen2 models if available
                    try:
                        from vllm.model_executor.models.qwen2_moe import (
                            Qwen2MoeDecoderLayer,
                        )

                        _patch_decoder_layer_init(
                            Qwen2MoeDecoderLayer,
                            extract_layer_index,
                            "Qwen2MoeDecoderLayer",
                            is_worker=True,
                        )
                    except ImportError:
                        pass

                    # Patch Qwen3 MoE models if available (for Qwen3-30B-A3B and similar)
                    try:
                        from vllm.model_executor.models.qwen3_moe import (
                            Qwen3MoeDecoderLayer,
                        )

                        _patch_decoder_layer_init(
                            Qwen3MoeDecoderLayer,
                            extract_layer_index,
                            "Qwen3MoeDecoderLayer",
                            is_worker=True,
                        )
                    except ImportError:
                        pass

                    # Patch DeepSeek models if available
                    try:
                        from vllm.model_executor.models.deepseek_v2 import (
                            DeepseekV2DecoderLayer,
                        )

                        _patch_decoder_layer_init(
                            DeepseekV2DecoderLayer,
                            extract_layer_index,
                            "DeepseekV2DecoderLayer",
                            is_worker=True,
                        )
                    except ImportError:
                        pass

                    # CRITICAL: Patch FusedMoE.select_experts BEFORE torch.compile
                    # torch.compile happens during model loading, so we must patch before that
                    try:
                        from vllm.model_executor.layers.fused_moe import (
                            FusedMoE,
                        )

                        if not hasattr(
                            FusedMoE.select_experts,
                            "_expert_dist_patched_worker",
                        ):
                            original_select_experts_worker = (
                                FusedMoE.select_experts
                            )
                            patched_func = _create_select_experts_patcher(
                                original_select_experts_worker
                            )
                            FusedMoE.select_experts = staticmethod(patched_func)
                            FusedMoE.select_experts._expert_dist_patched_worker = True
                            # print(f"[ExpertDist-Worker] Patched FusedMoE.select_experts BEFORE torch.compile", flush=True)
                    except Exception as e:
                        print(
                            f"[ExpertDist-Worker] Warning: Could not patch FusedMoE.select_experts before compile: {e}",
                            flush=True,
                        )

                except Exception as e:
                    print(
                        f"[ExpertDist-Worker] Warning: Could not patch decoder layers before model load: {e}"
                    )

                # Only pass Worker.__init__ parameters to avoid passing them to model constructors
                # Worker.__init__ accepts: vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
                worker_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in [
                        "vllm_config",
                        "local_rank",
                        "rank",
                        "distributed_init_method",
                        "is_driver_worker",
                    ]
                }

                # Apply additional monkey patching BEFORE initialization (for forward methods, etc.)
                # This ensures model methods are patched BEFORE they are used/compiled in Worker.__init__
                try:
                    self._apply_worker_monkey_patching()
                except Exception as e:
                    # Don't crash if patching fails, but log it
                    print(
                        f"[ExpertDist] Warning: Worker monkey patching failed: {e}"
                    )

                # Call the original Worker.__init__ (this loads the model, which now uses patched decoder layers)
                # Use Worker._original_init directly (not super()) to avoid calling our patched version recursively
                Worker._original_init(self, **worker_kwargs)

                # Initialize expert distribution recorder (will be done lazily when needed)
                self.expert_distribution_recorder = None

                # Auto-configure and start expert distribution recording if flag is set (similar to sglang.py)
                try:
                    import tempfile

                    EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE = os.path.join(
                        tempfile.gettempdir(),
                        "vllm_expert_distribution_auto_start.flag",
                    )
                    auto_start_enabled = os.path.exists(
                        EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE
                    )

                    if (
                        auto_start_enabled
                        and hasattr(self, "model_runner")
                        and self.model_runner is not None
                    ):
                        # Auto-configure with per_pass mode (CUDA graph compatible)
                        self.configure_expert_distribution_recorder(
                            recording_mode="per_pass",
                            enable_metrics=True,
                            buffer_size=-1,
                        )
                        # Auto-start recording
                        if (
                            hasattr(
                                self.model_runner,
                                "expert_distribution_recorder",
                            )
                            and self.model_runner.expert_distribution_recorder
                            is not None
                        ):
                            self.model_runner.expert_distribution_recorder.start_record()
                            import torch

                            rank = (
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            )
                            if rank == 0:
                                # print(f"[ExpertDist-Worker PID {os.getpid()}] Auto-configured and started expert distribution recording", flush=True)
                                pass
                except Exception as e:
                    # Don't fail if auto-start fails
                    print(
                        f"[ExpertDist-Worker PID {os.getpid()}] Warning: Could not auto-start expert distribution recording: {e}",
                        flush=True,
                    )

            # Monkey patch __init__ (methods are already added above)
            Worker.__init__ = patched_init

            # print("Successfully monkey-patched Worker with expert distribution methods")

            # Now monkey patch GPUModelRunner with expert distribution methods
            try:
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                # Add expert_distribution_recorder attribute
                if not hasattr(GPUModelRunner, "expert_distribution_recorder"):
                    GPUModelRunner.expert_distribution_recorder = None

                def _get_expert_location_metadata(self):
                    """Extract expert location metadata from model config."""
                    try:
                        model = self.get_model()
                        hf_config = self.model_config.hf_config

                        # Check if it's a MoE model
                        def is_mixture_of_experts(model):
                            return hasattr(
                                model, "num_logical_experts"
                            ) and hasattr(model, "num_expert_layers")

                        if is_mixture_of_experts(model):
                            num_logical_experts = model.num_logical_experts
                            num_layers = model.num_expert_layers
                            ep_size = getattr(model, "ep_size", 1)

                            num_physical_experts = num_logical_experts
                            num_local_physical_experts = (
                                num_logical_experts // ep_size
                                if ep_size > 1
                                else num_logical_experts
                            )
                        else:
                            # Fallback to config-based extraction
                            num_experts = getattr(
                                hf_config, "num_experts", 60
                            )  # Qwen default
                            num_layers = 24  # Qwen 1.5 MoE layers

                            num_physical_experts = num_experts
                            num_local_physical_experts = num_experts
                            ep_size = 1

                        from expert_distribution_recorder import (
                            ExpertLocationMetadata,
                        )

                        return ExpertLocationMetadata(
                            num_layers=num_layers,
                            num_logical_experts=num_experts,
                            num_physical_experts=num_physical_experts,
                            num_local_physical_experts=num_local_physical_experts,
                            ep_size=ep_size,
                        )
                    except Exception as e:
                        print(
                            f"Failed to extract expert location metadata: {e}"
                        )
                        return None

                def configure_expert_distribution_recording(
                    self,
                    recording_mode=None,
                    enable_metrics=False,
                    buffer_size=-1,
                ):
                    """Configure expert distribution recording."""
                    from expert_distribution_recorder import (
                        ExpertDistributionRecorder,
                        set_global_expert_distribution_recorder,
                    )

                    expert_location_metadata = (
                        self._get_expert_location_metadata()
                    )
                    if expert_location_metadata is None:
                        print(
                            "[ExpertRecorder] Could not extract expert location metadata from model"
                        )

                        return

                    import torch

                    rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )

                    # Reuse existing recorder if it exists to preserve CUDA graph buffer references.
                    # Creating a new recorder would break CUDA graphs captured with the old buffer.
                    if (
                        hasattr(self, "expert_distribution_recorder")
                        and self.expert_distribution_recorder is not None
                    ):
                        existing_mode = getattr(
                            self.expert_distribution_recorder,
                            "_recording_mode",
                            None,
                        )
                        if existing_mode == recording_mode:
                            print(
                                f"[ExpertRecorder] Recorder already exists with mode={existing_mode}, reusing"
                            )
                        else:
                            print(
                                f"[ExpertRecorder] Changing recording mode from {existing_mode} to {recording_mode}. This may break CUDA graphs if they were already captured."
                            )
                            self.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                                recording_mode=recording_mode,
                                expert_location_metadata=expert_location_metadata,
                                rank=rank,
                                device=str(self.device),
                                buffer_size=buffer_size,
                                enable_metrics=enable_metrics,
                            )
                            set_global_expert_distribution_recorder(
                                self.expert_distribution_recorder
                            )
                    else:
                        self.expert_distribution_recorder = ExpertDistributionRecorder.init_new(
                            recording_mode=recording_mode,
                            expert_location_metadata=expert_location_metadata,
                            rank=rank,
                            device=str(self.device),
                            buffer_size=buffer_size,
                            enable_metrics=enable_metrics,
                        )
                        set_global_expert_distribution_recorder(
                            self.expert_distribution_recorder
                        )
                    print(
                        f"[ExpertRecorder] Expert distribution recording configured with mode={recording_mode}"
                    )

                def start_expert_distribution_recording(self):
                    """Start recording expert distributions."""
                    if self.expert_distribution_recorder:
                        self.expert_distribution_recorder.start_record()

                def stop_expert_distribution_recording(self):
                    """Stop recording expert distributions."""
                    if self.expert_distribution_recorder:
                        self.expert_distribution_recorder.stop_record()

                def dump_expert_distribution_record(self, output_path=None):
                    """Dump recorded expert distribution data."""
                    if self.expert_distribution_recorder:
                        return self.expert_distribution_recorder.dump_record(
                            output_path
                        )
                    return {}

                def check_expert_buffer(self):
                    """Check if expert buffer is available."""
                    if self.expert_distribution_recorder:
                        buffer = self.expert_distribution_recorder.get_expert_counts_buffer()
                        return {
                            "buffer_available": buffer is not None,
                            "buffer_shape": buffer.shape
                            if buffer is not None
                            else None,
                        }
                    return {"buffer_available": False}

                # Monkey patch the methods
                def patched_load_model(self, *args, **kwargs):
                    # Call original load_model
                    original_load_model(self, *args, **kwargs)

                    # Auto-initialize recorder immediately after load
                    try:
                        import os
                        # print(f"[ExpertDist-Worker PID {os.getpid()}] patched_load_model: Model loaded. Attempting auto-init...", flush=True)

                        # Verify if forward patch is effective on the loaded model
                        try:
                            model = self.get_model()
                            if (
                                hasattr(model, "model")
                                and hasattr(model.model, "layers")
                                and len(model.model.layers) > 0
                            ):
                                layer0 = model.model.layers[0]
                                # print(f"[ExpertDist-Worker PID {os.getpid()}] Layer 0 type: {type(layer0)}", flush=True)
                                # print(f"[ExpertDist-Worker PID {os.getpid()}] Layer 0 forward: {layer0.forward}", flush=True)
                                if hasattr(layer0.forward, "__name__"):
                                    # print(f"[ExpertDist-Worker PID {os.getpid()}] Layer 0 forward name: {layer0.forward.__name__}", flush=True)
                                    pass
                        except Exception as e:
                            print(
                                f"[ExpertDist-Worker PID {os.getpid()}] Failed to inspect model layers: {e}",
                                flush=True,
                            )

                        # We can use self._get_expert_location_metadata() now since we patched it into the class
                        metadata = self._get_expert_location_metadata()

                        if metadata:
                            import torch
                            from expert_distribution_recorder import (
                                ExpertDistributionRecorder,
                                set_global_expert_distribution_recorder,
                            )

                            rank = (
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            )

                            # Default to per_pass mode (most common and graph-compatible)
                            # print(f"[ExpertDist-Worker PID {os.getpid()}] Auto-initializing default PER_PASS recorder for CUDAGraph capture", flush=True)
                            recorder = ExpertDistributionRecorder.init_new(
                                recording_mode="per_pass",
                                expert_location_metadata=metadata,
                                rank=rank,
                                device=str(self.device),
                            )

                            self.expert_distribution_recorder = recorder
                            set_global_expert_distribution_recorder(recorder)

                            # CRITICAL: Start recording immediately so it's active during CUDAGraph capture
                            recorder.start_record()

                            # print(f"[ExpertDist-Worker PID {os.getpid()}] Recorder initialized, started, and set globally.", flush=True)
                        else:
                            print(
                                f"[ExpertDist-Worker PID {os.getpid()}] Metadata is None",
                                flush=True,
                            )
                    except Exception as e:
                        import os

                        print(
                            f"[ExpertDist-Worker PID {os.getpid()}] CRITICAL: Failed to auto-initialize recorder in load_model: {e}",
                            flush=True,
                        )
                        import traceback

                        traceback.print_exc()
                        # Re-raise exception to ensure we don't proceed with broken state
                        raise e

                GPUModelRunner._get_expert_location_metadata = (
                    _get_expert_location_metadata
                )
                GPUModelRunner.configure_expert_distribution_recording = (
                    configure_expert_distribution_recording
                )
                GPUModelRunner.start_expert_distribution_recording = (
                    start_expert_distribution_recording
                )
                GPUModelRunner.stop_expert_distribution_recording = (
                    stop_expert_distribution_recording
                )
                GPUModelRunner.dump_expert_distribution_record = (
                    dump_expert_distribution_record
                )
                GPUModelRunner.check_expert_buffer = check_expert_buffer

                # Patch load_model
                original_load_model = GPUModelRunner.load_model
                GPUModelRunner.load_model = patched_load_model

                # Patch execute_model for Cuda Graph support
                _patch_execute_model(GPUModelRunner)

                # print("Successfully monkey-patched GPUModelRunner with expert distribution methods")

            except ImportError as e:
                print(f"Could not monkey-patch GPUModelRunner: {e}")

        except ImportError as e:
            print(f"vLLM not available: {e}")

    except ImportError as e:
        print(f"vLLM not available: {e}")

    # Patch FusedMoE.select_experts and SharedFusedMoE.select_experts in main process
    # Note: Worker processes will patch these again during initialization
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE

        if not hasattr(FusedMoE.select_experts, "_expert_dist_patched_main"):
            original_select_experts = FusedMoE.select_experts
            patched_func = _create_select_experts_patcher(
                original_select_experts
            )
            FusedMoE.select_experts = staticmethod(patched_func)
            FusedMoE.select_experts._expert_dist_patched_main = True
    except Exception:
        pass

    # Also patch SharedFusedMoE.select_experts
    try:
        from vllm.model_executor.layers.shared_fused_moe.shared_fused_moe import (
            SharedFusedMoE,
        )

        if not hasattr(
            SharedFusedMoE.select_experts, "_expert_dist_patched_main"
        ):
            original_shared_select_experts = SharedFusedMoE.select_experts
            patched_func = _create_select_experts_patcher(
                original_shared_select_experts
            )
            SharedFusedMoE.select_experts = staticmethod(patched_func)
            SharedFusedMoE.select_experts._expert_dist_patched_main = True
    except Exception:
        pass

    # Patch API server to add expert distribution endpoints
    _patch_api_server()


def _patch_api_server():
    """Patch vLLM API server to add expert distribution endpoints."""
    try:
        # print(f"[ExpertDist] Patching API server endpoints...", flush=True)
        from fastapi import Response
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.openai import api_server

        # 1. Add methods to AsyncLLMEngine to support expert distribution RPCs
        if not hasattr(AsyncLLMEngine, "start_expert_distribution_record"):

            async def start_expert_distribution_record(self, mode="per_pass"):
                # Calls configure_expert_distribution_recording on workers
                # We use configure because it handles init + start
                return await self.engine.model_executor.collective_rpc(
                    "configure_expert_distribution_recording",
                    args=(mode, True, -1),
                )

            AsyncLLMEngine.start_expert_distribution_record = (
                start_expert_distribution_record
            )

        if not hasattr(AsyncLLMEngine, "stop_expert_distribution_record"):

            async def stop_expert_distribution_record(self):
                return await self.engine.model_executor.collective_rpc(
                    "stop_expert_distribution_recording"
                )

            AsyncLLMEngine.stop_expert_distribution_record = (
                stop_expert_distribution_record
            )

        if not hasattr(AsyncLLMEngine, "dump_expert_distribution_record"):

            async def dump_expert_distribution_record(self):
                # We probably want to dump to a specific path derived from model path or config
                # For now, let the worker decide the path or use default
                return await self.engine.model_executor.collective_rpc(
                    "dump_expert_distribution_record"
                )

            AsyncLLMEngine.dump_expert_distribution_record = (
                dump_expert_distribution_record
            )

        # 2. Register routes on the FastAPI app
        # Note: api_server.app is the FastAPI instance
        app = api_server.app

        # Define route handlers
        @app.post("/start_expert_distribution")
        async def api_start_record(mode: str = "per_pass"):
            # Get engine from app state or global
            engine = (
                app.state.engine
                if hasattr(app.state, "engine")
                else api_server.engine
            )
            if engine and hasattr(engine, "start_expert_distribution_record"):
                await engine.start_expert_distribution_record(mode)
                return {"status": "started", "mode": mode}
            return Response(
                content="Engine not ready or patching failed", status_code=500
            )

        @app.post("/stop_expert_distribution")
        async def api_stop_record():
            engine = (
                app.state.engine
                if hasattr(app.state, "engine")
                else api_server.engine
            )
            if engine and hasattr(engine, "stop_expert_distribution_record"):
                await engine.stop_expert_distribution_record()
                return {"status": "stopped"}
            return Response(
                content="Engine not ready or patching failed", status_code=500
            )

        @app.post("/dump_expert_distribution")
        async def api_dump_record():
            engine = (
                app.state.engine
                if hasattr(app.state, "engine")
                else api_server.engine
            )
            if engine and hasattr(engine, "dump_expert_distribution_record"):
                result = await engine.dump_expert_distribution_record()
                return {"status": "dumped", "result": result}
            return Response(
                content="Engine not ready or patching failed", status_code=500
            )

        print(
            f"[ExpertDist] API server endpoints patched successfully.",
            flush=True,
        )

    except ImportError:
        # API server modules might not be available in worker process
        pass
    except Exception as e:
        print(
            f"[ExpertDist] Warning: Failed to patch API server: {e}", flush=True
        )
