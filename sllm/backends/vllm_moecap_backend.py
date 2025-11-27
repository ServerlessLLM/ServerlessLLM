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
"""
VllmMoeCapBackend: Extended vLLM backend with MoE-CAP batch recording support.

This backend extends the standard VllmBackend to add batch statistics recording
capabilities required for MoE-CAP evaluation. It patches the GPUModelRunner's
execute_model method to track batch size, latency, sequence lengths, and expert
activation patterns.
"""

import asyncio
import gc
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from moe_cap.utils.hardware_utils import get_gpu_details
from transformers import AutoTokenizer
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.inputs import TokensPrompt
from vllm.sequence import IntermediateTensors
from vllm.utils import Counter
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    AsyncGPUModelRunnerOutput,
    GPUModelRunner,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from sllm.backends.backend_utils import BackendStatus, SllmBackend

GLOBAL_GPU_TYPE = get_gpu_details()
logger = logging.getLogger("ray")

# ============================================================================
# CRITICAL: Apply expert distribution monkey patching BEFORE any other vLLM imports
# ============================================================================
import sys

# Add path to extracted_expert_dist for imports
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_expert_dist_path = os.path.join(_current_file_dir, "extracted_expert_dist")
if _expert_dist_path not in sys.path:
    sys.path.insert(0, _expert_dist_path)

try:
    from vllm_integration import apply_vllm_monkey_patching

    apply_vllm_monkey_patching()
    logger.info(
        f"[PID {os.getpid()}] Expert distribution monkey patching applied successfully!"
    )
except ImportError as e:
    logger.warning(
        f"[PID {os.getpid()}] Could not import expert distribution patching: {e}"
    )
except Exception as e:
    logger.warning(
        f"[PID {os.getpid()}] Failed to apply expert distribution patching: {e}"
    )
    import traceback

    traceback.print_exc()

# Expert distribution recording state
EXPERT_DISTRIBUTION_RECORDING_FLAG_FILE = os.path.join(
    tempfile.gettempdir(), "sllm_expert_distribution_recording.flag"
)
EXPERT_DISTRIBUTION_AUTO_START_FLAG_FILE = os.path.join(
    tempfile.gettempdir(), "sllm_expert_distribution_auto_start.flag"
)
EXPERT_DISTRIBUTION_OUTPUT_DIR = os.path.join(
    os.getcwd(), "logs/expert_distribution"
)
_expert_record_lock = threading.Lock()
_forward_pass_id_counter = 0
_forward_pass_id_lock = threading.Lock()

# ============================================================================
# Helper functions from VllmBackend
# ============================================================================


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# ============================================================================
# Global recording state - using file-based flags for multiprocessing safety
# Each instance has its own recording files to avoid conflicts
# ============================================================================
_record_locks = {}  # Dict of locks per instance_id
_recording_states = {}  # Dict of RecordingState per instance_id

# Environment variable name for passing instance_id to worker processes
INSTANCE_ID_ENV_VAR = "SLLM_MOECAP_INSTANCE_ID"


def _get_lock(instance_id: str) -> threading.Lock:
    """Get or create a lock for the given instance_id."""
    if instance_id not in _record_locks:
        _record_locks[instance_id] = threading.Lock()
    return _record_locks[instance_id]


def set_current_instance_id(instance_id: str):
    """Set the current instance_id via environment variable so workers can access it."""
    os.environ[INSTANCE_ID_ENV_VAR] = instance_id


def get_current_instance_id() -> Optional[str]:
    """Get the current instance_id from environment variable."""
    return os.environ.get(INSTANCE_ID_ENV_VAR)


class RecordingState:
    """
    Global state for recording batch statistics - multiprocessing safe.

    Each instance has its own recording files identified by instance_id to prevent
    conflicts when multiple models are running in parallel.
    """

    def __init__(self, instance_id: str):
        """
        Initialize recording state for a specific instance.

        Args:
            instance_id: Unique identifier for this model instance
        """
        self.instance_id = instance_id
        # Create instance-specific file paths
        safe_id = instance_id.replace("/", "_").replace(":", "_")
        self.flag_file = os.path.join(
            tempfile.gettempdir(), f"sllm_moecap_{safe_id}_recording.flag"
        )
        self.data_file = os.path.join(
            tempfile.gettempdir(), f"sllm_moecap_{safe_id}_records.jsonl"
        )
        # Clean up any stale files on init
        self._cleanup_files()

    def _cleanup_files(self):
        """Remove recording flag and data files."""
        try:
            if os.path.exists(self.flag_file):
                os.remove(self.flag_file)
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
        except Exception:
            pass

    def is_recording(self):
        """Check if recording is active (file-based flag)."""
        return os.path.exists(self.flag_file)

    def start_recording(self, output_file: str = None):
        """Start recording batch statistics."""
        # Create flag file
        with open(self.flag_file, "w") as f:
            f.write("1")
        # Clear data file
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        logger.info(
            f"Started recording batch statistics for {self.instance_id} "
            f"(flag: {self.flag_file}, data: {self.data_file})"
        )

    def stop_recording(self):
        """Stop recording batch statistics."""
        if os.path.exists(self.flag_file):
            os.remove(self.flag_file)
        count = self.get_record_count()
        logger.info(
            f"Stopped recording for {self.instance_id}. Total records: {count}"
        )

    def add_record(self, record: dict):
        """Add a record to the data file (thread-safe, process-safe)."""
        lock = _get_lock(self.instance_id)
        with lock:
            with open(self.data_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    def get_records(self):
        """Get all recorded statistics."""
        if not os.path.exists(self.data_file):
            return []
        records = []
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_record_count(self):
        """Get count of records without loading all."""
        if not os.path.exists(self.data_file):
            return 0
        count = 0
        with open(self.data_file, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def clear_records(self):
        """Clear all recorded statistics."""
        count = self.get_record_count()
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        logger.info(f"Cleared {count} records for {self.instance_id}")
        return count


# ============================================================================
# Expert Distribution Recording State
# ============================================================================
class ExpertDistributionRecordingState:
    """State for expert distribution recording with automatic JSONL output."""

    def __init__(self, instance_id: str = None):
        self.expert_record_list = []
        self.output_dir = EXPERT_DISTRIBUTION_OUTPUT_DIR
        self.model_path = None
        self.enabled = False
        self.checked_auto_start = False
        self.instance_id = instance_id or str(uuid.uuid4())
        # Instance-specific flag files
        safe_id = self.instance_id.replace("/", "_").replace(":", "_")
        self.flag_file = os.path.join(
            tempfile.gettempdir(), f"sllm_expert_{safe_id}_recording.flag"
        )

    def set_model_path(self, model_path: str):
        """Set the model path for output file naming."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_name = model_path.replace("/", "_")
        self.model_path = f"{sanitized_name}_{timestamp}"

    def enable(self):
        """Enable automatic expert distribution recording."""
        self.enabled = True
        # Create flag file
        with open(self.flag_file, "w") as f:
            f.write(self.model_path or "")
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Expert distribution recording enabled for {self.instance_id}"
        )

    def disable(self):
        """Disable automatic expert distribution recording."""
        self.enabled = False
        if os.path.exists(self.flag_file):
            os.remove(self.flag_file)

    def is_recording(self):
        """Check if recording is active."""
        return os.path.exists(self.flag_file) or self.enabled

    def add_record(self, record: dict):
        """Add a record and write to JSONL file."""
        if not self.enabled:
            return

        with _expert_record_lock:
            self.expert_record_list.append(record)

            # Write to JSONL file immediately
            if self.model_path:
                output_file = os.path.join(
                    self.output_dir,
                    f"{self.model_path}/expert_distribution_record.jsonl",
                )
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()

    def get_records(self):
        """Get all recorded records."""
        return self.expert_record_list.copy()

    def clear_records(self):
        """Clear all recorded records."""
        count = len(self.expert_record_list)
        self.expert_record_list = []
        return count


# Global expert distribution recording states per instance
_expert_distribution_states = {}


def get_or_create_expert_distribution_state(
    instance_id: str,
) -> ExpertDistributionRecordingState:
    """Get or create expert distribution recording state for an instance."""
    if instance_id not in _expert_distribution_states:
        _expert_distribution_states[instance_id] = (
            ExpertDistributionRecordingState(instance_id)
        )
    return _expert_distribution_states[instance_id]


# ============================================================================
# Custom execute_model implementation with MoE-CAP batch tracking
# ============================================================================
# Global dict to store recording state instances per model
_recording_states = {}


@torch.inference_mode()
def execute_model_moecap(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    """Custom execute_model with latency and batch statistics tracking."""

    # ========================================================================
    # Lazy initialization of expert_distribution_recorder if not already done
    # This is critical for expert activation recording to work
    # ========================================================================
    if not hasattr(self, "_moecap_recorder_initialized"):
        self._moecap_recorder_initialized = True
        try:
            # Add path to extracted_expert_dist if not already in sys.path
            import os
            import sys

            _current_file_dir = os.path.dirname(os.path.abspath(__file__))
            _expert_dist_path = os.path.join(
                _current_file_dir, "extracted_expert_dist"
            )
            if _expert_dist_path not in sys.path:
                sys.path.insert(0, _expert_dist_path)

            # Try to initialize expert distribution recorder
            from expert_distribution_recorder import (
                ExpertDistributionRecorder,
                ExpertLocationMetadata,
                set_global_expert_distribution_recorder,
            )

            # Get model info for recorder initialization
            hf_config = self.model_config.hf_config

            # Detect number of experts
            num_experts = getattr(hf_config, "num_experts", None)
            if num_experts is None:
                num_experts = getattr(hf_config, "n_routed_experts", None)
            if num_experts is None:
                num_experts = getattr(
                    hf_config, "num_local_experts", 8
                )  # Default

            num_layers = getattr(hf_config, "num_hidden_layers", 28)

            expert_location_metadata = ExpertLocationMetadata(
                num_layers=num_layers,
                num_logical_experts=num_experts,
                num_physical_experts=num_experts,
                num_local_physical_experts=num_experts,
                ep_size=1,
            )

            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )

            self.expert_distribution_recorder = (
                ExpertDistributionRecorder.init_new(
                    recording_mode="per_pass",
                    expert_location_metadata=expert_location_metadata,
                    rank=rank,
                    device=str(self.device),
                    buffer_size=-1,
                    enable_metrics=True,
                )
            )
            set_global_expert_distribution_recorder(
                self.expert_distribution_recorder
            )

            # Auto-start recording
            self.expert_distribution_recorder.start_record()

            if rank == 0:
                print(
                    f"[MOECAP] ✓ Expert distribution recorder initialized: {num_layers} layers, {num_experts} experts",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[MOECAP] ⚠️ Failed to initialize expert distribution recorder: {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            self.expert_distribution_recorder = None

    world_size = self.vllm_config.parallel_config.world_size
    gpu_raw_type = GLOBAL_GPU_TYPE
    with record_function_or_nullcontext("Preprocess"):
        with self.synchronize_input_prep():
            # Update persistent batch states.
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(
                    scheduler_output, self.vllm_config
                )
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.input_batch.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )
            # Prepare the decoder inputs.
            (
                attn_metadata,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens_np,
                spec_decode_common_attn_metadata,
                max_query_len,
                ubatch_slices,
                num_tokens_after_padding,
            ) = self._prepare_inputs(scheduler_output)
        (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        ) = self._preprocess(
            scheduler_output,
            intermediate_tensors,
            ubatch_slices,
            num_tokens_after_padding,
        )
        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
        )
        batch_descriptor = BatchDescriptor(
            num_tokens=num_input_tokens, uniform_decode=uniform_decode
        )
        cudagraph_runtime_mode, batch_descriptor = (
            self.cudagraph_dispatcher.dispatch(batch_descriptor)
        )

    if ubatch_slices is not None:
        num_input_tokens = ubatch_slices[0].num_tokens

    # ======== START TIMING ========
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Run the model
    with (
        set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
        ),
        record_function_or_nullcontext("Forward"),
        self.maybe_get_kv_connector_output(
            scheduler_output
        ) as kv_connector_output,
    ):
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    with record_function_or_nullcontext("Postprocess"):
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None
        if not self.broadcast_pp_output:
            if not get_pp_group().is_last_rank:
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            if self.is_pooling_model:
                output = self._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                )
                output.kv_connector_output = kv_connector_output
                return output
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            assert not self.is_pooling_model
            if not get_pp_group().is_last_rank:
                all_gather_tensors = {
                    "residual": not is_residual_scattered_for_sp(
                        self.vllm_config, num_input_tokens
                    )
                }
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            model_output_broadcast_data = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        if scheduler_output.grammar_bitmask is not None:
            apply_grammar_bitmask(
                scheduler_output, self.input_batch, logits, self.device
            )

    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    def propose_draft_token_ids(sampled_token_ids):
        assert spec_decode_common_attn_metadata is not None
        with record_function_or_nullcontext("Draft"):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

    use_padded_batch_for_eagle = (
        self.speculative_config
        and self.speculative_config.use_eagle()
        and not self.speculative_config.disable_padded_drafter_batch
    )
    effective_drafter_max_model_len = self.max_model_len
    if effective_drafter_max_model_len is None:
        effective_drafter_max_model_len = self.model_config.max_model_len
    if (
        self.speculative_config
        and self.speculative_config.draft_model_config is not None
        and self.speculative_config.draft_model_config.max_model_len is not None
    ):
        effective_drafter_max_model_len = (
            self.speculative_config.draft_model_config.max_model_len
        )
    input_fits_in_drafter = spec_decode_common_attn_metadata and (
        spec_decode_common_attn_metadata.seq_lens.max()
        + self.speculative_config.num_speculative_tokens
        <= effective_drafter_max_model_len
    )
    if use_padded_batch_for_eagle and input_fits_in_drafter:
        propose_draft_token_ids(sampler_output.sampled_token_ids)

    with record_function_or_nullcontext("Bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            num_scheduled_tokens,
        )

    if (
        self.speculative_config
        and not use_padded_batch_for_eagle
        and input_fits_in_drafter
    ):
        propose_draft_token_ids(valid_sampled_token_ids)

    with record_function_or_nullcontext("EPLB"):
        self.eplb_step()

    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )

    # ======== END TIMING ========
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    latency = end_time - start_time
    batch_size = input_ids.size(0)
    forward_mode = "decode" if uniform_decode else "prefill"
    sum_seq_len = num_input_tokens

    # Track forward pass ID
    global _forward_pass_id_counter
    with _forward_pass_id_lock:
        _forward_pass_id_counter += 1
        forward_pass_id = _forward_pass_id_counter

    # Collect expert distribution data (per_pass mode)
    expert_activation = 0
    expert_utilization = 0
    num_experts_total = 0
    try:
        # Try multiple ways to get expert distribution data
        recorder = None

        # Method 1: Check self.expert_distribution_recorder (standard location)
        if (
            hasattr(self, "expert_distribution_recorder")
            and self.expert_distribution_recorder is not None
        ):
            recorder = self.expert_distribution_recorder

        # # Method 2: Check global recorder
        # if recorder is None:
        #     try:
        #         from extracted_expert_dist.expert_distribution_recorder import get_global_expert_distribution_recorder
        #         recorder = get_global_expert_distribution_recorder()
        #     except ImportError:
        #         pass

        if (
            recorder is not None
            and hasattr(recorder, "_recording")
            and recorder._recording
        ):
            # Get number of experts for utilization calculation
            if (
                hasattr(recorder, "_expert_location_metadata")
                and recorder._expert_location_metadata is not None
            ):
                num_experts_total = (
                    recorder._expert_location_metadata.num_logical_experts
                )

            # Try to collect from gatherer
            if (
                hasattr(recorder, "_gatherer")
                and recorder._gatherer is not None
            ):
                gatherer = recorder._gatherer

                # Collect the data
                collected_data = gatherer.collect()
                collected_data["forward_mode"] = forward_mode

                # Reset gatherer for next pass
                gatherer.reset()

                # Append to accumulator
                if (
                    hasattr(recorder, "_accumulator")
                    and recorder._accumulator is not None
                ):
                    recorder._accumulator.append(
                        forward_pass_id, collected_data
                    )

                # Calculate expert activation from collected data
                # Try different keys that might contain expert counts
                counts = None
                for key in [
                    "expert_count",
                    "expert_counts",
                    "logical_count",
                    "counts",
                ]:
                    if (
                        key in collected_data
                        and collected_data[key] is not None
                    ):
                        counts = collected_data[key]
                        break

                if (
                    counts is not None
                    and isinstance(counts, torch.Tensor)
                    and counts.numel() > 0
                ):
                    # Shape could be (num_layers, num_experts) or (1, num_layers, num_experts)
                    if counts.dim() == 3:
                        counts = counts.squeeze(0)
                    if counts.dim() == 2:
                        # Average activated experts per layer
                        activated_per_layer = (
                            (counts > 0).float().sum(dim=1)
                        )  # (num_layers,)
                        expert_activation = activated_per_layer.mean().item()
                    elif counts.dim() == 1:
                        expert_activation = (counts > 0).float().sum().item()

                # Calculate utilization
                if expert_activation > 0 and num_experts_total > 0:
                    expert_utilization = expert_activation / num_experts_total

    except Exception as e:
        # Log the error for debugging but don't crash
        import traceback

        print(f"[MOECAP] Could not collect expert distribution data: {e}")
        print(traceback.format_exc())

    # Only warn about missing expert data after CUDA graph warmup (first ~10-15 passes)
    # During warmup, the recorder may not be fully active yet
    WARMUP_PASSES = 15
    if expert_activation == 0 and forward_pass_id > WARMUP_PASSES:
        print(
            f"[MOECAP] ⚠️ Expert distribution data not available for forward pass {forward_pass_id}"
        )

    # Record batch statistics if recording is enabled
    # Get instance_id from global variable (set by VllmMoeCapBackend in this process)
    instance_id = get_current_instance_id()

    if instance_id:
        # Get or create recording state for this instance
        if instance_id not in _recording_states:
            _recording_states[instance_id] = RecordingState(instance_id)
            print(
                f"[MOECAP] Created RecordingState for instance {instance_id} in PID {os.getpid()}"
            )

        rec_state = _recording_states[instance_id]
        if rec_state.is_recording():
            rec_dict = {
                "batch_size": batch_size,
                "latency": latency,
                "seq_lens_sum": sum_seq_len,
                "forward_mode": forward_mode,
                "expert_activation": expert_activation,
                "forward_pass_id": forward_pass_id,
                "gpu_num": world_size,
                "gpu_raw_type": gpu_raw_type,
            }
            rec_state.add_record(rec_dict)
            print(
                f"[MOECAP] ✓ Recorded {forward_mode}: batch_size={batch_size}, "
                f"latency={latency:.4f}s, expert_act={expert_activation:.2f}"
                f"forward_pass_id={forward_pass_id}"
            )

    if not self.use_async_scheduling:
        return output
    return AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
    )


# ============================================================================
# Apply monkey patch at MODULE LOAD TIME
# This ensures Ray workers get the patched version when they import GPUModelRunner
# ============================================================================
GPUModelRunner.execute_model = execute_model_moecap


class VllmMoeCapBackend(SllmBackend):
    """
    Standalone vLLM backend with MoE-CAP batch recording capabilities.

    This backend directly initializes AsyncLLMEngine without using VllmBackend,
    ensuring the monkey patch is applied before any vLLM Ray workers are created.

    Additional Methods:
    - start_batch_recording: Start recording batch statistics
    - stop_batch_recording: Stop recording batch statistics
    - dump_batch_recording: Get all recorded batch statistics
    - batch_recording_status: Check current recording status
    - clear_batch_recording: Clear all recorded statistics
    """

    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        # CRITICAL: Set environment variable to make vLLM workers import our patch module
        # This must be done BEFORE creating AsyncEngineArgs or initializing the engine
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = (
            "spawn"  # Ensure clean worker processes
        )
        # Add our patch module to Python path for worker imports
        import sys

        sllm_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if sllm_path not in sys.path:
            sys.path.insert(0, sllm_path)

        logger.info(
            f"[PID {os.getpid()}] VllmMoeCapBackend: Environment configured for worker patching"
        )

        # Create instance-specific recording state with unique UUID
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_name = model
        self.instance_id = str(uuid.uuid4())
        self.recording_state = RecordingState(self.instance_id)
        _recording_states[self.instance_id] = self.recording_state

        # CRITICAL: Set environment variable BEFORE creating engine
        # So Ray workers inherit it when they're spawned
        set_current_instance_id(self.instance_id)

        # Initialize status and config (replicated from VllmBackend)
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        self.trace_debug = backend_config.get("trace_debug", False)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        self.task = backend_config.get("task", "auto")

        # Build engine args (replicated from VllmBackend)
        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        # CRITICAL: Use custom worker class that applies the monkey patch
        # Pass model_name to workers so they can look up instance_id
        filtered_engine_config["worker_cls"] = (
            "sllm.backends.moecap_worker.MoeCapGPUWorker"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)
        self.engine = None

    async def init_backend(self) -> None:
        """Initialize the backend by creating AsyncLLMEngine (with monkey patch already applied)."""
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            # The monkey patch was applied at module load time, so any Ray workers
            # created here will use the patched GPUModelRunner.execute_model
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        """Generate completion for the given request."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        valid_massages = [
            message for message in messages if "content" in message
        ]
        construct_prompt = self.tokenizer.apply_chat_template(
            valid_massages, tokenize=False, add_generation_prompt=True
        )
        # construct_prompt: str = "\n".join(
        #     [
        #         f"{message['role']}: {message['content']}"
        #         for message in messages
        #         if "content" in message
        #     ]
        # )

        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # Non-stream case
        final_output = None
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

        return process_output(final_output, model_name)

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache for the given requests."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def encode(self, request_data: Dict[str, Any]):
        """Encode input for embeddings."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)

    async def start_batch_recording(self) -> Dict[str, Any]:
        """Start recording batch statistics."""
        self.recording_state.start_recording()
        return {
            "status": "success",
            "message": f"Started recording batch statistics for {self.model_name}",
            "instance_id": self.instance_id,
        }

    async def stop_batch_recording(self) -> Dict[str, Any]:
        """Stop recording batch statistics."""
        self.recording_state.stop_recording()
        return {
            "status": "success",
            "message": f"Stopped recording batch statistics for {self.model_name}",
            "instance_id": self.instance_id,
            "total_records": self.recording_state.get_record_count(),
        }

    async def dump_batch_recording(self) -> Dict[str, Any]:
        """Dump batch statistics to file and return as JSON."""
        records = self.recording_state.get_records()
        return {
            "status": "success",
            "model": self.model_name,
            "instance_id": self.instance_id,
            "records": records,
            "total_records": len(records),
        }

    async def batch_recording_status(self) -> Dict[str, Any]:
        """Get current recording status."""
        return {
            "model": self.model_name,
            "instance_id": self.instance_id,
            "is_recording": self.recording_state.is_recording(),
            "total_records": self.recording_state.get_record_count(),
        }

    async def clear_batch_recording(self) -> Dict[str, Any]:
        """Clear all recorded batch statistics."""
        count = self.recording_state.clear_records()
        return {
            "status": "success",
            "model": self.model_name,
            "instance_id": self.instance_id,
            "message": f"Cleared {count} records",
            "cleared_count": count,
        }

    # =========================================================================
    # Expert Distribution Recording Methods
    # =========================================================================

    async def configure_expert_distribution(
        self,
        recording_mode: str = "per_pass",
        enable_metrics: bool = True,
        buffer_size: int = -1,
    ) -> Dict[str, Any]:
        """
        Configure expert distribution recording on all workers.

        Args:
            recording_mode: One of "per_token", "per_pass", "stat", "stat_approx"
            enable_metrics: Whether to compute and log metrics
            buffer_size: Size of recording buffer (-1 for unlimited)

        Returns:
            Configuration status
        """
        if self.engine is None:
            return {"status": "error", "message": "Engine not initialized"}

        # Get or create expert distribution state for this instance
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        exp_state.set_model_path(self.model_name)

        try:
            # Use collective_rpc to configure on all workers
            result = await self.engine.collective_rpc(
                "configure_expert_distribution_recorder",
                args=(recording_mode, enable_metrics, buffer_size),
            )
            return {
                "status": "success",
                "message": f"Configured expert distribution recording with mode={recording_mode}",
                "instance_id": self.instance_id,
                "worker_results": result,
            }
        except Exception as e:
            logger.error(f"Failed to configure expert distribution: {e}")
            return {
                "status": "error",
                "message": str(e),
                "instance_id": self.instance_id,
            }

    async def start_expert_distribution_recording(
        self,
        recording_mode: str = "per_pass",
    ) -> Dict[str, Any]:
        """
        Start expert distribution recording on all workers.

        Args:
            recording_mode: One of "per_token", "per_pass", "stat", "stat_approx"

        Returns:
            Status of the operation
        """
        if self.engine is None:
            return {"status": "error", "message": "Engine not initialized"}

        # Enable local recording state
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        exp_state.set_model_path(self.model_name)
        exp_state.enable()

        try:
            # Configure and start on workers
            await self.engine.collective_rpc(
                "configure_expert_distribution_recorder",
                args=(recording_mode, True, -1),
            )
            await self.engine.collective_rpc(
                "start_expert_distribution_recording",
            )
            return {
                "status": "success",
                "message": f"Started expert distribution recording with mode={recording_mode}",
                "instance_id": self.instance_id,
            }
        except Exception as e:
            logger.error(f"Failed to start expert distribution recording: {e}")
            return {
                "status": "error",
                "message": str(e),
                "instance_id": self.instance_id,
            }

    async def stop_expert_distribution_recording(self) -> Dict[str, Any]:
        """Stop expert distribution recording on all workers."""
        if self.engine is None:
            return {"status": "error", "message": "Engine not initialized"}

        # Disable local recording state
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        exp_state.disable()

        try:
            await self.engine.collective_rpc(
                "stop_expert_distribution_recording",
            )
            return {
                "status": "success",
                "message": "Stopped expert distribution recording",
                "instance_id": self.instance_id,
                "total_records": len(exp_state.get_records()),
            }
        except Exception as e:
            logger.error(f"Failed to stop expert distribution recording: {e}")
            return {
                "status": "error",
                "message": str(e),
                "instance_id": self.instance_id,
            }

    async def dump_expert_distribution(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dump expert distribution data from all workers.

        Args:
            output_path: Path to save the recording (optional)

        Returns:
            Dictionary containing recorded data from all workers
        """
        if self.engine is None:
            return {"status": "error", "message": "Engine not initialized"}

        # Get local recording state data (fast)
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        local_records = exp_state.get_records()

        # Try to get worker data with timeout
        worker_results = []
        try:
            worker_results = await asyncio.wait_for(
                self.engine.collective_rpc(
                    "dump_expert_distribution_record",
                    args=(output_path,),
                ),
                timeout=5.0,  # 5 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout waiting for worker expert distribution data"
            )
        except Exception as e:
            logger.warning(
                f"Could not get worker expert distribution data: {e}"
            )

        return {
            "status": "success",
            "model": self.model_name,
            "instance_id": self.instance_id,
            "worker_data": worker_results,
            "local_records": local_records,
            "total_local_records": len(local_records),
        }

    async def expert_distribution_status(self) -> Dict[str, Any]:
        """Get current expert distribution recording status."""
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        return {
            "model": self.model_name,
            "instance_id": self.instance_id,
            "is_recording": exp_state.is_recording(),
            "total_records": len(exp_state.get_records()),
        }

    async def clear_expert_distribution(self) -> Dict[str, Any]:
        """Clear all recorded expert distribution data."""
        exp_state = get_or_create_expert_distribution_state(self.instance_id)
        count = exp_state.clear_records()
        return {
            "status": "success",
            "model": self.model_name,
            "instance_id": self.instance_id,
            "message": f"Cleared {count} expert distribution records",
            "cleared_count": count,
        }
