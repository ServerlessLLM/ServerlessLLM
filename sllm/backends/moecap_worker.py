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
Custom vLLM worker that applies MoE-CAP monkey patch on initialization.

This worker is used by VllmMoeCapBackend to ensure the execute_model patch
is applied in every worker process (including Ray workers).
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL: Apply expert distribution monkey patching BEFORE any other imports
# ============================================================================
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_expert_dist_path = os.path.join(_current_file_dir, "extracted_expert_dist")
if _expert_dist_path not in sys.path:
    sys.path.insert(0, _expert_dist_path)

try:
    from vllm_integration import apply_vllm_monkey_patching

    apply_vllm_monkey_patching()
    logger.info(
        f"[PID {os.getpid()}] MoeCapWorker: Expert distribution patching applied"
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

# Apply execute_model monkey patch BEFORE importing Worker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from sllm.backends.vllm_moecap_backend import execute_model_moecap

GPUModelRunner.execute_model = execute_model_moecap

# Now import the actual worker class (it's called Worker in vLLM v1, not GPUWorker)
from vllm.v1.worker.gpu_worker import Worker


class MoeCapGPUWorker(Worker):
    """
    Custom GPU worker that ensures MoE-CAP monkey patch is applied.

    The patch is applied at module import time (above), so by the time
    this class is instantiated, GPUModelRunner.execute_model is already patched.

    The instance_id is passed via a global variable that's inherited from the
    parent process when Ray workers are created.

    Expert distribution recording is handled by the vllm_integration patching
    applied at module import time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize expert distribution recorder after model is loaded
        self._init_expert_distribution_recorder()

    def _init_expert_distribution_recorder(self):
        """Initialize expert distribution recorder for this worker."""
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                # The recorder will be initialized by configure_expert_distribution_recorder
                # when called via collective_rpc
                self.model_runner.expert_distribution_recorder = None
                logger.info(
                    f"[PID {os.getpid()}] MoeCapWorker: Expert distribution recorder placeholder set"
                )
        except Exception as e:
            logger.warning(
                f"[PID {os.getpid()}] Failed to init expert recorder: {e}"
            )

    def configure_expert_distribution_recorder(
        self,
        recording_mode: str = "per_pass",
        enable_metrics: bool = True,
        buffer_size: int = -1,
    ):
        """Configure expert distribution recording on this worker."""
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                # Use the method added by vllm_integration patching
                if hasattr(
                    self.model_runner, "configure_expert_distribution_recording"
                ):
                    self.model_runner.configure_expert_distribution_recording(
                        recording_mode=recording_mode,
                        enable_metrics=enable_metrics,
                        buffer_size=buffer_size,
                    )
                    return {"success": True, "mode": recording_mode}
            return {"success": False, "error": "model_runner not available"}
        except Exception as e:
            logger.error(f"Failed to configure expert recorder: {e}")
            return {"success": False, "error": str(e)}

    def start_expert_distribution_recording(self):
        """Start expert distribution recording."""
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                if hasattr(
                    self.model_runner, "start_expert_distribution_recording"
                ):
                    self.model_runner.start_expert_distribution_recording()
                    return {"success": True}
            return {"success": False, "error": "model_runner not available"}
        except Exception as e:
            logger.error(f"Failed to start expert recording: {e}")
            return {"success": False, "error": str(e)}

    def stop_expert_distribution_recording(self):
        """Stop expert distribution recording."""
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                if hasattr(
                    self.model_runner, "stop_expert_distribution_recording"
                ):
                    self.model_runner.stop_expert_distribution_recording()
                    return {"success": True}
            return {"success": False, "error": "model_runner not available"}
        except Exception as e:
            logger.error(f"Failed to stop expert recording: {e}")
            return {"success": False, "error": str(e)}

    def dump_expert_distribution_record(self, output_path=None):
        """Dump expert distribution recording data."""
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                if hasattr(
                    self.model_runner, "dump_expert_distribution_record"
                ):
                    return self.model_runner.dump_expert_distribution_record(
                        output_path
                    )
            return {}
        except Exception as e:
            logger.error(f"Failed to dump expert recording: {e}")
            return {"error": str(e)}
