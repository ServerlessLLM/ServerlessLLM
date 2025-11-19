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

logger = logging.getLogger(__name__)

# Apply monkey patch BEFORE importing Worker
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
    """

    pass  # No special initialization needed - patch and global var are sufficient
