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
Worker initialization module for MoE-CAP monkey patching.

This module is imported by each vLLM Ray worker to apply the execute_model patch.
It must be imported BEFORE any vLLM worker initialization happens.
"""

import logging
import os
import sys

logger = logging.getLogger("ray")

# Apply the monkey patch immediately when this module is imported
try:
    # Import and patch BEFORE any other vLLM components
    from sllm.backends.vllm_moecap_backend import (
        GPUModelRunner,
        execute_model_moecap,
    )

    GPUModelRunner.execute_model = execute_model_moecap

except Exception as e:
    logger.error(
        f"[PID {os.getpid()}] vllm_moecap_worker_patch: Failed to apply monkey patch: {e}"
    )
    import traceback

    traceback.print_exc()
    raise
