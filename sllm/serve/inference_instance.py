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

import ray

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


@ray.remote
def start_instance(
    instance_id, backend, model_name, backend_config, startup_config
):
    logger.info(f"Starting instance {instance_id} with backend {backend}")
    if backend == "vllm":
        from sllm.serve.backends import VllmBackend

        model_backend_cls = ray.remote(VllmBackend)
    elif backend == "dummy":
        from sllm.serve.backends import DummyBackend

        model_backend_cls = ray.remote(DummyBackend)
    elif backend == "transformers":
        from sllm.serve.backends import TransformersBackend

        model_backend_cls = ray.remote(TransformersBackend)
    else:
        logger.error(f"Unknown backend: {backend}")
        raise ValueError(f"Unknown backend: {backend}")

    return model_backend_cls.options(
        name=instance_id, **startup_config, max_concurrency=10
    ).remote(model_name, backend_config)
