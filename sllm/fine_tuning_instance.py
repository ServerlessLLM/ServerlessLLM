# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2025                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
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
def start_ft_instance(
    instance_id, backend, model_name, backend_config, startup_config
):
    logger.info(f"Starting instance {instance_id} with backend {backend}")
    if backend == "peft_lora":
        from sllm.serve.ft_backends import PeftLoraBackend

        model_backend_cls = ray.remote(PeftLoraBackend)
    else:
        logger.error(f"Unknown backend: {backend}")
        raise ValueError(f"Unknown backend: {backend}")

    return model_backend_cls.options(
        name=instance_id,
        namespace="fine_tuning",
        **startup_config,
        max_concurrency=10,
    ).remote(model_name, backend_config)
