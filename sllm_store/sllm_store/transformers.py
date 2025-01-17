# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import concurrent.futures
import json
import os
import time
import uuid
from typing import Optional, Union

import torch
from accelerate import dispatch_model, init_empty_weights

# from accelerate.hooks import add_hook_to_module
from accelerate.utils import set_module_tensor_to_device
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
)
from sllm_store.client import SllmStoreClient
from sllm_store.device_map_utils import (
    DeviceMapType,
    _compute_device_placement_from_map,
    _compute_device_placement_from_map_fast,
    _expand_tensor_name,
    _transform_device_map_to_dict,
)
from sllm_store.logger import init_logger
from sllm_store.torch import load_dict_non_blocking, save_dict
from sllm_store.utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
    get_no_split_modules,
    get_tied_no_split_modules,
    send_module_buffers_to_device,
    get_quantization_config_and_type,
    get_quantization_fn,
    replace_linear_with_quantized,
)
from torch import nn
from transformers import AutoConfig, GenerationConfig
from transformers.quantizers.quantizers_utils import get_module_from_name
import bitsandbytes as bnb
import importlib

logger = init_logger(__name__)


def _get_uuid():
    return str(uuid.uuid4())


def save_model(model: nn.Module, model_path: str):
    """
    Args:
        model: nn.Module
            a model to be saved
        storage_path: str
            a local path to save the converted model
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model = model.cpu()
    model_state_dict = model.state_dict()

    save_dict(model_state_dict, model_path)

    # This section of code was adopted from the Hugging Face Transformers project under Apache-2.0 License. # noqa: E501
    # Source: https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/modeling_utils.py#L2425-L2447
    # Modifications made: Removed the support for '_hf_peft_config_loaded'
    #
    # Save the config
    model.config.save_pretrained(model_path)
    if model.can_generate():
        # generation config built from the model config + the model config holds generation kwargs -> generate # noqa: E501
        # may revert to legacy behavior if the two don't match
        if (
            model.generation_config._from_model_config
            and model.config._has_non_default_generation_parameters()
        ):
            new_generation_config = GenerationConfig.from_model_config(
                model.config
            )
            if new_generation_config != model.generation_config:
                logger.warning(
                    "Your generation config was originally created from the model config, but the model "  # noqa: E501
                    "config has changed since then. Unless you pass the `generation_config` argument to this "  # noqa: E501
                    "model's `generate` calls, they will revert to the legacy behavior where the base "  # noqa: E501
                    "`generate` parameterization is loaded from the model config instead. "  # noqa: E501
                    "To avoid this behavior and this warning, we recommend you to overwrite the generation "  # noqa: E501
                    "config model attribute before calling the model's `save_pretrained`, preferably also "  # noqa: E501
                    "removing any generation kwargs from the model config. This warning will be raised to an "  # noqa: E501
                    "exception in v4.41."
                )
        model.generation_config.save_pretrained(model_path)

    # save module index
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    # save tied parameters
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)


def load_model(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization: Optional[str] = None,
    storage_path: Optional[str] = None,
    fully_parallel: bool = False,
    hf_model_class: str = "AutoModelForCausalLM",
):
    if fully_parallel:
        return fully_parallel_load(
            model_path=model_path,
            hf_model_class=hf_model_class,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization=quantization,
            storage_path=storage_path,
        )
    # if fully_parallel is disabled, we still try to parallelize the model
    # initialization and data loading in the best effort
    return best_effort_load(
        model_path=model_path,
        hf_model_class=hf_model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization=quantization,
        storage_path=storage_path,
    )


def fully_parallel_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization: Optional[str] = None,
    storage_path: Optional[str] = None,
):
    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")
    start = time.time()
    device_map = _transform_device_map_to_dict(device_map)
    with open(
        os.path.join(storage_path, model_path, "tied_no_split_modules.json"),
        "r",
    ) as f:
        tied_no_split_modules = json.load(f)

    if isinstance(device_map, str):
        with open(
            os.path.join(storage_path, model_path, "no_split_modules.json"),
            "r",
        ) as f:
            no_split_modules = json.load(f)
        device_map = _compute_device_placement_from_map_fast(
            no_split_modules, tied_no_split_modules, device_map
        )
    # TODO: offload `load_dict_non_blocking` to c++ for real parallelism
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            load_dict_non_blocking, model_path, device_map, storage_path
        )
        logger.debug(
            f"load_dict_non_blocking takes {time.time() - start} seconds"
        )

        start = time.time()
        config = AutoConfig.from_pretrained(
            f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
        )
        if torch_dtype is not None:
            config.torch_dtype = torch_dtype

        if quantization:
            quantization_config, quant_type = get_quantization_config_and_type(
                quantization
            )

        logger.debug(f"load config takes {time.time() - start} seconds")
        start = time.time()
        with init_empty_weights():
            module = importlib.import_module("transformers")
            _class = getattr(module, hf_model_class)
            model = _class.from_config(
                config,
                trust_remote_code=True,
            ).to(config.torch_dtype)

        model.tie_weights()
        logger.debug(f"load model takes {time.time() - start} seconds")

        replica_uuid, state_dict = future.result()

    with torch.no_grad():
        if quantization:
            quantization_fn = get_quantization_fn(quantization)

            if quantization == "int8":

                def quantize(x):
                    quantized_weights, scales, _ = quantization_fn(x)
                    return quantized_weights, scales

            else:

                def quantize(x):
                    quantized_weights, quant_state = quantization_fn(
                        x, quant_type=quant_type
                    )
                    return quantized_weights, quant_state

            for name, param in state_dict.items():
                module = get_module_from_name(
                    model, name
                )  # gets the specific layer from the model
                if isinstance(module[0], torch.nn.Linear) and name.endswith(
                    ".weight"
                ):
                    replace_linear_with_quantized(model, name, quantization)
                    module, _ = get_module_from_name(model, name)

                print(module)

                if param.dtype in [
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                ] and name.endswith(".weight"):
                    if isinstance(
                        module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
                    ):
                        print("weight was quantized")

                        param_fp16 = param.data.to(torch.float16)
                        quantized_weights, scales_or_state = quantize(
                            param_fp16
                        )

                        module._parameters["weight"] = quantized_weights
                        print(f"weights {module._parameters["weight"]}")

                        if isinstance(module, bnb.nn.Linear4bit):
                            module.weight_state = scales_or_state
                        else:
                            module.weight_scale.data = scales_or_state

                        print(f"quantized {quantized_weights.dtype}")
                        set_module_tensor_to_device(
                            model,
                            name,
                            quantized_weights.device,
                            quantized_weights,
                        )
                    else:
                        print(
                            f"skipped 2, module is {type(module)} and {module}"
                        )
                        set_module_tensor_to_device(
                            model, name, param.device, param
                        )

                else:
                    print("skipped")
                    set_module_tensor_to_device(
                        model, name, param.device, param
                    )
        else:
            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)

        send_module_buffers_to_device(model, device_map)

    remaining_meta = [
        name for name, param in model.named_parameters() if param.is_meta
    ]
    if remaining_meta:
        logger.warning(f"Found remaining meta tensors: {remaining_meta}")

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )
    client = SllmStoreClient("127.0.0.1:8073")
    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map

    return model


def best_effort_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    storage_path: Optional[str] = None,
):
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    replica_uuid = _get_uuid()
    device_map = _transform_device_map_to_dict(device_map)

    if isinstance(device_map, dict) and (
        torch.device("cpu") in device_map.values()
        or "cpu" in device_map.values()
    ):
        raise ValueError("CPU is not supported in device_map.")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")
    start = time.time()
    config = AutoConfig.from_pretrained(
        f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
    )
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype

    logger.debug(f"load config takes {time.time() - start} seconds")
    start = time.time()
    with init_empty_weights():
        module = importlib.import_module("transformers")
        _class = getattr(module, hf_model_class)
        model = _class.from_config(config, trust_remote_code=True).to(
            config.torch_dtype
        )

    model.tie_weights()
    logger.debug(f"load model takes {time.time() - start} seconds")

    start = time.time()
    if isinstance(device_map, str):
        device_map = _compute_device_placement_from_map(
            model, device_map, config.torch_dtype
        )
        logger.debug(f"device_map: {device_map}")
    # check if 'cpu' is in device_map values and raise an exception
    if "cpu" in device_map.values():
        raise ValueError(
            "The GPUs are either unavailable or do not have enough memory. Please ensure they are available and ready for use."  # noqa: E501
        )

    logger.debug(
        f"compute_device_placement takes {time.time() - start} seconds"
    )

    with open(
        os.path.join(storage_path, model_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    start = time.time()
    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    device_memory = calculate_device_memory(
        expanded_device_map, tensor_data_index
    )
    # logger.debug(f"calculate_device_memory {device_memory}")
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    # cuda_memory_ptrs = { k: [v] for k,v in cuda_memory_ptrs.items()}
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    device_uuid_map = get_device_uuid_map()
    # logger.debug(f"determine device_uuid_map {device_uuid_map}")
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )
    logger.debug(f"allocate_cuda_memory takes {time.time() - start} seconds")

    ret = client.load_into_gpu(
        model_path,
        replica_uuid,
        {
            device_uuid_map[device_id]: v
            for device_id, v in tensor_copy_chunks.items()
        },
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into GPU")

    # load model state_dict
    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )
    logger.info(f"restore state_dict takes {time.time() - start} seconds")

    with torch.no_grad():
        for name, param in state_dict.items():
            set_module_tensor_to_device(
                model, name, expanded_device_map[name], param
            )
        send_module_buffers_to_device(model, device_map)

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )

    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map

    return model
