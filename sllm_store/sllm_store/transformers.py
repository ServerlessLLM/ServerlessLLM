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
)
from torch import nn
from transformers import AutoConfig
from transformers.integrations.bitsandbytes import (
    set_module_quantized_tensor_to_device,
    replace_with_bnb_linear,
)
import importlib
from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
)
from peft.utils import set_peft_model_state_dict

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
    # Source: https://github.com/huggingface/transformers/blob/241c04d36867259cdf11dbb4e9d9a60f9cb65ebc/src/transformers/modeling_utils.py#L2812-L2856
    # Modifications made: Removed the support for '_hf_peft_config_loaded'
    #
    # Save the config
    model.config.save_pretrained(model_path)
    if model.can_generate():
        model.generation_config.save_pretrained(model_path)

    # save module index
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    # save tied parameters
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)


def save_lora(model: PeftModel, lora_path: str):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path, exist_ok=True)

    model = model.cpu()

    lora_state_dict = get_peft_model_state_dict(model)

    save_dict(lora_state_dict, lora_path)

    # save the config
    if hasattr(model, "peft_config") and model.peft_config:
        adapter_name = getattr(model, "active_adapter", None)
        if adapter_name is None:
            logger.warning("No active_adapter found")
            return

        config = model.peft_config.get(adapter_name, None)
        if config is None:
            logger.warning(f"No config found for adapter: {adapter_name}")
            return

        config.save_pretrained(lora_path)
        logger.info(
            f"Saved LoRA config for adapter: {adapter_name} to {lora_path}"
        )


def load_model(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
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
            quantization_config=quantization_config,
            storage_path=storage_path,
        )
    # if fully_parallel is disabled, we still try to parallelize the model
    # initialization and data loading in the best effort
    return best_effort_load(
        model_path=model_path,
        hf_model_class=hf_model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        storage_path=storage_path,
    )


def fully_parallel_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
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
        if quantization_config and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            if not isinstance(quantization_config, BitsAndBytesConfig):
                raise ValueError(
                    f"Invalid config type: {type(quantization_config)}"
                )

            logger.debug(
                f"using precision: {quantization_config.quantization_method()}"
            )

            if quantization_config.llm_int8_enable_fp32_cpu_offload:
                logger.debug("Offloading is not supported yet")
                quantization_config.llm_int8_enable_fp32_cpu_offload = False

            has_torch_dtype = torch_dtype is not None
            model = replace_with_bnb_linear(
                model, quantization_config=quantization_config
            )

            # synchronize
            client = SllmStoreClient("127.0.0.1:8073")
            client.confirm_model_loaded(model_path, replica_uuid)

            for name, param in state_dict.items():
                final_device = param.device
                if not has_torch_dtype:
                    param = param.to(torch.float16)

                set_module_quantized_tensor_to_device(
                    model, name, final_device, param.to("cpu")
                )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        send_module_buffers_to_device(model, device_map)

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )

    client = SllmStoreClient("127.0.0.1:8073")
    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    return model


def best_effort_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
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
        if quantization_config and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            if not isinstance(quantization_config, BitsAndBytesConfig):
                raise ValueError(
                    f"Invalid config type: {type(quantization_config)}"
                )

            logger.debug(
                f"using precision: {quantization_config.quantization_method()}"
            )

            if quantization_config.llm_int8_enable_fp32_cpu_offload:
                logger.debug("Offloading is not supported yet")
                quantization_config.llm_int8_enable_fp32_cpu_offload = False

            model = replace_with_bnb_linear(
                model, quantization_config=quantization_config
            )

            client.confirm_model_loaded(model_path, replica_uuid)

            for name, param in state_dict.items():
                if (
                    param.dtype not in [torch.uint8, torch.int8]
                    and torch_dtype is None
                ):
                    param = param.to(torch.float16)

                set_module_quantized_tensor_to_device(
                    model, name, param.device, param
                )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        send_module_buffers_to_device(model, device_map)

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )

    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map

    return model


def load_lora(
    model: nn.Module,
    adapter_name: str,
    adapter_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    storage_path: Optional[str] = None,
    is_trainable: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
):
    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")

    config_path = os.path.join(
        storage_path, adapter_path, "adapter_config.json"
    )
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    lora_config = LoraConfig(**config_dict)

    if lora_config.is_prompt_learning and is_trainable:
        raise ValueError(
            "Cannot set a prompt learning adapter to trainable\
            when loading pretrained adapter."
        )

    lora_config.inference_mode = not is_trainable

    client = SllmStoreClient("127.0.0.1:8073")
    client.register_model(adapter_path)

    model.add_adapter(lora_config, adapter_name=adapter_name)

    replica_uuid, state_dict = load_dict_non_blocking(
        adapter_path, {"": 0}, storage_path
    )

    # https://github.com/huggingface/transformers/blob/de182ba2690fe6c3466f6463c7f4b3a61694b885/src/transformers/integrations/peft.py#L228-L265
    processed_adapter_state_dict = {}
    prefix = "base_model.model."
    for key, value in state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        processed_adapter_state_dict[new_key] = value

    incompatible_keys = set_peft_model_state_dict(
        model, processed_adapter_state_dict, adapter_name
    )
    if incompatible_keys is not None:
        err_msg = ""
        origin_name = adapter_path if adapter_path is not None else "state_dict"
        # Check for unexpected keys.
        if (
            hasattr(incompatible_keys, "unexpected_keys")
            and len(incompatible_keys.unexpected_keys) > 0
        ):
            err_msg = (
                f"Loading adapter weights from {origin_name} led to \
                    unexpected keys not found in the model: "
                f"{', '.join(incompatible_keys.unexpected_keys)}. "
            )

        # Check for missing keys.
        missing_keys = getattr(incompatible_keys, "missing_keys", None)
        if missing_keys:
            # Filter missing keys specific to the current adapter, \
            # as missing base model keys are expected.
            lora_missing_keys = [
                k for k in missing_keys if "lora_" in k and adapter_name in k
            ]
            if lora_missing_keys:
                err_msg += (
                    f"Loading adapter weights from {origin_name} led to \
                        missing keys in the model: "
                    f"{', '.join(lora_missing_keys)}"
                )

        if err_msg:
            logger.warning(err_msg)

    # convert base model to PeftModel
    peft_model = get_peft_model(model, lora_config)

    # synchronize
    client.confirm_model_loaded(adapter_path, replica_uuid)

    if lora_config.inference_mode:
        peft_model.eval()

    logger.info(f"Available adapters: {peft_model.peft_config.keys()}")

    return peft_model
