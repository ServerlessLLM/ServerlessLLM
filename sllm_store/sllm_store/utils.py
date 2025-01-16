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
from functools import reduce

import torch
from torch import nn
from accelerate.utils import find_tied_parameters
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from transformers.quantizers.quantizers_utils import get_module_from_name


def set_module_buffer_to_device(
    module: nn.Module,
    target: str,
    device: torch.device,
):
    module_path, _, buffer_name = target.rpartition(".")

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(
            mod._get_name() + " has no attribute `" + buffer_name + "`"
        )

    buffer = mod._buffers[buffer_name]
    mod._buffers[buffer_name] = buffer.to(device)


def send_module_buffers_to_device(
    module: nn.Module,
    device_map: dict,
):
    if "" in device_map and len(device_map) != 1:
        raise RuntimeError(
            f"Device map {device_map} is invalid. If you want to specify the default device, use key ''."  # noqa: E501
        )

    buffer_names = [name for name, _ in module.named_buffers()]
    for tensor_or_module, device_id in device_map.items():
        if tensor_or_module == "":
            for buffer_name in buffer_names:
                set_module_buffer_to_device(module, buffer_name, device_id)
        else:
            for buffer_name in buffer_names:
                if buffer_name.startswith(tensor_or_module):
                    set_module_buffer_to_device(module, buffer_name, device_id)


def calculate_device_memory(device_map, tensor_index):
    device_memory = {}
    tensor_record = {}
    for tensor_name, device in device_map.items():
        if tensor_name in tensor_index:
            if device not in device_memory:
                device_memory[device] = 0
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                continue  # Skip duplicate tensors
            tensor_record[(offset, size)] = True
            device_memory[device] += tensor_index[tensor_name][1]
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return device_memory


def calculate_tensor_device_offsets(device_map, tensor_index):
    tensor_device_offsets = {}
    tensor_copy_chunks = {}
    device_offset = {}
    tensor_record = {}
    for tensor_name, device in device_map.items():
        if device not in tensor_device_offsets:
            tensor_device_offsets[device] = {}
            tensor_copy_chunks[device] = []
            device_offset[device] = 0
        if tensor_name in tensor_index:
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                tensor_device_offsets[device][tensor_name] = tensor_record[
                    (offset, size)
                ]
            else:
                tensor_record[(offset, size)] = device_offset[device]
                tensor_device_offsets[device][tensor_name] = device_offset[
                    device
                ]
                tensor_copy_chunks[device].append(
                    (offset, size, device_offset[device], 0)
                )
                device_offset[device] += size
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return tensor_device_offsets, tensor_copy_chunks


def get_total_parameter_size(module):
    total_param_size = 0
    for param in module.parameters():
        total_param_size += param.numel() * dtype_byte_size(param.dtype)
    return total_param_size


def get_parameter_size(model: nn.Module, param_path: str):
    # Split the parameter path by dots
    attributes = param_path.split(".")

    # Use reduce to traverse the model's attributes
    param = reduce(getattr, attributes, model)

    # Return the size of the parameter
    return param.numel() * dtype_byte_size(param.dtype)


def get_no_split_modules(model, no_split_modules_list, parent_name=""):
    no_split_modules = {}
    for name, submodule in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        module_class_name = submodule.__class__.__name__
        # If the module is a leaf module or in the no_split_modules_list, we don't split it # noqa: E501
        if (
            not list(submodule.children())
            or module_class_name in no_split_modules_list
        ):
            no_split_modules[full_name] = get_total_parameter_size(submodule)
            continue
        no_split_modules.update(
            get_no_split_modules(submodule, no_split_modules_list, full_name)
        )

    return no_split_modules


def get_tied_no_split_modules(model, no_split_modules):
    tied_parameters = find_tied_parameters(model)
    tied_modules = []
    for tied_param_group in tied_parameters:
        tied_module_group = []
        shared_size = None
        for tied_param in tied_param_group:
            param_size = get_parameter_size(model, tied_param)
            if shared_size is None:
                shared_size = param_size
            else:
                assert (
                    shared_size == param_size
                ), f"Parameter {tied_param} does not have the same size as the other parameters in the group"  # noqa: E501
            tied_module = None
            while "." in tied_param:
                tied_param = tied_param.rsplit(".", 1)[0]
                if tied_param in no_split_modules:
                    tied_module = tied_param
                    break
            if tied_module is None:
                raise ValueError(
                    f"Parameter {tied_param} is not in the no_split_modules list"  # noqa: E501
                )
            tied_module_group.append(tied_module)
        tied_modules.append((tied_module_group, shared_size))

    return tied_modules


def dtype_byte_size(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8


def get_quantization_config_and_type(precision: str):
    if precision == "int4":
        return BitsAndBytesConfig(load_in_4bit=True), "nf4"
    elif precision == "fp4":
        return BitsAndBytesConfig(load_in_4bit=True), "fp4"
    elif precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True), "nf4"
    elif precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), "nf4"
    else:
        raise ValueError(f"Unsupported quantization type: {precision}")


def get_quantization_fn(precision: str):
    if precision in ["fp4", "nf4", "int4"]:
        return bnb.functional.quantize_4bit
    elif precision == "int8":
        return bnb.functional.int8_vectorwise_quant
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def replace_linear_with_quantized(model, name, quantization):
    module_name = name[:-7] if name.endswith('.weight') else name
    
    # Get the full module directly
    module, _ = get_module_from_name(model, module_name)

    if isinstance(module, torch.nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        if quantization == "int8":
            new_layer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias,
                has_fp16_weights=True,
                threshold=6.0,
            )
        else:  # 4bit (fp4, nf4, int4)
            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias,
                compute_dtype=torch.float16,
                quant_type=quantization,
            )

        # Get parent module and child name for setting
        parent_path, child_name = module_name.rsplit(".", 1)
        parent_module, _ = get_module_from_name(model, parent_path)
        setattr(parent_module, child_name, new_layer)
        return new_layer

    print(f"Module {module_name} is type {type(module)}, not Linear")
    print("not quantized")
    return module
