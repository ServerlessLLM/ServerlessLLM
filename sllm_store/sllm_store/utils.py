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
from accelerate.utils import find_tied_parameters
from torch import nn
import re


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


def to_num_bytes(value: str) -> int:
    """
    Convert a string representing a data size to its equivalent number of bytes.

    The input must strictly follow the format:
        <number><unit>

    - <number>: A positive integer.
    - <unit>: One of the following case-sensitive units:
        B, KB, MB, GB, TB, PB, EB, ZB, YB

    No leading, trailing, or middle spaces or other characters are allowed.

    Examples:
        "1GB"  -> 1073741824
        "512MB" -> 536870912

    Args:
        value (str): The string to convert.

    Returns:
        int: The equivalent number of bytes.

    Raises:
        ValueError: If the input format is incorrect.
    """
    # Define the regular expression pattern for validation
    pattern = r"^(\d+)(B|KB|MB|GB|TB|PB|EB|ZB|YB)$"
    match = re.fullmatch(pattern, value)

    if not match:
        error_message = (
            "Invalid format. The input must be a positive integer "
            "followed immediately by a unit "
            "(B, KB, MB, GB, TB, PB, EB, ZB, YB), case sensitive, "
            "with no spaces or other characters."
        )
        raise ValueError(error_message)

    number_str, unit = match.groups()
    number = int(number_str)

    # Define the multiplier for each unit
    unit_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "EB": 1024**6,
        "ZB": 1024**7,
        "YB": 1024**8,
    }

    bytes_value = number * unit_multipliers[unit]
    return bytes_value
