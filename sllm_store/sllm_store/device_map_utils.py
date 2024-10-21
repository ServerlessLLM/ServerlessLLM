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
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from sllm_store.logger import init_logger

logger = init_logger(__name__)

DeviceMapType = Union[
    str, Dict[str, Union[int, str, torch.device]], int, torch.device
]


def _transform_device_map_to_dict(
    device_map: DeviceMapType,
) -> Dict[str, Union[int, torch.device]]:
    """Transforms the device_map to a dictionary if it is not already a dictionary."""  # noqa: E501

    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif isinstance(device_map, str) and device_map not in [
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
    ]:
        try:
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "  # noqa: E501
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."  # noqa: E501
            ) from None
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "  # noqa: E501
            )
        else:
            device_map = {"": device_map}
    return device_map


def _expand_tensor_name(
    device_map: DeviceMapType, tensor_names: List[str]
) -> Dict[str, Union[int, torch.device]]:
    if "" in device_map and len(device_map) != 1:
        raise RuntimeError(
            f"Device map {device_map} is invalid. If you want to specify the default device, use key ''."  # noqa: E501
        )

    expanded_device_map = {}
    for tensor_or_module, device_id in device_map.items():
        if tensor_or_module == "":
            return {k: device_id for k in tensor_names}

        for name in tensor_names:
            # TODO: use trie to speed up prefix match
            if name.startswith(tensor_or_module):
                expanded_device_map[name] = device_id

    return expanded_device_map


def _compute_device_placement_from_map(
    model: torch.nn.Module,
    device_map: DeviceMapType,
    target_dtype: torch.dtype,
    max_memory: Optional[int] = None,
) -> Dict[str, Union[int, torch.device]]:
    """
    Computes the device placement for the model based on the device_map.
    """

    if isinstance(device_map, str):
        no_split_modules = model._get_no_split_modules(device_map)
        if device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "  # noqa: E501
                "'sequential'."
            )

        device_map_kwargs = {"no_split_module_classes": no_split_modules}

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                dtype=target_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        else:
            max_memory = get_max_memory(max_memory)
        device_map_kwargs["max_memory"] = max_memory

        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(
            model, dtype=target_dtype, **device_map_kwargs
        )

    return device_map


def _compute_device_placement_from_map_fast(
    no_split_modules: Dict[str, int],
    tied_modules: List[Tuple[List[str], int]],
    device_map: DeviceMapType,
) -> Dict[str, Union[int, torch.device]]:
    """
    Computes the device placement for no split modules based on the device_map.
    """

    if isinstance(device_map, str):
        if device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "  # noqa: E501
                "'sequential'."
            )

        max_memory = get_max_memory()
        # we don't support loading to cpu
        max_memory.pop("cpu")

        # tied modules are treated as a single module
        for tied_groups, shared_size in tied_modules:
            modules = list(no_split_modules.keys())
            for module in modules:
                if module in tied_groups:
                    tied_size = no_split_modules[module]
                    for m in modules:
                        if m in tied_groups and m != module:
                            tied_size += no_split_modules[m] - shared_size
                            no_split_modules.pop(m)
                    no_split_modules[module] = tied_size
                    break

        while next(iter(no_split_modules.values())) > next(
            iter(max_memory.values())
        ):
            device_id, memory = max_memory.popitem()
            logger.warning(
                f"Device {device_id} has insufficient memory {memory} for the first module."  # noqa: E501
            )

        total_size = sum(no_split_modules.values())

        if total_size > sum(max_memory.values()):
            raise RuntimeError(
                "The total size of the model is greater than the maximum memory available."  # noqa: E501
            )

        placement = None
        if device_map == "auto" or device_map == "balanced":
            # 1. use dynamic programming to find the best balanced placement
            placement = _get_balanced_placement(no_split_modules, max_memory)
        elif device_map == "balanced_low_0":
            # 2.1 decide minimum modules on device 0
            # 2.2 use dynamic programming to find the best balanced placement for the rest on other devices # noqa: E501
            raise NotImplementedError
        else:
            # 3. use greedy algorithm to find the best sequential placement
            placement = _get_sequential_placement(no_split_modules, max_memory)

        if placement is None:
            raise RuntimeError(
                "Failed to find a valid placement for the model."
            )

        # reassign tied modules to the same device
        for tied_groups, shared_size in tied_modules:  # noqa: B007
            modules = list(placement.keys())
            for module in modules:
                if module in tied_groups:
                    for m in tied_groups:
                        if m != module:
                            placement[m] = placement[module]

        return placement

    return device_map


def _get_balanced_placement(
    module_size: Dict[str, int],
    device_memory: Dict[torch.device, int],
) -> Dict[str, Union[int, torch.device]]:
    """
    Computes the balanced placement for no split modules based on the given device_memory.
    """  # noqa: E501

    module_names = list(module_size.keys())
    assert len(module_names) > 0 and len(module_names) >= len(device_memory)

    length = len(module_names)
    n = len(device_memory)
    if n <= 0 or length == 0:
        logger.error("No device memory or no modules to place.")
        return None

    # "balanced" means that the gap between the sums of the partitions is minimized # noqa: E501
    # Initialize DP table
    dp = [
        [[float("inf"), float("inf"), 0, []] for _ in range(n + 1)]
        for _ in range(length + 1)
    ]
    dp[0][0] = [
        0,
        float("inf"),
        0,
        [[] for _ in range(n)],
    ]  # [gap, min_size, max_size, partitions]

    # Fill DP table
    for i in range(1, length + 1):
        for k in range(1, n + 1):
            for j in range(i):
                current_partition = module_names[j:i]
                current_size = sum(
                    [module_size[module] for module in current_partition]
                )
                # check if this partition can fit in the device memory
                if current_size > device_memory[k - 1]:
                    # print(f"Partition {current_partition} is too large for device {k-1}, memory {device_memory[k-1]}") # noqa: E501
                    continue
                if dp[j][k - 1][0] < float("inf"):
                    if dp[j][k - 1][1] == float("inf"):
                        max_gap = 0
                    else:
                        max_gap = max(
                            abs(dp[j][k - 1][1] - current_size),
                            abs(dp[j][k - 1][2] - current_size),
                        )
                    if max_gap < dp[i][k][0]:
                        dp[i][k][0] = max_gap
                        dp[i][k][1] = min(dp[j][k - 1][1], current_size)
                        dp[i][k][2] = max(dp[j][k - 1][2], current_size)
                        dp[i][k][-1] = dp[j][k - 1][-1][:]  # Copy partitions
                        dp[i][k][-1][k - 1] = (
                            current_partition  # Update last partition
                        )

    # Result
    result = dp[length][n][-1]
    result_device_map = {}
    for i, partition in enumerate(result):
        for module in partition:
            result_device_map[module] = i
    return result_device_map


def _get_sequential_placement(
    module_size: Dict[str, int],
    device_memory: Dict[torch.device, int],
) -> Dict[str, Union[int, torch.device]]:
    """
    Computes the sequential placement for no split modules based on the given device_memory.
    """  # noqa: E501

    module_names = list(module_size.keys())
    assert len(module_names) > 0 and len(module_names) >= len(device_memory)

    length = len(module_names)
    n = len(device_memory)
    if n <= 0 or length == 0:
        return None

    result_device_map = {}
    current_device = 0
    for module, size in module_size.items():
        while current_device < n and size > device_memory[current_device]:
            current_device += 1
        if size <= device_memory[current_device]:
            result_device_map[module] = current_device
        else:
            raise RuntimeError(
                f"Module {module} is too large for device {current_device}, memory {device_memory[current_device]}"  # noqa: E501
            )

    return result_device_map
