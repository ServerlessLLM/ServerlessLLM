import os
from typing import Optional, Union, Dict

from serverless_llm_store.torch import save_dict, load_dict
from serverless_llm_store.device_map_utils import DeviceMapType

import torch
from torch import nn


def save_model(model: nn.Module, model_path: str, rank: int, key_list):
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

    # move all tensors to CPU
    saved_state_dict = {}
    for key, tensor in model_state_dict.items():
        if key in key_list:
          saved_state_dict[key] = tensor.cpu().contiguous()
        
    save_dict(saved_state_dict, os.path.join(model_path, f"rank_{rank}"))


def load_model(
    model: nn.Module,
    model_path: Optional[Union[str, os.PathLike]],
    rank: int,
    key_list,
    device_map: DeviceMapType,
    storage_path: str = "./models",
):
  model_path = os.path.join(model_path, f"rank_{rank}")
  for key, param in model.named_parameters(recurse=True):
      if key in key_list:
          param.data = torch.empty(1)
  
  sllm_state_dict = load_dict(model_path, device_map, storage_path)
  
  for key, param in model.named_parameters(recurse=True):
      if key in key_list:
          tensor = sllm_state_dict[key]
          param.data = tensor