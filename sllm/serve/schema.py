from pydantic import BaseModel, Field
from typing import Dict, List, Any, Literal, Optional, Union
from datetime import datetime

# =============================================================================
# Heartbeat (Worker -> Head)
# =============================================================================
class HeartbeatPayload(BaseModel):
    node_id: str
    ip_address: str
    instances_on_device: Dict[str, List[str]]
    hardware_info: Dict[str, Any]


# =============================================================================
# Model Deployment (User -> Head)
# =============================================================================
class AutoScalingConfig(BaseModel):
    min_instances: int
    max_instances: int
    target: float
    keep_alive: int

class ModelDeploymentRequest(BaseModel):
    model: str
    backend: Literal["vllm"]
    backend_config: Dict[str, Any]
    auto_scaling_config: AutoScalingConfig


# =============================================================================
# Start Instance (Head -> Worker)
# =============================================================================
class StartInstanceRequest(BaseModel):
    instance_id: str
    model_config: ModelDeploymentRequest


# =============================================================================
# Inference Request (User -> Head)
# =============================================================================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

u
# =============================================================================
# Fine-Tuning Job (User -> Head)
# =============================================================================
class DatasetConfig(BaseModel):
    dataset_source: str
    hf_dataset_name: str
    split: str

class LoraConfig(BaseModel):
    r: int
    lora_alpha: int
    lora_dropout: float
    task_type: str

class TrainingConfig(BaseModel):
    auto_find_batch_size: bool
    num_train_epochs: int
    learning_rate: float
    use_cpu: bool

class FineTuningRequest(BaseModel):
    model: str
    ft_backend: Literal["peft"]
    output_dir: str
    dataset_config: DatasetConfig
    lora_config: LoraConfig
    training_config: TrainingConfig


# =============================================================================
# Internal Invoke (Head -> Worker)
# =============================================================================
class InvokeRequest(BaseModel):
    instance_id: str
    payload: ChatCompletionRequest


# =============================================================================
# KV Store Schemas (Head -> Head)
# =============================================================================
# Worker
class GpuDetail(BaseModel):
    name: str
    total_memory: str

class GpuInfo(BaseModel):
    load: float
    memory_free: Union[int, str]
    memory_used: Union[int, str]

class HardwareInfo(BaseModel):
    pcie_bandwidth: int
    disk_total_space: Union[int, str]
    disk_write_bandwidth: Union[int, str]
    disk_read_bandwidth: Union[int, str]
    gpus: List[GpuDetail]
    cpu_percent: float
    gpu_info: GpuInfo

class Worker(BaseModel):
    node_id: str
    node_ip: str
    hardware_info: HardwareInfo
    instances_alive: Dict[str, List[str]]
    last_heartbeat_time: datetime

# Model
class BackendConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = None
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None
    hf_model_class: Optional[str] = None
    quantization_config: Optional[Dict[str, Any]] = None
    enable_lora: Optional[bool] = None
    lora_adapters: Optional[Dict[str, Any]] = None

class AutoScalingConfig(BaseModel):
    metric: Optional[str] = None
    target: Optional[int] = None
    min_instances: Optional[int] = None
    max_instances: Optional[int] = None
    keep_alive: Optional[int] = None

class Model(BaseModel):
    model_name: str
    backend: Literal["vllm", "transformers", "sglang"]
    num_gpus: int
    backend_config: Optional[BackendConfig] = Field(default_factory=BackendConfig)
    auto_scaling_config: Optional[AutoScalingConfig] = Field(default_factory=AutoScalingConfig)
    instances: List[str] = Field(default_factory=list)
