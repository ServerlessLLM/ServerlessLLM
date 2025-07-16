from pydantic import BaseModel, Field
from typing import Dict, List, Any

class HeartbeatPayload(BaseModel):
    node_id: str
    ip_address: str
    instances_on_device: Dict[str, List[str]] = Field(default_factory=dict)
    hardware_info: Dict[str, Any]
