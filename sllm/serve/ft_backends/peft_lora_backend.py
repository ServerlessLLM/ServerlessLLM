# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2025                                       #
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
import json
import os
import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import peft
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from sllm.serve.ft_backends.backend_utils import (
    FineTuningBackendStatus,
    SllmFineTuningBackend,
)
from sllm.serve.logger import init_logger
from sllm_store.transformers import save_lora

logger = init_logger(__name__)


class DeletingException(Exception):
    pass


class FineTuningStatus:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.state = FineTuningBackendStatus.PENDING
        self.logs = []
        self.metrics = {}  # e.g., {"epoch": 1, "loss": 0.231}
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.state = FineTuningBackendStatus.RUNNING
            self.updated_at = time.time()

    def update_metrics(self, epoch=None, loss=None):
        with self.lock:
            if epoch is not None:
                self.metrics["epoch"] = epoch
            if loss is not None:
                self.metrics["loss"] = loss
            self.updated_at = time.time()

    def log(self, message: str):
        with self.lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.logs.append(f"[{timestamp}] {message}")
            self.updated_at = time.time()

    def complete(self):
        with self.lock:
            self.state = FineTuningBackendStatus.COMPLETED
            self.updated_at = time.time()

    def fail(self, reason=""):
        with self.lock:
            self.state = FineTuningBackendStatus.FAILED
            self.logs.append(f"Job failed: {reason}")
            self.updated_at = time.time()

    def abort(self):
        with self.lock:
            self.state = FineTuningBackendStatus.ABORTED
            self.logs.append("Job aborted by user")
            self.updated_at = time.time()

    def get_status(self):
        with self.lock:
            return {
                "job_id": self.job_id,
                "state": self.state.value,
                "metrics": self.metrics,
                "logs": self.logs[-10:],  # return last 10 logs
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


class PeftLoraBackend(SllmFineTuningBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing PeftLoraBackend for {model_name} with config: {backend_config}"
        )
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.status: FineTuningBackendStatus = (
            FineTuningBackendStatus.UNINITIALIZED
        )
        self.ft_status = FineTuningStatus(self.status)
        self.status_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != FineTuningBackendStatus.UNINITIALIZED:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            hf_model_class = self.backend_config.get("hf_model_class", None)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            if hf_model_class is None:
                logger.error(
                    f"hf_model_class cannot be None. Please provide a valid model class"
                )
                raise ValueError(
                    "hf_model_class cannot be None. Please provide a valid model class"
                )

            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join("transformers", self.model_name)
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
            )
            tokenizer_path = os.path.join(
                storage_path, "transformers", self.model_name, "tokenizer"
            )
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to load from system's cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path
                )
            self.status = FineTuningBackendStatus.RUNNING

    def _load_dataset(
        self,
        dataset_config: Optional[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        dataset_source = dataset_config.get("dataset_source")
        hf_dataset_name = dataset_config.get("hf_dataset_name")
        tokenization_field = dataset_config.get("tokenization_field")
        split = dataset_config.get("split", None)
        data_files = dataset_config.get("data_files", None)
        extension_type = dataset_config.get("extension_type")

        if dataset_source not in {"hf_hub", "local"}:
            logger.error(
                "Invalid 'dataset_source'. Must be 'hf_hub' or 'local'."
            )
            raise ValueError(
                "Invalid 'dataset_source'. Must be 'hf_hub' or 'local'."
            )

        if dataset_source == "hf_hub":
            if not hf_dataset_name:
                logger.error(
                    "hf_dataset_name must be provided in the dataset configuration."
                )
                raise ValueError(
                    "hf_dataset_name must be provided in the dataset configuration."
                )
            data = load_dataset(hf_dataset_name, split=split)
            data = data.map(
                lambda samples: tokenizer(samples[tokenization_field]),
                batched=True,
            )
            return data
        elif dataset_source == "local":
            if not extension_type:
                logger.error(
                    "extension_type must be provided in the dataset configuration."
                )
                raise ValueError(
                    "extension_type must be provided in the dataset configuration."
                )
            if not data_files:
                logger.error(
                    "data_files must be provided in the dataset configuration."
                )
                raise ValueError(
                    "data_files must be provided in the dataset configuration."
                )
            data = load_dataset(
                extension_type, data_files=data_files, split=split
            )
            data = data.map(
                lambda samples: tokenizer(samples[tokenization_field]),
                batched=True,
            )
            return data

    def fine_tuning(self, request_data: Optional[Dict[str, Any]]):
        logger.debug("Receive fine tune request on peft lora backend")
        with self.status_lock:
            if self.status != FineTuningBackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        dataset_config = request_data.get("dataset_config")
        try:
            dataset = self._load_dataset(dataset_config, self.tokenizer)
        except ValueError as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": str(e)}

        lora_config = request_data.get("lora_config")
        try:
            lora_config = LoraConfig(**lora_config)
        except TypeError as e:
            logger.error(f"Failed to load lora_config: {e}")
            raise e
        peft_model = get_peft_model(self.model, lora_config)

        training_config = request_data.get("training_config")
        storage_path = os.getenv("STORAGE_PATH", "./models")
        output_dir = request_data.get(
            "output_dir", f"ft_{self.model_name}_adapter"
        )
        lora_save_path = os.path.join(
            storage_path,
            "transformers",
            output_dir,
        )
        try:
            training_args = TrainingArguments(
                output_dir=lora_save_path, **training_config
            )
        except TypeError as e:
            logger.error(f"Failed to load training_config: {e}")
            raise e
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        trainer.train()

        save_lora(peft_model, lora_save_path)
        logger.info(
            f"Fine-tuning completed. LoRA adapter and config saved to {lora_save_path}"
        )

        response = {
            "model": self.model_name,
            "lora_save_path": lora_save_path,
        }

        return response

    def shutdown(self):
        """Abort all requests and shutdown the backend."""
        with self.status_lock:
            if self.status == FineTuningBackendStatus.DELETING:
                return
            self.status = FineTuningBackendStatus.DELETING
            if self.ft_status:
                self.ft_status.status = FineTuningBackendStatus.DELETING

        while self.ft_status and len(self.ft_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= FineTuningBackendStatus.STOPPING.value:
                return
            self.status = FineTuningBackendStatus.STOPPING
        while self.ft_status and len(self.ft_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()
