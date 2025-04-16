# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.generation.streamers import BaseStreamer

from sllm.serve.backends.backend_utils import BackendStatus, SllmBackend
from sllm.serve.logger import init_logger
from sllm_store.transformers import load_lora, load_model, save_lora

logger = init_logger(__name__)


class DeletingException(Exception):
    pass


class InferenceStatus(BaseStreamer):
    def __init__(self, status: BackendStatus):
        super().__init__()
        self.status = status
        self.intermediate = []

    def put(self, value):
        value = value.tolist()
        if not self.intermediate:
            self.intermediate = value
        else:
            # NOTE: This does not support in-flight batching
            # or dynamic batch size
            for i, v in enumerate(value):
                self.intermediate[i].append(v)
        logger.warning(f"Intermediate output: {self.intermediate}")
        if self.status == BackendStatus.DELETING:
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")

    def get(self):
        return deepcopy(self.intermediate)

    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend for {model_name} with config: {backend_config}"
        )
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.enable_lora = backend_config.get("enable_lora", False)
        self.lora_adapters = backend_config.get("lora_adapters", [])
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.inf_status = InferenceStatus(self.status)
        self.status_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
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
            if self.enable_lora and self.lora_adapters:
                for i, (lora_name, lora_path) in enumerate(
                    self.lora_adapters.items()
                ):
                    lora_path = os.path.join(
                        storage_path, "transformers", lora_path
                    )
                    self.model = load_lora(
                        self.model,
                        lora_name,
                        lora_path,
                        device_map,
                        storage_path,
                    )
            self.status = BackendStatus.RUNNING

    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    def _encoder_tokenize(self, query: str, max_length: int):
        return self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")

    def encode(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        def last_token_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
            )
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery: {query}"

        model_name = request_data.get("model", "dummy-model")
        task_instruct = request_data.get("task_instruct", "")
        max_length = request_data.get("max_length", 4096)
        query = request_data.get("input", [])

        if not query:
            return {"error": "Missing query in request data"}

        query = [get_detailed_instruct(task_instruct, q) for q in query]

        batch_dict = self._encoder_tokenize(query, max_length)
        with torch.no_grad():
            output = self.model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(
            output.hidden_states[-1], batch_dict["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_tokens = sum([len(self.tokenizer.tokenize(q)) for q in query])
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "query_tokens": query_tokens,
                "total_tokens": query_tokens,
            },
        }

        return response

    def generate(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        lora_adapter_name = request_data.get("lora_adapter_name", None)

        # Combine messages to form the prompt
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            prompt = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in messages
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        inputs = self._tokenize(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate response
        try:
            with torch.no_grad():
                # TODO: check current active adapters.
                if self.enable_lora and lora_adapter_name is None:
                    self.model.disable_adapters()
                elif self.enable_lora and lora_adapter_name:
                    self.model.set_adapter(lora_adapter_name)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    streamer=self.inf_status,
                )
        except DeletingException:
            logger.info("Backend is shutting down. Aborting request")
            output_tokens = self.inf_status.get()
            self.inf_status.delete()
            return {
                "preempted": "True",
                "current_output": output_tokens,
                "completed_tokens": len(output_tokens[0]) - prompt_tokens,
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens
            # FIXME: consider corner case when max_tokens is reached
            finish_reason = (
                "stop" if completion_tokens < max_tokens else "length"
            )

            # Generate response compatible with OpenAI's API
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            self.inf_status.delete()

            return response

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
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset_config = request_data.get("dataset_config")
        try:
            dataset = self._load_dataset(dataset_config, tokenizer)
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
                tokenizer, mlm=False
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
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING
            if self.inf_status:
                self.inf_status.status = BackendStatus.DELETING

        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()

    def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        status = self.inf_status.get()
        logger.info(f"Current tokens: {status}")
        return status

    def resume_kv_cache(self, request_datas):
        logger.info(f"Resuming cache for {request_datas}")
        with torch.no_grad():
            device = self.model.device
            input_ids = torch.tensor(request_datas).to(device)
            logger.info(input_ids)
            output = self.model.generate(
                input_ids,
                past_key_values=self.past_key_values,
                max_new_tokens=1,
                return_dict_in_generate=True,
                return_legacy_cache=True,
            )
            self.past_key_values = output.past_key_values
            self.current_tokens = output.sequences
        logger.info(f"Resumed {len(self.past_key_values[0][0][0][0])} tokens")

    def resume_generate(
        self, request_data: Optional[Dict[str, Any]], current_output
    ):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)

        # Combine messages to form the prompt
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            prompt = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in messages
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        inputs = self._tokenize(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate response
        try:
            with torch.no_grad():
                device = self.model.device
                current_output = torch.tensor(current_output).to(device)
                if len(current_output[0]) < len(self.current_tokens[0]):
                    current_output = self.current_tokens
                outputs = self.model.generate(
                    current_output,
                    past_key_values=self.past_key_values,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    streamer=self.inf_status,
                )
        except DeletingException:
            logger.error("Backend is shutting down. Aborting request")
            raise DeletingException("Backend is shutting down")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens
            # FIXME: consider corner case when max_tokens is reached
            finish_reason = (
                "stop" if completion_tokens < max_tokens else "length"
            )

            # Generate response compatible with OpenAI's API
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            self.inf_status.delete()

            return response
