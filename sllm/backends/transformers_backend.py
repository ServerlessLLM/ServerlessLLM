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

import asyncio
import json
import os
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import peft
import torch
import torch.nn.functional as F
import transformers
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.generation.streamers import BaseStreamer

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.logger import init_logger
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
        if self.status == BackendStatus.DELETING:
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")

    def get(self):
        return deepcopy(self.intermediate)

    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []


# Request models for FastAPI
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    task_id: Optional[str] = None
    request_id: Optional[str] = None
    lora_adapter_name: Optional[str] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    task_id: Optional[str] = None
    request_id: Optional[str] = None
    lora_adapter_name: Optional[str] = None


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]
    task_instruct: Optional[str] = ""
    max_length: Optional[int] = 4096


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.backend_config = backend_config
        logger.info(f"Initializing TransformersBackend for {model_name}")
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None
        self.current_tokens = None
        self.inf_status = None

        # Port allocation and server setup
        self.port = backend_config.get("port") or allocate_backend_port(
            "transformers"
        )
        self.host = backend_config.get("host", "0.0.0.0")
        self.base_url = f"http://{self.host}:{self.port}"

        # Update backend_config with allocated port for reference
        self.backend_config["port"] = self.port
        self.backend_config["host"] = self.host

        # FastAPI app
        self.app = FastAPI()
        self.server = None
        self.server_task = None

        # Set up routes
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes"""

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            logger.info(f"Chat completions request: task_id={request.task_id}")
            request_dict = {
                "model": request.model or self.model_name,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "task_id": request.task_id,
                "lora_adapter_name": request.lora_adapter_name,
            }
            return await self.generate(request_dict)

        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            logger.info(f"Completions request: task_id={request.task_id}")
            request_dict = {
                "model": request.model or self.model_name,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "task_id": request.task_id,
                "lora_adapter_name": request.lora_adapter_name,
            }
            return await self.generate(request_dict)

        @self.app.post("/v1/embeddings")
        async def embeddings(request: EmbeddingRequest):
            request_dict = {
                "model": request.model or self.model_name,
                "input": request.input,
                "task_instruct": request.task_instruct,
                "max_length": request.max_length,
            }
            return await self.encode(request_dict)

        @self.app.get("/get_current_tokens")
        async def get_current_tokens():
            tokens = await self.get_current_tokens()
            return {"tokens": tokens}

        @self.app.post("/resume_kv_cache")
        async def resume_kv_cache(request: Dict[str, Any]):
            request_datas = request.get("request_datas", [])
            await self.resume_kv_cache(request_datas)
            return {"status": "ok"}

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            logger.info(f"Starting backend for {self.model_name}")

            try:
                # Initialize the model
                await self._load_model()

                # Start FastAPI server
                await self._start_server()

                self.status = BackendStatus.RUNNING
                logger.info(f"Backend started on {self.base_url}")

            except Exception as e:
                logger.error(f"Failed to start Transformers backend: {e}")
                await self._cleanup()
                raise

    async def _load_model(self):
        """Load the transformers model and tokenizer"""
        device_map = self.backend_config.get("device_map", "auto")
        torch_dtype = self.backend_config.get("torch_dtype", "float16")

        # Convert string to torch dtype
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, torch.float16)

        hf_model_class = self.backend_config.get(
            "hf_model_class", "AutoModelForCausalLM"
        )
        if hf_model_class is None:
            raise ValueError("hf_model_class cannot be None")

        quantization_config = self.backend_config.get(
            "quantization_config", None
        )

        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join("transformers", self.model_name)
        full_model_path = os.path.join(storage_path, model_path)

        # Validate model files exist
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Transformers model not found at {full_model_path}")

        # Load model using sllm_store
        try:
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
                quantization_config=quantization_config,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load transformers model: {e}")

        # Load tokenizer
        tokenizer_path = os.path.join(
            storage_path, "transformers", self.model_name, "tokenizer"
        )
        try:
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize inference status
        self.inf_status = InferenceStatus(self.status)

        logger.info(f"Model loaded: {self.model_name}")

    async def _start_server(self):
        """Start the FastAPI server in a separate task"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="error",  # Reduce uvicorn logging
        )
        self.server = uvicorn.Server(config)

        # Start server in background task
        self.server_task = asyncio.create_task(self.server.serve())

        # Wait a moment for server to start
        await asyncio.sleep(1)

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

    async def encode(self, request_data: Optional[Dict[str, Any]]):
        async with self.status_lock:
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

        model_name = request_data.get("model", self.model_name)
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

    async def generate(self, request_data: Optional[Dict[str, Any]]):
        task_id = (
            request_data.get("task_id", "unknown")
            if request_data
            else "unknown"
        )
        logger.info(f"Generate request: task_id={task_id}")

        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                logger.error(f"Model not initialized: task_id={task_id}")
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", self.model_name)
        # Handle both chat completions (messages) and completions (prompt)
        if "messages" in request_data:
            messages = request_data.get("messages", [])
        else:
            messages = []
            # For completions, we'll handle the prompt directly later
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        lora_adapter_name = request_data.get("lora_adapter_name", None)

        # Generate prompt based on request type
        if "messages" in request_data:
            # Chat completions format
            logger.debug(
                f"Chat completion: task_id={task_id}, messages={len(messages)}"
            )
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
        else:
            # Completions format - direct prompt
            prompt = request_data.get("prompt", "")
            logger.debug(
                f"Completion: task_id={task_id}, prompt_len={len(prompt)}"
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "streamer": self.inf_status,
        }

        if lora_adapter_name:
            if (
                not hasattr(self.model, "peft_config")
                or lora_adapter_name not in self.model.peft_config
            ):
                return {"error": f"LoRA adapter {lora_adapter_name} not found"}
            generate_kwargs["adapter_names"] = [lora_adapter_name]

        inputs = self._tokenize(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate response
        logger.info(
            f"Generation start: task_id={task_id}, tokens={prompt_tokens}"
        )
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
        except DeletingException:
            logger.info(f"Aborting request due to shutdown: task_id={task_id}")
            output_tokens = self.inf_status.get()
            self.inf_status.delete()
            return {
                "preempted": "True",
                "current_output": output_tokens,
                "completed_tokens": len(output_tokens[0]) - prompt_tokens,
            }
        except Exception as e:
            logger.error(f"Generation failed: task_id={task_id}, error={e}")
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
            if "messages" in request_data:
                # Chat completions format
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
            else:
                # Completions format
                response = {
                    "id": f"cmpl-{uuid.uuid4()}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": output_text,
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

            logger.debug(
                f"Generation complete: task_id={task_id}, tokens={completion_tokens}"
            )
            self.inf_status.delete()
            return response

    async def fine_tuning(self, request_data: Optional[Dict[str, Any]]):
        async with self.status_lock:
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
        logger.info(f"Fine-tuning completed: {lora_save_path}")

        response = {
            "model": self.model_name,
            "lora_save_path": lora_save_path,
        }

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

    async def load_lora_adapter(self, lora_name: str, lora_path: str):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        if (
            hasattr(self.model, "peft_config")
            and lora_name in self.model.peft_config
        ):
            logger.info(f"LoRA adapter {lora_name} already loaded")
            return

        lora_path = os.path.join("transformers", lora_path)
        storage_path = os.getenv("STORAGE_PATH", "./models")
        device_map = self.backend_config.get("device_map", "auto")
        torch_dtype = self.backend_config.get("torch_dtype", "float16")

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, torch.float16)

        self.model = load_lora(
            self.model,
            lora_name,
            lora_path,
            device_map=device_map,
            storage_path=storage_path,
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded LoRA adapter {lora_name} from {lora_path}")

    async def _cleanup(self):
        """Clean up resources"""
        if self.server_task and not self.server_task.done():
            if self.server:
                self.server.should_exit = True
            try:
                await asyncio.wait_for(self.server_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Server task did not stop gracefully")
                self.server_task.cancel()

        if self.model is not None:
            del self.model
            self.model = None

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING
            if self.inf_status:
                self.inf_status.status = BackendStatus.DELETING

        # Wait for ongoing requests to finish
        while self.inf_status and len(self.inf_status.get()) > 0:
            await asyncio.sleep(1)

        await self._cleanup()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING

        while self.inf_status and len(self.inf_status.get()) > 0:
            await asyncio.sleep(1)
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        if self.inf_status:
            status = self.inf_status.get()
            logger.debug(f"Current tokens: {status}")
            return status
        return []

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume KV cache for given request token sequences."""
        if self.status != BackendStatus.RUNNING:
            return

        with torch.no_grad():
            device = self.model.device
            input_ids = torch.tensor(request_datas).to(device)
            output = self.model.generate(
                input_ids,
                past_key_values=self.past_key_values,
                max_new_tokens=1,
                return_dict_in_generate=True,
                return_legacy_cache=True,
            )
            self.past_key_values = output.past_key_values
            self.current_tokens = output.sequences
