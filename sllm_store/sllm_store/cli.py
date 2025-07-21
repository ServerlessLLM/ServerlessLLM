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

import asyncio
import importlib
import os
import shutil
import sys
import logging
import time
from typing import Optional
import torch
import click

from sllm_store.server import serve
from sllm_store.logger import init_logger
from sllm_store.utils import to_num_bytes
from sllm_store.transformers import save_model, save_lora, load_model, load_lora
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        storage_path: str = "./models",
        local_model_path: Optional[str] = None,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        import gc
        from tempfile import TemporaryDirectory

        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM

        # set the model storage path
        storage_path = os.getenv("STORAGE_PATH", storage_path)

        def _run_writer(input_dir, model_name):
            # load models from the input directory
            llm_writer = LLM(
                model=input_dir,
                download_dir=input_dir,
                dtype=torch_dtype,
                tensor_parallel_size=tensor_parallel_size,
                num_gpu_blocks_override=1,
                enforce_eager=True,
                max_model_len=1,
            )
            model_path = os.path.join(storage_path, model_name)
            model_executer = llm_writer.llm_engine.engine_core  # For engine V1
            # save the models in the ServerlessLLM format
            model_executer.save_serverless_llm_state(
                path=model_path, pattern=pattern, max_size=max_size
            )
            for file in os.listdir(input_dir):
                # Copy the metadata files into the output directory
                if os.path.splitext(file)[1] not in (
                    ".bin",
                    ".pt",
                    ".safetensors",
                ):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(model_path, file)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy(src_path, dest_path)
            del model_executer
            del llm_writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = local_model_path
                # download from huggingface
                if local_model_path is None:
                    input_dir = snapshot_download(
                        model_name,
                        cache_dir=cache_dir,
                        allow_patterns=[
                            "*.safetensors",
                            "*.bin",
                            "*.json",
                            "*.txt",
                        ],
                    )
                _run_writer(input_dir, model_name)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            shutil.rmtree(os.path.join(storage_path, model_name))
            raise RuntimeError(
                f"Failed to save {model_name} for vllm backend: {e}"
            ) from e


logger = init_logger(__name__)


def check_vllm():
    """Simple check if ServerlessLLM patch is applied."""
    try:
        # Check if the LoadFormat enum has SERVERLESS_LLM
        from vllm.config import LoadFormat

        if not hasattr(LoadFormat, "SERVERLESS_LLM"):
            return False

        # Check if ExecutorBase has the save method
        from vllm.executor.executor_base import ExecutorBase

        return hasattr(ExecutorBase, "save_serverless_llm_state")
    except ImportError:
        return False


@click.group()
def cli():
    """sllm-store CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8073, help="Port")
@click.option("--storage-path", default="./models", help="Storage path")
@click.option("--num-thread", default=4, help="Number of I/O threads")
@click.option(
    "--chunk-size", default="32MB", help="Chunk size, e.g., 4KB, 1MB, 1GB"
)
@click.option(
    "--mem-pool-size",
    default="4GB",
    help="Memory pool size, e.g., 1GB, 4GB, 1TB",
)
@click.option(
    "--disk-size", default="128GB", help="Disk size, e.g., 1GB, 4GB, 1TB"
)
@click.option(
    "--registration-required",
    default=False,
    help="Require registration before loading model",
)
def start(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    disk_size,
    registration_required,
):
    """Start the gRPC server."""
    # Convert the chunk size to bytes
    chunk_size = to_num_bytes(chunk_size)

    # Convert the memory pool size to bytes
    mem_pool_size = to_num_bytes(mem_pool_size)

    try:
        logger.info("Starting gRPC server")
        asyncio.run(
            serve(
                host=host,
                port=port,
                storage_path=storage_path,
                num_thread=num_thread,
                chunk_size=chunk_size,
                mem_pool_size=mem_pool_size,
                # disk size is not used
                # disk_size=disk_size,
                registration_required=registration_required,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)


@cli.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub",
)
@click.option("--backend", type=str, required=True, help="Backend")
@click.option("--adapter-name", type=str, help="Name of the LoRA adapter")
@click.option(
    "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
)
@click.option(
    "--local-model-path", type=str, help="Local path to the model snapshot"
)
@click.option(
    "--storage-path",
    default="./models",
    help="Local path to save the model",
)
def save(
    model_name,
    backend,
    adapter_name,
    tensor_parallel_size,
    local_model_path,
    storage_path,
):
    """
    Saves a model to the sllm-store's storage.

    This command is for adding new models to the sllm-store's local storage.
    """

    logger.info(
        f"Saving model {adapter_name if adapter_name else model_name} "
        f"to {storage_path}"
    )

    try:
        if backend == "vllm":
            if not check_vllm():
                logger.error(
                    "vLLM is not patched. Please run "
                    "`./sllm_store/vllm_patch/patch.sh` first."
                )
                sys.exit(1)
            downloader = VllmModelDownloader()
            downloader.download_vllm_model(
                model_name,
                "float16",
                tensor_parallel_size=tensor_parallel_size,
                storage_path=storage_path,
                local_model_path=local_model_path,
            )
        elif backend == "transformers":
            if adapter_name:
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                config.torch_dtype = torch.float16
                module = importlib.import_module("transformers")
                hf_model_cls = module.AutoModelForCausalLM
                base_model = hf_model_cls.from_config(
                    config,
                    trust_remote_code=True,
                ).to(config.torch_dtype)

                # Load a lora adapter from HuggingFace model hub
                model = PeftModel.from_pretrained(base_model, adapter_name)

                # Save the model to the local path
                model_path = os.path.join(storage_path, adapter_name)
                save_lora(model, model_path)
            else:
                # Load a model from HuggingFace model hub
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16
                )

                # Save the model to the local path
                model_path = os.path.join(storage_path, model_name)
                save_model(model, model_path)
        else:
            logger.error(f"Unsupported backend '{backend}'")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to save model {model_name}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(
        f"Model {adapter_name if adapter_name else model_name} "
        f"saved successfully to {storage_path}"
    )


@cli.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub",
)
@click.option("--backend", type=str, required=True, help="Backend")
@click.option("--adapter-name", type=str, help="Name of the LoRA adapter")
@click.option(
    "--precision",
    type=str,
    help="Precision of quantized model. Supports int8, fp4, and nf4",
)
@click.option(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path where model is saved",
)
def load(
    model_name,
    backend,
    adapter_name,
    precision,
    storage_path,
):
    """
    Loads a model from the sllm-store's storage.

    This command is for loading new models from the sllm-store's local storage.
    """

    logger.info(
        f"Loading model {adapter_name if adapter_name else model_name} "
        f"from {storage_path}"
    )

    quantization_config = None
    if precision:
        if precision == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "fp4":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif precision == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4"
            )
        else:
            logger.error(
                f"Unsupported precision: {precision}. "
                f"Supports int8, fp4, and nf4."
            )
            sys.exit(1)

    try:
        start_load_time = time.time()

        if backend == "vllm":
            from vllm import LLM

            if not check_vllm():
                logger.error(
                    "vLLM is not patched. Please run "
                    "`./sllm_store/vllm_patch/patch.sh` first."
                )
                sys.exit(1)
            model_full_path = os.path.join(storage_path, model_name)
            llm = LLM(
                model=model_full_path,
                load_format="serverless_llm",
                dtype="float16",
            )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )

            example_inferences("vllm", model=llm)

        elif backend == "transformers":
            # warm up the GPU
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                torch.ones(1).to(f"cuda:{i}")
                torch.cuda.synchronize()
            if adapter_name:
                model = load_model(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    storage_path=storage_path,
                    fully_parallel=True,
                )

                model = load_lora(
                    model,
                    adapter_name,
                    adapter_path=adapter_name,
                    device_map="auto",
                    storage_path=storage_path,
                    torch_dtype=torch.float16,
                )
            else:
                model = load_model(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    storage_path=storage_path,
                    fully_parallel=True,
                    quantization_config=quantization_config,
                )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )

            example_inferences(
                "transformers",
                model=model,
                model_name=model_name,
                adapter_name=adapter_name,
            )

        else:
            logger.error(f"Unsupported backend '{backend}'")
            sys.exit(1)

    except Exception as e:
        logger.error(
            f"Failed to load model or perform inference: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info(
        f"Model {adapter_name if adapter_name else model_name} "
        f"loaded successfully from {storage_path}"
    )


def example_inferences(backend, model=None, model_name=None, adapter_name=None):
    if backend == "vllm":
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = model.generate(prompts, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    elif backend == "transformers":
        from pathlib import Path

        parts = Path(model_name).parts
        if len(parts) >= 2:
            model_name = f"{parts[-2]}/{parts[-1]}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(
            "cuda"
        )
        generate_kwargs = {}
        if adapter_name:
            generate_kwargs["adapter_names"] = [adapter_name]
        outputs = model.generate(**inputs, **generate_kwargs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        logger.error(f"Unsupported backend '{backend}' for example inferences")
        sys.exit(1)


# Entry point for the 'sllm-store' CLI
def main():
    cli()
