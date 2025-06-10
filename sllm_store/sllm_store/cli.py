import asyncio
import importlib
import os
import sys
import logging
import time
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
from vllm import LLM, SamplingParams
from peft import PeftModel

logger = init_logger(__name__)


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
    # Convert the chunk size to bytes
    chunk_size = to_num_bytes(chunk_size)

    # Convert the memory pool size to bytes
    mem_pool_size = to_num_bytes(mem_pool_size)

    """Start the gRPC server"""
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
@click.option(
    "--backend", type=str, required=True, default="vllm", help="Backend"
)
@click.option(
    "--adapter", is_flag=True, help="Indicate if the model is an adapter"
)
@click.option("--adapter-name", type=str, help="Name of the LoRA adapter")
@click.option(
    "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
)
@click.option(
    "--local-model-path", type=str, help="Local path to the model snapshot"
)
@click.option(
    "--storage-path",
    required=True,
    default="./models",
    help="Local path to save the model",
)
def save(
    model_name,
    backend,
    adapter,
    adapter_name,
    tensor_parallel_size,
    local_model_path,
    storage_path,
):
    """
    Saves a model to the sllm-store's storage.

    This command is for adding new models to the sllm-store's local storage.
    """

    logger.info(f"Saving model {model_name} to {storage_path}")

    if adapter:
        config = AutoConfig.from_pretrained(
            os.path.join(storage_path, "transformers", model_name),
            trust_remote_code=True,
        )
        config.torch_dtype = torch.float16
        module = importlib.import_module("transformers")
        hf_model_cls = getattr(module, AutoModelForCausalLM)
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

    logger.info(f"Model {model_name} saved successfully to {storage_path}")


@cli.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub",
)
@click.option(
    "--backend", type=str, required=True, default="vllm", help="Backend"
)
@click.option(
    "--adapter", is_flag=True, help="Indicate if the model is an adapter"
)
@click.option("--adapter-name", type=str, help="Name of the LoRA adapter")
@click.option("--adapter-path", type=str, help="Path to the LoRA adapter")
@click.option(
    "--precision",
    type=str,
    default="int8",
    help="Precision of quantized model. Supports int8, fp4, and nf4",
)
@click.option(
    "--storage-path",
    required=True,
    default="./models",
    help="Local path to save the model",
)
def load(
    model_name,
    backend,
    adapter,
    adapter_name,
    adapter_path,
    precision,
    storage_path,
):
    """
    Loads a model from the sllm-store's storage.

    This command is for loading new models from the sllm-store's local storage.
    """

    logger.info(f"Loading model {model_name} from {storage_path}")

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
            model_full_path = os.path.join(storage_path, model_name)
            llm = LLM(
                model=model_full_path,
                load_format="serverless_llm",
                dtype="float16",
            )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
            outputs = llm.generate(prompts, sampling_params)

            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        elif backend == "transformers":
            # warm up the GPU
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                torch.ones(1).to(f"cuda:{i}")
                torch.cuda.synchronize()
            if adapter:
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
                    adapter_path,
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
                    quantization_config=quantization_config
                    if precision
                    else None,
                )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(
                "cuda"
            )
            generate_kwargs = {}
            if adapter and adapter_name:
                generate_kwargs["adapter_names"] = [adapter_name]
            outputs = model.generate(**inputs, **generate_kwargs)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        else:
            logger.error(f"Unsupported backend '{backend}'")
            sys.exit(1)

    except Exception as e:
        logger.error(
            f"Failed to load model or perform inference: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info(f"Model {model_name} loaded successfully from {storage_path}")


# Entry point for the 'sllm-store' CLI
def main():
    cli()
