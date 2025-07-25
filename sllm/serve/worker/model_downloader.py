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
import importlib
import os
import shutil
from typing import Optional

import torch
from torch import nn
from transformers import AutoTokenizer

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


async def download_transformers_model(
    model_name: str,
    pretrained_model_name_or_path: str,
    torch_dtype: str,
    hf_model_class: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", "./models")
    model_path = os.path.join(storage_path, "transformers", model_name)
    tokenizer_path = os.path.join(
        storage_path, "transformers", model_name, "tokenizer"
    )

    if os.path.exists(model_path):
        logger.info(f"{model_path} already exists")
        return True

    try:
        torch_dtype = getattr(torch, torch_dtype)
    except AttributeError:
        logger.error(
            f"Invalid torch dtype: {torch_dtype}, defaulting to float16"
        )
        torch_dtype = torch.float16

    logger.info(f"Downloading {model_path}")

    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    model = hf_model_cls.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from sllm_store.transformers import save_model

    logger.info(f"Saving {model_path}")
    try:
        save_model(model, model_path)
        tokenizer.save_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to save {model_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {model_name} for transformer backend: {e}"
        )

    return True


async def download_lora_adapter(
    base_model_name: str,
    adapter_name: str,
    adapter_name_or_path: str,
    hf_model_class: str,
    torch_dtype: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", "./models")
    adapter_path = os.path.join(
        storage_path, "transformers", adapter_name_or_path
    )

    if os.path.exists(adapter_path):
        logger.info(f"{adapter_path} already exists")
        return True

    try:
        torch_dtype = getattr(torch, torch_dtype)
    except AttributeError:
        logger.error(
            f"Invalid torch dtype: {torch_dtype}, defaulting to float16"
        )
        torch_dtype = torch.float16

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        os.path.join(storage_path, "transformers", base_model_name),
        trust_remote_code=True,
    )
    config.torch_dtype = torch_dtype
    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    base_model = hf_model_cls.from_config(
        config,
        trust_remote_code=True,
    ).to(config.torch_dtype)

    logger.info(f"Downloading {adapter_path}")
    from peft import PeftModel

    try:
        model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise RuntimeError(f"LoRA adapter load failed: {e}")

    from sllm_store.transformers import save_lora

    logger.info(f"Saving {adapter_path}")
    try:
        save_lora(model, adapter_path)
    except Exception as e:
        logger.error(f"Failed to save {adapter_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {adapter_name} for transformer backend: {e}"
        )

    return True


class VllmModelDownloader:
    def __init__(self):
        pass

    async def download_vllm_model(
        self,
        model_name: str,
        pretrained_model_name_or_path: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        import gc
        from tempfile import TemporaryDirectory

        from huggingface_hub import snapshot_download
        from vllm import LLM

        # set the storage path
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(storage_path, "vllm", model_name)
        if os.path.exists(model_path):
            logger.info(f"{model_path} already exists")
            return

        cache_dir = TemporaryDirectory()
        try:
            if os.path.exists(pretrained_model_name_or_path):
                input_dir = pretrained_model_name_or_path
            else:
                # download from huggingface
                input_dir = snapshot_download(
                    model_name,
                    cache_dir=cache_dir.name,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.json",
                        "*.txt",
                    ],
                )
            logger.info(f"Loading model from {input_dir}")

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
            # model_executer = llm_writer.llm_engine.model_executor #V0
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
                    logger.info(src_path)
                    logger.info(dest_path)
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
        except Exception as e:
            logger.info(f"An error occurred while saving the model: {e}")
            # remove the output dir
            shutil.rmtree(model_path)
            raise RuntimeError(
                f"Failed to save {model_name} for vllm backend: {e}"
            )
        finally:
            cache_dir.cleanup()
