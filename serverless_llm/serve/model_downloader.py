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
import logging
import os
import shutil
from typing import Optional

import ray

logger = logging.getLogger("ray")


# def get_directory_size(directory):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             if not os.path.islink(file_path):
#                 total_size += os.path.getsize(file_path)
#     return total_size


@ray.remote(num_cpus=1)
def download_transformers_model(model_name: str, torch_dtype: str) -> bool:
    storage_path = os.getenv("STORAGE_PATH", "./models")
    model_dir = os.path.join(storage_path, "transformers", model_name)
    if os.path.exists(model_dir):
        logger.info(f"Model {model_name} already exists in {model_dir}")
        return

    if os.path.exists(model_dir):
        # model_size = get_directory_size(model_dir)
        logger.info(f"Model {model_name} already exists")
        return True

    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    logger.info(f"Downloading model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    from serverless_llm_store.transformers import save_model

    logger.info(f"Saving model {model_name} to {model_dir}")
    try:
        save_model(model, model_dir)
    except Exception as e:
        logger.error(f"Failed to save model {model_name}: {e}")
        shutil.rmtree(model_dir)
        raise RuntimeError(
            f"Failed to save model {model_name} for transformer backend: {e}"
        )

    # model_size = get_directory_size(model_dir)
    # logger.info(f"Model {model_name} (size: {model_size}) downloaded")

    return True


@ray.remote
class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        import gc
        import shutil
        from tempfile import TemporaryDirectory

        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM

        # set the model storage path
        storage_path = os.getenv("STORAGE_PATH", "./models")

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
            model_executer = llm_writer.llm_engine.model_executor
            # save the models in the ServerlessLLM format
            model_executer.save_serverless_llm_state(
                path=model_name, pattern=pattern, max_size=max_size
            )
            for file in os.listdir(input_dir):
                # Copy the metadata files into the output directory
                if os.path.splitext(file)[1] not in (
                    ".bin",
                    ".pt",
                    ".safetensors",
                ):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(storage_path, model_name, file)
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

        try:
            with TemporaryDirectory() as cache_dir:
                # download model from huggingface
                input_dir = snapshot_download(
                    model_name,
                    cache_dir=cache_dir,
                    allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt"],
                )
                logger.info(input_dir)
                _run_writer(input_dir, model_name)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            shutil.rmtree(os.path.join(storage_path, model_name))
            raise RuntimeError(
                f"Failed to save model {model_name} for vllm backend: {e}"
            )