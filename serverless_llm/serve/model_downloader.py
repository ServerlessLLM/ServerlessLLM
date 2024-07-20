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
from typing import Optional
import shutil

import ray

logger = logging.getLogger("ray")


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


@ray.remote(num_cpus=1)
def download_transformers_model(model_name: str, torch_dtype: str) -> int:
    storage_path = os.getenv("STORAGE_PATH", "./models")
    # TODO: storage_path = os.path.join(storage_path, "transformers")
    model_dir = os.path.join(storage_path, model_name)

    if os.path.exists(model_dir):
        model_size = get_directory_size(model_dir)
        logger.info(f"Model {model_name} (size: {model_size}) already exists")
        return model_size

    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    logger.info(f"Downloading model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    from serverless_llm_store import save_model

    logger.info(f"Saving model {model_name} to {model_dir}")
    save_model(model, model_dir)

    model_size = get_directory_size(model_dir)
    logger.info(f"Model {model_name} (size: {model_size}) downloaded")

    return model_size


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

        def _run_writer(input_dir, output_dir):
            llm_writer = LLM(
                model=input_dir,
                download_dir=input_dir,
                dtype=torch_dtype,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="mp",
            )
            model_executer = llm_writer.llm_engine.model_executor
            # TODO: change the `save_sharded_state` to `save_serverless_llm_state`
            model_executer.save_sharded_state(
                path=output_dir, pattern=pattern, max_size=max_size
            )
            for file in os.listdir(input_dir):
                if os.path.splitext(file)[1] not in (
                    ".bin",
                    ".pt",
                    ".safetensors",
                ):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(output_dir, file)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy(src_path, output_dir)
            del model_executer
            del llm_writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        storage_path = os.getenv("STORAGE_PATH", "./models")
        # TODO: storage_path = os.path.join(storage_path, "vllm")
        output_dir = os.path.join(storage_path, model_name)
        if os.path.exists(output_dir):
            model_size = get_directory_size(output_dir)
            print(f"Model {model_name} (size: {model_size}) already exists")
            return model_size
        # create the output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = snapshot_download(model_name, cache_dir=cache_dir)
                _run_writer(input_dir, output_dir)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            shutil.rmtree(output_dir)
            raise e

        model_size = get_directory_size(output_dir)
        print(f"Model {model_name} (size: {model_size}) downloaded")
        return model_size
