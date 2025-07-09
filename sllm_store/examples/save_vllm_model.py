import argparse
import os
import shutil
from typing import Optional


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
            model_executer = llm_writer.llm_engine.model_executor
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


parser = argparse.ArgumentParser(
    description="Save a model from HuggingFace model hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub.",
)
parser.add_argument(
    "--local-model-path",
    type=str,
    required=False,
    help="Local path to the model snapshot.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path to save the model.",
)
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Tensor parallel size.",
)

args = parser.parse_args()

model_name = args.model_name
local_model_path = args.local_model_path
storage_path = args.storage_path
tensor_parallel_size = args.tensor_parallel_size

downloader = VllmModelDownloader()
downloader.download_vllm_model(
    model_name,
    "float16",
    tensor_parallel_size=tensor_parallel_size,
    storage_path=storage_path,
    local_model_path=local_model_path,
)
