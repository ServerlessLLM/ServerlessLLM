import argparse
import os
import shutil
from typing import Optional


class SGLangModelSaver:
    def __init__(self):
        pass

    def save_from_hf(
        self,
        model_name: str,
        torch_dtype: str,
        storage_path: str = "./models",
        local_model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        device: Optional[str] = None,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """
        Load a model using sglang's loaders and save it in ServerlessLLM format.

        - If local_model_path is provided, use it as the model snapshot directory.
        - Otherwise, download a HF snapshot (limited to relevant files) into a temp cache.
        - Load weights on CPU to keep GPU memory usage minimal.
        - Save tensors via ServerlessLLM store format under STORAGE_PATH/model_name.
        - Copy non-weight metadata files into the output directory.
        """
        import gc
        from tempfile import TemporaryDirectory

        import torch
        from huggingface_hub import snapshot_download

        from sglang.srt.configs.load_config import LoadFormat

        # set the model storage path
        storage_path = os.getenv("STORAGE_PATH", storage_path)

        def _copy_metadata(src_dir: str, dst_dir: str) -> None:
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                ext = os.path.splitext(fname)[1]
                if ext in (".bin", ".pt", ".safetensors"):
                    continue
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                if os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)

        def _load_and_save(input_dir: str, save_name: str, tp_size: int, device: Optional[str], pattern: Optional[str], max_size: Optional[int]) -> None:
            # Launch an sglang Engine instance (in-process) and save via engine API
            from sglang.srt.entrypoints.engine import Engine

            out_dir = os.path.join(storage_path, save_name)
            os.makedirs(out_dir, exist_ok=True)

            # Start engine with minimal args; use CPU by default to minimize GPU usage
            engine_args = {
                "model_path": input_dir,
                "dtype": torch_dtype,
                "tp_size": max(1, int(tp_size)),
                "load_format": LoadFormat.AUTO.value,
                "log_level": "error",
                "skip_server_warmup": True,
                "disable_cuda_graph": True,
            }
            if device == "cpu":
                engine_args["device"] = "cpu"
            with Engine(**engine_args) as engine:
                engine.save_serverless_llm_state(path=out_dir, pattern=pattern, max_size=max_size)

            # Copy metadata files
            _copy_metadata(input_dir, out_dir)

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = local_model_path
                # download from huggingface
                if input_dir is None:
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
                _load_and_save(input_dir, model_name, tensor_parallel_size, device, pattern, max_size)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            try:
                shutil.rmtree(os.path.join(storage_path, model_name))
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to save {model_name} for sglang backend: {e}"
            ) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a model from HuggingFace model hub using sglang."
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
        "--dtype",
        type=str,
        default="auto",
        help="Torch dtype to use for loading (e.g., float16, bfloat16, auto).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size to launch the engine with.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for loading the model.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional filename pattern for shards (ignored by ServerlessLLM save).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional max shard size in bytes (ignored by ServerlessLLM save).",
    )

    args = parser.parse_args()

    model_name = args.model_name
    local_model_path = args.local_model_path
    storage_path = args.storage_path
    torch_dtype = args.dtype
    tensor_parallel_size = args.tensor_parallel_size
    device = args.device
    pattern = args.pattern
    max_size = args.max_size

    saver = SGLangModelSaver()
    saver.save_from_hf(
        model_name=model_name,
        torch_dtype=torch_dtype,
        storage_path=storage_path,
        local_model_path=local_model_path,
        tensor_parallel_size=tensor_parallel_size,
        device=device,
        pattern=pattern,
        max_size=max_size,
    )


