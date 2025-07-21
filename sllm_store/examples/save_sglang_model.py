import argparse
import os
import shutil
from typing import Optional
from huggingface_hub import snapshot_download
from sglang.srt.entrypoints.engine import Engine


def save_sllm_model(
    model_name: str,
    storage_path: str = "./models",
    local_model_path: Optional[str] = None,
    tp_size: int = 1,
    skip_save: bool = False,
):
    if local_model_path is None:
        print(f"🔄 Downloading model {model_name} from HuggingFace...")
        input_dir = snapshot_download(model_name)
    else:
        input_dir = local_model_path

    model_path = os.path.join(storage_path, model_name)
    os.makedirs(model_path, exist_ok=True)

    print(f"🚀 Initializing SGLang Engine with model path: {input_dir}")
    engine = Engine(model_path=input_dir, tp_size=tp_size)

    if not skip_save:
        print(f"💾 Saving SLLM state to {model_path} ...")
        engine.save_serverless_llm_state(path=model_path)
    else:
        print("⚠️ Skip saving SLLM state (debug mode)")

    print("📂 Copying tokenizer/config files ...")
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1] not in (
            ".bin",
            ".pt",
            ".safetensors",
        ):
            src_path = os.path.join(input_dir, file)
            dest_path = os.path.join(model_path, file)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy(src_path, dest_path)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save HuggingFace model to SLLM format using SGLang Engine."
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
        "--tp-size", type=int, default=1, help="Tensor parallel size."
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip saving serverless LLM state (for debugging).",
    )
    args = parser.parse_args()

    save_sllm_model(
        model_name=args.model_name,
        storage_path=args.storage_path,
        local_model_path=args.local_model_path,
        tp_size=args.tp_size,
        skip_save=args.skip_save,
    )
