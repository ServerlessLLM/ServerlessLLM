import argparse
import os
from sglang.srt.entrypoints.engine import Engine


def main():
    parser = argparse.ArgumentParser(
        description="Load ServerlessLLM model with SGLang and run inference."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (same as save path).",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./models",
        help="Local path to saved model.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    storage_path = args.storage_path
    model_path = os.path.join(storage_path, model_name)

    engine = Engine(
        model_path=model_path, tp_size=1, load_format="serverless_llm"
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    for prompt in prompts:
        result = engine.generate(prompt=prompt)
        print(f"Prompt: {prompt!r}, Generated text: {result['text']!r}")


if __name__ == "__main__":
    main()
