import argparse
import os

from sglang.srt.entrypoints.engine import Engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model from ServerlessLLM storage using sglang."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name from ServerlessLLM storage.",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./models",
        help="Local path to the model storage.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights and activations.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph.",
    )
    parser.add_argument(
        "--tp-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size to launch the engine with.",
    )
    args = parser.parse_args()

    model_name = args.model_name
    storage_path = args.storage_path
    model_path = os.path.join(storage_path, model_name)

    engine = Engine(
        model_path=model_path,
        load_format="serverless_llm",
        dtype=args.dtype,
        disable_cuda_graph=args.disable_cuda_graph,
        tp_size=args.tp_size,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}
    outputs = engine.generate(prompts, sampling_params=sampling_params)

    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output["text"]
        print(
            f"""===============================\nPrompt: {prompt}\nGenerated text: {generated_text}\n===============================\n"""
        )
