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

import argparse
import asyncio
import os
import time
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

from sllm.serve.logger import init_logger
from sllm_store.transformers import load_model

logger = init_logger(__name__)

app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
model_name = None


def create_chat_request(
    messages,
    max_tokens=100,
    temperature=0.7,
    top_p=1.0,
    model=None,
    request_id=None,
    task_id=None,
):
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "model": model,
        "request_id": request_id,
        "task_id": task_id,
    }


def create_embedding_request(
    input_data, model=None, request_id=None, task_id=None
):
    return {
        "input": input_data,
        "model": model,
        "request_id": request_id,
        "task_id": task_id,
    }


def initialize_model(
    model_name_param: str,
    device_map: str = "auto",
    torch_dtype: str = "float16",
    hf_model_class: str = "AutoModelForCausalLM",
    quantization_config: Optional[Dict[str, Any]] = None,
):
    """Initialize the model and tokenizer using sllm_store."""
    global model, tokenizer, model_name

    model_name = model_name_param
    storage_path = os.getenv("STORAGE_PATH", "/models")
    model_path = os.path.join("transformers", model_name)
    tokenizer_path = os.path.join(storage_path, "transformers", model_name)

    logger.info(f"Loading model from {model_path}")

    # Convert torch_dtype string to actual dtype
    try:
        torch_dtype_obj = getattr(torch, torch_dtype)
    except AttributeError:
        logger.error(
            f"Invalid torch dtype: {torch_dtype}, defaulting to float16"
        )
        torch_dtype_obj = torch.float16

    # Load model using sllm_store
    model = load_model(
        model_path=model_path,
        device_map=device_map,
        torch_dtype=torch_dtype_obj,
        quantization_config=quantization_config,
        storage_path=storage_path,
        hf_model_class=hf_model_class,
    )

    # Load tokenizer (use same path as model)
    full_model_path = os.path.join(storage_path, model_path)
    tokenizer = AutoTokenizer.from_pretrained(full_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model and tokenizer loaded successfully for {model_name}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request):
    global model, tokenizer

    if not model or not tokenizer:
        return {"error": "Model not loaded"}

    try:
        # Use request_id from task_id if available
        response_id = (
            request.task_id or request.request_id or "chatcmpl-transformers"
        )

        # Convert messages to prompt
        prompt = ""
        for message in request.messages:
            if message.get("role") == "user":
                prompt += f"User: {message.get('content', '')}\n"
            elif message.get("role") == "assistant":
                prompt += f"Assistant: {message.get('content', '')}\n"
        prompt += "Assistant: "

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        # Move to same device as model
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            # Get the device of the first layer
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                device = f"cuda:{first_device}"
            else:
                device = first_device
            input_ids = input_ids.to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
                if request.temperature > 0
                else 1.0,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode response
        response_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
        )

        # Count tokens for usage
        prompt_tokens = input_ids.shape[-1]
        completion_tokens = (
            len(tokenizer.encode(response_text)) if response_text else 0
        )

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        return {"error": str(e)}


@app.post("/v1/embeddings")
async def embeddings(request):
    return {"error": "Embeddings not implemented for Transformers backend"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, help="Name of the model to load"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind to"
    )
    parser.add_argument(
        "--device_map", default="auto", help="Device map for model"
    )
    parser.add_argument("--torch_dtype", default="float16", help="Torch dtype")
    parser.add_argument(
        "--hf_model_class",
        default="AutoModelForCausalLM",
        help="HuggingFace model class",
    )

    args = parser.parse_args()

    # Initialize model on startup
    initialize_model(
        model_name_param=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        hf_model_class=args.hf_model_class,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)
