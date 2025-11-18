import argparse
import time

import torch
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info
from sllm_store.transformers import load_model

parser = argparse.ArgumentParser(
    description="Load a saved Qwen2-VL model from ServerlessLLM storage."
)
parser.add_argument(
    "--model-name",
    type=str,
    default="Qwen/Qwen2-VL-2B-Instruct",
    help="Model name stored locally.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path where the model is stored.",
)

args = parser.parse_args()

model_name = args.model_name
storage_path = args.storage_path

# Warm up available GPUs.
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

start = time.time()
# Load via AutoModel so the right VLM implementation is selected automatically.
model = load_model(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    storage_path=storage_path,
    fully_parallel=True,
    hf_model_class="AutoModelForVision2Seq",
)
print(f"Model loading time: {time.time() - start:.2f}s")

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
