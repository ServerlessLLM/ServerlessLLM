from sllm_store.transformers import save_model

# Load a model from HuggingFace model hub.
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16)

model_folder = '/home/ryan/ServerlessLLM/models'
# Replace './models' with your local path.
save_model(model, model_folder)

import time
from sllm_store.transformers import load_model

# warm up the GPU
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

start = time.time()
model = load_model("facebook/opt-1.3b", device_map="auto", torch_dtype=torch.float16, storage_path=model_folder, fully_parallel=True)
# Please note the loading time depends on the model size and the hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt').to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
