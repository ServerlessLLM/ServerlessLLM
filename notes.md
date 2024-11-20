# serverlessllm notes

# small talk at meeting: 
* onboarding experience
  * i basically read the code and paper and talked to claude a lot
  * yao was very helpful ngl
* end-to-end automation pipeline, challenges/interesting things when working on it
  * most of the challenges were just learning the tools. like actions, docker, yaml code, working in ssh (and configuring my usual tools like tmux and conda)
  * also simplifying and understanding the work. originally i overcomplicated it way too much by thinking that i had to test every function individually. 
  * very helpful in that i learned how to write better commits
* next steps
  * i wanna try everything; every part of it so i get better at programming and understanding the systems side of ml
  * adding support for different models, especially multimodal and SAM
  * it's like implementing papers, but more useful and also forces me to write good code and commits
  * refining the testing pipeline - batching, more focused tests, that

## issue 83
* adding quantization methods in sllm store
* integration with bitsandbytes, gptq, awq
  * pre-quantization: quantization before saving as checkpoints
  * quantization while the model is being loaded - full precision loads quickly and gets quantized while being loaded into memory 

how can this be done: 
* flags? 
* 

(from alpin) AQLM, AWQ, Exllamav2, FP6_LLM, FP8, GPTQ, Marlin, GGUF, QuIP#, PTQ, SqueezeLLM

## issue #84
* testing that sllm works for ssms
* performing inference with an ssm-based model
* make a quickstart test for ssms

## issue #116
backends: 
* onnx
* tensorRT
* vllm
* torch/HF default (automodelforcausalLM)

tests: 
* onnx test  via onnx/backend/test -> run using pytest

todos later: 
* format the markdown so it renders a csv or a list using markdown formatting, so adding new models to the list is much easier

## issue #83
* add quanitzation support 
* bitsandbytes, gptq, awq integration
* either pre-convert model checkpoints into sllm-store format so you can load them in fast, or enable fast loading of original checkpoints while quantizing during runtime
* focus is bitsandbytes

## issue breakdown (so it's easier to read: 
* type: docuemntation/implement or test models/engine or systems work
* description
* tasks involved

## notes
* partial support means basic inference works, but further features such as streaming/batching/etc doesn't necessarily work
* backend is just how the model inference is actually ran (like how raw torch is different to onnx) 

```
curl -L https://github.com/nektos/act/releases/latest/download/act_Linux_x86_64.tar.gz -o act.tar.gz
tar -xzf act.tar.gz act

act testing (have to make a json for pull requests called event.json btw)
./act pull_request -e event.json     -W .github/workflows/inference_test.yml     --container-architecture linux/amd64     -P ubuntu-latest=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
```


```
<start the docker or whatever>
docker run -it --gpus all pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel /bin/bash
docker ps
docker exec -it <id> /bin/bash

apt-get update
apt-get install docker npm wget git

mkdir models
export MODEL_FOLDER=~/models

git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM

# now we split into two worldlines 
# head
conda create -n sllm python=3.10 -y
conda activate sllm
pip install -e .
cd sllm_store && rm -rf build

ray start --head --port=6379 --num-cpus=4 --num-gpus=0 \
--resources='{"control_node": 1}' --block

# worker
conda create -n sllm-worker python=3.10 -y
conda activate sllm-worker
pip install -e ".[worker]"
cd sllm_store && rm -rf build

# start head ray node 
conda activate sllm
export CUDA_VISIBLE_DEVICES=0
ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block

# start worker ray node 
conda activate sllm-worker
export CUDA_VISIBLE_DEVICES=0
sllm-store-server

# actual serving worldline
conda activate sllm
sllm-serve start

# testing worldline
export MODEL_FOLDER=/workspace/models (varies depending on where you start it) 
cd serverlessllm/tests/inference_test
python test_store.py
python test_inference.py
```

potential errors: 
* missing imports from the model packages
* gated repos
* missing config files
* missing HF token

testing cases:
* unsupported models
* models with other imports 
